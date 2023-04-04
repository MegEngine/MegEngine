#include "megbrain/imperative/physical_tensor.h"
#include "megbrain/common.h"
#include "megbrain/comp_node.h"
#include "megbrain/imperative.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/resource_manager.h"

#include "./event_pool.h"

#include "./profiler/events.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifndef WIN32
#include <pthread.h>
#endif

#include "range/v3/all.hpp"

namespace views = ranges::views;

namespace mgb {
namespace imperative {

namespace {

struct CompNodeHash {
    auto operator()(CompNode cn) const { return mgb::hash(cn); }
};

template <typename T>
struct NoThrowMovable : T {
    using T::T;
    NoThrowMovable(NoThrowMovable&&) noexcept = default;
};

template <typename... Ts>
using Map = NoThrowMovable<std::map<Ts...>>;

class CompNodeSyncManager {
    //! synchronization information for each compnode
    struct CompNodeData {
        template <typename T>
        class ReleaseQueue {
            Map<uint64_t, T> map;

        public:
            template <typename A>
            void emplace(uint64_t t, A&& a) {
                map.emplace_hint(map.end(), t, std::forward<A>(a));
            }
            void release(uint64_t t) { map.erase(map.begin(), map.upper_bound(t)); }
        };

        //! next virtual event
        uint64_t next = 1;
        //! last completed virtual event
        uint64_t completed = 0;
        //! virtual event to real event, the map is ORDERLY, that means, in events {a:
        //! e1, b: e2}, if b>a, e2 is later than e1
        Map<uint64_t, EventPtr> events;
        //! ordering information at some virtual events: what virtual events on other
        //! comp nodes is _sequenced before_ this virtual event. concretely, the key is
        //! the virtual event id `t` on this comp node, and the value is the ordering
        //! information. the ordering is a vector, the index of the vector is the
        //! compnode id, and the value of the vector is the event id which the event `t`
        //! wait for on the compnode
        Map<uint64_t, std::vector<uint64_t>> ordering;
        //! in megengine, each compnode manager their own resources, and the resource
        //! can be used by other compnode. for example, we have compnodes: cn1, cn2,
        //! cn1 can allocate a tensor alpha, and the tensor alpha can be used on cn2. we
        //! want to release the tensor alpha after cn2 used, and the release maybe
        //! asynchronized. so in each CompNodeData cn-i, we setup release queues for
        //! each compnode. if other compnode cn-j use the resource on cn-i, we hold the
        //! resource (refcnt) in the release queue cn-j in CompNodeData cn-i. if cn-j
        //! have complete its task and do not need the resource on cn-i, we release the
        //! resource(refcnt) in the queue cn-j in CompNodeData cn-i. here is the release
        //! queues, which is a vector. the index of the vector is the compnode id, and
        //! the value of the vector is the release queue
        std::vector<ReleaseQueue<BlobPtr>> release_queues;
        //! different from the device resource, if the host resource is used by a
        //! compnode, then the compnode is responsible for the resource release rather
        //! than the host. that means if cn-i used a host resource, then cn-i add the
        //! resource(refcnt) to its host_release_queue. if the task is completed, cn-i
        //! remove the resource in its host_release_queue
        ReleaseQueue<HostTensorStorage::RawStorage> host_release_queue;
    };

    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_should_stop = false;
    //! to realize the async release, we create a new thread to polling and release
    std::thread m_polling_thread;
    std::unordered_map<CompNode, size_t, CompNodeHash> m_cn2id;
    std::vector<CompNodeData> m_cndata;

    auto do_record(CompNode cn, size_t cnid, std::unique_lock<std::mutex>& lock) {
        // CAUSION: don't keep reference across locking boundary
        lock.unlock();
        auto e = EventPool::without_timer().alloc(cn);
        e->record();
        lock.lock();
        auto& cndata = m_cndata[cnid];
        //! add the event to the event map of compnodedata[cnid]
        return cndata.events.emplace_hint(cndata.events.end(), cndata.next++, e);
    }

    // get a real event t' such that t <= t'
    std::pair<uint64_t, CompNode::Event*> get_event(
            CompNode cn, size_t cnid, uint64_t t, std::unique_lock<std::mutex>& lock) {
        auto& cndata = m_cndata[cnid];
        auto it = cndata.events.lower_bound(t);
        if (it == cndata.events.end()) {
            it = do_record(cn, cnid, lock);
        }
        return {it->first, it->second.get()};
    }

    size_t get_cnid_unsafe(CompNode cn) {
        auto [it, unseen] = m_cn2id.try_emplace(cn, m_cndata.size());
        if (unseen) {
            m_cndata.emplace_back();
        }
        return it->second;
    }

    //! the implementation of polling thread
    void monitor_events() {
#if defined(__APPLE__)
        pthread_setname_np("CompNodeSync");
#elif defined(__unix__)
        pthread_setname_np(pthread_self(), "CompNodeSync");
#endif

        // poll events in rounds. sleep for a fixed duration between rounds.
        // number of events to query is decided by the number of successful queries in
        // last round, independently for each comp node:
        // a. all -> double
        // b. 0 -> 1
        // c. otherwise -> #successful

        struct Item {
            size_t cnid;
            decltype(CompNodeData::events)::iterator it;
        };

        struct Stat {
            size_t num_success = 0;
            size_t num_attempts = 0;
            // iterator to the last finished event
            decltype(CompNodeData::events)::iterator it;
        };

        std::vector<Stat> stats;
        std::vector<Item> todos;
        std::vector<bool> updated;
        std::unique_lock lock(m_mtx);
        for (;;) {
            updated.clear();
            updated.resize(m_cndata.size(), false);
            // copy events to a temporary storage so that we may unlock while polling
            stats.resize(m_cndata.size());
            // for each compnode
            for (size_t cnid = 0; cnid < m_cndata.size(); ++cnid) {
                // decide max number of events to query for each compnode
                // rule c: #successful
                size_t n = stats[cnid].num_success;
                if (n == stats[cnid].num_attempts) {
                    // rule a: double
                    n *= 2;
                }
                if (n == 0) {
                    // rule b: 1
                    n = 1;
                }
                // now copy upto n events to todos for each compnode
                auto& events = m_cndata[cnid].events;
                size_t i = 0;
                for (auto it = events.begin(); i < n && it != events.end(); ++i, ++it) {
                    todos.push_back({cnid, it});
                }
                // reset stats for this round
                stats[cnid].num_success = 0;
                stats[cnid].num_attempts = n;
            }

            lock.unlock();

            bool last_result = false;
            size_t last_cnid = -1;
            for (auto item : todos) {
                if (item.cnid == last_cnid && !last_result) {
                    // previous failed, this one almost certainly should fail
                    continue;
                }
                last_cnid = item.cnid;
                last_result = item.it->second->finished();
                if (last_result) {
                    stats[item.cnid].num_success++;
                    // the last finished event iterator
                    stats[item.cnid].it = item.it;
                }
            }
            todos.clear();

            lock.lock();

            // update completed
            for (auto [cnid, stat] : views::enumerate(stats)) {
                if (stat.num_success == 0) {
                    continue;
                }
                // the last finished event id of compnode `cnid`
                auto t = stat.it->first;
                auto& cndata = m_cndata[cnid];
                // update the complete information of compnode `cnid`
                if (cndata.completed < t) {
                    cndata.completed = t;
                    updated[cnid] = true;
                    // also propagate by the transitive <= relation to ensure that
                    // we can safely delete ordering information without performance
                    // degradation even if some completion events are missed by our query
                    auto it = cndata.ordering.upper_bound(t);
                    if (it != cndata.ordering.begin()) {
                        // get the ordering information of event t, if event t is
                        // finished, that means the events on other compnode which event
                        // t wait for are also finished, so we can update these compnode
                        // complete information
                        it = std::prev(it);
                        // for each compnode and event which event t wait for
                        for (auto [cnid, t] : views::enumerate(it->second)) {
                            auto& cndata = m_cndata[cnid];
                            if (cndata.completed < t) {
                                cndata.completed = t;
                                updated[cnid] = true;
                            }
                        }
                    }
                }
            }

            // release dev storage
            // receiver is the resource owner and the releaser is the resource user
            // for each resource owner
            for (size_t receiver_cnid = 0; receiver_cnid < m_cndata.size();
                 ++receiver_cnid) {
                // for each resource user
                for (size_t releaser_cnid = 0;
                     releaser_cnid < m_cndata[receiver_cnid].release_queues.size();
                     ++releaser_cnid) {
                    // if the user has not updated its completed, that means no event
                    // are finished on resource user, the resource owner still should
                    // hold the resource for these unfinished events, skip
                    if (!(releaser_cnid < updated.size() && updated[releaser_cnid])) {
                        continue;
                    }
                    // if some events are finished on resource user, the resource owner
                    // does not need hold resource for these events, release them
                    auto& q = m_cndata[receiver_cnid].release_queues[releaser_cnid];
                    q.release(m_cndata[releaser_cnid].completed);
                }
            }

            // for each compnode
            for (size_t cnid = 0; cnid < updated.size(); ++cnid) {
                if (!updated[cnid]) {
                    continue;
                }
                auto& cndata = m_cndata[cnid];
                // if event `t` on compnode `cnid` is finished, the host resource which
                // reserve for the events `<=t` can be release
                auto t = cndata.completed;
                // release host storage
                cndata.host_release_queue.release(t);
                // remove completed events
                [&](auto& map) {
                    map.erase(map.begin(), map.upper_bound(t));
                }(cndata.events);
                // delete ordering information
                [&](auto& map) {
                    map.erase(map.begin(), map.upper_bound(t));
                }(cndata.ordering);
            }

            using namespace std::literals;
            if (m_cv.wait_for(lock, 10us, [&] { return m_should_stop; })) {
                return;
            }
        }
    }

    CompNodeSyncManager() {
        m_polling_thread = std::thread([this] { monitor_events(); });
    }

public:
    ~CompNodeSyncManager() {
        {
            MGB_LOCK_GUARD(m_mtx);
            m_should_stop = true;
            m_cv.notify_all();
        }
        m_polling_thread.join();
    }

    static CompNodeSyncManager& inst();

    //! record an event on cn
    uint64_t record(CompNode cn, bool doitnow = false) {
        std::unique_lock lock(m_mtx);
        auto cnid = get_cnid_unsafe(cn);
        if (doitnow) {
            return do_record(cn, cnid, lock)->first;
        }
        //! if we do not DOITNOW, we only increase the counter, and then the get_event()
        //! function will do the actual recording
        return m_cndata[cnid].next++;
    }

    //! try to async release a resource until `cn` complete event `t`
    void async_release(CompNode cn, uint64_t t, BlobPtr blob) {
        MGB_LOCK_GUARD(m_mtx);
        //! the releaser can be seen as a resource user, the receiver can be seen as a
        //! resource owner so we represent releaser as user and represent
        //! receiver as owner
        auto releaser_cnid = get_cnid_unsafe(cn);
        //! if the user has complete event t, so we do not need to hold a blob
        //! reference
        if (t <= m_cndata[releaser_cnid].completed) {
            return;
        }
        //! the owner is the compnode of the blob
        auto receiver_cnid = get_cnid_unsafe(blob->comp_node());
        //! resource owner hold queues for each resource user, which
        //! is represented as a vector, the index is the user compnode id, and
        //! the value is the queue which hold the blobs for the correspond user
        //! compnode to use until the event `t` of user compnode completed
        auto& qs = m_cndata[receiver_cnid].release_queues;
        if (releaser_cnid >= qs.size()) {
            qs.resize(releaser_cnid + 1);
        }
        //! get the releaser/user queue of compnode `releaser_cnid`
        auto& q = qs[releaser_cnid];
        //! add the blob and the event `t` to the compnode `releaser_cnid` queue
        q.emplace(t, std::move(blob));
    }

    void async_release(CompNode cn, uint64_t t, HostTensorStorage::RawStorage storage) {
        MGB_LOCK_GUARD(m_mtx);
        // the releaser is the resource user
        auto releaser_cnid = get_cnid_unsafe(cn);
        // if the resource user have complete event `t`, do not hold anything
        if (t <= m_cndata[releaser_cnid].completed) {
            return;
        }
        // hold the host tensor resource in the user compnode host_release_queue
        auto& q = m_cndata[releaser_cnid].host_release_queue;
        q.emplace(t, std::move(storage));
    }

    //! the `waiter` compnode wait the completion of the event `t` on `waitee` compnode
    void device_wait(CompNode waiter, CompNode waitee, uint64_t t) {
        std::unique_lock lock(m_mtx);

        auto waiter_id = get_cnid_unsafe(waiter);
        auto waitee_id = get_cnid_unsafe(waitee);
        auto& waiter_data = m_cndata.at(waiter_id);
        auto& waitee_data = m_cndata.at(waitee_id);

        //! waitee has already completed the event t, so waiter does not need to wait
        if (t <= waitee_data.completed) {
            return;
        }

        //! the ordering are orderly, so the rbegin() of ordering is the last event of
        //! compnode. if the last event of waiter is alreadty waiting for the event
        //! which is later than event t(>=t) of waitee, we do not need to add the new
        //! device_wait information
        if (waiter_data.ordering.size() &&
            waitee_id < waiter_data.ordering.rbegin()->second.size() &&
            t <= waiter_data.ordering.rbegin()->second[waitee_id]) {
            return;
        }

        //! get the virtual event t and the corresponding real event of waitee
        //! you can think of the t_waitee as the virtual event t on the waitee compnode
        auto [t_waitee, e] = get_event(waitee, waitee_id, t, lock);

        //! DO NOT unlock around this line! Event* could be invalidated!
        e->device_wait_by(waiter);

        //! add a new event t_waiter on the waiter compnode, and t_waiter event wait for
        //! t_waitee event
        auto t_waiter = waiter_data.next++;
        //! try to add an ordering information to the waiter.ordering, this ordering
        //! describe the event t_waiter wait what events on other compnodes, the
        //! ordering is a vector, the index of the vector is the compnode id, and
        //! the value of the vector is the event id which the event t_waiter wait for
        //! on the compnode
        std::vector<uint64_t> t_waiter_ordering(m_cndata.size(), 0);
        if (!waiter_data.ordering.empty()) {
            //! if the last event of waiter has already waited for some events, and the
            //! new event t_waiter is later than the last event,so the new event
            //! t_waiter also need to wait for these events
            auto& o = waiter_data.ordering.rbegin()->second;
            std::copy(o.begin(), o.end(), t_waiter_ordering.begin());
        }
        //! on the waitee compnode, the new event t_waiter wait for the event t_waitee
        t_waiter_ordering[waitee_id] = t_waitee;
        //! on the waiter compnode itself, the new event t_waiter wait for itself
        t_waiter_ordering[waiter_id] = t_waiter;
        {
            //! get all the events ordering information on the waitee compnode which are
            //! ahead of t_waitee, event t_waiter should also wait for all the events
            //! which are ahead of t_waitee on the waitee compnode,
            auto it = waitee_data.ordering.upper_bound(t_waitee);
            if (it != waitee_data.ordering.begin()) {
                //! these events ahead of t_waitee maybe wait for other events which is
                //! recorded in their ordering information, so we update the ordering
                //! information of event t_waiter according to the ordering information
                //! of these events
                for (auto [a, b] :
                     views::zip(t_waiter_ordering, std::prev(it)->second)) {
                    static_assert(std::is_lvalue_reference_v<decltype(a)>);
                    a = std::max(a, b);
                }
            }
        }
        //! add the new event t_waiter and its ordering information to the waiter compnode
        waiter_data.ordering.emplace_hint(
                waiter_data.ordering.end(), t_waiter, t_waiter_ordering);

        //! the event t_waiter is completed because the above code
        //! `e->wait_device_by()`, that means all its depentent events on other compnode
        //! are also completed, so we can release
        for (auto [t, q] : views::zip(t_waiter_ordering, waiter_data.release_queues)) {
            q.release(t);
        }
    }
};

CompNodeSyncManager& CompNodeSyncManager::inst() {
    static std::mutex mtx;
    static std::unique_ptr<CompNodeSyncManager> inst;

    struct Guard final : CompNodeDepedentObject {
        std::shared_ptr<void> on_comp_node_finalize() override {
            MGB_LOCK_GUARD(mtx);
            inst.reset();
            return {};
        }
    };

    static std::optional<Guard> guard;

#ifndef WIN32
    static bool broken = false;
    static struct ForkGuard {
        ForkGuard() {
            mgb_assert(0 == pthread_atfork(NULL, NULL, [] {
                           if (inst) {
                               inst.release();  // deliberate leak, unfixable
                               broken = true;
                           }
                       }));
        }
    } fork_guard;
#endif

    MGB_LOCK_GUARD(mtx);
    if (!inst) {
#ifndef WIN32
        mgb_assert(!broken);
#endif
        EventPool::without_timer();
        inst.reset(new CompNodeSyncManager);
        guard.emplace();
    }
    return *inst;
}

}  // namespace

uint64_t record_event(CompNode cn, bool doitnow) {
    return CompNodeSyncManager::inst().record(cn, doitnow);
}

void device_wait_event(CompNode waiter, CompNode waitee, uint64_t event) {
    CompNodeSyncManager::inst().device_wait(waiter, waitee, event);
}

void async_release(CompNode cn, uint64_t event, BlobPtr blob) {
    CompNodeSyncManager::inst().async_release(cn, event, std::move(blob));
}

void async_release(CompNode cn, uint64_t event, HostTensorStorage::RawStorage storage) {
    CompNodeSyncManager::inst().async_release(cn, event, std::move(storage));
}

void EventDeleter::operator()(CompNode::Event* event) {
    EventPool::without_timer().free(event);
}

namespace {
std::atomic_uint64_t next_blob_id = 0;
}

OwnedBlob::OwnedBlob(const DeviceTensorStorage& s)
        : Blob(s.comp_node(), s.size() + s.offset()),
          m_storage{s.raw_storage()},
          m_id{next_blob_id++} {
    BlobManager::inst()->register_blob(this);
}

OwnedBlob::OwnedBlob(CompNode cn, size_t sz)
        : Blob(cn, sz), m_storage{}, m_id{next_blob_id++} {
    BlobManager::inst()->register_blob(this);
}

OwnedBlob::~OwnedBlob() {
    BlobManager::inst()->unregister_blob(this);
}

const Blob::RawStorage& OwnedBlob::storage() {
    if (!m_storage && m_size) {
        BlobManager::inst()->alloc_with_defrag(this, m_size);
    }
    return m_storage;
}

BlobPtr OwnedBlob::borrow_to(CompNode cn) {
    return std::make_shared<BorrowedBlob>(
            cn, std::static_pointer_cast<OwnedBlob>(shared_from_this()));
}

bool OwnedBlob::storage_is_unique() {
    return m_storage.unique();
}

void* OwnedBlob::raw_ptr_not_for_readwrite() {
    return m_storage.get();
}

BorrowedBlob::BorrowedBlob(CompNode cn, std::shared_ptr<OwnedBlob> owner)
        : Blob(cn, owner->size()),
          m_owner(std::move(owner)),
          m_event(record_event(m_owner->comp_node(), true)) {}

BorrowedBlob::~BorrowedBlob() {
    async_release(m_comp_node, record_event(m_comp_node, true), std::move(m_owner));
}

const Blob::RawStorage& BorrowedBlob::storage() {
    {
        MGB_LOCK_GUARD(m_mtx);
        if (!m_initialized) {
            device_wait_event(m_comp_node, m_owner->comp_node(), m_event);
            m_initialized = true;
        }
    }
    return m_owner->storage();
}

BlobPtr BorrowedBlob::borrow_to(CompNode cn) {
    return std::make_shared<BorrowedBlob>(cn, m_owner);
}

bool BorrowedBlob::storage_is_unique() {
    return m_owner.unique() && m_owner->storage_is_unique();
}

void* BorrowedBlob::raw_ptr_not_for_readwrite() {
    return m_owner->raw_ptr_not_for_readwrite();
}

Tensor::Tensor(
        BlobPtr blob, const TensorLayout& layout, size_t offset, const HostTensorND& hv)
        : m_cn(blob->comp_node()),
          m_shape(layout),
          m_dtype(layout.dtype),
          m_layout(layout),
          m_blob(std::move(blob)),
          m_offset(offset),
          m_value(hv) {}

Tensor::Tensor(const HostTensorND& hv) : Tensor(hv.layout(), hv.comp_node()) {
    constexpr int size_threshold = TensorShape::MAX_NDIM;
    size_t nr_elems = hv.layout().total_nr_elems();
    if (nr_elems <= size_threshold) {
        m_value = hv;
    }
    if (nr_elems) {
        MGB_RECORD_EVENT(
                profiler::HostToDeviceEvent, hv.layout(), hv.comp_node(), hv.raw_ptr(),
                dev_tensor().raw_ptr());
        dev_tensor(false).copy_from_fixlayout(hv);
        // even though hv is saved in m_value, Tensor itself could be
        // released before copy completes
        MGB_RECORD_EVENT(
                profiler::HostToDeviceFinishEvent, hv.layout(), hv.comp_node(),
                hv.raw_ptr(), dev_tensor().raw_ptr());
        async_release(hv);
    }
}

Tensor::Tensor(const DeviceTensorND& dv, const HostTensorND& hv)
        : m_offset(dv.storage().offset()),
          m_cn(dv.comp_node()),
          m_shape(dv.layout()),
          m_dtype(dv.layout().dtype),
          m_blob(Blob::make(dv.storage())),
          m_layout(dv.layout()) {
    if (!hv.empty()) {
        mgb_assert(dv.comp_node() == hv.comp_node());
        mgb_assert(dv.dtype() == hv.dtype());
        mgb_assert(dv.shape().eq_shape(hv.shape()));
        m_value = hv;
    }
}

Tensor::Tensor(const TensorLayout& layout, const CompNode& cn)
        : m_layout{layout},
          m_blob{Blob::make(cn, layout.span().dist_byte())},
          m_offset{0},
          m_cn(cn),
          m_shape(layout),
          m_dtype(layout.dtype) {}

Tensor::Tensor(const BlobPtr blob, const size_t offset, const TensorLayout& layout)
        : m_layout{layout},
          m_blob{blob},
          m_offset{offset},
          m_cn(blob->comp_node()),
          m_shape(layout),
          m_dtype(layout.dtype) {}

TensorPtr Tensor::make(const HostTensorND& hv) {
    auto&& blob = MultiCNConstTensorCache::inst().lookup(hv);
    if (blob) {
        return make(std::forward<decltype(blob)>(blob), hv.layout(), hv);
    }
    return std::make_shared<Tensor>(hv);
}

void Tensor::to_contiguous_inplace(VarNode::LayoutConstraintCallback& layout_checker) {
    MGB_LOCK_GUARD(m_blob_mtx);
    if (!m_layout.is_empty() && !layout_checker(m_layout)) {
        DeviceTensorStorage storage;
        storage.reset(m_cn, m_blob->size(), m_blob->storage());
        storage = storage.sub(m_offset);
        DeviceTensorND dv;
        dv.reset(storage, m_layout);

        DeviceTensorND dv_contig;
        dv_contig.copy_from(dv);
        m_layout = dv_contig.layout();
        std::atomic_store(&m_blob, BlobPtr(Blob::make(dv_contig.storage())));
        mgb_assert(m_layout.is_contiguous());
        m_offset = 0;
    }
}

void Tensor::to_contiguous_inplace() {
    static VarNode::LayoutConstraintCallback default_cb =
            [](const TensorLayout& layout) { return layout.is_contiguous(); };
    to_contiguous_inplace(default_cb);
}

void Tensor::assign_from_dev_tensor(DeviceTensorND dv) {
    MGB_LOCK_GUARD(m_blob_mtx);
    std::atomic_store(&m_blob, BlobPtr(Blob::make(dv.storage())));
    m_offset = dv.storage().offset();
    m_layout = dv.layout();
}

DeviceTensorND Tensor::dev_tensor(bool contiguous) {
    mgb_assert(m_blob, "uninitialized tensor.");
    if (contiguous) {
        to_contiguous_inplace();
    }
    MGB_LOCK_GUARD(m_blob_mtx);
    DeviceTensorStorage storage;
    storage.reset(m_cn, m_blob->size(), m_blob->storage());
    storage = storage.sub(m_offset);
    DeviceTensorND ret;
    ret.reset(storage, m_layout);
    return ret;
}

bool Tensor::empty() {
    return !m_blob->size();
}

DnnTensorND Tensor::dnn_tensor() {
    mgb_assert(m_blob, "uninitialized tensor.");
    mgb_assert(m_layout.ndim, "dnn don't support scalar");
    return DnnTensorND{m_layout, m_blob->storage(), m_offset};
}

DnnTensorND Tensor::dnn_tensor(TensorShape new_shape) {
    mgb_assert(m_blob, "uninitialized tensor.");
    return DnnTensorND{m_layout.reshape(new_shape), m_blob->storage(), m_offset};
}

void Tensor::fetch_value() {
    MGB_LOCK_GUARD(m_value_mtx);
    if (m_value.empty()) {
        m_value.copy_from(dev_tensor(false));
        m_value_ready.reset(EventPool::without_timer().alloc(comp_node()));
        m_value_ready->record();
    }
}

bool Tensor::value_fetched() {
    MGB_LOCK_GUARD(m_value_mtx);
    return m_value.layout().ndim != 0;
}

const HostTensorND& Tensor::get_value() {
    fetch_value();
    if (m_value_ready) {
        m_value_ready->host_wait();
    }
    return m_value;
}

const HostTensorND* Tensor::try_get_value() {
    MGB_LOCK_GUARD(m_value_mtx);
    if (!m_value.empty() && (!m_value_ready || m_value_ready->finished())) {
        return &m_value;
    }
    return nullptr;
}

TensorPtr Tensor::make_scalar(DTypeScalar value, CompNode cn) {
    HostTensorND hv{cn, value.dtype()};
    hv.resize({1});
    memcpy(hv.raw_ptr(), value.storage(), value.dtype().size(1));
    return make(hv);
}

TensorPtr Tensor::sub(size_t offset, TensorShape shape) {
    TensorLayout layout(shape, m_dtype);
    return Tensor::make(m_blob, offset + m_offset, layout);
}

uint64_t Tensor::get_ready_event() {
    if (m_produced_at == 0) {
        m_produced_at = record_event(comp_node());
    }
    return m_produced_at;
}

bool Tensor::storage_is_unique() {
    return m_blob.unique() && m_blob->storage_is_unique();
}

void Tensor::static_initialize() {
    EventPool::with_timer();
    EventPool::without_timer();
    MultiCNConstTensorCache::inst();
}

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
