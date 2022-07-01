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
    struct CompNodeData {
        template <typename T>
        class ReleaseQueue {
            Map<uint64_t, T> map;

        public:
            template <typename A>
            void emplace(uint64_t t, A&& a) {
                map.emplace_hint(map.end(), t, std::forward<A>(a));
            }
            void release(uint64_t t) {
                auto it = map.upper_bound(t);
                map.erase(map.begin(), it);
            }
        };

        //! next virtual event
        uint64_t next = 1;
        //! last completed virtual event
        uint64_t completed = 0;
        //! virtual event to real event
        Map<uint64_t, EventPtr> events;
        //! ordering information at some virtual events:
        //! what virtual events on other comp nodes is _sequenced before_ this virtual
        //! event
        Map<uint64_t, std::vector<uint64_t>> ordering;
        //! release queue for dev storage, keyed by releaser. this comp node is the
        //! **receiver**
        std::vector<ReleaseQueue<BlobPtr>> release_queues;
        //! release queue for host storage. this comp node is the **releaser**
        ReleaseQueue<HostTensorStorage::RawStorage> host_release_queue;
    };

    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_should_stop = false;
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
        return cndata.events.emplace_hint(cndata.events.end(), cndata.next++, e);
    }

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
        std::unique_lock lock(m_mtx);
        for (;;) {
            // copy events to a temporary storage so that we may unlock while polling
            stats.resize(m_cndata.size());
            for (size_t cnid = 0; cnid < m_cndata.size(); ++cnid) {
                // decide max number of events to query
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
                // now copy upto n events
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
                    stats[item.cnid].it = item.it;
                }
            }
            todos.clear();

            lock.lock();

            // release dev storage
            for (size_t receiver_cnid = 0; receiver_cnid < m_cndata.size();
                 ++receiver_cnid) {
                for (size_t releaser_cnid = 0;
                     releaser_cnid < m_cndata[receiver_cnid].release_queues.size();
                     ++releaser_cnid) {
                    if (releaser_cnid >= stats.size() ||
                        stats[releaser_cnid].num_success == 0) {
                        continue;
                    }
                    auto& q = m_cndata[receiver_cnid].release_queues[releaser_cnid];
                    q.release(stats[releaser_cnid].it->first);
                }
            }

            for (size_t cnid = 0; cnid < stats.size(); ++cnid) {
                if (stats[cnid].num_success == 0) {
                    continue;
                }
                auto& cndata = m_cndata[cnid];
                auto it = stats[cnid].it;
                auto t = it->first;
                // update completed
                cndata.completed = t;
                // release host storage
                cndata.host_release_queue.release(t);
                // remove completed events
                auto& events = cndata.events;
                events.erase(events.begin(), std::next(it));
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

    uint64_t record(CompNode cn, bool doitnow = false) {
        std::unique_lock lock(m_mtx);
        auto cnid = get_cnid_unsafe(cn);
        if (doitnow) {
            return do_record(cn, cnid, lock)->first;
        }
        return m_cndata[cnid].next++;
    }

    void async_release(CompNode cn, uint64_t t, BlobPtr blob) {
        MGB_LOCK_GUARD(m_mtx);
        auto releaser_cnid = get_cnid_unsafe(cn);
        if (t <= m_cndata[releaser_cnid].completed) {
            return;
        }
        auto receiver_cnid = get_cnid_unsafe(blob->comp_node());
        auto& qs = m_cndata[receiver_cnid].release_queues;
        if (releaser_cnid >= qs.size()) {
            qs.resize(releaser_cnid + 1);
        }
        auto& q = qs[releaser_cnid];
        q.emplace(t, std::move(blob));
    }

    void async_release(CompNode cn, uint64_t t, HostTensorStorage::RawStorage storage) {
        MGB_LOCK_GUARD(m_mtx);
        auto releaser_cnid = get_cnid_unsafe(cn);
        if (t <= m_cndata[releaser_cnid].completed) {
            return;
        }
        auto& q = m_cndata[releaser_cnid].host_release_queue;
        q.emplace(t, std::move(storage));
    }

    void device_wait(CompNode waiter, CompNode waitee, uint64_t t) {
        std::unique_lock lock(m_mtx);

        auto waiter_id = get_cnid_unsafe(waiter);
        auto waitee_id = get_cnid_unsafe(waitee);
        auto& waiter_data = m_cndata.at(waiter_id);
        auto& waitee_data = m_cndata.at(waitee_id);
        auto [t_waitee, e] = get_event(waitee, waitee_id, t, lock);

        // DO NOT unlock around this line! Event* could be invalidated!
        e->device_wait_by(waiter);

        auto t_waiter = waiter_data.next++;
        std::vector<uint64_t> ordering(m_cndata.size(), 0);
        if (!waiter_data.ordering.empty()) {
            auto& o = waiter_data.ordering.rbegin()->second;
            std::copy(o.begin(), o.end(), ordering.begin());
        }
        ordering[waitee_id] = t_waitee;
        ordering[waiter_id] = t_waiter;
        {
            auto it = waitee_data.ordering.lower_bound(t_waitee);
            if (it != waitee_data.ordering.begin()) {
                for (auto [a, b] : views::zip(ordering, std::prev(it)->second)) {
                    static_assert(std::is_lvalue_reference_v<decltype(a)>);
                    a = std::max(a, b);
                }
            }
        }
        waiter_data.ordering.emplace_hint(
                waiter_data.ordering.end(), t_waiter, ordering);

        for (auto [t, q] : views::zip(ordering, waiter_data.release_queues)) {
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
