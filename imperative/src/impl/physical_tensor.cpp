/**
 * \file imperative/src/impl/physical_tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative.h"
#include "megbrain/imperative/blob_manager.h"
#include "./event_pool.h"
#include <mutex>

namespace mgb {
namespace imperative {

namespace {

class AsyncReleaser : public CompNodeDepedentObject {
    struct WaiterParam {
        CompNode cn;
        CompNode::Event* event;
        BlobPtr blob;
        HostTensorStorage::RawStorage storage;
    };
    class Waiter final : public AsyncQueueSC<WaiterParam, Waiter> {
        AsyncReleaser* m_par_releaser;

    public:
        // disable busy wait by set max_spin=0 to save CPU cycle
        Waiter(AsyncReleaser* releaser)
                : AsyncQueueSC<WaiterParam, Waiter>(0),
                  m_par_releaser(releaser) {}

        void process_one_task(WaiterParam& param) {
            if (param.event->finished()) {
                param.blob.reset();
                param.storage.reset();
                EventPool::without_timer().free(param.event);
                return;
            }

            using namespace std::literals;
            std::this_thread::sleep_for(1us);
            add_task(std::move(param));
        }
        void on_async_queue_worker_thread_start() override {
            sys::set_thread_name("releaser");
        }
    };
    Waiter m_waiter{this};

protected:
    std::shared_ptr<void> on_comp_node_finalize() override {
        m_waiter.wait_task_queue_empty();
        return {};
    }

public:
    static AsyncReleaser* inst() {
        static AsyncReleaser releaser;
        return &releaser;
    }

    ~AsyncReleaser() {
        m_waiter.wait_task_queue_empty();
    }

    void add(BlobPtr blob, CompNode cn) { add(cn, std::move(blob), {}); }

    void add(const HostTensorND& hv) {
        add(hv.comp_node(), {}, hv.storage().raw_storage());
    }

    void add(CompNode cn, BlobPtr blob,
             HostTensorStorage::RawStorage storage = {}) {
        auto event = EventPool::without_timer().alloc(cn);
        event->record();
        m_waiter.add_task({cn, event, std::move(blob), std::move(storage)});
    }
};

class CompNodeSyncManager : public CompNodeDepedentObject {
    ThinHashMap<Blob*, std::unique_ptr<CompNode::Event>> m_blob2event;
    std::mutex m_mtx;
public:
    std::shared_ptr<void> on_comp_node_finalize() override {
        MGB_LOCK_GUARD(m_mtx);
        m_blob2event.clear();
        return {};
    }

    static CompNodeSyncManager& inst() {
        static CompNodeSyncManager sl_inst;
        return sl_inst;
    }

    CompNode::Event* get_or_create_event(Blob* blob) {
        mgb_assert(!is_finalized());
        MGB_LOCK_GUARD(m_mtx);
        auto&& e = m_blob2event[blob];
        if (!e) {
            e = blob->comp_node().create_event();
        }
        return e.get();
    }

    void remove(Blob* blob) {
        MGB_LOCK_GUARD(m_mtx);
        m_blob2event.erase(blob);
    }
};

// Cache for small blobs
// 1. A blob has to be seen twice (within a window) to be eligible for cache
// 2. Cache eviction occurs when cache size reaches a threshold, in least frequently used order
class ConstTensorCache {
public:
    struct Entry {
        size_t hitcnt = 0;
        std::unique_ptr<dt_byte[]> data;
        size_t size;
        BlobPtr blob;

        Entry(const dt_byte* ptr, size_t size_, BlobPtr blob_)
                : data(new dt_byte[size_]), size(size_), blob(blob_) {
            memcpy(data.get(), ptr, size);
        }

        // does not check input
        bool match(const HostTensorND& hv) {
            return 0 == memcmp(data.get(), hv.raw_ptr(), hv.layout().span().high_byte);
        }
    };

    bool check(const HostTensorND& hv) {
        auto&& layout = hv.layout();
        auto&& span = layout.span();
        return hv.format().is_default() && !hv.empty() &&
            layout.is_contiguous() && span.low_byte == 0 &&
            span.high_byte <= max_bytes;
    }

    // hash storage; does not check input
    static uint64_t hash(const HostTensorND& hv) {
        auto&& span = hv.layout().span();
        return XXHash{}
            .update(hv.raw_ptr(), span.high_byte)
            .digest();
    }

    BlobPtr lookup(const HostTensorND& hv) {
        if (!check(hv)) {
            return {};
        }
        auto h = hash(hv);
        MGB_LOCK_GUARD(mtx);
        // lookup in g1
        auto it = g1.find(h);
        if (it != g1.end()) {
            if (!it->second.match(hv)) {
                mgb_log_warn("hash collision in const tensor cache");
                return {};
            }
            it->second.hitcnt += 1;
            return it->second.blob;
        }
        // lookup in g0
        if (!g0.extract(h) && !g0b.extract(h)) {
            maybe_collect_g0();
            g0.emplace(h);
            return {};
        }
        // add new entry to g1
        maybe_collect_g1();
        Entry entry(hv.raw_ptr(), hv.layout().span().high_byte, Tensor(hv).blob());
        it = g1.emplace_hint(it, h, std::move(entry));
        it->second.hitcnt += 1;
        return it->second.blob;
    }

    void clear() {
        MGB_LOCK_GUARD(mtx);
        g0.clear();
        g0b.clear();
        g1.clear();
    }

    std::mutex mtx;
    size_t hwm = 1024, lwm = 512, max_bytes = TensorShape::MAX_NDIM * 8, window = 65536;

private:
    void maybe_collect_g0() {
        if (g0.size() > window) {
            std::swap(g0, g0b);
            g0.clear();
        }
    }
    void maybe_collect_g1() {
        if (g1.size() <= hwm) return;

        using KV = std::pair<uint64_t, Entry>;
        std::vector<KV> tmp;
        tmp.reserve(g1.size());
        for (auto&& kv : g1) {
            tmp.emplace_back(kv.first, std::move(kv.second));
        }
        std::nth_element(tmp.begin(), tmp.begin() + lwm, tmp.end(), [](const KV& lhs, const KV& rhs) {
                return lhs.second.hitcnt > rhs.second.hitcnt;
            });
        g1.clear();
        for (auto&& kv : tmp) {
            kv.second.hitcnt = 0;
            g1.emplace(std::move(kv));
        }
    }
    std::unordered_set<uint64_t> g0, g0b;
    std::unordered_map<uint64_t, Entry> g1;
};

struct MultiCNConstTensorCache : CompNodeDepedentObject {
    std::mutex mtx;
    CompNode::UnorderedMap<ConstTensorCache> cn2cache;

    std::shared_ptr<void> on_comp_node_finalize() {
        MGB_LOCK_GUARD(mtx);
        cn2cache.clear();
        return {};
    }

    BlobPtr lookup(const HostTensorND& hv) {
        MGB_LOCK_GUARD(mtx);
        return cn2cache[hv.comp_node()].lookup(hv);
    }

    static MultiCNConstTensorCache& inst() {
        static MultiCNConstTensorCache sl_inst;
        return sl_inst;
    }
};

}  // namespace

void EventDeleter::operator()(CompNode::Event* event) {
    EventPool::without_timer().free(event);
}

namespace {
    std::atomic_uint64_t next_blob_id = 0;
}

Blob::Blob(const DeviceTensorStorage& s):
    m_comp_node{s.comp_node()}, m_storage{s.raw_storage()},
    m_size{s.size()} {
    m_id = next_blob_id++;
    BlobManager::inst()->register_blob(this);
}

Blob::Blob(CompNode cn, size_t sz):
    m_comp_node{cn}, m_storage{}, m_size{sz} {
    m_id = next_blob_id++;
    BlobManager::inst()->register_blob(this);
}

Blob::~Blob() {
    BlobManager::inst()->unregister_blob(this);
    CompNodeSyncManager::inst().remove(this);
}

const Blob::RawStorage& Blob::storage() {
    if (!m_storage) {
        BlobManager::inst()->alloc_with_defrag(this, m_size);
    }
    return m_storage;
}

Tensor::Tensor(BlobPtr blob, const TensorLayout& layout, size_t offset, const HostTensorND& hv)
        : m_layout(layout), m_blob(std::move(blob)), m_offset(offset), m_value(hv) {
}

Tensor::Tensor(const HostTensorND &hv)
    : Tensor(hv.layout(), hv.comp_node()) {
    m_value = hv;
    dev_tensor().copy_from_fixlayout(hv);
    // even though hv is saved in m_value, Tensor itself could be
    // released before copy completes
    AsyncReleaser::inst()->add(hv);
}

Tensor::Tensor(const DeviceTensorND &dv, const HostTensorND& hv) {
    if (!hv.empty()) {
        mgb_assert(dv.comp_node() == hv.comp_node());
        mgb_assert(dv.dtype() == hv.dtype());
        mgb_assert(dv.shape().eq_shape(hv.shape()));
        m_value = hv;
    }
    m_layout = dv.layout();
    m_blob = Blob::make(dv.storage());
    m_offset = 0;
}

Tensor::Tensor(const TensorLayout& layout, const CompNode& cn)
    : m_layout{layout}, m_blob{Blob::make(cn, layout.dtype.size(layout.total_nr_elems()))},
    m_offset{0} {}

Tensor::Tensor(const BlobPtr blob, const size_t offset, const TensorLayout& layout)
    : m_layout{layout}, m_blob{blob}, m_offset{offset} {}

TensorPtr Tensor::make(const HostTensorND& hv) {
    auto&& blob = MultiCNConstTensorCache::inst().lookup(hv);
    if (blob) {
        return make(std::forward<decltype(blob)>(blob), hv.layout(), hv);
    }
    return std::make_shared<Tensor>(hv);
}

DeviceTensorND Tensor::dev_tensor() {
    mgb_assert(m_blob, "uninitialized tensor.");
    DeviceTensorStorage storage;
    storage.reset(m_blob->comp_node(), m_blob->size(), m_blob->storage());
    storage = storage.sub(m_offset);
    DeviceTensorND ret;
    ret.reset(storage, m_layout);
    return ret;
}

void Tensor::fetch_value() {
    MGB_LOCK_GUARD(m_mtx);
    if (m_value.empty()) {
        m_value.copy_from(dev_tensor());
        m_value_ready.reset(EventPool::without_timer().alloc(comp_node()));
        m_value_ready->record();
    }
}

bool Tensor::value_fetched() {
    MGB_LOCK_GUARD(m_mtx);
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
    MGB_LOCK_GUARD(m_mtx);
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
    TensorLayout layout(shape, m_layout.dtype);
    return Tensor::make(m_blob, offset + m_offset, layout);
}

void Tensor::add_release_callback(CompNode cn) {
    AsyncReleaser::inst()->add(m_blob, cn);
}

CompNode::Event* Tensor::get_or_create_event() {
    auto e = CompNodeSyncManager::inst().get_or_create_event(m_blob.get());
    e->record();
    return e;
}

void Tensor::static_initialize() {
    EventPool::with_timer();
    EventPool::without_timer();
    AsyncReleaser::inst();
    CompNodeSyncManager::inst();
    MultiCNConstTensorCache::inst();
}

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
