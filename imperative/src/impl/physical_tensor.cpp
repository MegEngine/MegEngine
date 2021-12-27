/**
 * \file imperative/src/impl/physical_tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative.h"
#include "megbrain/imperative/blob_manager.h"
#include "megbrain/imperative/profiler.h"

#include "./async_releaser.h"
#include "./event_pool.h"

#include "./profiler/events.h"

#include <mutex>

namespace mgb {
namespace imperative {

namespace {

class CompNodeSyncManager : public CompNodeDepedentObject {
    ThinHashMap<Blob*, std::unique_ptr<CompNode::Event>> m_blob2event;
    std::mutex m_mtx;

public:
#if MGB_CUDA && defined(WIN32)
    //! FIXME: windows cuda driver shutdown before call atexit function even
    //! register atexit function after init cuda driver! as a workround
    //! recovery resource by OS temporarily, may need remove this after
    //! upgrade cuda runtime
    static bool is_into_atexit;
#endif
    std::shared_ptr<void> on_comp_node_finalize() override {
        MGB_LOCK_GUARD(m_mtx);
        m_blob2event.clear();
        return {};
    }

    static CompNodeSyncManager& inst() {
        static CompNodeSyncManager* sl_inst = new CompNodeSyncManager();
#if MGB_CUDA && defined(WIN32)
        //! FIXME: windows cuda driver shutdown before call atexit function even
        //! register atexit function after init cuda driver! as a workround
        //! recovery resource by OS temporarily, may need remove this after
        //! upgrade cuda runtime
        if (!is_into_atexit) {
            auto err = atexit([] { is_into_atexit = true; });
            mgb_assert(!err, "failed to register atexit function");
        }
#endif
        return *sl_inst;
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
#if MGB_CUDA && defined(WIN32)
//! FIXME: windows cuda driver shutdown before call atexit function even
//! register atexit function after init cuda driver! as a workround
//! recovery resource by OS temporarily, may need remove this after
//! upgrade cuda runtime
bool CompNodeSyncManager::is_into_atexit = false;
#endif

}  // namespace

void EventDeleter::operator()(CompNode::Event* event) {
    EventPool::without_timer().free(event);
}

namespace {
std::atomic_uint64_t next_blob_id = 0;
}

Blob::Blob(const DeviceTensorStorage& s)
        : m_comp_node{s.comp_node()},
          m_storage{s.raw_storage()},
          m_size{s.size() + s.offset()} {
    m_id = next_blob_id++;
    BlobManager::inst()->register_blob(this);
}

Blob::Blob(CompNode cn, size_t sz) : m_comp_node{cn}, m_storage{}, m_size{sz} {
    m_id = next_blob_id++;
    BlobManager::inst()->register_blob(this);
}

Blob::~Blob() {
    BlobManager::inst()->unregister_blob(this);

#if MGB_CUDA && defined(WIN32)
    //! FIXME: windows cuda driver shutdown before call atexit function even
    //! register atexit function after init cuda driver! as a workround
    //! recovery resource by OS temporarily, may need remove this after
    //! upgrade cuda runtime
    if (CompNodeSyncManager::is_into_atexit)
        return;
#endif
    CompNodeSyncManager::inst().remove(this);
}

const Blob::RawStorage& Blob::storage() {
    if (!m_storage) {
        BlobManager::inst()->alloc_with_defrag(this, m_size);
    }
    return m_storage;
}

Tensor::Tensor(
        BlobPtr blob, const TensorLayout& layout, size_t offset, const HostTensorND& hv)
        : m_layout(layout), m_blob(std::move(blob)), m_offset(offset), m_value(hv) {}

Tensor::Tensor(const HostTensorND& hv) : Tensor(hv.layout(), hv.comp_node()) {
    constexpr int size_threshold = TensorShape::MAX_NDIM;
    if (hv.layout().total_nr_elems() <= size_threshold) {
        m_value = hv;
    }
    MGB_RECORD_EVENT(
            profiler::HostToDeviceEvent, hv.layout(), hv.comp_node(), hv.raw_ptr(),
            dev_tensor().raw_ptr());
    dev_tensor().copy_from_fixlayout(hv);
    // even though hv is saved in m_value, Tensor itself could be
    // released before copy completes
    MGB_RECORD_EVENT(
            profiler::HostToDeviceFinishEvent, hv.layout(), hv.comp_node(),
            hv.raw_ptr(), dev_tensor().raw_ptr());
    AsyncReleaser::inst()->add(hv);
}

Tensor::Tensor(const DeviceTensorND& dv, const HostTensorND& hv) {
    if (!hv.empty()) {
        mgb_assert(dv.comp_node() == hv.comp_node());
        mgb_assert(dv.dtype() == hv.dtype());
        mgb_assert(dv.shape().eq_shape(hv.shape()));
        m_value = hv;
    }
    m_layout = dv.layout();
    m_blob = Blob::make(dv.storage());
    m_offset = dv.storage().offset();
}

Tensor::Tensor(const TensorLayout& layout, const CompNode& cn)
        : m_layout{layout},
          m_blob{Blob::make(cn, layout.span().dist_byte())},
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
