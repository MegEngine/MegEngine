/**
 * \file imperative/src/include/megbrain/imperative/physical_tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <memory>
#include <mutex>

#include "megbrain/imperative/resource_manager.h"
#include "megbrain/tensor.h"

namespace mgb {
namespace imperative {

/************************** Tensor *****************************/
class Blob;
using BlobPtr = std::shared_ptr<Blob>;

class BlobManagerImpl;

class Blob : public NonCopyableObj {
public:
    Blob(const DeviceTensorStorage& s);
    Blob(CompNode cn, size_t sz);
    ~Blob();

    template <typename... Args>
    static BlobPtr make(Args&&... args) {
        return std::make_shared<Blob>(std::forward<Args>(args)...);
    }

    using RawStorage = DeviceTensorStorage::RawStorage;
    const RawStorage& storage();

    const CompNode& comp_node() const { return m_comp_node; }

    size_t size() const { return m_size; }

    size_t id() const { return m_id; }

private:
    friend class BlobManagerImpl;
    CompNode m_comp_node;
    mutable RawStorage m_storage;
    size_t m_size = 0;
    size_t m_id;
};

struct EventDeleter {
    void operator()(CompNode::Event*);
};
using EventPtr = std::unique_ptr<CompNode::Event, EventDeleter>;

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
class Tensor : public NonCopyableObj {
public:
    Tensor() = default;
    Tensor(BlobPtr blob, const TensorLayout& layout, size_t offset = 0,
           const HostTensorND& hv = {});
    Tensor(BlobPtr blob, const TensorLayout& layout, const HostTensorND& hv = {})
            : Tensor(std::move(blob), layout, 0, hv){};
    Tensor(const HostTensorND& hv);
    Tensor(const DeviceTensorND& dv, const HostTensorND& hv = {});
    Tensor(const TensorLayout& layout, const CompNode& cn);
    Tensor(const BlobPtr blob, const size_t offset, const TensorLayout& layout);

    static TensorPtr make(const HostTensorND& hv);

    template <
            typename T,
            typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, HostTensorND>>>
    static TensorPtr make(T&& hv) {
        TensorPtr (*f)(const HostTensorND&) = &make;
        return f(std::forward<T>(hv));
    };

    template <typename... Args>
    static TensorPtr make(Args&&... args) {
        return std::make_shared<Tensor>(std::forward<Args>(args)...);
    }

    CompNode comp_node() const {
        mgb_assert(m_blob, "uninitialized tensor.");
        return m_blob->comp_node();
    }

    DType dtype() const { return m_layout.dtype; }

    TensorLayout layout() const { return m_layout; }

    const TensorShape& shape() const { return m_layout; }

    size_t offset() const { return m_offset; }

    DeviceTensorND dev_tensor();

    static TensorPtr make_scalar(DTypeScalar value, CompNode cn);

    TensorPtr make_scalar(DTypeScalar value) const {
        mgb_assert(m_blob, "uninitialized tensor.");
        return make_scalar(value, m_blob->comp_node());
    }

    BlobPtr& blob() { return m_blob; }

    void fetch_value();
    bool value_fetched();
    TensorPtr sub(size_t offset, TensorShape shape);

    // m_value is set once readonly afterwards
    // so the return value is thread safe
    const HostTensorND& get_value();
    // return a pointer instead of a reference to ensure thread safety
    const HostTensorND* try_get_value();

    void add_release_callback(CompNode cn);
    CompNode::Event* get_or_create_event();

    // Make sure all static objects required to destruct a tensor has completed
    // construction. All static storage duration object that holds tensors must
    // call this method before their constructors completes.
    static void static_initialize();

private:
    TensorLayout m_layout;
    BlobPtr m_blob;
    size_t m_offset;
    std::mutex m_mtx;
    HostTensorND m_value;
    EventPtr m_value_ready = nullptr;
};

// Cache for small blobs
// 1. A blob has to be seen twice (within a window) to be eligible for cache
// 2. Cache eviction occurs when cache size reaches a threshold, in least frequently
// used order
class ConstTensorCache {
public:
    struct Entry {
        size_t hitcnt = 0;
        std::unique_ptr<dt_byte[]> data;
        size_t size;
        BlobPtr blob;

        Entry() = default;
        Entry(const dt_byte* ptr, size_t size_, BlobPtr blob_)
                : data(new dt_byte[size_]), size(size_), blob(blob_) {
            memcpy(data.get(), ptr, size);
        }

        // does not check input
        bool match(const HostTensorND& hv) {
            return 0 == memcmp(data.get(), hv.raw_ptr(), hv.layout().span().high_byte);
        }
    };

    using KV = std::pair<uint64_t, Entry>;

    bool check(const HostTensorND& hv) {
        auto&& layout = hv.layout();
        auto&& span = layout.span();
        return hv.format().is_default() && !hv.empty() && layout.is_contiguous() &&
               span.low_byte == 0 && span.high_byte <= max_bytes;
    }

    // hash storage; does not check input
    static uint64_t hash(const HostTensorND& hv) {
        auto&& span = hv.layout().span();
        return XXHash{}.update(hv.raw_ptr(), span.high_byte).digest();
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
    const size_t hwm = 1024, lwm = 512, max_bytes = TensorShape::MAX_NDIM * 8,
                 window = 65536;

private:
    void maybe_collect_g0() {
        if (g0.size() > window) {
            std::swap(g0, g0b);
            g0.clear();
        }
    }
    void maybe_collect_g1() {
        if (g1.size() < hwm)
            return;

        tmp.clear();
        for (auto&& kv : g1) {
            tmp.emplace_back(kv.first, std::move(kv.second));
        }
        std::nth_element(
                tmp.begin(), tmp.begin() + lwm, tmp.end(),
                [](const KV& lhs, const KV& rhs) {
                    return lhs.second.hitcnt > rhs.second.hitcnt;
                });
        tmp.resize(lwm);
        g1.clear();
        for (auto&& kv : tmp) {
            kv.second.hitcnt = 0;
            g1.emplace(std::move(kv));
        }
    }

    // g0: records blobs which have been seen at least once (within a window)
    // g0b: backup of g0
    // g1: records the most frequently used blobs which have been seen at least
    // twice. When `g1.size() == hwm`, it will be refreshed and only the top
    // `lhw` frequently used blobs will be kept.
    std::unordered_set<uint64_t> g0, g0b;
    std::unordered_map<uint64_t, Entry> g1;
    std::vector<KV> tmp;

public:
    ConstTensorCache() {
        g0.reserve(window), g0b.reserve(window);
        g1.reserve(hwm), tmp.reserve(hwm);
    }
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
        static auto* sl_inst =
                ResourceManager::create_global<MultiCNConstTensorCache>();
        return *sl_inst;
    }
};

struct LogicalTensorDesc {
    TensorLayout layout;
    CompNode comp_node;
    DeviceTensorND value;  // cpu:default
};
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
