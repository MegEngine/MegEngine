/**
 * \file imperative/src/include/megbrain/imperative/physical_tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <mutex>
#include <memory>

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

    template<typename ...Args>
    static BlobPtr make(Args&& ...args) {
        return std::make_shared<Blob>(std::forward<Args>(args)...);
    }

    using RawStorage = DeviceTensorStorage::RawStorage;
    const RawStorage& storage();

    const CompNode& comp_node() const {
        return m_comp_node;
    }

    size_t size() const {
        return m_size;
    }
private:
    friend class BlobManagerImpl;
    CompNode m_comp_node;
    mutable RawStorage m_storage;
    size_t m_size = 0;
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
    Tensor(BlobPtr blob, const TensorLayout& layout, size_t offset = 0, const HostTensorND& hv = {});
    Tensor(BlobPtr blob, const TensorLayout& layout, const HostTensorND& hv = {})
        : Tensor(std::move(blob), layout, 0, hv) {};
    Tensor(const HostTensorND &hv);
    Tensor(const DeviceTensorND &dv, const HostTensorND& hv = {});
    Tensor(const TensorLayout& layout, const CompNode& cn);
    Tensor(const BlobPtr blob, const size_t offset, const TensorLayout& layout);

    static TensorPtr make(const HostTensorND& hv);

    template<typename T, typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, HostTensorND>>>
    static TensorPtr make(T&& hv) {
        TensorPtr (*f)(const HostTensorND&) = &make;
        return f(std::forward<T>(hv));
    };

    template<typename ...Args>
    static TensorPtr make(Args&& ...args) {
        return std::make_shared<Tensor>(std::forward<Args>(args)...);
    }

    CompNode comp_node() const {
        mgb_assert(m_blob, "uninitialized tensor.");
        return m_blob->comp_node();
    }

    DType dtype() const {
        return m_layout.dtype;
    }

    TensorLayout layout() const {
        return m_layout;
    }

    const TensorShape& shape() const {
        return m_layout;
    }

    DeviceTensorND dev_tensor();

    static TensorPtr make_scalar(DTypeScalar value, CompNode cn);

    TensorPtr make_scalar(DTypeScalar value) const {
        mgb_assert(m_blob, "uninitialized tensor.");
        return make_scalar(value, m_blob->comp_node());
    }

    BlobPtr& blob() {
        return m_blob;
    }

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
private:

    TensorLayout m_layout;
    BlobPtr m_blob;
    size_t m_offset;
    std::mutex m_mtx;
    HostTensorND m_value;
    EventPtr m_value_ready = nullptr;
};

struct LogicalTensorDesc {
    TensorLayout layout;
    CompNode comp_node;
    DeviceTensorND value; // cpu:default
};

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
