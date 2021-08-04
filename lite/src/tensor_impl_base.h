/**
 * \file src/tensor_impl_base.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once

#include "lite/tensor.h"
#include "misc.h"
#include "type_info.h"

#include <unordered_map>

namespace lite {

/*!
 * \brief implement the Tensor
 */
class Tensor::TensorImplBase : public DynTypeObj {
public:
    virtual ~TensorImplBase() = default;

    virtual LiteDeviceType get_device_type() const = 0;

    virtual int get_device_id() const = 0;

    virtual LiteBackend get_backend_type() const = 0;

    virtual Layout get_layout() const = 0;

    virtual bool is_pinned_host() const = 0;

    virtual void* get_memory_ptr() const = 0;

    virtual void* get_memory_ptr(const std::vector<size_t>& idx) const = 0;

    virtual void set_layout(const Layout& layout) = 0;

    //! use the user allocated data to reset the memory of the tensor, the
    //! memory will not be managed by the lite, later, the user should delete
    //! it.
    virtual void reset(void* prepared_data) = 0;

    //! use the user allocated data and corresponding layout to reset the data
    //! and layout of the tensor, the memory will not be managed by lite, later,
    //! the user should delete it.
    virtual void reset(void* prepared_data, const Layout& layout) = 0;

    //! reshape the tensor with new shape, keep the data_type the same
    virtual void reshape(const Layout& layout) = 0;

    //! get a new tensor slice from the origin tensor
    virtual std::shared_ptr<Tensor> slice(
            const std::vector<size_t>& start, const std::vector<size_t>& end,
            const std::vector<size_t>& step = {}) = 0;

    //! set the tensor memory with zero
    virtual void fill_zero() = 0;

    //! copy tensor form other tensor
    //! Note: the best way for tensor copy is just set the dst device, left
    //! layout empty, when copying the dst layout will be set the same with
    //! src
    virtual void copy_from(const TensorImplBase* src_impl) = 0;

    //! share memory with other tensor
    virtual void share_memory_with(const TensorImplBase* src_impl) = 0;

    //! whether the memory of tensor is continue
    virtual bool is_continue_memory() const = 0;
};

/*!
 * \brief friend class of Tensor, for convenient accessing the Network members
 */
class TensorHelper {
public:
    static inline std::shared_ptr<Tensor::TensorImplBase> implement(
            const std::shared_ptr<Tensor> tensor) {
        LITE_ASSERT(tensor);
        return tensor->m_tensor_impl;
    }
    static inline std::shared_ptr<Tensor::TensorImplBase> implement(
            const Tensor* tensor) {
        LITE_ASSERT(tensor);
        return tensor->m_tensor_impl;
    }
    static inline void implement(const std::shared_ptr<Tensor> tensor,
                                 std::shared_ptr<Tensor::TensorImplBase> impl) {
        LITE_ASSERT(tensor);
        tensor->m_tensor_impl = impl;
    }
};

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
