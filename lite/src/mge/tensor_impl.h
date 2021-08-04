/**
 * \file src/mge/tensor_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "lite/tensor.h"
#include "tensor_impl_base.h"

#include "megbrain/tensor.h"

#include <unordered_map>

namespace lite {

/*!
 * \brief implement the Tensor in mge
 */
class TensorImplDft final : public Tensor::TensorImplBase {
    LITE_DYN_TYPE_OBJ_FINAL_DECL;

public:
    TensorImplDft();
    TensorImplDft(LiteDeviceType device, bool is_pinned_host = false);
    TensorImplDft(LiteDeviceType device, const Layout& layout,
                  bool is_pinned_host = false);
    TensorImplDft(int device_id, LiteDeviceType device,
                  const Layout& layout = {}, bool is_pinned_host = false);
    TensorImplDft(int device_id, int stream_id, LiteDeviceType device,
                  bool is_pinned_host = false);

    virtual ~TensorImplDft() = default;

    LiteDeviceType get_device_type() const override;

    int get_device_id() const override;

    LiteBackend get_backend_type() const override {
        return LiteBackend::LITE_DEFAULT;
    }
    Layout get_layout() const override;

    bool is_pinned_host() const override;

    //! which will trigger memory alloc in tensor implement
    void* get_memory_ptr() const override;

    //! which will trigger memory alloc in tensor implement if memory is not
    //! allocated, and compute the ptr in the gaven idx
    void* get_memory_ptr(const std::vector<size_t>& idx) const override;

    //! set layout will change the layout and reallocate memory of the tensor
    void set_layout(const Layout& layout) override;

    //! use the user allocated data to reset the memory of the tensor, the
    //! memory will not be managed by the lite, later, the user should delete
    //! it.
    void reset(void* prepared_data) override;

    //! use the user allocated data and corresponding layout to reset the data
    //! and layout of the tensor, the memory will not be managed by lite, later,
    //! the user should delete it.
    void reset(void* prepared_data, const Layout& layout) override;

    //! get a new tensor slice from the origin tensor
    std::shared_ptr<Tensor> slice(
            const std::vector<size_t>& start, const std::vector<size_t>& end,
            const std::vector<size_t>& step = {}) override;

    //! set the tensor memory with zero
    void fill_zero() override;

    //! reshape the tensor with new shape, keep the data_type the same
    void reshape(const Layout& layout) override;

    //! copy tensor form other tensor
    //! Note: the best way for tensor copy is just set the dst device, left
    //! layout empty, when copying the dst layout will be set the same with
    //! src
    void copy_from(const TensorImplBase* src_impl) override;

    //! share memory with other tensor
    void share_memory_with(const TensorImplBase* src_impl) override;

    //! whether the memory of tensor is continue
    bool is_continue_memory() const override;

    //! get host tensor
    std::shared_ptr<mgb::HostTensorND> host_tensor() const {
        return m_host_tensor;
    }
    //! get device tensor
    std::shared_ptr<mgb::DeviceTensorND> dev_tensor() const {
        return m_dev_tensor;
    }
    //! copy from mgb tensor
    void copy_from_mge_tensor(const mgb::DeviceTensorND& dv);

public:
    friend class NetworkImplDft;

private:
    bool is_host() const { return m_host_tensor != nullptr; };

    void copy_from_continue(const TensorImplBase* src_impl);

    void copy_from_fixlayout(const TensorImplBase* src_impl);

    void set_mge_tensor_compnode(const mgb::CompNode& comp_node);

private:
    std::shared_ptr<mgb::HostTensorND> m_host_tensor;
    std::shared_ptr<mgb::DeviceTensorND> m_dev_tensor;
};

}  // namespace lite

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
