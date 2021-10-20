/**
 * \file dnn/src/cuda/batch_normalization/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

namespace batch_normalization {

struct BNTensorDescHolder {
    using ParamDim = param::BN::ParamDim;
    using FwdMode = param::BN::FwdMode;
    using Format = param::Convolution::Format;

    TensorDesc xy_desc;
    BNParamDesc param_desc;
    cudnnBatchNormMode_t bn_mode;

    BNTensorDescHolder(
            const TensorLayout& x, const ParamDim& param_dim, const FwdMode& fwd_mode);
};

size_t get_reserve_size(
        const cudnnHandle_t& handle, const BNTensorDescHolder& tensor_desc);

}  // namespace batch_normalization

class BNForwardImpl final : public BNForward {
public:
    using BNForward::BNForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
            _megdnn_tensor_in bn_bias, _megdnn_tensor_out mean,
            _megdnn_tensor_out variance, _megdnn_tensor_out batch_mean,
            _megdnn_tensor_out batch_inv_variance, _megdnn_tensor_out reserve,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override;
    size_t get_reserve_in_bytes(const TensorLayout& src) override;
};

class BNBackwardImpl final : public BNBackward {
public:
    using BNBackward::BNBackward;
    void exec(
            _megdnn_tensor_in x, _megdnn_tensor_in dy,
            _megdnn_tensor_in saved_batch_mean,
            _megdnn_tensor_in saved_batch_inv_variance, _megdnn_tensor_in bn_scale,
            _megdnn_tensor_in reserve, _megdnn_tensor_out d_bn_scale,
            _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& x, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override;
    size_t get_reserve_in_bytes(const TensorLayout& src) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
