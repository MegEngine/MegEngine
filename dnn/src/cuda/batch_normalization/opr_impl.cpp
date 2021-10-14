/**
 * \file dnn/src/cuda/batch_normalization/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./opr_impl.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

namespace batch_normalization {

BNTensorDescHolder::BNTensorDescHolder(
        const TensorLayout& x, const ParamDim& param_dim, const FwdMode& fwd_mode) {
    TensorShape xy_shape(x);
    Format xy_format = Format::NCHW;

    switch (param_dim) {
        case ParamDim::DIM_11HW:
            // xy: N, C, H, W --> (N*C), 1, H, W
            xy_shape.shape[0] = xy_shape.shape[0] * xy_shape.shape[1];
            xy_shape.shape[1] = 1;
            bn_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
            break;
        case ParamDim::DIM_1CHW:
            bn_mode = CUDNN_BATCHNORM_PER_ACTIVATION;
            break;
        case ParamDim::DIM_1C11:
            bn_mode = CUDNN_BATCHNORM_SPATIAL;
            break;
        case ParamDim::DIM_111C:
            bn_mode = CUDNN_BATCHNORM_SPATIAL;
            xy_format = Format::NHWC;
#if CUDNN_VERSION >= 7410
            if (fwd_mode == FwdMode::TRAINING) {
                bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
            }
#endif  // CUDNN_VERSION >= 7400
            break;
        default:
            megdnn_throw("Unknown param dim type of batch normalization.");
    }
    xy_desc.set(TensorLayout(xy_shape, x.dtype), xy_format);
    param_desc.set(xy_desc.desc, bn_mode);
}

size_t get_reserve_size(
        const cudnnHandle_t& handle, const BNTensorDescHolder& tensor_desc) {
#if CUDNN_VERSION >= 7410
    size_t reserve_size;
    cudnn_check(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
            handle, tensor_desc.bn_mode, CUDNN_BATCHNORM_OPS_BN,
            nullptr,                   // activationDesc
            tensor_desc.xy_desc.desc,  // xDesc
            &reserve_size));
    return reserve_size;
#else
    return 0;
#endif  // CUDNN_VERSION >= 7410
}
}  // namespace batch_normalization

using batch_normalization::BNTensorDescHolder;

size_t BNForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&) {
#if CUDNN_VERSION >= 7410
    auto handle = cudnn_handle(this->handle());
    BNTensorDescHolder tensor_desc(src, m_param.param_dim, m_param.fwd_mode);

    size_t workspace_size;
    cudnn_check(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
            handle, tensor_desc.bn_mode, CUDNN_BATCHNORM_OPS_BN,
            tensor_desc.xy_desc.desc,     // xDesc
            tensor_desc.xy_desc.desc,     // yDesc
            tensor_desc.xy_desc.desc,     // zDesc
            tensor_desc.param_desc.desc,  // bnScaleBiasMeanVarDesc
            nullptr,                      // activationDesc
            &workspace_size));
    return workspace_size;
#else
    return 0;
#endif  // CUDNN_VERSION >= 7410
}

size_t BNForwardImpl::get_reserve_in_bytes(const TensorLayout& src) {
    BNTensorDescHolder tensor_desc(src, m_param.param_dim, m_param.fwd_mode);
    return batch_normalization::get_reserve_size(
            cudnn_handle(this->handle()), tensor_desc);
}

void BNForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in bn_scale, _megdnn_tensor_in bn_bias,
        _megdnn_tensor_out mean, _megdnn_tensor_out variance,
        _megdnn_tensor_out batch_mean, _megdnn_tensor_out batch_inv_variance,
        _megdnn_tensor_out reserve, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(
            src.layout, bn_scale.layout, bn_bias.layout, mean.layout, variance.layout,
            batch_mean.layout, batch_inv_variance.layout, dst.layout, workspace.size,
            reserve.layout.access_bytes());
    auto handle = cudnn_handle(this->handle());
    BNTensorDescHolder tensor_desc(src.layout, m_param.param_dim, m_param.fwd_mode);

    float alpha = 1.0f, beta = 0.0f;
    switch (m_param.fwd_mode) {
        case param::BN::FwdMode::TRAINING:
#if CUDNN_VERSION >= 7410
            cudnn_check(cudnnBatchNormalizationForwardTrainingEx(
                    handle, tensor_desc.bn_mode, CUDNN_BATCHNORM_OPS_BN, &alpha,
                    &beta,                                  // one & zero
                    tensor_desc.xy_desc.desc, src.raw_ptr,  // xDesc & x
                    nullptr, nullptr,                       // zDesc & z
                    tensor_desc.xy_desc.desc, dst.raw_ptr,  // yDesc & y
                    tensor_desc.param_desc.desc,            // bnScaleBiasMeanVarDesc
                    bn_scale.raw_ptr, bn_bias.raw_ptr, m_param.avg_factor, mean.raw_ptr,
                    variance.raw_ptr, m_param.epsilon, batch_mean.raw_ptr,
                    batch_inv_variance.raw_ptr, nullptr, workspace.raw_ptr,
                    workspace.size, reserve.raw_ptr, reserve.layout.access_bytes()));
#else
            cudnn_check(cudnnBatchNormalizationForwardTraining(
                    handle, tensor_desc.bn_mode, &alpha, &beta,
                    tensor_desc.xy_desc.desc, src.raw_ptr,  // xDesc & x
                    tensor_desc.xy_desc.desc, dst.raw_ptr,  // yDesc & y
                    tensor_desc.param_desc.desc,            // bnScaleBiasMeanVarDesc
                    bn_scale.raw_ptr, bn_bias.raw_ptr, m_param.avg_factor, mean.raw_ptr,
                    variance.raw_ptr, m_param.epsilon, batch_mean.raw_ptr,
                    batch_inv_variance.raw_ptr));
#endif  // CUDNN_VERSION >= 7410
            break;
        case param::BN::FwdMode::INFERENCE:
            cudnn_check(cudnnBatchNormalizationForwardInference(
                    handle, tensor_desc.bn_mode, &alpha, &beta,
                    tensor_desc.xy_desc.desc, src.raw_ptr, tensor_desc.xy_desc.desc,
                    dst.raw_ptr, tensor_desc.param_desc.desc, bn_scale.raw_ptr,
                    bn_bias.raw_ptr, mean.raw_ptr, variance.raw_ptr, m_param.epsilon));
            break;
        default:
            megdnn_throw("Unknown forward mode type of batch normalization.");
    }
}

size_t BNBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&) {
#if CUDNN_VERSION >= 7410
    auto handle = cudnn_handle(this->handle());
    BNTensorDescHolder tensor_desc(x, m_param.param_dim, m_param.fwd_mode);

    size_t workspace_size;
    cudnn_check(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
            handle, tensor_desc.bn_mode, CUDNN_BATCHNORM_OPS_BN,
            tensor_desc.xy_desc.desc,     // xDesc
            tensor_desc.xy_desc.desc,     // yDesc
            tensor_desc.xy_desc.desc,     // dyDesc
            nullptr,                      // dzDesc
            tensor_desc.xy_desc.desc,     // dxDesc
            tensor_desc.param_desc.desc,  // dBnScaleBiasDesc
            nullptr,                      // activationDesc
            &workspace_size));
    return workspace_size;
#else
    return 0;
#endif  // CUDNN_VERSION >= 7410
}

size_t BNBackwardImpl::get_reserve_in_bytes(const TensorLayout& src) {
    BNTensorDescHolder tensor_desc(src, m_param.param_dim, m_param.fwd_mode);
    return batch_normalization::get_reserve_size(
            cudnn_handle(this->handle()), tensor_desc);
}

void BNBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in dy, _megdnn_tensor_in saved_batch_mean,
        _megdnn_tensor_in saved_batch_inv_variance, _megdnn_tensor_in bn_scale,
        _megdnn_tensor_in reserve, _megdnn_tensor_out d_bn_scale,
        _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx,
        _megdnn_workspace workspace) {
    check_exec(
            x.layout, dy.layout, saved_batch_mean.layout,
            saved_batch_inv_variance.layout, bn_scale.layout, d_bn_scale.layout,
            d_bn_bias.layout, dx.layout, workspace.size, reserve.layout.access_bytes());
    auto handle = cudnn_handle(this->handle());
    BNTensorDescHolder tensor_desc(x.layout, m_param.param_dim, m_param.fwd_mode);

    float alpha = 1.0, beta = 0.0;
#if CUDNN_VERSION >= 7410
    cudnn_check(cudnnBatchNormalizationBackwardEx(
            handle, tensor_desc.bn_mode, CUDNN_BATCHNORM_OPS_BN, &alpha, &beta, &alpha,
            &beta, tensor_desc.xy_desc.desc,
            x.raw_ptr,                                      // xDesc & x
            nullptr, nullptr,                               // yDesc & y
            tensor_desc.xy_desc.desc, dy.raw_ptr,           // dyDesc & dy
            nullptr, nullptr,                               // dzDesc & dz
            tensor_desc.xy_desc.desc, dx.raw_ptr,           // dxDesc & dx
            tensor_desc.param_desc.desc, bn_scale.raw_ptr,  // bnScale
            nullptr,                                        // bnBias
            d_bn_scale.raw_ptr, d_bn_bias.raw_ptr,          // dScale, dBias
            m_param.epsilon, saved_batch_mean.raw_ptr, saved_batch_inv_variance.raw_ptr,
            nullptr, workspace.raw_ptr, workspace.size, reserve.raw_ptr,
            reserve.layout.access_bytes()));
#else
    cudnn_check(cudnnBatchNormalizationBackward(
            handle, tensor_desc.bn_mode, &alpha, &beta, &alpha, &beta,
            tensor_desc.xy_desc.desc, x.raw_ptr,            // xDesc & x
            tensor_desc.xy_desc.desc, dy.raw_ptr,           // dyDesc & dy
            tensor_desc.xy_desc.desc, dx.raw_ptr,           // dxDesc & dx
            tensor_desc.param_desc.desc, bn_scale.raw_ptr,  // bnScale
            d_bn_scale.raw_ptr, d_bn_bias.raw_ptr,          // dScale, dBias
            m_param.epsilon, saved_batch_mean.raw_ptr,
            saved_batch_inv_variance.raw_ptr));
#endif
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
