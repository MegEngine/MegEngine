/**
 * \file dnn/src/rocm/batch_normalization/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "./opr_impl.h"

#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

namespace batch_normalization {

void BNTensorDescHolder::setup(const TensorLayout& x,
                               const ParamDim& param_dim) {
    TensorShape xy_shape(x);

    switch (param_dim) {
        case ParamDim::DIM_11HW:
            // xy: N, C, H, W --> (N*C), 1, H, W
            xy_shape.shape[0] = xy_shape.shape[0] * xy_shape.shape[1];
            xy_shape.shape[1] = 1;
            bn_mode = miopenBNPerActivation;
            break;
        case ParamDim::DIM_1CHW:
            bn_mode = miopenBNPerActivation;
            break;
        case ParamDim::DIM_1C11:
            bn_mode = miopenBNSpatial;
            break;
        default:
            megdnn_throw(megdnn_mangle(
                    "Unknown param dim type of batch normalization."));
    }
    xy_desc.set(TensorLayout(xy_shape, x.dtype));
    param_desc.set(xy_desc.desc, bn_mode);
}

}  // namespace batch_normalization

void BNForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
                         _megdnn_tensor_in bn_bias, _megdnn_tensor_out mean,
                         _megdnn_tensor_out variance,
                         _megdnn_tensor_out batch_mean,
                         _megdnn_tensor_out batch_inv_variance,
                         _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, bn_scale.layout, bn_bias.layout, mean.layout,
               variance.layout, batch_mean.layout, batch_inv_variance.layout,
               dst.layout, workspace.size);
    auto handle = concrete_handle(this->handle())->miopen_handle();
    m_tensor_desc.setup(src.layout, m_param.param_dim);

    float alpha = 1.0f, beta = 0.0f;
    switch (m_param.fwd_mode) {
        case param::BN::FwdMode::TRAINING:
            miopen_check(miopenBatchNormalizationForwardTraining(
                    handle, m_tensor_desc.bn_mode, &alpha, &beta,
                    m_tensor_desc.xy_desc.desc,     // xDesc
                    src.raw_ptr,                    // x
                    m_tensor_desc.xy_desc.desc,     // yDesc
                    dst.raw_ptr,                    // y
                    m_tensor_desc.param_desc.desc,  // bnScaleBiasMeanVarDesc
                    bn_scale.raw_ptr, bn_bias.raw_ptr, m_param.avg_factor,
                    mean.raw_ptr, variance.raw_ptr, m_param.epsilon,
                    batch_mean.raw_ptr, batch_inv_variance.raw_ptr));

            break;
        case param::BN::FwdMode::INFERENCE:
            miopen_check(miopenBatchNormalizationForwardInference(
                    handle, m_tensor_desc.bn_mode, &alpha, &beta,
                    m_tensor_desc.xy_desc.desc, src.raw_ptr,
                    m_tensor_desc.xy_desc.desc, dst.raw_ptr,
                    m_tensor_desc.param_desc.desc, bn_scale.raw_ptr,
                    bn_bias.raw_ptr, mean.raw_ptr, variance.raw_ptr,
                    m_param.epsilon));
            break;
        default:
            megdnn_throw(megdnn_mangle(
                    "Unknown forward mode type of batch normalization."));
    }
}

void BNBackwardImpl::exec(_megdnn_tensor_in x, _megdnn_tensor_in dy,
                          _megdnn_tensor_in saved_batch_mean,
                          _megdnn_tensor_in saved_batch_inv_variance,
                          _megdnn_tensor_in bn_scale,
                          _megdnn_tensor_out d_bn_scale,
                          _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx,
                          _megdnn_workspace workspace) {
    check_exec(x.layout, dy.layout, saved_batch_mean.layout,
               saved_batch_inv_variance.layout, bn_scale.layout,
               d_bn_scale.layout, d_bn_bias.layout, dx.layout, workspace.size);
    auto handle = concrete_handle(this->handle())->miopen_handle();
    m_tensor_desc.setup(x.layout, m_param.param_dim);

    float alpha = 1.0, beta = 0.0;
    miopen_check(miopenBatchNormalizationBackward(
            handle, m_tensor_desc.bn_mode, &alpha, &beta, &alpha, &beta,
            m_tensor_desc.xy_desc.desc, x.raw_ptr, m_tensor_desc.xy_desc.desc,
            dy.raw_ptr, m_tensor_desc.xy_desc.desc, dx.raw_ptr,
            m_tensor_desc.param_desc.desc, bn_scale.raw_ptr, d_bn_scale.raw_ptr,
            d_bn_bias.raw_ptr, m_param.epsilon, saved_batch_mean.raw_ptr,
            saved_batch_inv_variance.raw_ptr));
}

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
