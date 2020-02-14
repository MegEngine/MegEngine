/**
 * \file dnn/src/cuda/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/pooling/opr_impl.h"

#include "src/cuda/utils.h"
#include "./pooling2d_int8_cdiv4hwn4.cuh"

namespace megdnn {
namespace cuda {

void PoolingForwardImpl::setup_descs(const TensorLayout &src,
        const TensorLayout &dst)
{
    src_desc.set(src, param().format);
    dst_desc.set(dst, param().format);
    pooling_desc.set(this->param());
}

void PoolingForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    using Format = param::Pooling::Format;
    if (param().format == Format::CHWN4) {
        pooling2d::Param kern_param;
        size_t c = src.layout[0], hi = src.layout[1], wi = src.layout[2],
               n = src.layout[3], ho = dst.layout[1], wo = dst.layout[2];
        c = c * 4;
        size_t ph = param().pad_h, pw = param().pad_w;
        size_t window_h = param().window_h, window_w = param().window_w;
        size_t sh = param().stride_h, sw = param().stride_w;
        kern_param.n = n, kern_param.c = c, kern_param.hi = hi,
        kern_param.wi = wi, kern_param.ho = ho, kern_param.wo = wo,
        kern_param.ph = ph, kern_param.pw = pw, kern_param.window_h = window_h,
        kern_param.window_w = window_w, kern_param.sh = sh, kern_param.sw = sw;
        auto&& stream = cuda_stream(handle());
        return pooling2d::_do_pooling2d_int8_cdiv4hwn4(
                src.compatible_ptr<int8_t>(), dst.compatible_ptr<int8_t>(),
                kern_param, stream, static_cast<uint32_t>(param().mode));
    }
    auto handle = cudnn_handle(this->handle());
    setup_descs(src.layout, dst.layout);
    dt_float32 alpha = 1.0f, beta = 0.0f;
    cudnn_check(cudnnPoolingForward(handle,
                pooling_desc.desc,
                &alpha,
                src_desc.desc, src.raw_ptr,
                &beta,
                dst_desc.desc, dst.raw_ptr));
}

void PoolingBackwardImpl::setup_descs(const TensorLayout &src,
        const TensorLayout &dst,
        const TensorLayout &diff,
        const TensorLayout &grad)
{
    src_desc.set(src);
    dst_desc.set(dst);
    diff_desc.set(diff);
    grad_desc.set(grad);
    pooling_desc.set(this->param());
}

void PoolingBackwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in dst,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, diff.layout, grad.layout, workspace.size);
    auto handle = cudnn_handle(this->handle());
    setup_descs(src.layout, dst.layout, diff.layout, grad.layout);
    float alpha = 1.0f, beta = 0.0f;
    cudnn_check(cudnnPoolingBackward(handle,
                pooling_desc.desc,
                &alpha,
                dst_desc.desc, dst.raw_ptr,
                diff_desc.desc, diff.raw_ptr,
                src_desc.desc, src.raw_ptr,
                &beta,
                grad_desc.desc, grad.raw_ptr));
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
