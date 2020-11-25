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

#include "./pooling2d_int8.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void PoolingForwardImpl::setup_descs(const TensorLayout& src,
                                     const TensorLayout& dst) {
    src_desc.set(src, param().format);
    dst_desc.set(dst, param().format);
    pooling_desc.set(this->param());
}

WorkspaceBundle PoolingForwardImpl::get_workspace_bundle(
        void* ptr, const TensorLayout& src, const TensorLayout& dst) const {
    SmallVector<size_t> sizes;
    TensorLayout fsrc = src;
    TensorLayout fdst = dst;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    get_workspace(fsrc);
    get_workspace(fdst);
    return {ptr, std::move(sizes)};
}

void PoolingForwardImpl::exec(_megdnn_tensor_in ssrc, _megdnn_tensor_out sdst,
                              _megdnn_workspace sworkspace) {
    check_exec(ssrc.layout, sdst.layout, sworkspace.size);
    TensorND src = ssrc;
    TensorND dst = sdst;
    auto wsb =
            get_workspace_bundle(sworkspace.raw_ptr, ssrc.layout, sdst.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(this->handle()), &wsb);
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(ssrc, src).src_to_comp_type(sdst, dst);
    }
    {
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
            kern_param.ph = ph, kern_param.pw = pw,
            kern_param.window_h = window_h, kern_param.window_w = window_w,
            kern_param.sh = sh, kern_param.sw = sw;
            auto&& stream = cuda_stream(handle());
            return pooling2d::do_pooling2d_int8_cdiv4hwn4(
                    src.compatible_ptr<int8_t>(), dst.compatible_ptr<int8_t>(),
                    kern_param, stream, static_cast<uint32_t>(param().mode));
        } else if (param().format == Format::NCHW4) {
            pooling2d::Param kern_param;
            size_t n = src.layout[0], hi = src.layout[2], wi = src.layout[3],
                   c = src.layout[1], ho = dst.layout[2], wo = dst.layout[3];
            c = c * 4;
            size_t ph = param().pad_h, pw = param().pad_w;
            size_t window_h = param().window_h, window_w = param().window_w;
            size_t sh = param().stride_h, sw = param().stride_w;
            kern_param.n = n, kern_param.c = c, kern_param.hi = hi,
            kern_param.wi = wi, kern_param.ho = ho, kern_param.wo = wo,
            kern_param.ph = ph, kern_param.pw = pw,
            kern_param.window_h = window_h, kern_param.window_w = window_w,
            kern_param.sh = sh, kern_param.sw = sw;
            auto&& stream = cuda_stream(handle());
            return pooling2d::do_pooling2d_int8_ncdiv4hw4(
                    src.compatible_ptr<int8_t>(), dst.compatible_ptr<int8_t>(),
                    kern_param, stream, static_cast<uint32_t>(param().mode));
        } else if (param().format == Format::NCHW32) {
            pooling2d::Param kern_param;
            size_t n = src.layout[0], hi = src.layout[2], wi = src.layout[3],
                   c = src.layout[1], ho = dst.layout[2], wo = dst.layout[3];
            c = c * 32;
            size_t ph = param().pad_h, pw = param().pad_w;
            size_t window_h = param().window_h, window_w = param().window_w;
            size_t sh = param().stride_h, sw = param().stride_w;
            kern_param.n = n, kern_param.c = c, kern_param.hi = hi,
            kern_param.wi = wi, kern_param.ho = ho, kern_param.wo = wo,
            kern_param.ph = ph, kern_param.pw = pw,
            kern_param.window_h = window_h, kern_param.window_w = window_w,
            kern_param.sh = sh, kern_param.sw = sw;
            auto&& stream = cuda_stream(handle());
            return pooling2d::do_pooling2d_int8_ncdiv32hw32(
                    src.compatible_ptr<int8_t>(), dst.compatible_ptr<int8_t>(),
                    kern_param, stream, static_cast<uint32_t>(param().mode));
        }
        auto handle = cudnn_handle(this->handle());
        setup_descs(src.layout, dst.layout);
        dt_float32 alpha = 1.0f, beta = 0.0f;
        cudnn_check(cudnnPoolingForward(handle, pooling_desc.desc, &alpha,
                                        src_desc.desc, src.raw_ptr, &beta,
                                        dst_desc.desc, dst.raw_ptr));
    }
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(dst, sdst);
    }
}

void PoolingBackwardImpl::setup_descs(const TensorLayout& src,
                                      const TensorLayout& dst,
                                      const TensorLayout& diff,
                                      const TensorLayout& grad) {
    src_desc.set(src);
    dst_desc.set(dst);
    diff_desc.set(diff);
    grad_desc.set(grad);
    pooling_desc.set(this->param());
}

WorkspaceBundle PoolingBackwardImpl::get_workspace_bundle(
        void* ptr, const TensorLayout& src, const TensorLayout& dst,
        const TensorLayout& diff, const TensorLayout& grad) const {
    SmallVector<size_t> sizes;
    TensorLayout fsrc = src;
    TensorLayout fdst = dst;
    TensorLayout fdiff = diff;
    TensorLayout fgrad = grad;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    get_workspace(fsrc);
    get_workspace(fdst);
    get_workspace(fdiff);
    get_workspace(fgrad);
    return {ptr, std::move(sizes)};
}

void PoolingBackwardImpl::exec(_megdnn_tensor_in ssrc, _megdnn_tensor_in sdst,
                               _megdnn_tensor_in sdiff,
                               _megdnn_tensor_out sgrad,
                               _megdnn_workspace sworkspace) {
    check_exec(ssrc.layout, sdst.layout, sdiff.layout, sgrad.layout,
               sworkspace.size);
    auto handle = cudnn_handle(this->handle());
    TensorND src = ssrc;
    TensorND dst = sdst;
    TensorND diff = sdiff;
    TensorND grad = sgrad;
    auto wsb = get_workspace_bundle(sworkspace.raw_ptr, ssrc.layout,
                                    sdst.layout, sdiff.layout, sgrad.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(this->handle()), &wsb);
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(ssrc, src)
                .src_to_comp_type(sdst, dst)
                .src_to_comp_type(sdiff, diff)
                .src_to_comp_type(sgrad, grad);
    }
    {
        setup_descs(src.layout, dst.layout, diff.layout, grad.layout);
        float alpha = 1.0f, beta = 0.0f;
        cudnn_check(cudnnPoolingBackward(
                handle, pooling_desc.desc, &alpha, dst_desc.desc, dst.raw_ptr,
                diff_desc.desc, diff.raw_ptr, src_desc.desc, src.raw_ptr, &beta,
                grad_desc.desc, grad.raw_ptr));
    }
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(grad, sgrad);
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
