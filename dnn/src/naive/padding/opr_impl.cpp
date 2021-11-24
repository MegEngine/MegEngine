/**
 * \file dnn/src/naive/padding/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/naive/padding/opr_impl.h"
#include <math.h>
#include <stdio.h>
#include "src/common/utils.h"
#include "src/naive/handle.h"
namespace megdnn {
namespace naive {

struct ShapeParams {
    size_t src_shape[MEGDNN_MAX_NDIM];
    size_t dst_shape[MEGDNN_MAX_NDIM];
    ptrdiff_t src_stride[MEGDNN_MAX_NDIM];
    ptrdiff_t dst_stride[MEGDNN_MAX_NDIM];
    size_t offsets[MEGDNN_MAX_NDIM * 2];
};

template <typename T>
void exec_const_internal(
        const size_t ndim, const size_t total_out_nr, const T* const src, T* const dst,
        ShapeParams params, const T padding_val) MEGDNN_NOEXCEPT {
    rep(out_index, total_out_nr) {
        bool in_src_valid_area = true;
        size_t in_index = 0;
        size_t out_index_tmp = out_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            size_t dim_index = out_index_tmp / params.dst_stride[dim];
            out_index_tmp -= dim_index * params.dst_stride[dim];
            in_src_valid_area &=
                    (dim_index >= params.offsets[dim * 2] &&
                     dim_index < params.offsets[dim * 2] + params.src_shape[dim]);
            in_index += (dim_index - params.offsets[dim * 2]) * params.src_stride[dim];
        }

        if (in_src_valid_area) {
            dst[out_index] = src[in_index];
        } else {
            dst[out_index] = padding_val;
        }
    }
}

template <typename T>
void exec_replicate_internal(
        const size_t ndim, const size_t total_out_nr, const T* const src, T* const dst,
        ShapeParams params) MEGDNN_NOEXCEPT {
    rep(out_index, total_out_nr) {
        size_t in_index = 0;
        size_t out_index_tmp = out_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            size_t dim_index = out_index_tmp / params.dst_stride[dim];
            out_index_tmp -= dim_index * params.dst_stride[dim];
            dim_index = (size_t)std::min(
                    (long long)params.src_shape[dim] - 1,
                    std::max(
                            (long long)dim_index - (long long)params.offsets[dim * 2],
                            (long long)0));
            in_index += dim_index * params.src_stride[dim];
        }
        dst[out_index] = src[in_index];
    }
}

template <typename T>
void exec_reflect_internal(
        const size_t ndim, const size_t total_out_nr, const T* const src, T* const dst,
        ShapeParams params) MEGDNN_NOEXCEPT {
    rep(out_index, total_out_nr) {
        size_t in_index = 0;
        size_t out_index_tmp = out_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            long long dim_index = out_index_tmp / params.dst_stride[dim];
            out_index_tmp -= dim_index * params.dst_stride[dim];
            dim_index -= (long long)params.offsets[dim * 2];
            dim_index = std::max(dim_index, -dim_index);
            dim_index = std::min(
                    dim_index, 2 * (long long)params.src_shape[dim] - dim_index - 2);
            in_index += size_t(dim_index) * (size_t)params.src_stride[dim];
        }
        dst[out_index] = src[in_index];
    }
}

template <typename T>
void backward_exec_const_internal(
        const size_t ndim, const size_t total_in_nr, const T* const src, T* const dst,
        ShapeParams params) MEGDNN_NOEXCEPT {
    rep(in_index, total_in_nr) {
        bool in_dst_valid_area = true;
        size_t out_index = 0;
        size_t in_index_tmp = in_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            size_t dim_index = in_index_tmp / params.src_stride[dim];
            in_index_tmp -= dim_index * params.src_stride[dim];
            in_dst_valid_area &=
                    (dim_index >= params.offsets[dim * 2] &&
                     dim_index < params.offsets[dim * 2] + params.dst_shape[dim]);
            out_index += (dim_index - params.offsets[dim * 2]) * params.dst_stride[dim];
        }
        if (in_dst_valid_area) {
            dst[out_index] = src[in_index];
        }
    }
}

template <typename T>
void backward_exec_replicate_internal(
        const size_t ndim, const size_t total_in_nr, const T* const src, T* const dst,
        ShapeParams params) MEGDNN_NOEXCEPT {
    rep(in_index, total_in_nr) {
        size_t out_index = 0;
        size_t in_index_tmp = in_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            size_t dim_index = in_index_tmp / params.src_stride[dim];
            in_index_tmp -= dim_index * params.src_stride[dim];
            dim_index = (size_t)std::min(
                    (long long)params.dst_shape[dim] - 1,
                    std::max(
                            (long long)dim_index - (long long)params.offsets[dim * 2],
                            (long long)0));
            out_index += dim_index * params.dst_stride[dim];
        }
        dst[out_index] += src[in_index];
    }
}

template <typename T>
void backward_exec_reflect_internal(
        const size_t ndim, const size_t total_in_nr, const T* const src, T* const dst,
        ShapeParams params) MEGDNN_NOEXCEPT {
    rep(in_index, total_in_nr) {
        size_t out_index = 0;
        size_t in_index_tmp = in_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            long long dim_index = in_index_tmp / params.src_stride[dim];
            in_index_tmp -= dim_index * params.src_stride[dim];
            dim_index -= (long long)params.offsets[dim * 2];
            dim_index = std::max(dim_index, -dim_index);
            dim_index = std::min(
                    dim_index, 2 * (long long)params.dst_shape[dim] - dim_index - 2);
            out_index += size_t(dim_index) * (size_t)params.dst_stride[dim];
        }
        dst[out_index] += src[in_index];
    }
}

void PaddingForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    forward_check_exec(src.layout, dst.layout);
    SmallVector<size_t> offsets(get_offsets());
    ShapeParams params;
    for (size_t i = 0; i < src.layout.ndim; ++i) {
        params.src_shape[i] = src.layout.shape[i];
        params.dst_shape[i] = dst.layout.shape[i];
        params.src_stride[i] = src.layout.stride[i];
        params.dst_stride[i] = dst.layout.stride[i];
        params.offsets[i * 2] = offsets[i * 2];
        params.offsets[i * 2 + 1] = offsets[i * 2 + 1];
    }

    size_t n = dst.layout.total_nr_elems();
    switch (param().padding_mode) {
        case param::Padding::PaddingMode::CONSTANT:
#define cb(DType)                                                       \
    if (src.layout.dtype == DType()) {                                  \
        using T = typename DTypeTrait<DType>::ctype;                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_const_internal<T>(            \
                src.layout.ndim, n, src.ptr<T>(), dst.ptr<T>(), params, \
                T(param().padding_val)));                               \
        return;                                                         \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
            break;
        case param::Padding::PaddingMode::REPLICATE:
#define cb(DType)                                                         \
    if (src.layout.dtype == DType()) {                                    \
        using T = typename DTypeTrait<DType>::ctype;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_replicate_internal<T>(          \
                src.layout.ndim, n, src.ptr<T>(), dst.ptr<T>(), params)); \
        return;                                                           \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
            break;
        case param::Padding::PaddingMode::REFLECT:
#define cb(DType)                                                         \
    if (src.layout.dtype == DType()) {                                    \
        using T = typename DTypeTrait<DType>::ctype;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_reflect_internal<T>(            \
                src.layout.ndim, n, src.ptr<T>(), dst.ptr<T>(), params)); \
        return;                                                           \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
            break;
        default:
            megdnn_assert(false, "unsupported padding mode!");
    }
}

void PaddingBackwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    backward_check_exec(src.layout, dst.layout);
    SmallVector<size_t> offsets(get_offsets());
    ShapeParams params;
    for (size_t i = 0; i < src.layout.ndim; ++i) {
        params.src_shape[i] = src.layout.shape[i];
        params.dst_shape[i] = dst.layout.shape[i];
        params.src_stride[i] = src.layout.stride[i];
        params.dst_stride[i] = dst.layout.stride[i];
        params.offsets[i * 2] = offsets[i * 2];
        params.offsets[i * 2 + 1] = offsets[i * 2 + 1];
    }
    size_t n = src.layout.total_nr_elems();

    memset(dst.raw_ptr(), 0, dst.layout.access_bytes());

    switch (param().padding_mode) {
        case param::Padding::PaddingMode::CONSTANT:
#define cb(DType)                                                         \
    if (src.layout.dtype == DType()) {                                    \
        using T = typename DTypeTrait<DType>::ctype;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(backward_exec_const_internal<T>(     \
                src.layout.ndim, n, src.ptr<T>(), dst.ptr<T>(), params)); \
        return;                                                           \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
            break;
        case param::Padding::PaddingMode::REPLICATE:
#define cb(DType)                                                         \
    if (src.layout.dtype == DType()) {                                    \
        using T = typename DTypeTrait<DType>::ctype;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(backward_exec_replicate_internal<T>( \
                src.layout.ndim, n, src.ptr<T>(), dst.ptr<T>(), params)); \
        return;                                                           \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
            break;
        case param::Padding::PaddingMode::REFLECT:
#define cb(DType)                                                         \
    if (src.layout.dtype == DType()) {                                    \
        using T = typename DTypeTrait<DType>::ctype;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(backward_exec_reflect_internal<T>(   \
                src.layout.ndim, n, src.ptr<T>(), dst.ptr<T>(), params)); \
        return;                                                           \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
            break;
        default:
            megdnn_assert(false, "unsupported padding mode!");
    }
}

size_t PaddingForwardImpl::get_workspace_in_bytes(
        const TensorLayout& /* src */, const TensorLayout& /* dst */) {
    return 0;
}

size_t PaddingBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& /* src */, const TensorLayout& /* dst */) {
    return 0;
}
}  // namespace naive
}  // namespace megdnn