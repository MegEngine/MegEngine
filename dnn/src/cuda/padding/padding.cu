/**
 * \file dnn/src/cuda/padding/padding.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include <algorithm>
#include <cstring>
#include <iostream>
#include "megdnn/basic_types.h"
#include "padding.cuh"
#include "src/cuda/int_fastdiv.cuh"
#include "src/cuda/query_blocksize.cuh"

namespace megdnn {
namespace cuda {
namespace padding {

struct ShapeParams {
    size_t src_shape[MEGDNN_MAX_NDIM];
    size_t dst_shape[MEGDNN_MAX_NDIM];
    Uint32Fastdiv src_stride[MEGDNN_MAX_NDIM];
    Uint32Fastdiv dst_stride[MEGDNN_MAX_NDIM];
    size_t offsets[MEGDNN_MAX_NDIM * 2];
};

template <typename T>
__global__ void paddingConst_kernel(const size_t ndim,
                                    const size_t total_out_nr,
                                    const T* const src, T* const dst,
                                    ShapeParams params,
                                    const float_t padding_val) {
    KERN_FOR(out_index, total_out_nr) {
        bool in_src_valid_area = true;
        size_t in_index = 0;
        size_t out_index_tmp = out_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            Uint32Fastdiv dst_stride = params.dst_stride[dim], src_stride = params.src_stride[dim];
            size_t src_shape = params.src_shape[dim];
            size_t offset = params.offsets[dim*2];

            size_t dim_index = out_index_tmp / dst_stride;
            in_src_valid_area &= (dim_index >= offset && dim_index < offset+src_shape);
            if(!in_src_valid_area) break;
            out_index_tmp -= dim_index * dst_stride.divisor();
            in_index += (dim_index - offset)*src_stride.divisor();
            /*
            size_t dim_index = out_index_tmp / params.dst_stride[dim];
            out_index_tmp -= dim_index * params.dst_stride[dim].divisor();
            in_src_valid_area &= (dim_index >= params.offsets[dim * 2] &&
                                  dim_index < params.offsets[dim * 2] +
                                                      params.src_shape[dim]);
            in_index += (dim_index - params.offsets[dim * 2]) *
                        params.src_stride[dim].divisor();
            */
        }
        dst[out_index] = in_src_valid_area ? src[in_index] : padding_val;
    }
}

template <typename T>
__global__ void paddingReplicate_kernel(const size_t ndim,
                                        const size_t total_out_nr,
                                        const T* const src, T* const dst,
                                        ShapeParams params, const float_t) {
    KERN_FOR(out_index, total_out_nr) {
        size_t in_index = 0;
        size_t out_index_tmp = out_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            size_t dim_index = out_index_tmp / params.dst_stride[dim];
            out_index_tmp -= dim_index * params.dst_stride[dim].divisor();
            dim_index = (size_t)llmin(
                    (long long)params.src_shape[dim] - 1,
                    llmax((long long)dim_index -
                                  (long long)params.offsets[dim * 2],
                          (long long)0));
            in_index += dim_index * params.src_stride[dim].divisor();
        }
        dst[out_index] = src[in_index];
    }
}

template <typename T>
__global__ void paddingReflect_kernel(const size_t ndim,
                                      const size_t total_out_nr,
                                      const T* const src, T* const dst,
                                      ShapeParams params, const float_t) {
    KERN_FOR(out_index, total_out_nr) {
        size_t in_index = 0;
        size_t out_index_tmp = out_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            long long dim_index = out_index_tmp / params.dst_stride[dim];
            out_index_tmp -= dim_index * params.dst_stride[dim].divisor();
            dim_index -= (long long)params.offsets[dim * 2];
            dim_index = llmax(dim_index, -dim_index);
            dim_index = llmin(dim_index, 2 * (long long)params.src_shape[dim] -
                                                 dim_index - 2);
            in_index += size_t(dim_index) *
                        (size_t)params.src_stride[dim].divisor();
        }
        dst[out_index] = src[in_index];
    }
}

template <typename T>
__global__ void paddingConstBackward_kernel(const size_t ndim,
                                            const size_t total_in_nr,
                                            const T* const src, T* const dst,
                                            ShapeParams params) {
    KERN_FOR(in_index, total_in_nr) {
        bool in_dst_valid_area = true;
        size_t out_index = 0;
        size_t in_index_tmp = in_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            size_t dim_index = in_index_tmp / params.src_stride[dim];
            in_index_tmp -= dim_index * params.src_stride[dim].divisor();
            in_dst_valid_area &= (dim_index >= params.offsets[dim * 2] &&
                                  dim_index < params.offsets[dim * 2] +
                                                      params.dst_shape[dim]);
            out_index += (dim_index - params.offsets[dim * 2]) *
                         params.dst_stride[dim].divisor();
        }
        if (in_dst_valid_area) {
            dst[out_index] = src[in_index];
        }
    }
}

template <typename T>
__global__ void paddingReplicateBackward_kernel(const size_t ndim,
                                                const size_t total_in_nr,
                                                const T* const src,
                                                T* const dst,
                                                ShapeParams params) {
    KERN_FOR(in_index, total_in_nr) {
        size_t out_index = 0;
        size_t in_index_tmp = in_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            size_t dim_index = in_index_tmp / params.src_stride[dim];
            in_index_tmp -= dim_index * params.src_stride[dim].divisor();
            dim_index = (size_t)llmin(
                    (long long)params.dst_shape[dim] - 1,
                    llmax((long long)dim_index -
                                  (long long)params.offsets[dim * 2],
                          (long long)0));
            out_index += dim_index * params.dst_stride[dim].divisor();
        }
        atomic_add(&dst[out_index], src[in_index]);
    }
}

template <typename T>
__global__ void paddingReflectBackward_kernel(const size_t ndim,
                                              const size_t total_in_nr,
                                              const T* const src, T* const dst,
                                              ShapeParams params) {
    KERN_FOR(in_index, total_in_nr) {
        size_t out_index = 0;
        size_t in_index_tmp = in_index;
        for (size_t dim = 0; dim <= ndim - 1; ++dim) {
            long long dim_index = in_index_tmp / params.src_stride[dim];
            in_index_tmp -= dim_index * params.src_stride[dim].divisor();
            dim_index -= (long long)params.offsets[dim * 2];
            dim_index = llmax(dim_index, -dim_index);
            dim_index = llmin(dim_index, 2 * (long long)params.dst_shape[dim] -
                                                 dim_index - 2);
            out_index += size_t(dim_index) *
                         (size_t)params.dst_stride[dim].divisor();
        }
        atomic_add(&dst[out_index], src[in_index]);
    }
}

template <typename T>
void padding_forward_proxy(const TensorND& src, const TensorND& dst,
                           size_t offsets[MEGDNN_MAX_NDIM * 2], uint32_t mode,
                           const float_t padding_val, cudaStream_t stream) {
    ShapeParams params;
    for (size_t i = 0; i < src.layout.ndim; ++i) {
        params.src_shape[i] = src.layout.shape[i];
        params.dst_shape[i] = dst.layout.shape[i];
        params.src_stride[i] = src.layout.stride[i];
        params.dst_stride[i] = dst.layout.stride[i];
        params.offsets[i * 2] = offsets[i * 2];
        params.offsets[i * 2 + 1] = offsets[i * 2 + 1];
    }

    void (*fwd_kern)(const size_t, const size_t, const T* const, T* const,
                     ShapeParams, const float_t);
    switch (mode) {
        case param_enumv::Padding::PaddingMode::CONSTANT:
            fwd_kern = paddingConst_kernel<T>;
            break;
        case param_enumv::Padding::PaddingMode::REPLICATE:
            fwd_kern = paddingReplicate_kernel<T>;
            break;
        case param_enumv::Padding::PaddingMode::REFLECT:
            fwd_kern = paddingReflect_kernel<T>;
            break;
        default:
            megdnn_assert(false, "invalid padding mode");
    }

    size_t total_nr = dst.layout.total_nr_elems();

    uint32_t nr_threads = query_blocksize_for_kernel(fwd_kern);
    dim3 threads(nr_threads);
    dim3 blocks(DIVUP(total_nr, nr_threads));
    fwd_kern<<<blocks, threads, 0, stream>>>(src.layout.ndim, total_nr,
                                             src.ptr<T>(), dst.ptr<T>(), params,
                                             padding_val);
    after_kernel_launch();
}

template <typename T>
void padding_backward_proxy(const TensorND& src, const TensorND& dst,
                            size_t offsets[MEGDNN_MAX_NDIM * 2], uint32_t mode,
                            cudaStream_t stream) {
    ShapeParams params;

    for (size_t i = 0; i < src.layout.ndim; ++i) {
        params.src_shape[i] = src.layout.shape[i];
        params.dst_shape[i] = dst.layout.shape[i];
        params.src_stride[i] = src.layout.stride[i];
        params.dst_stride[i] = dst.layout.stride[i];
        params.offsets[i * 2] = offsets[i * 2];
        params.offsets[i * 2 + 1] = offsets[i * 2 + 1];
    }

    cudaMemset(dst.raw_ptr, 0, dst.layout.access_bytes());

    void (*bwd_kern)(const size_t, const size_t, const T* const, T* const,
                     ShapeParams);

    switch (mode) {
        case param_enumv::Padding::PaddingMode::CONSTANT:
            bwd_kern = paddingConstBackward_kernel<T>;
            break;
        case param_enumv::Padding::PaddingMode::REPLICATE:
            bwd_kern = paddingReplicateBackward_kernel<T>;
            break;
        case param_enumv::Padding::PaddingMode::REFLECT:
            bwd_kern = paddingReflectBackward_kernel<T>;
            break;
        default:
            megdnn_assert(false, "invalid padding mode");
    }
    size_t total_nr = src.layout.total_nr_elems();
    uint32_t nr_threads = query_blocksize_for_kernel(bwd_kern);
    dim3 threads(nr_threads);
    dim3 blocks(DIVUP(total_nr, nr_threads));
    bwd_kern<<<blocks, threads, 0, stream>>>(
            src.layout.ndim, total_nr, src.ptr<T>(), dst.ptr<T>(), params);
    after_kernel_launch();
}

#define INST(T)                                                 \
    template void padding_forward_proxy<T>(                     \
            const TensorND& src, const TensorND& dst,           \
            size_t offsets[MEGDNN_MAX_NDIM * 2], uint32_t mode, \
            const float_t padding_val, cudaStream_t stream);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
#undef INST

#define INST(T)                                                 \
    template void padding_backward_proxy<T>(                    \
            const TensorND& src, const TensorND& dst,           \
            size_t offsets[MEGDNN_MAX_NDIM * 2], uint32_t mode, \
            cudaStream_t stream);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
#undef INST

}  // namespace padding
}  // namespace cuda
}  // namespace megdnn