/**
 * \file dnn/src/cuda/tensor_remap/tensor_remap.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/tensor_remap/tensor_remap.cuh"

namespace megdnn {
namespace cuda {
namespace {

template <typename ctype>
__global__ void forward_kernel(const ctype* src, const int* map, ctype* dst,
                               uint32_t sdim, uint32_t ddim,
                               array_wrapper<int, MEGDNN_MAX_NDIM> sstride,
                               array_wrapper<int, MEGDNN_MAX_NDIM> dstride,
                               array_wrapper<uint32_t, MEGDNN_MAX_NDIM> dshape,
                               uint32_t total) {
    uint32_t didx_cont = threadIdx.x + blockIdx.x * blockDim.x;
    if (didx_cont < total) {
        uint32_t midx = didx_cont * sdim;
        uint32_t didx = 0u;
        for (uint32_t j = ddim; j > 0u; --j) {
            uint32_t i = j - 1u;
            uint32_t didx_cur = didx_cont % dshape.data[i];
            didx_cont /= dshape.data[i];
            didx += didx_cur * dstride.data[i];
        }
        uint32_t sidx = 0u;
        for (uint32_t i = 0u; i < sdim; ++i) {
            uint32_t sidx_cur = map[midx + i];
            sidx += sidx_cur * sstride.data[i];
        }
        dst[didx] = src[sidx];
    }
}

template <typename ctype>
__global__ void fill_zero_kernel(ctype* a, uint32_t dim,
                                 array_wrapper<int, MEGDNN_MAX_NDIM> stride,
                                 array_wrapper<uint32_t, MEGDNN_MAX_NDIM> shape,
                                 uint32_t total) {
    uint32_t idx_cont = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx_cont < total) {
        uint32_t idx = 0u;
        for (uint32_t j = dim; j > 0u; --j) {
            uint32_t i = j - 1u;
            uint32_t idx_cur = idx_cont % shape.data[i];
            idx_cont /= shape.data[i];
            idx += idx_cur * stride.data[i];
        }
        a[idx] = 0.0f;
    }
}

template <typename ctype>
__global__ void backward_kernel(const ctype* diff, const int* map, ctype* grad,
                                uint32_t sdim, uint32_t ddim,
                                array_wrapper<int, MEGDNN_MAX_NDIM> sstride,
                                array_wrapper<int, MEGDNN_MAX_NDIM> dstride,
                                array_wrapper<uint32_t, MEGDNN_MAX_NDIM> dshape,
                                uint32_t total) {
    uint32_t didx_cont = threadIdx.x + blockIdx.x * blockDim.x;
    if (didx_cont < total) {
        uint32_t midx = didx_cont * sdim;
        uint32_t didx = 0u;
        for (uint32_t j = ddim; j > 0u; --j) {
            uint32_t i = j - 1u;
            uint32_t didx_cur = didx_cont % dshape.data[i];
            didx_cont /= dshape.data[i];
            didx += didx_cur * dstride.data[i];
        }
        uint32_t sidx = 0u;
        for (uint32_t i = 0u; i < sdim; ++i) {
            uint32_t sidx_cur = map[midx + i];
            sidx += sidx_cur * sstride.data[i];
        }
        atomicAdd(&grad[sidx], diff[didx]);
    }
}

template <typename ctype>
__global__ void backward_kernel_non_overlapping(
        const ctype* diff, const int* map, ctype* grad, uint32_t sdim,
        uint32_t ddim, array_wrapper<int, MEGDNN_MAX_NDIM> sstride,
        array_wrapper<int, MEGDNN_MAX_NDIM> dstride,
        array_wrapper<uint32_t, MEGDNN_MAX_NDIM> dshape, uint32_t total) {
    uint32_t didx_cont = threadIdx.x + blockIdx.x * blockDim.x;
    if (didx_cont < total) {
        uint32_t midx = didx_cont * sdim;
        uint32_t didx = 0u;
        for (uint32_t j = ddim; j > 0u; --j) {
            uint32_t i = j - 1u;
            uint32_t didx_cur = didx_cont % dshape.data[i];
            didx_cont /= dshape.data[i];
            didx += didx_cur * dstride.data[i];
        }
        uint32_t sidx = 0u;
        for (uint32_t i = 0u; i < sdim; ++i) {
            uint32_t sidx_cur = map[midx + i];
            sidx += sidx_cur * sstride.data[i];
        }
        grad[sidx] = diff[didx];
    }
}

}  // anonymous namespace

namespace tensor_remap {
template <typename ctype>
void forward(const ctype* src, const int* map, ctype* dst, uint32_t sdim,
             uint32_t ddim, const array_wrapper<int, MEGDNN_MAX_NDIM>& sstride,
             const array_wrapper<int, MEGDNN_MAX_NDIM>& dstride,
             const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& dshape,
             cudaStream_t stream) {
    uint32_t total = 1u;
    for (uint32_t i = 0u; i < ddim; ++i)
        total *= dshape.data[i];
    uint32_t threads =
            query_blocksize_for_kernel((void*)&forward_kernel<ctype>);
    uint32_t blocks = DIVUP(total, threads);
    forward_kernel<ctype><<<blocks, threads, 0, stream>>>(
            src, map, dst, sdim, ddim, sstride, dstride, dshape, total);
    after_kernel_launch();
}

template <typename ctype>
void backward(const ctype* diff, const int* map, ctype* grad, uint32_t sdim,
              uint32_t ddim, const array_wrapper<int, MEGDNN_MAX_NDIM>& sstride,
              const array_wrapper<int, MEGDNN_MAX_NDIM>& dstride,
              const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& sshape,
              const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& dshape,
              bool is_non_overlapping, cudaStream_t stream) {
    {
        // Fill grad with zeros.
        uint32_t total = 1u;
        for (uint32_t i = 0u; i < sdim; ++i)
            total *= sshape.data[i];
        uint32_t threads =
                query_blocksize_for_kernel((void*)&fill_zero_kernel<ctype>);
        uint32_t blocks = DIVUP(total, threads);
        fill_zero_kernel<ctype><<<blocks, threads, 0, stream>>>(
                grad, sdim, sstride, sshape, total);
        after_kernel_launch();
    }
    {
        // Update grad.
        uint32_t total = 1u;
        for (uint32_t i = 0u; i < ddim; ++i)
            total *= dshape.data[i];
        if (is_non_overlapping) {
            uint32_t threads = query_blocksize_for_kernel(
                    (void*)&backward_kernel_non_overlapping<ctype>);
            uint32_t blocks = DIVUP(total, threads);
            backward_kernel_non_overlapping<ctype>
                    <<<blocks, threads, 0, stream>>>(diff, map, grad, sdim,
                                                     ddim, sstride, dstride,
                                                     dshape, total);
        } else {
            uint32_t threads =
                    query_blocksize_for_kernel((void*)&backward_kernel<ctype>);
            uint32_t blocks = DIVUP(total, threads);
            backward_kernel<ctype><<<blocks, threads, 0, stream>>>(
                    diff, map, grad, sdim, ddim, sstride, dstride, dshape,
                    total);
        }
        after_kernel_launch();
    }
}

#define INST(T)                                                                \
    template void forward<T>(                                                  \
            const T* src, const int* map, T* dst, uint32_t sdim,               \
            uint32_t ddim, const array_wrapper<int, MEGDNN_MAX_NDIM>& sstride, \
            const array_wrapper<int, MEGDNN_MAX_NDIM>& dstride,                \
            const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& dshape,            \
            cudaStream_t stream);                                              \
    template void backward<T>(                                                 \
            const T* diff, const int* map, T* grad, uint32_t sdim,             \
            uint32_t ddim, const array_wrapper<int, MEGDNN_MAX_NDIM>& sstride, \
            const array_wrapper<int, MEGDNN_MAX_NDIM>& dstride,                \
            const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& sshape,            \
            const array_wrapper<uint32_t, MEGDNN_MAX_NDIM>& dshape,            \
            bool is_non_overlapping, cudaStream_t stream);
INST(dt_float32)
INST(dt_int32)

#undef INST

}  // namespace tensor_remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
