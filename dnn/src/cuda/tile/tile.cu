/**
 * \file dnn/src/cuda/tile/tile.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/tile/tile.cuh"

#include "src/cuda/utils.cuh"
#include <numeric>
#include <functional>
#include <stdint.h>
#include "megdnn/dtype.h"

namespace megdnn {
namespace cuda {
namespace tile {

template <typename T>
__global__ void forward_kernel_1d(const T *src, T *dst,
        uint32_t sshape, uint32_t dshape, uint32_t tshape)
{
    uint32_t di = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t si = di % sshape;
    if (di < dshape) {
        dst[di] = src[si];
    }
}

template <typename T>
void forward_proxy_1d(const T *src, T *dst,
        size_t sshape, size_t dshape, size_t tshape,
        cudaStream_t stream)
{
    size_t NR_BLOCKS = DIVUP(dshape, NR_THREADS);
    forward_kernel_1d<<<NR_BLOCKS, NR_THREADS, 0, stream>>>(src, dst,
            sshape, dshape, tshape);
}

template <typename T>
__global__ void forward_kernel_2d(const T *src, T *dst,
        uint32_t sshape0, uint32_t sshape1,
        uint32_t dshape0, uint32_t dshape1,
        uint32_t tshape0, uint32_t tshape1)
{
    uint32_t dix = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t diy = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t six = dix % sshape0;
    uint32_t siy = diy % sshape1;
    uint32_t diz = diy * dshape0 + dix;
    uint32_t siz = siy * sshape0 + six;
    if (dix < dshape0 && diy < dshape1) {
        dst[diz] = src[siz];
    }
}

template <typename T>
void forward_proxy_2d(const T *src, T *dst,
        size_t sshape0, size_t sshape1,
        size_t dshape0, size_t dshape1,
        size_t tshape0, size_t tshape1,
        cudaStream_t stream)
{
    dim3 threads(NR_THREADS_X, NR_THREADS_Y);
    dim3 blocks(DIVUP(dshape0, threads.x), DIVUP(dshape1, threads.y));
    forward_kernel_2d<<<blocks, threads, 0, stream>>>(src, dst,
            sshape0, sshape1,
            dshape0, dshape1,
            tshape0, tshape1);
}

template <typename T, uint32_t ndim>
__global__ void forward_kernel_generic_tpl(const T * __restrict__ src,
        T * __restrict__ dst,
        uint32_t n,
        array_wrapper<uint32_t, ndim> sshape,
        array_wrapper<uint32_t, ndim> dshape,
        array_wrapper<uint32_t, ndim> tshape)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < n) {
        uint32_t didx = tidx;
        uint32_t sidx = 0;
        uint32_t base = 1;
        // calculate index
#pragma unroll
        for (size_t i = ndim; i > 0; --i) {
            size_t cidx = didx % sshape.data[i-1];
            sidx += cidx * base;
            base *= sshape.data[i-1];
            didx /= dshape.data[i-1];
        }
        dst[tidx] = src[sidx];
    }
}

template <typename T, size_t ndim>
void forward_proxy_generic_tpl(const T *src, T *dst,
        const size_t *sshape_, const size_t *dshape_, const size_t *tshape_,
        cudaStream_t stream)
{
    array_wrapper<uint32_t, ndim> sshape, dshape, tshape;
    for (size_t i = 0; i < ndim; ++i) sshape.data[i] = sshape_[i];
    for (size_t i = 0; i < ndim; ++i) dshape.data[i] = dshape_[i];
    for (size_t i = 0; i < ndim; ++i) tshape.data[i] = tshape_[i];
    size_t n = std::accumulate(dshape_, dshape_ + ndim, size_t(1),
            std::multiplies<size_t>());
    size_t NR_BLOCKS = DIVUP(n, NR_THREADS);
    forward_kernel_generic_tpl<T, ndim><<<NR_BLOCKS, NR_THREADS, 0, stream>>>(
            src, dst, n,
            sshape, dshape, tshape);
}

template <typename T>
void forward_proxy_generic(const T *src, T *dst, size_t ndim,
        const size_t *sshape_, const size_t *dshape_, const size_t *tshape_,
        cudaStream_t stream)
{
#define CASE(ndim) \
    case ndim: \
        forward_proxy_generic_tpl<T, ndim>(src, dst, \
                sshape_, dshape_, tshape_, stream); \
        break;
    switch (ndim) {
        CASE(2);
        CASE(3);
        CASE(4);
        CASE(5);
        CASE(6);
        default:
            megdnn_assert_internal(false);
    }
#undef CASE
}

template <typename T>
void forward_proxy(const T *src, T *dst, size_t ndim,
        const size_t *sshape_, const size_t *dshape_, const size_t *tshape_,
        cudaStream_t stream)
{
    if (ndim == 1) {
        forward_proxy_1d<T>(src, dst, sshape_[0], dshape_[0], tshape_[0], stream);
    } else if (ndim == 2 && dshape_[0] <= 65535 * NR_THREADS_Y) {
        // CUDA can launch 65535 blocks along axis Y at most.
        // Note that the index 1 and 0 are swapped, it is because in the kernel,
        // index zero corresponds to axis X (which is the lowest adjacent axis),
        // and index one corresponds to axis Y. However, outside the kernel,
        // our representation is the opposite.
        forward_proxy_2d<T>(src, dst,
                sshape_[1], sshape_[0],
                dshape_[1], dshape_[0],
                tshape_[1], tshape_[0],
                stream);
    } else {
        forward_proxy_generic<T>(src, dst,
                ndim, sshape_, dshape_, tshape_,
                stream);
    }
    after_kernel_launch();
}

#define INST(T) \
template void forward_proxy<T>(const T *src, T *dst, size_t ndim, \
        const size_t *sshape_, const size_t *dshape_, const size_t *tshape_, \
        cudaStream_t stream);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

#undef cb
#undef INST

} // namespace tile
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

