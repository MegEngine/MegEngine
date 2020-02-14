/**
 * \file dnn/src/cuda/dot/dot.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/dot/dot.cuh"

#include "src/cuda/utils.cuh"
#include "src/cuda/cub/util_ptx.cuh"

namespace {

using namespace megdnn;

template <typename T> __global__ void kernel(const T *a, const T *b,
        dt_float32 *c,
        uint32_t n, int32_t strideA, int32_t strideB)
{
    uint32_t tid = threadIdx.x;
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    volatile __shared__ dt_float32 sdata[256];
    sdata[tid] = (gid < n ?
            dt_float32(a[gid*strideA]) * dt_float32(b[gid*strideB])
            : 0);
    __syncthreads();
    if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        cub::WARP_SYNC(0xffffffff);
        if (tid < 16)
            sdata[tid] += sdata[tid + 16];
        cub::WARP_SYNC(0xffffffff);
        if (tid < 8)
            sdata[tid] += sdata[tid + 8];
        cub::WARP_SYNC(0xffffffff);
        if (tid < 4)
            sdata[tid] += sdata[tid + 4];
        cub::WARP_SYNC(0xffffffff);
        if (tid < 2)
            sdata[tid] += sdata[tid + 2];
        cub::WARP_SYNC(0xffffffff);
        if (tid < 1)
            sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0)
        atomicAdd(c, sdata[0]);
}

template <typename T> __global__ void cvt_kernel(const dt_float32 *src, T *dst)
{
    dst[0] = T(src[0]);
}

} // anonymous namespace

namespace megdnn {
namespace cuda {
namespace dot {

template <typename T> void run(const T *a, const T *b, T *c, float *workspace,
        uint32_t n, int32_t strideA, int32_t strideB,
        cudaStream_t stream)
{
    cuda_check(cudaMemsetAsync(workspace, 0, sizeof(dt_float32), stream));
    // each block add 256 entries
    uint32_t blocks = DIVUP(n, 256);
    uint32_t threads = 256;
    kernel<T><<<blocks, threads, 0, stream>>>(a, b,
            workspace,
            n, strideA, strideB);
    cvt_kernel<T><<<1, 1, 0, stream>>>(workspace, c);
    after_kernel_launch();
}

template void run<dt_float16>(const dt_float16 *a, const dt_float16 *b,
        dt_float16 *c, dt_float32 *workspace,
        uint32_t n, int32_t strideA, int32_t strideB,
        cudaStream_t stream);

} // namespace dot
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
