/**
 * \file dnn/src/cuda/elemwise_multi_type/kern_iXxf32xf32xi8.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./kern.cuh"

#include "megdnn/dtype.h"
#include "src/common/elemwise_multi_type/kern_defs.cuh"
#include "src/cuda/utils.cuh"

using namespace megdnn;

namespace {

template <typename T>
struct __builtin_align__(sizeof(T) * 4) Packed4 {
    T v[4];
};

template <typename stype, typename dtype>
__global__ void kern_1d(const stype* x, const float* k, const float* b,
                        dtype* y, uint32_t n) {
    elemwise_multi_type::Fma3iXxf32xf32xiYOp<stype, dtype> op;
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        y[i] = op(x[i], k[i], b[i]);
    }
}

template <typename stype, typename dtype>
void invoke_kern_1d(const stype* x, const float* k, const float* b, dtype* y,
                    uint32_t n, cudaStream_t stream) {
    dim3 threads = NR_THREADS;
    dim3 blocks = DIVUP(n, NR_THREADS);
    kern_1d<stype, dtype><<<blocks, threads, 0, stream>>>(x, k, b, y, n);
    after_kernel_launch();
}

template <typename stype, typename dtype>
__global__ void kern_2d_fallback(const stype* x, const float* k, const float* b,
                                 dtype* y, uint32_t m, uint32_t n) {
    uint32_t i = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t j = threadIdx.x + blockIdx.x * blockDim.x;
    elemwise_multi_type::Fma3iXxf32xf32xiYOp<stype, dtype> op;
    if (i < m && j < n) {
        y[i * n + j] = op(x[i * n + j], k[j], b[j]);
    }
}

template <typename stype, typename dtype>
__global__ void kern_2d_mul4(const stype* __restrict x,
                             const float* __restrict k,
                             const float* __restrict b, dtype* y_, uint32_t m,
                             uint32_t n) {
    uint32_t i = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t j = threadIdx.x + blockIdx.x * blockDim.x;
    elemwise_multi_type::Fma3iXxf32xf32xiYOp<stype, dtype> op;
    Packed4<dtype>* __restrict__ y = (Packed4<dtype>*)y_;
    if (i < m && j < n) {
        stype x0 = x[(i * n + j) * 4 + 0];
        stype x1 = x[(i * n + j) * 4 + 1];
        stype x2 = x[(i * n + j) * 4 + 2];
        stype x3 = x[(i * n + j) * 4 + 3];
        float k0 = k[j * 4 + 0];
        float k1 = k[j * 4 + 1];
        float k2 = k[j * 4 + 2];
        float k3 = k[j * 4 + 3];
        float b0 = b[j * 4 + 0];
        float b1 = b[j * 4 + 1];
        float b2 = b[j * 4 + 2];
        float b3 = b[j * 4 + 3];
        Packed4<dtype> pack;
        pack.v[0] = op(x0, k0, b0);
        pack.v[1] = op(x1, k1, b1);
        pack.v[2] = op(x2, k2, b2);
        pack.v[3] = op(x3, k3, b3);
        y[i * n + j] = pack;
    }
}

template <typename stype, typename dtype>
void invoke_kern_2d(const stype* x, const float* k, const float* b, dtype* y,
                    uint32_t m, uint32_t n, cudaStream_t stream) {
    if (n % 4 == 0 && is_same<dtype, int8_t>::value) {
        dim3 threads(NR_THREADS_X, NR_THREADS_Y);
        dim3 blocks(DIVUP(n / 4, NR_THREADS_X), DIVUP(m, NR_THREADS_Y));
        // each thread process 4 elems
        // template to avoid compile error
        kern_2d_mul4<stype, dtype>
                <<<blocks, threads, 0, stream>>>(x, k, b, y, m, n / 4);
    } else {
        dim3 threads(NR_THREADS_X, NR_THREADS_Y);
        dim3 blocks(DIVUP(n, NR_THREADS_X), DIVUP(m, NR_THREADS_Y));
        kern_2d_fallback<stype, dtype>
                <<<blocks, threads, 0, stream>>>(x, k, b, y, m, n);
        after_kernel_launch();
    }
}

}  // anonymous namespace

using namespace megdnn;

template <typename stype>
void cuda::elemwise_multi_type::fma3_iXxf32xf32xi8_bcast_1x(
        const stype* x, const float* k, const float* b, dt_int8* y, uint32_t m,
        uint32_t n, cudaStream_t stream) {
    if (m == 1) {
        invoke_kern_1d(x, k, b, y, n, stream);
    } else {
        invoke_kern_2d(x, k, b, y, m, n, stream);
    }
}

#define INST(stype)                                                       \
    template void                                                         \
    cuda::elemwise_multi_type::fma3_iXxf32xf32xi8_bcast_1x<stype>(        \
            const stype*, const float*, const float*, dt_int8*, uint32_t, \
            uint32_t, cudaStream_t)
#define cb(t) INST(DTypeTrait<t>::ctype);
MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
#undef INST
