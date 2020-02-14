/**
 * \file dnn/src/cuda/param_pack/param_pack.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/dtype.h"
#include "src/cuda/param_pack/param_pack.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace param_pack {

template <typename T>
__global__ void concat_kernel(const T** srcs, T* dst,
                                      const int32_t* table_outer,
                                      const int32_t* table_inner,
                                      size_t total_size) {
    size_t addr = threadIdx.x + blockIdx.x * blockDim.x;
    if (addr < total_size) {
        int32_t i = table_outer[addr];
        int32_t idx = table_inner[addr];
        if (idx != -1)
            dst[addr] = srcs[i][idx];
        else
            dst[addr] = 0;
    }
}

template <typename T>
__global__ void split_kernel(const T* src, T** dsts,
                                     const int32_t* table_outer,
                                     const int32_t* table_inner,
                                     size_t total_size) {
    size_t addr = threadIdx.x + blockIdx.x * blockDim.x;
    if (addr < total_size) {
        int32_t i = table_outer[addr];
        int32_t idx = table_inner[addr];
        if (idx != -1) {
            dsts[i][idx] = src[addr];
        }
    }
}

template <typename T>
void split_proxy(const T* src, T** dsts, size_t total_size,
                         const int32_t* table_outer, const int32_t* table_inner,
                         cudaStream_t stream) {
    size_t NR_BLOCKS = DIVUP(total_size, NR_THREADS);
    split_kernel<<<NR_BLOCKS, NR_THREADS, 0, stream>>>(
            src, dsts, table_outer, table_inner, total_size);
    after_kernel_launch();
}

template <typename T>
void concat_proxy(const T** srcs, T* dst, size_t total_size,
                          const int32_t* table_outer,
                          const int32_t* table_inner, cudaStream_t stream) {
    size_t NR_BLOCKS = DIVUP(total_size, NR_THREADS);
    concat_kernel<<<NR_BLOCKS, NR_THREADS, 0, stream>>>(
            srcs, dst, table_outer, table_inner, total_size);
    after_kernel_launch();
}

#define INST(T)                                                           \
    template void concat_proxy<T>(const T**, T*, size_t,          \
                                          const int32_t*, const int32_t*, \
                                          cudaStream_t);                  \
    template void split_proxy<T>(const T*, T**, size_t,           \
                                         const int32_t*, const int32_t*,  \
                                         cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
#undef INST

}  // namespace param_pack
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
