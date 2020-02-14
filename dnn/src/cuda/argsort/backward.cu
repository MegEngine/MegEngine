/**
 * \file dnn/src/cuda/argsort/backward.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./argsort.cuh"
#include "./backward.cuh"

#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace argsort;

namespace {

template <typename T>
__global__ void backward_kernel(uint32_t dst_w, uint32_t src_w,
                                uint32_t src_size, T* dst, const T* src_data,
                                const int* src_idx) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < src_size) {
        uint32_t r = idx / src_w;
        dst[r * dst_w + src_idx[idx]] = src_data[idx];
    }
}

}  // namespace

template <typename T>
void argsort::backward_proxy(uint32_t dst_h, uint32_t dst_w, uint32_t src_w,
                             T* dst, const T* src_data, const int* src_idx,
                             cudaStream_t stream) {
    if (dst_w != src_w) {
        cudaMemsetAsync(dst, 0, dst_h * dst_w * sizeof(T), stream);
    }

    uint32_t src_size = dst_h * src_w;
    backward_kernel<<<DIVUP(src_size, 512), 512, 0, stream>>>(
            dst_w, src_w, src_size, dst, src_data, src_idx);
    after_kernel_launch();
}

namespace megdnn {
namespace cuda {
namespace argsort {

#define INST(T)                                                             \
    template void backward_proxy(uint32_t dst_h, uint32_t dst_w,            \
                                 uint32_t src_w, T* dst, const T* src_data, \
                                 const int* src_idx, cudaStream_t stream);
ARGSORT_FOREACH_CTYPE(INST)
#undef INST

}  // namespace argsort
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
