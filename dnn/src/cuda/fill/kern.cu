/**
 * \file dnn/src/cuda/fill/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/fill/kern.cuh"
#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

namespace {

template <typename T>
__global__ void kernel(T *dst, T value, uint32_t size) {
    int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        dst[i] = value;
    }
}

} // anonymous namespace

namespace megdnn {
namespace cuda {
namespace fill {

template <typename T>
void exec_internal(T *dst, T value, size_t size, cudaStream_t stream) {
    kernel<T><<<DIVUP(size, NR_THREADS), NR_THREADS, 0, stream>>>(dst, value, size);
    after_kernel_launch();
}

#define INST(T) template void exec_internal<T>(T *, \
        T, size_t, cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

} // namespace fill
} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
