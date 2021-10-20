/**
 * \file dnn/src/rocm/eye/eye.cpp.hip
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "hip_header.h"
#include "megdnn/dtype.h"
#include "src/rocm/eye/eye.h.hip"
#include "src/rocm/utils.h.hip"

namespace {

template <typename T>
__global__ void kernel(T* dst, uint32_t m, uint32_t n, int k) {
    int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t x = i % n;
    int32_t y = i / n;
    if (i < m * n) {
        dst[i] = (y + k == x);
    }
}

}  // anonymous namespace

namespace megdnn {
namespace rocm {
namespace eye {

template <typename T>
void exec_internal(T* dst, size_t m, size_t n, int k, hipStream_t stream) {
    hipLaunchKernelGGL((kernel<T>), dim3(DIVUP(m * n, NR_THREADS)),
                       dim3(NR_THREADS), 0, stream, dst, m, n, k);
    after_kernel_launch();
}

#define INST(T) \
    template void exec_internal<T>(T*, size_t, size_t, int, hipStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

}  // namespace eye
}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
