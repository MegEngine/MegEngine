/**
 * \file dnn/src/rocm/linspace/linspace.cpp.hip
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
#include "./linspace.h.hip"
#include "src/rocm/utils.h.hip"
#include "megdnn/dtype.h"

namespace {

template <typename T>
__global__ void kernel(T *dst, double start, double step, uint32_t n)
{
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        dst[i] = T(start + step*i);
    }
}

} // anonymous namespace

namespace megdnn {
namespace rocm {
namespace linspace {

template <typename T>
void exec_internal(T *dst, double start, double step, size_t n,
        hipStream_t stream)
{
    uint32_t threads = NR_THREADS;
    uint32_t blocks = DIVUP(n, threads);
    hipLaunchKernelGGL(kernel, 
                       dim3(blocks), dim3(threads), 0, stream,
                       dst, start, step, n);
    after_kernel_launch();
}

#define INST(T) template void exec_internal<T>(T *dst, \
        double start, double step, size_t n, hipStream_t stream);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

} // namespace linspace
} // namespace rocm
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
