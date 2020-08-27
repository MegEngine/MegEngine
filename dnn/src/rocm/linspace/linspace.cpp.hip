/**
 * \file src/rocm/linspace/linspace.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
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
