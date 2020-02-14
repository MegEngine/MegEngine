/**
 * \file dnn/src/cuda/split/split.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/split/split.cuh"

#include "src/cuda/utils.cuh"
#include "megdnn/dtype.h"

namespace megdnn {
namespace cuda {
namespace split {

template <typename T>
__global__ void forward_kernel(const T *src, T **dsts,
        size_t nr_dsts,
        size_t A, size_t B, size_t C,
        const size_t *Bv,
        const size_t *table_outer,
        const size_t *table_inner)
{
    size_t addr = threadIdx.x + blockIdx.x * blockDim.x;
    if (addr < A*B*C) {
        size_t c = addr % C;
        size_t b = addr / C % B;
        size_t a = addr / (B*C);
        size_t i = table_outer[b];
        size_t B_dst = Bv[i];
        size_t b_dst = table_inner[b];
        size_t addr_dst = (a*B_dst + b_dst)*C + c;
        dsts[i][addr_dst] = src[addr];
    }
}

template <typename T>
void forward_proxy(const T *src,
        T **dsts,
        size_t nr_dsts,
        size_t A, size_t B, size_t C,
        const size_t *Bv,
        const size_t *table_outer,
        const size_t *table_inner,
        cudaStream_t stream)
{
    size_t total_nr_elem = A * B * C;
    size_t NR_BLOCKS = DIVUP(total_nr_elem, NR_THREADS);
    forward_kernel<<<NR_BLOCKS, NR_THREADS, 0, stream>>>(src, dsts,
            nr_dsts,
            A, B, C,
            Bv,
            table_outer,
            table_inner);
    after_kernel_launch();
}

#define INST(T) \
template void forward_proxy<T>(const T *, T **, size_t, size_t, size_t, size_t, \
        const size_t *, const size_t *, const size_t *, cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

#undef cb
#undef INST

} // namespace split
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
