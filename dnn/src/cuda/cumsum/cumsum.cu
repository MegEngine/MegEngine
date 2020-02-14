/**
 * \file dnn/src/cuda/cumsum/cumsum.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern_impl.cuinl"

namespace megdnn {
namespace cuda {
namespace cumsum {

#define INST_(T, Op, exclusive, reverse)                                  \
    template void run_kern<T, Op, exclusive, reverse>(                    \
            T*, void*, uint32_t, uint32_t, uint32_t, uint32_t, const Op&, \
            cudaStream_t)
#define INST(T)                      \
    INST_(T, SumOp<T>, true, true);  \
    INST_(T, SumOp<T>, false, true); \
    INST_(T, SumOp<T>, true, false); \
    INST_(T, SumOp<T>, false, false);

#define cb(DType) INST(typename DTypeTrait<DType>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

}  // namespace cumsum
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen
