/**
 * \file dnn/src/cuda/relayout/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"
#include "src/cuda/relayout/kern.cuh"

namespace megdnn {
namespace cuda {

void copy_noncontig_general(const TensorND &dst, const TensorND &src, cudaStream_t stream) {
    ElemwiseOpParamN<2> param;
    param[0] = dst;
    param[1] = src;

#define RUN(_dt)                                                        \
    do {                                                                \
        typedef DTypeTrait<dtype::_dt>::ctype ctype;                    \
        param[0].layout.dtype = param[1].layout.dtype = dtype::_dt();   \
        param.init_from_given_tensor();                                 \
        param.assert_initialized();                                     \
        noncontig_general_intl::UserOpInvoker<ctype, 2>(param, stream); \
        return;                                                         \
    } while (0)

    switch (dst.layout.dtype.size()) {
        case 1:
            RUN(Byte);
        case 2:
            RUN(Float16);
        case 4:
            RUN(Int32);
    }
    megdnn_assert(0, "bad dtype size");
}

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
