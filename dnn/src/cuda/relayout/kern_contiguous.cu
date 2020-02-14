/**
 * \file dnn/src/cuda/relayout/kern_contiguous.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/relayout/kern_contiguous.cuh"

namespace megdnn {
namespace cuda {

// dst is contiguous
void copy_last_contiguous(const TensorND &dst, const TensorND &src,
                          size_t contiguous_size, cudaStream_t stream) {
    ElemwiseOpParamN<2> param;
    param[0] = dst;
    param[1] = src;

#define RUN(_dt)                                                      \
    do {                                                              \
        typedef DTypeTrait<dtype::_dt>::ctype ctype;                  \
        param[0].layout.dtype = param[1].layout.dtype = dtype::_dt(); \
        param.init_from_given_tensor();                               \
        param.assert_initialized();                                   \
        contiguous_intl::UserOpInvoker<ctype, 2>(param, stream,       \
                contiguous_size);                                     \
        return;                                                       \
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

}  // namespace megdnn
}  // namespace cuda

// vim: syntax=cpp.doxygen
