/**
 * \file dnn/src/rocm/relayout/kern_contiguous.cpp.hip
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

#include "./relayout_contiguous.h.hip"
#include "src/rocm/elemwise_helper.h.hip"

#include <stdio.h>

namespace megdnn {
namespace rocm {

// dst is contiguous
void copy_last_contiguous(const TensorND& dst, const TensorND& src,
                          size_t contiguous_size, hipStream_t stream) {
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
                                                 contiguous_size);    \
        return;                                                       \
    } while (0)

    switch (dst.layout.dtype.size()) {
        case 1:
            RUN(Byte);
#if !MEGDNN_DISABLE_FLOAT16
        case 2:
            RUN(Float16);
#endif
        case 4:
            RUN(Int32);
    }
    megdnn_assert(0, "bad dtype size");
}

}  // namespace rocm
}  // namespace megdnn


// vim: syntax=cpp.doxygen
