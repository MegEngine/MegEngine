/**
 * \file src/rocm/relayout/kern_contiguous.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
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
