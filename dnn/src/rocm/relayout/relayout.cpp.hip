/**
 * \file src/rocm/relayout/relayout.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "./relayout.h.hip"
#include "megdnn/basic_types.h"
#include "src/rocm/elemwise_helper.h.hip"

namespace {
    template<typename ctype>
    struct CopyOp {
        __device__ __forceinline__ void operator() (
                uint32_t idx, ctype &dst, ctype src) {
            MEGDNN_MARK_USED_VAR(idx);
            dst = src;
        }
    };

} // anonymous namespace

namespace megdnn {
namespace rocm {

void copy_noncontig_general(const TensorND& dst, const TensorND& src,
                            hipStream_t stream) {
    ElemwiseOpParamN<2> param;
    param[0] = dst;
    param[1] = src;

#define RUN(_dt)                                                      \
    do {                                                              \
        typedef DTypeTrait<dtype::_dt>::ctype ctype;                  \
        param[0].layout.dtype = param[1].layout.dtype = dtype::_dt(); \
        param.init_from_given_tensor();                               \
        run_elemwise<CopyOp<ctype>, ctype, 2>(param, stream);         \
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


// vim: ft=cpp syntax=cpp.doxygen
