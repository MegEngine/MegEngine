/**
 * \file dnn/src/rocm/relayout/relayout.cpp.hip
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
