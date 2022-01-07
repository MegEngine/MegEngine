/**
 * \file dnn/src/fallback/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./opr_impl.h"

#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_elemwise_exec_UNARY_INT)
MIDOUT_DECL(megdnn_fallback_elemwise_exec_UNARY_FLOAT)
MIDOUT_DECL(megdnn_fallback_elemwise_exec_BINARY_INT)
MIDOUT_DECL(megdnn_fallback_elemwise_exec_BINARY_FLOAT)

namespace megdnn {
namespace fallback {

void ElemwiseImpl::exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) {
    if (!dst.layout.is_contiguous()) {
        return naive::ElemwiseForwardImpl::exec(srcs, dst);
    }

    m_src = &srcs;
    m_dst = &dst;

#define CONCAT2(a, b, c) a##_##b##_##c
#define CONCAT(a, b, c)  CONCAT2(a, b, c)
#define SWITCH_MODE_CB(_mode)                                                      \
    case Mode::_mode:                                                              \
        MIDOUT_BEGIN(                                                              \
                CONCAT(megdnn_fallback_elemwise_exec, ARITY, CAT),                 \
                midout_iv(Mode::_mode)) {                                          \
            return CONCAT(exec, ARITY, CAT)<param_enumv::Elemwise::Mode::_mode>(); \
        }                                                                          \
        MIDOUT_END();
#define SWITCH_MODE                                          \
    switch (m_param.mode) {                                  \
        CONCAT(MEGDNN_FOREACH_ELEMWISE_MODE, ARITY, CAT)     \
        (SWITCH_MODE_CB) default : megdnn_throw("bad mode"); \
    }

    if (dst.layout.dtype.category() == DTypeCategory::INT) {
#define CAT INT
        if (srcs.size() == 1) {
#define ARITY UNARY
            SWITCH_MODE
#undef ARITY
        }

        if (srcs.size() == 2) {
#define ARITY BINARY
            SWITCH_MODE
#undef ARITY
        }
#undef CAT
    } else if (dst.layout.dtype.category() == DTypeCategory::FLOAT) {
#define CAT FLOAT
        if (srcs.size() == 1) {
#define ARITY UNARY
            SWITCH_MODE
#undef ARITY
        }

        if (srcs.size() == 2) {
#define ARITY BINARY
            SWITCH_MODE
#undef ARITY
        }
#undef CAT
    }

#undef cb
    naive::ElemwiseForwardImpl::exec(srcs, dst);
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
