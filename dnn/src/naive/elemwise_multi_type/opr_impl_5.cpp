/**
 * \file dnn/src/naive/elemwise_multi_type/opr_impl_5.cpp
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
#include "src/common/elemwise_multi_type/kern_defs.cuh"

using namespace megdnn;
using namespace naive;

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<3>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED &&
            param[0].layout.dtype.category() == param[1].layout.dtype.category() &&
            param[0].layout.dtype.category() == param[2].layout.dtype.category());
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

    switch (mode) {
#define DISPATCH(_mode)                                                        \
    case Elemwise::Mode::_mode: {                                              \
        typedef ElemwiseKern<                                                  \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode, float> \
                KernImpl;                                                      \
        dispatch_qint_op_dtype<KernImpl, ElemwiseOpParamN<3>>(param, dst);     \
        break;                                                                 \
    }

        DISPATCH(FUSE_MUL_ADD3);
        DISPATCH(COND_LEQ_MOV);
#undef DISPATCH
        default:
            megdnn_assert_internal(0);
    }
}

// vim: syntax=cpp.doxygen
