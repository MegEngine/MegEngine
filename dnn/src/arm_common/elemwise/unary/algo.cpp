/**
 * \file dnn/src/arm_common/elemwise/unary/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/elemwise/unary/algo.h"
#include "src/arm_common/elemwise_op.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_elemwise_unary)

using namespace megdnn;
using namespace arm_common;

bool ElemwiseImpl::AlgoUnary::is_available(const KernParam& kern_param) const {
    if (BcastType::VEC != kern_param.broad_cast_type)
        return false;

    if (kern_param.m_dst->layout.dtype.category() != DTypeCategory::FLOAT &&
        (kern_param.mode == Mode::EXP || kern_param.mode == Mode::SIGMOID ||
         kern_param.mode == Mode::TANH || kern_param.mode == Mode::FAST_TANH ||
         kern_param.mode == Mode::H_SWISH)) {
        return false;
    }
    //! As `NEGATE` mode is so simple, that the code generate by compiler is
    //! vectorized optimized, while other mode such as `ABS` has branch, the
    //! compiler may not generate code as good as user intrinsic.
    if (kern_param.mode == Mode::NEGATE) {
        return false;
    }

    auto& elparam = kern_param.unary_elparam;
    if (!elparam[0].layout.is_contiguous())
        return false;
    megdnn_assert(elparam[0].layout.ndim == 1);
    auto& src0 = elparam[0];

#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id)                    \
    auto mode = kern_param.mode;                                              \
    if (mode == Mode::RELU || mode == Mode::ABS || mode == Mode::SIGMOID ||   \
        mode == Mode::EXP || mode == Mode::TANH || mode == Mode::FAST_TANH || \
        mode == Mode::H_SWISH)                                                \
        return true;

#define DISPATCH_MODE_INT(_case, _type, _type_midout_id) \
    auto mode = kern_param.mode;                         \
    if (mode == Mode::RELU || mode == Mode::ABS)         \
        return true;

    DISPATCH_TYPE("AlgoUnary::is_available"_hash);
    return false;
#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT
}

void ElemwiseImpl::AlgoUnary::exec(const KernParam& kern_param) const {
#define DISPATCH_UNARY(_mode, _case, _type, _type_midout_id, _op)           \
    case Mode::_mode:                                                       \
        MIDOUT_BEGIN(megdnn_arm_common_elemwise_unary, midout_iv(_case),    \
                     midout_iv(Mode::_mode), midout_iv(_type_midout_id)) {  \
            thin_function<void(const _type*, _type*, DType, DType, size_t)> \
                    run = OpCallerUnary<_op<_type, _type>,                  \
                                        BcastType::VEC>::run;               \
            MEGDNN_DISPATCH_CPU_KERN(                                       \
                    static_cast<naive::HandleImpl*>(kern_param.handle),     \
                    run(static_cast<const _type*>(src0.raw_ptr),            \
                        static_cast<_type*>(dst_tensor.raw_ptr),            \
                        src0.layout.dtype, dst_tensor.layout.dtype,         \
                        nr_elems));                                         \
        }                                                                   \
        MIDOUT_END();                                                       \
        return

    auto& elparam = kern_param.unary_elparam;
    megdnn_assert(elparam[0].layout.ndim == 1);
    auto& src0 = elparam[0];
    auto& dst_tensor = *(kern_param.m_dst);

    size_t nr_elems = src0.layout.total_nr_elems();

#define DISPATCH_MODE_FLOAT(_case, _type, _type_midout_id)                    \
    switch (kern_param.mode) {                                                \
        DISPATCH_UNARY(RELU, _case, _type, _type_midout_id, ReluOp);          \
        DISPATCH_UNARY(ABS, _case, _type, _type_midout_id, AbsOp);            \
        DISPATCH_UNARY(SIGMOID, _case, _type, _type_midout_id, SigmoidOp);    \
        DISPATCH_UNARY(EXP, _case, _type, _type_midout_id, ExpOp);            \
        DISPATCH_UNARY(TANH, _case, _type, _type_midout_id, TanhOp);          \
        DISPATCH_UNARY(FAST_TANH, _case, _type, _type_midout_id, FastTanhOp); \
        DISPATCH_UNARY(H_SWISH, _case, _type, _type_midout_id, HSwishOp);     \
        default:                                                              \
            megdnn_throw(ssprintf("No avaiable algo find for: %d",            \
                                  static_cast<int>(kern_param.mode)));        \
    }

#define DISPATCH_MODE_INT(_case, _type, _type_midout_id)               \
    switch (kern_param.mode) {                                         \
        DISPATCH_UNARY(RELU, _case, _type, _type_midout_id, ReluOp);   \
        DISPATCH_UNARY(ABS, _case, _type, _type_midout_id, AbsOp);     \
        default:                                                       \
            megdnn_throw(ssprintf("No avaiable algo find for: %d",     \
                                  static_cast<int>(kern_param.mode))); \
    }

    DISPATCH_TYPE("AlgoUnary::exec"_hash);
#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT
#undef DISPATCH_UNARY
}

// vim: syntax=cpp.doxygen
