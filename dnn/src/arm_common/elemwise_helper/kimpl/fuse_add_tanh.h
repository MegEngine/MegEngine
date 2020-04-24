/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/fuse_add_tanh.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/arm_common/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddTanhOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        float tmpf = exp(src0 + (src1));
        float tmpf2 = 1 / tmpf;
        return (tmpf - tmpf2) / (tmpf + tmpf2);
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddTanhOp;

#define OP(_ctype, _neon_type, _func_suffix, _simd_width)                     \
    template <>                                                               \
    struct FuseAddTanhOp<_ctype> : FuseAddTanhOpBase<_ctype> {                \
        using FuseAddTanhOpBase::FuseAddTanhOpBase;                           \
        using FuseAddTanhOpBase::operator();                                  \
        constexpr static size_t SIMD_WIDTH = _simd_width;                     \
        void operator()(const _neon_type& src0, const _neon_type& src1,       \
                        dst_ctype* dst) const {                               \
            auto vitem = operator()(src0, src1);                              \
            vst1q_##_func_suffix(dst, vitem.val[0]);                          \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);             \
        }                                                                     \
        _neon_type operator()(const _neon_type& src0,                         \
                              const _neon_type& src1) const {                 \
            auto val1 = src0.val[0];                                          \
            auto val2 = src0.val[1];                                          \
            auto val3 = src1.val[0];                                          \
            auto val4 = src1.val[1];                                          \
            val1 = vaddq_##_func_suffix(val1, val3);                          \
            val2 = vaddq_##_func_suffix(val2, val4);                          \
            auto exp1 = exp_ps_##_func_suffix(val1);                          \
            auto exp2 = exp_ps_##_func_suffix(val2);                          \
            auto rexp1 = vrecpeq_##_func_suffix(exp1);                        \
            auto rexp2 = vrecpeq_##_func_suffix(exp2);                        \
            rexp1 = vmulq_##_func_suffix(vrecpsq_##_func_suffix(exp1, rexp1), \
                                         rexp1);                              \
            rexp2 = vmulq_##_func_suffix(vrecpsq_##_func_suffix(exp2, rexp2), \
                                         rexp2);                              \
            val1 = vsubq_##_func_suffix(exp1, rexp1);                         \
            val2 = vsubq_##_func_suffix(exp2, rexp2);                         \
            exp1 = vaddq_##_func_suffix(exp1, rexp1);                         \
            exp2 = vaddq_##_func_suffix(exp2, rexp2);                         \
            rexp1 = vrecpeq_##_func_suffix(exp1);                             \
            rexp2 = vrecpeq_##_func_suffix(exp2);                             \
            rexp1 = vmulq_##_func_suffix(vrecpsq_##_func_suffix(exp1, rexp1), \
                                         rexp1);                              \
            rexp2 = vmulq_##_func_suffix(vrecpsq_##_func_suffix(exp2, rexp2), \
                                         rexp2);                              \
            val1 = vmulq_##_func_suffix(val1, rexp1);                         \
            val2 = vmulq_##_func_suffix(val2, rexp2);                         \
            return {{val1, val2}};                                            \
        }                                                                     \
    };
OP(dt_float32, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8x2_t, f16, 8)
#endif
#undef OP

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
