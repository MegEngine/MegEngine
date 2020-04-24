/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/fast_tanh.h
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

//! tanh = x * (27 + x^2) / (27 + 9 * x^2)
template <typename src_ctype, typename dst_ctype = src_ctype>
struct FastTanhOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float x = src;
        return x * (27.f + x * x) / (27.f + 9.f * x * x);
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FastTanhOp;

#define OP(_ctype, _neon_type, _func_suffix, _fix_func_suffix, _simd_width)  \
    template <>                                                              \
    struct FastTanhOp<_ctype> : FastTanhOpBase<_ctype> {                     \
        using FastTanhOpBase::FastTanhOpBase;                                \
        using FastTanhOpBase::operator();                                    \
        constexpr static size_t SIMD_WIDTH = _simd_width;                    \
        void operator()(const _neon_type& src, _ctype* dst) const {          \
            auto vitem = operator()(src);                                    \
            vst1q_##_func_suffix(dst, vitem.val[0]);                         \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);            \
        }                                                                    \
        _neon_type operator()(const _neon_type& src) const {                 \
            auto val_27 = vdupq_n_##_func_suffix(27.f);                      \
            auto val_9 = vdupq_n_##_func_suffix(9.f);                        \
            auto valx = src.val[0];                                          \
            auto valx1 = src.val[1];                                         \
            auto valxp2 = vmulq_##_fix_func_suffix(valx, valx);              \
            auto valx1p2 = vmulq_##_fix_func_suffix(valx1, valx1);           \
            auto denominator = vaddq_##_fix_func_suffix(valxp2, val_27);     \
            auto denominator1 = vaddq_##_fix_func_suffix(valx1p2, val_27);   \
            valx = vmulq_##_fix_func_suffix(valx, denominator);              \
            valx1 = vmulq_##_fix_func_suffix(valx1, denominator1);           \
            denominator = vmlaq_##_fix_func_suffix(val_27, valxp2, val_9);   \
            denominator1 = vmlaq_##_fix_func_suffix(val_27, valx1p2, val_9); \
            auto r_denominator = vrecpeq_##_func_suffix(denominator);        \
            auto r_denominator1 = vrecpeq_##_func_suffix(denominator1);      \
            r_denominator = vmulq_##_fix_func_suffix(                        \
                    vrecpsq_##_func_suffix(denominator, r_denominator),      \
                    r_denominator);                                          \
            r_denominator1 = vmulq_##_fix_func_suffix(                       \
                    vrecpsq_##_func_suffix(denominator1, r_denominator1),    \
                    r_denominator1);                                         \
            valx = vmulq_##_fix_func_suffix(valx, r_denominator);            \
            valx1 = vmulq_##_fix_func_suffix(valx1, r_denominator1);         \
            return {{valx, valx1}};                                          \
        }                                                                    \
    };
OP(dt_float32, float32x4x2_t, f32, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8x2_t, f16, fix_f16, 8)
#endif
#undef OP

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
