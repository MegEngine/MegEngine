/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/true_div.h
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

//! use a couple Newton-Raphson steps to refine the estimate.
//! A / B => 1. rB = vrecpeq_f32(B) 2. rB= vmulq_f32(vrecpsq_f32(B, rB), rB)
//! 3. A * rB
template <typename src_ctype, typename dst_ctype = src_ctype>
struct TrueDivOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return src0 / src1;
    }
};

#if MEGDNN_AARCH64
template <typename src_ctype, typename dst_ctype = src_ctype>
struct TrueDivOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)    \
    template <>                                                           \
    struct TrueDivOp<_ctype> : TrueDivOpBase<_ctype> {                    \
        using TrueDivOpBase::TrueDivOpBase;                               \
        using TrueDivOpBase::operator();                                  \
        constexpr static size_t SIMD_WIDTH = _simd_width;                 \
        void operator()(const _neon_type2& src0, const _neon_type2& src1, \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem.val[0]);                      \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);         \
        }                                                                 \
        _neon_type2 operator()(const _neon_type2& src0,                   \
                               const _neon_type2& src1) const {           \
            auto val1 = src0.val[0];                                      \
            auto val2 = src0.val[1];                                      \
            auto val3 = src1.val[0];                                      \
            auto val4 = src1.val[1];                                      \
            val1 = vdivq_##_func_suffix(val1, val3);                      \
            val2 = vdivq_##_func_suffix(val2, val4);                      \
            return {{val1, val2}};                                        \
        }                                                                 \
        void operator()(const _neon_type& src0, const _neon_type& src1,   \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem);                             \
        }                                                                 \
        _neon_type operator()(const _neon_type& src0,                     \
                              const _neon_type& src1) const {             \
            return vdivq_##_func_suffix(src0, src1);                      \
        }                                                                 \
    };
OP(dt_float32, float32x4_t, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8_t, float16x8x2_t, f16, 8)
#endif
#undef OP

#else

template <typename src_ctype, typename dst_ctype = src_ctype>
struct TrueDivOp : TrueDivOpBase<src_ctype, dst_ctype> {
    using TrueDivOpBase<src_ctype, dst_ctype>::TrueDivOpBase;
    using TrueDivOpBase<src_ctype, dst_ctype>::operator();
};

#endif

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
