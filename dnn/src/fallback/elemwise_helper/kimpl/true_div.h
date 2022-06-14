/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/true_div.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

//! use a couple Newton-Raphson steps to refine the estimate.
//! A / B => 1. rB = vrecpeq_f32(B) 2. rB= vmulq_f32(vrecpsq_f32(B, rB), rB)
//! 3. A * rB
template <typename src_ctype, typename dst_ctype = src_ctype>
struct TrueDivOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return src0 / src1;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct TrueDivOp;

#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width)                \
    template <>                                                                       \
    struct TrueDivOp<_ctype> : TrueDivOpBase<_ctype> {                                \
        using TrueDivOpBase::TrueDivOpBase;                                           \
        using TrueDivOpBase::operator();                                              \
        constexpr static size_t SIMD_WIDTH = _simd_width;                             \
        void operator()(                                                              \
                const _simd_type2& src0, const _simd_type2& src1,                     \
                dst_ctype* dst) const {                                               \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, GiGetSubVector##_func_suffix##V2(vitem, 0));   \
            GiStore##_func_suffix(                                                    \
                    dst + SIMD_WIDTH, GiGetSubVector##_func_suffix##V2(vitem, 1));    \
        }                                                                             \
        _simd_type2 operator()(                                                       \
                const _simd_type2& src0, const _simd_type2& src1) const {             \
            auto val1 = GiGetSubVector##_func_suffix##V2(src0, 0);                    \
            auto val2 = GiGetSubVector##_func_suffix##V2(src0, 1);                    \
            auto val3 = GiGetSubVector##_func_suffix##V2(src1, 0);                    \
            auto val4 = GiGetSubVector##_func_suffix##V2(src1, 1);                    \
            val1 = GiDivide##_func_suffix(val1, val3);                                \
            val2 = GiDivide##_func_suffix(val2, val4);                                \
            _simd_type2 ret;                                                          \
            GiSetSubVector##_func_suffix##V2(ret, 0, val1);                           \
            GiSetSubVector##_func_suffix##V2(ret, 1, val2);                           \
            return ret;                                                               \
        }                                                                             \
        void operator()(                                                              \
                const _simd_type& src0, const _simd_type& src1,                       \
                dst_ctype* dst) const {                                               \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, vitem);                                        \
        }                                                                             \
        _simd_type operator()(const _simd_type& src0, const _simd_type& src1) const { \
            return GiDivide##_func_suffix(src0, src1);                                \
        }                                                                             \
    };
OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
