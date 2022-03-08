/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/fast_tanh.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

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

#define OP(_ctype, _simd_type, _func_suffix, _fix_func_suffix, _simd_width)         \
    template <>                                                                     \
    struct FastTanhOp<_ctype> : FastTanhOpBase<_ctype> {                            \
        using FastTanhOpBase::FastTanhOpBase;                                       \
        using FastTanhOpBase::operator();                                           \
        constexpr static size_t SIMD_WIDTH = _simd_width;                           \
        void operator()(const _simd_type& src, _ctype* dst) const {                 \
            auto vitem = operator()(src);                                           \
            GiStore##_func_suffix(dst, vitem.val[0]);                               \
            GiStore##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);                  \
        }                                                                           \
        _simd_type operator()(const _simd_type& src) const {                        \
            auto val_27 = GiBroadcast##_func_suffix(27.f);                          \
            auto val_9 = GiBroadcast##_func_suffix(9.f);                            \
            auto valx = src.val[0];                                                 \
            auto valx1 = src.val[1];                                                \
            auto valxp2 = GiMultiply##_fix_func_suffix(valx, valx);                 \
            auto valx1p2 = GiMultiply##_fix_func_suffix(valx1, valx1);              \
            auto denominator = GiAdd##_fix_func_suffix(valxp2, val_27);             \
            auto denominator1 = GiAdd##_fix_func_suffix(valx1p2, val_27);           \
            valx = GiMultiply##_fix_func_suffix(valx, denominator);                 \
            valx1 = GiMultiply##_fix_func_suffix(valx1, denominator1);              \
            denominator = GiMultiplyAdd##_fix_func_suffix(val_27, valxp2, val_9);   \
            denominator1 = GiMultiplyAdd##_fix_func_suffix(val_27, valx1p2, val_9); \
            auto r_denominator = GiRecpe##_func_suffix(denominator);                \
            auto r_denominator1 = GiRecpe##_func_suffix(denominator1);              \
            r_denominator = GiMultiply##_fix_func_suffix(                           \
                    GiRecpeS##_func_suffix(denominator, r_denominator),             \
                    r_denominator);                                                 \
            r_denominator1 = GiMultiply##_fix_func_suffix(                          \
                    GiRecpeS##_func_suffix(denominator1, r_denominator1),           \
                    r_denominator1);                                                \
            valx = GiMultiply##_fix_func_suffix(valx, r_denominator);               \
            valx1 = GiMultiply##_fix_func_suffix(valx1, r_denominator1);            \
            return {{valx, valx1}};                                                 \
        }                                                                           \
    };
OP(dt_float32, GI_FLOAT32_V2_t, Float32, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
