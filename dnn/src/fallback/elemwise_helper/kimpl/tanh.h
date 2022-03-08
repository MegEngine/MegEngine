/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/tanh.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct TanhOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmp = src;
        return tanh(tmp);
    }
};

template <typename src_ctype, typename dst_type = src_ctype>
struct TanhOp;

#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width) \
    template <>                                                        \
    struct TanhOp<_ctype> : TanhOpBase<_ctype> {                       \
        using TanhOpBase::TanhOpBase;                                  \
        using TanhOpBase::operator();                                  \
        constexpr static size_t SIMD_WIDTH = _simd_width;              \
        void operator()(const _simd_type2& src, _ctype* dst) const {   \
            auto vitem = operator()(src);                              \
            GiStore##_func_suffix(dst, vitem.val[0]);                  \
            GiStore##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);     \
        }                                                              \
        _simd_type2 operator()(const _simd_type2& src) const {         \
            auto one_val = GiBroadcast##_func_suffix(1.f);             \
            auto two_val = GiBroadcast##_func_suffix(2.f);             \
            auto val1 = src.val[0];                                    \
            auto val2 = src.val[1];                                    \
            val1 = GiMultiply##_func_suffix(two_val, val1);            \
            val2 = GiMultiply##_func_suffix(two_val, val2);            \
            val1 = GiExpPs##_func_suffix(val1);                        \
            val2 = GiExpPs##_func_suffix(val2);                        \
            val1 = GiAdd##_func_suffix(one_val, val1);                 \
            val2 = GiAdd##_func_suffix(one_val, val2);                 \
            auto rval1 = GiRecpe##_func_suffix(val1);                  \
            auto rval2 = GiRecpe##_func_suffix(val2);                  \
            rval1 = GiMultiply##_func_suffix(                          \
                    GiRecpeS##_func_suffix(val1, rval1), rval1);       \
            rval2 = GiMultiply##_func_suffix(                          \
                    GiRecpeS##_func_suffix(val2, rval2), rval2);       \
            val1 = GiMultiply##_func_suffix(two_val, rval1);           \
            val2 = GiMultiply##_func_suffix(two_val, rval2);           \
            val1 = GiSubtract##_func_suffix(one_val, val1);            \
            val2 = GiSubtract##_func_suffix(one_val, val2);            \
            return {{val1, val2}};                                     \
        }                                                              \
        _simd_type operator()(const _simd_type& src) const {           \
            auto one_val = GiBroadcast##_func_suffix(1.f);             \
            auto two_val = GiBroadcast##_func_suffix(2.f);             \
            auto val1 = src;                                           \
            val1 = GiMultiply##_func_suffix(two_val, val1);            \
            val1 = GiExpPs##_func_suffix(val1);                        \
            val1 = GiAdd##_func_suffix(one_val, val1);                 \
            auto rval1 = GiRecpe##_func_suffix(val1);                  \
            rval1 = GiMultiply##_func_suffix(                          \
                    GiRecpeS##_func_suffix(val1, rval1), rval1);       \
            val1 = GiMultiply##_func_suffix(two_val, rval1);           \
            val1 = GiSubtract##_func_suffix(one_val, val1);            \
            return val1;                                               \
        }                                                              \
    };
OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
