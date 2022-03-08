/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/fuse_add_tanh.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddTanhOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
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

#define OP(_ctype, _simd_type, _func_suffix, _simd_width)                             \
    template <>                                                                       \
    struct FuseAddTanhOp<_ctype> : FuseAddTanhOpBase<_ctype> {                        \
        using FuseAddTanhOpBase::FuseAddTanhOpBase;                                   \
        using FuseAddTanhOpBase::operator();                                          \
        constexpr static size_t SIMD_WIDTH = _simd_width;                             \
        void operator()(                                                              \
                const _simd_type& src0, const _simd_type& src1,                       \
                dst_ctype* dst) const {                                               \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, vitem.val[0]);                                 \
            GiStore##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);                    \
        }                                                                             \
        _simd_type operator()(const _simd_type& src0, const _simd_type& src1) const { \
            auto val1 = src0.val[0];                                                  \
            auto val2 = src0.val[1];                                                  \
            auto val3 = src1.val[0];                                                  \
            auto val4 = src1.val[1];                                                  \
            val1 = GiAdd##_func_suffix(val1, val3);                                   \
            val2 = GiAdd##_func_suffix(val2, val4);                                   \
            auto exp1 = GiExpPs##_func_suffix(val1);                                  \
            auto exp2 = GiExpPs##_func_suffix(val2);                                  \
            auto rexp1 = GiRecpe##_func_suffix(exp1);                                 \
            auto rexp2 = GiRecpe##_func_suffix(exp2);                                 \
            rexp1 = GiMultiply##_func_suffix(                                         \
                    GiRecpeS##_func_suffix(exp1, rexp1), rexp1);                      \
            rexp2 = GiMultiply##_func_suffix(                                         \
                    GiRecpeS##_func_suffix(exp2, rexp2), rexp2);                      \
            val1 = GiSubtract##_func_suffix(exp1, rexp1);                             \
            val2 = GiSubtract##_func_suffix(exp2, rexp2);                             \
            exp1 = GiAdd##_func_suffix(exp1, rexp1);                                  \
            exp2 = GiAdd##_func_suffix(exp2, rexp2);                                  \
            rexp1 = GiRecpe##_func_suffix(exp1);                                      \
            rexp2 = GiRecpe##_func_suffix(exp2);                                      \
            rexp1 = GiMultiply##_func_suffix(                                         \
                    GiRecpeS##_func_suffix(exp1, rexp1), rexp1);                      \
            rexp2 = GiMultiply##_func_suffix(                                         \
                    GiRecpeS##_func_suffix(exp2, rexp2), rexp2);                      \
            val1 = GiMultiply##_func_suffix(val1, rexp1);                             \
            val2 = GiMultiply##_func_suffix(val2, rexp2);                             \
            return {{val1, val2}};                                                    \
        }                                                                             \
    };
OP(dt_float32, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
