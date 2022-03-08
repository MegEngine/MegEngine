/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/fuse_add_sigmoid.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddSigmoidOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        float tmpf = src0 + src1;
        tmpf = exp(-tmpf);
        tmpf = 1.f / (1.f + tmpf);
        return tmpf;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddSigmoidOp;

#define OP(_ctype, _simd_type, _func_suffix, _simd_width)                             \
    template <>                                                                       \
    struct FuseAddSigmoidOp<_ctype> : FuseAddSigmoidOpBase<_ctype> {                  \
        using FuseAddSigmoidOpBase::FuseAddSigmoidOpBase;                             \
        using FuseAddSigmoidOpBase::operator();                                       \
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
            val1 = GiSigmoidPs##_func_suffix(val1);                                   \
            val2 = GiSigmoidPs##_func_suffix(val2);                                   \
            return {{val1, val2}};                                                    \
        }                                                                             \
    };
OP(dt_float32, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
