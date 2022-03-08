/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/fuse_mul_add3.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseMulAdd3OpBase : TernaryOpBase<src_ctype, dst_ctype> {
    using TernaryOpBase<src_ctype, dst_ctype>::TernaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, const src_ctype src2,
            dst_ctype* dst) const {
        *dst = operator()(src0, src1, src2);
    }

    dst_ctype operator()(
            const src_ctype& src0, const src_ctype& src1, const src_ctype& src2) const {
        return (src0 * src1) + src2;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseMulAdd3Op;

#define OP(_ctype, _simd_type, _func_suffix, _simd_width)          \
    template <>                                                    \
    struct FuseMulAdd3Op<_ctype> : FuseMulAdd3OpBase<_ctype> {     \
        using FuseMulAdd3OpBase::FuseMulAdd3OpBase;                \
        using FuseMulAdd3OpBase::operator();                       \
        constexpr static size_t SIMD_WIDTH = _simd_width;          \
        void operator()(                                           \
                const _simd_type& src0, const _simd_type& src1,    \
                const _simd_type& src2, dst_ctype* dst) const {    \
            auto vitem = operator()(src0, src1, src2);             \
            GiStore##_func_suffix(dst, vitem.val[0]);              \
            GiStore##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]); \
        }                                                          \
        _simd_type operator()(                                     \
                const _simd_type& src0, const _simd_type& src1,    \
                const _simd_type& src2) const {                    \
            auto vitem0 = GiMultiplyAdd##_func_suffix(             \
                    src2.val[0], src0.val[0], src1.val[0]);        \
            auto vitem1 = GiMultiplyAdd##_func_suffix(             \
                    src2.val[1], src0.val[1], src1.val[1]);        \
            return {{vitem0, vitem1}};                             \
        }                                                          \
    };
OP(dt_float32, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
OP(dt_int32, GI_INT32_V2_t, Int32, GI_SIMD_LEN_BYTE / sizeof(int32_t))
OP(dt_int8, GI_INT8_V2_t, Int8, GI_SIMD_LEN_BYTE / sizeof(int8_t))
#undef OP

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
