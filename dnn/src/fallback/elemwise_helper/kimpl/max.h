/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/max.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {
template <typename src_ctype, typename dst_ctype = src_ctype>
struct MaxOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return src0 > src1 ? src0 : src1;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct MaxOp;

#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width)                \
    template <>                                                                       \
    struct MaxOp<_ctype> : MaxOpBase<_ctype> {                                        \
        using MaxOpBase::MaxOpBase;                                                   \
        using MaxOpBase::operator();                                                  \
        constexpr static size_t SIMD_WIDTH = _simd_width;                             \
        void operator()(                                                              \
                const _simd_type2& src0, const _simd_type2& src1,                     \
                dst_ctype* dst) const {                                               \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, vitem.val[0]);                                 \
            GiStore##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);                    \
        }                                                                             \
        _simd_type2 operator()(                                                       \
                const _simd_type2& src0, const _simd_type2& src1) const {             \
            auto vitem0 = GiMaximum##_func_suffix(src0.val[0], src1.val[0]);          \
            auto vitem1 = GiMaximum##_func_suffix(src0.val[1], src1.val[1]);          \
            return {{vitem0, vitem1}};                                                \
        }                                                                             \
        void operator()(                                                              \
                const _simd_type& src0, const _simd_type& src1,                       \
                dst_ctype* dst) const {                                               \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, vitem);                                        \
        }                                                                             \
        _simd_type operator()(const _simd_type& src0, const _simd_type& src1) const { \
            return GiMaximum##_func_suffix(src0, src1);                               \
        }                                                                             \
    };
OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
OP(dt_int32, GI_INT32_t, GI_INT32_V2_t, Int32, GI_SIMD_LEN_BYTE / sizeof(int32_t))
OP(dt_int8, GI_INT8_t, GI_INT8_V2_t, Int8, GI_SIMD_LEN_BYTE / sizeof(int8_t))
#undef OP

template <>
struct MaxOpBase<dt_qint8, dt_qint8> : BinaryOpBase<dt_qint8, dt_qint8> {
    using src_ctype = dt_qint8;
    using dst_ctype = dt_qint8;
    using BinaryOpBase::BinaryOpBase;

    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }

    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        float fsrc0 = src0.as_int8() * this->scale0;
        float fsrc1 = src1.as_int8() * this->scale1;
        return QConverter::convert<dst_ctype, float>(fsrc0 > fsrc1 ? fsrc0 : fsrc1);
    }
};

template <>
struct MaxOp<dt_qint8, dt_qint8> : MaxOpBase<dt_qint8, dt_qint8> {
    using MaxOpBase::MaxOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using MaxOpBase::operator();

    void operator()(
            const GI_INT8_V2_t& vsrc0, const GI_INT8_V2_t& vsrc1, dt_qint8* dst) const {
        OPERATOR_BINARY_QINT8_FALLBACK;
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1) const {
        auto vitem0 = GiMaximumFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[0]), this->vscale0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[0]), this->vscale1));
        auto vitem1 = GiMaximumFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[1]), this->vscale0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[1]), this->vscale1));
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>({{vitem0, vitem1}});
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
