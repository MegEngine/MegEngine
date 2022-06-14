/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/add.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct AddOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }

    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return src0 + src1;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct AddOp;

#define OP(_ctype, _gi_type, _gi_type2, _func_suffix, _simd_width)                    \
    template <>                                                                       \
    struct AddOp<_ctype> : AddOpBase<_ctype> {                                        \
        using AddOpBase::AddOpBase;                                                   \
        using AddOpBase::operator();                                                  \
        constexpr static size_t SIMD_WIDTH = _simd_width;                             \
        void operator()(                                                              \
                const _gi_type2& src0, const _gi_type2& src1, dst_ctype* dst) const { \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, GiGetSubVector##_func_suffix##V2(vitem, 0));   \
            GiStore##_func_suffix(                                                    \
                    dst + SIMD_WIDTH, GiGetSubVector##_func_suffix##V2(vitem, 1));    \
        }                                                                             \
        _gi_type2 operator()(const _gi_type2& src0, const _gi_type2& src1) const {    \
            auto vitem0 = GiAdd##_func_suffix(                                        \
                    GiGetSubVector##_func_suffix##V2(src0, 0),                        \
                    GiGetSubVector##_func_suffix##V2(src1, 0));                       \
            auto vitem1 = GiAdd##_func_suffix(                                        \
                    GiGetSubVector##_func_suffix##V2(src0, 1),                        \
                    GiGetSubVector##_func_suffix##V2(src1, 1));                       \
            _gi_type2 ret;                                                            \
            GiSetSubVector##_func_suffix##V2(ret, 0, vitem0);                         \
            GiSetSubVector##_func_suffix##V2(ret, 1, vitem1);                         \
            return ret;                                                               \
        }                                                                             \
        void operator()(                                                              \
                const _gi_type& src0, const _gi_type& src1, dst_ctype* dst) const {   \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, vitem);                                        \
        }                                                                             \
        _gi_type operator()(const _gi_type& src0, const _gi_type& src1) const {       \
            return GiAdd##_func_suffix(src0, src1);                                   \
        }                                                                             \
    };
OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32,
   GI_SIMD_LEN_BYTE / sizeof(dt_float32))
OP(dt_int32, GI_INT32_t, GI_INT32_V2_t, Int32, GI_SIMD_LEN_BYTE / sizeof(dt_int32))
OP(dt_int8, GI_INT8_t, GI_INT8_V2_t, Int8, GI_SIMD_LEN_BYTE / sizeof(dt_int8))
#undef OP

template <>
struct AddOpBase<dt_qint8, dt_qint8> : BinaryOpBase<dt_qint8, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint8& src0, const dt_qint8& src1, dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1) const {
        return QConverter::convert<dt_qint8, float>(
                src0.as_int8() * this->scale0 + src1.as_int8() * this->scale1);
    }
};

template <>
struct AddOp<dt_qint8, dt_qint8> : AddOpBase<dt_qint8, dt_qint8> {
    using AddOpBase::AddOpBase;
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    using AddOpBase::operator();

    void operator()(
            const GI_INT8_V2_t& vsrc0, const GI_INT8_V2_t& vsrc1, dt_qint8* dst) const {
        OPERATOR_BINARY_QINT8_FALLBACK;
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1) const {
        auto vitem0 = GiAddFloat32(
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc0, 0)),
                        GiFixLenType2GiFloat32Type(this->vscale0)),
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc1, 0)),
                        GiFixLenType2GiFloat32Type(this->vscale1)));
        auto vitem1 = GiAddFloat32(
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc0, 1)),
                        GiFixLenType2GiFloat32Type(this->vscale0)),
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc1, 1)),
                        GiFixLenType2GiFloat32Type(this->vscale1)));
        GI_FLOAT32_V2_t ret;
        GiSetSubVectorFloat32V2(ret, 0, vitem0);
        GiSetSubVectorFloat32V2(ret, 1, vitem1);

        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(ret);
    }
};

template <>
struct AddOpBase<dt_qint32, dt_qint8> : BinaryOpBase<dt_qint32, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1, dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        return QConverter::convert<dt_qint8, float>(
                src0.as_int32() * this->scale0 + src1.as_int32() * this->scale1);
    }
};

template <>
struct AddOp<dt_qint32, dt_qint8> : AddOpBase<dt_qint32, dt_qint8> {
    using AddOpBase::AddOpBase;
    using AddOpBase::operator();
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int32_t);

    void operator()(
            const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1,
            dt_qint8* dst) const {
        GiStoreLowInt8(reinterpret_cast<int8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1) const {
        auto vitem0 = GiAddFloat32(
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc0, 0)),
                        GiFixLenType2GiFloat32Type(this->vscale0)),
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc1, 0)),
                        GiFixLenType2GiFloat32Type(this->vscale1)));
        auto vitem1 = GiAddFloat32(
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc0, 1)),
                        GiFixLenType2GiFloat32Type(this->vscale0)),
                GiMultiplyFloat32(
                        GiCastToFloat32(GiGetSubVectorInt32V2(vsrc1, 1)),
                        GiFixLenType2GiFloat32Type(this->vscale1)));
        GI_FLOAT32_V2_t ret;
        GiSetSubVectorFloat32V2(ret, 0, vitem0);
        GiSetSubVectorFloat32V2(ret, 1, vitem1);

        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(ret);
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
