/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/abs.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct AbsOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const { return src > 0 ? src : (-src); }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct AbsOp;

#define OP(_ctype, _gi_type, _func_suffix, _simd_width)                             \
    template <>                                                                     \
    struct AbsOp<_ctype> : AbsOpBase<_ctype> {                                      \
        using AbsOpBase::AbsOpBase;                                                 \
        using AbsOpBase::operator();                                                \
        constexpr static size_t SIMD_WIDTH = _simd_width;                           \
        void operator()(const _gi_type& src, _ctype* dst) const {                   \
            auto vitem = operator()(src);                                           \
            GiStore##_func_suffix(dst, GiGetSubVector##_func_suffix##V2(vitem, 0)); \
            GiStore##_func_suffix(                                                  \
                    dst + SIMD_WIDTH, GiGetSubVector##_func_suffix##V2(vitem, 1));  \
        }                                                                           \
        _gi_type operator()(const _gi_type& src) const {                            \
            auto vitem0 =                                                           \
                    GiAbs##_func_suffix(GiGetSubVector##_func_suffix##V2(src, 0));  \
            auto vitem1 =                                                           \
                    GiAbs##_func_suffix(GiGetSubVector##_func_suffix##V2(src, 1));  \
            _gi_type ret;                                                           \
            GiSetSubVector##_func_suffix##V2(ret, 0, vitem0);                       \
            GiSetSubVector##_func_suffix##V2(ret, 1, vitem1);                       \
            return ret;                                                             \
        }                                                                           \
    };
OP(dt_float32, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(dt_float32))
OP(dt_int32, GI_INT32_V2_t, Int32, GI_SIMD_LEN_BYTE / sizeof(dt_int32))
OP(dt_int8, GI_INT8_V2_t, Int8, GI_SIMD_LEN_BYTE / sizeof(dt_int8))
#undef OP

template <>
struct AbsOpBase<dt_qint8, dt_qint8> : UnaryOpBase<dt_qint8, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint8& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }
    dt_qint8 operator()(const dt_qint8& src) const {
        float fsrc = src.as_int8() * this->scale;
        fsrc = fsrc > 0 ? fsrc : -fsrc;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
};

template <>
struct AbsOp<dt_qint8, dt_qint8> : AbsOpBase<dt_qint8, dt_qint8> {
    using AbsOpBase::AbsOpBase;
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    using AbsOpBase::operator();
    void operator()(const GI_INT8_V2_t& vsrc, dt_qint8* dst) const {
        OPERATOR_UNARY_QINT8_FALLBACK;
    }
    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc) const {
        auto vitem0 = GiMultiplyFloat32(
                GiCastToFloat32(GiGetSubVectorInt32V2(vsrc, 0)),
                GiFixLenType2GiFloat32Type(this->vscale));
        auto vitem1 = GiMultiplyFloat32(
                GiCastToFloat32(GiGetSubVectorInt32V2(vsrc, 1)),
                GiFixLenType2GiFloat32Type(this->vscale));
        vitem0 = GiAbsFloat32(vitem0);
        vitem1 = GiAbsFloat32(vitem1);
        GI_FLOAT32_V2_t tmp;
        GiSetSubVectorFloat32V2(tmp, 0, vitem0);
        GiSetSubVectorFloat32V2(tmp, 1, vitem1);
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(tmp);
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
