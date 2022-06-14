/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/hswish.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/kern_macro_prologue.h"
#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct HSwishOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmp = src;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        return (tmp);
    }
};

//! h_swish(x) = x * clip(x + 3, 0, 6) / 6
template <typename src_ctype, typename dst_ctype = src_ctype>
struct HSwishOp;

#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width)              \
    template <>                                                                     \
    struct HSwishOp<_ctype> : HSwishOpBase<_ctype> {                                \
        using HSwishOpBase::HSwishOpBase;                                           \
        using HSwishOpBase::operator();                                             \
        constexpr static size_t SIMD_WIDTH = _simd_width;                           \
        void operator()(const _simd_type2& src, _ctype* dst) const {                \
            auto vitem = operator()(src);                                           \
            GiStore##_func_suffix(dst, GiGetSubVector##_func_suffix##V2(vitem, 0)); \
            GiStore##_func_suffix(                                                  \
                    dst + SIMD_WIDTH, GiGetSubVector##_func_suffix##V2(vitem, 1));  \
        }                                                                           \
        void operator()(const _simd_type& src, _ctype* dst) const {                 \
            auto vitem = operator()(src);                                           \
            GiStore##_func_suffix(dst, vitem);                                      \
        }                                                                           \
        _simd_type2 operator()(const _simd_type2& src) const {                      \
            auto val1 = GiGetSubVector##_func_suffix##V2(src, 0);                   \
            auto val2 = GiGetSubVector##_func_suffix##V2(src, 1);                   \
            H_SWISH_KERN_FALLBACK(_func_suffix, val1, val2);                        \
            _simd_type2 ret;                                                        \
            GiSetSubVector##_func_suffix##V2(ret, 0, val1);                         \
            GiSetSubVector##_func_suffix##V2(ret, 1, val2);                         \
            return ret;                                                             \
        }                                                                           \
        _simd_type operator()(const _simd_type& src) const {                        \
            auto val_zero = GiBroadcast##_func_suffix(0.f);                         \
            auto val_six = GiBroadcast##_func_suffix(6.f);                          \
            auto val_three = GiBroadcast##_func_suffix(3.f);                        \
            auto val_rec_six = GiBroadcast##_func_suffix(1.f / 6.f);                \
            auto clip1 = GiMaximum##_func_suffix(                                   \
                    GiMinimum##_func_suffix(                                        \
                            GiAdd##_func_suffix(src, val_three), val_six),          \
                    val_zero);                                                      \
            return GiMultiply##_func_suffix(                                        \
                    GiMultiply##_func_suffix(src, clip1), val_rec_six);             \
        }                                                                           \
    };

OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

template <>
struct HSwishOpBase<dt_qint32, dt_qint8> : UnaryOpBase<dt_qint32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint32& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }

    dt_qint8 operator()(const dt_qint32& src) const {
        float tmp = src.as_int32() * this->scale_src;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        tmp *= this->scale_dst;
        return QConverter::convert<dt_qint8, float>(tmp);
    }
};

template <>
struct HSwishOp<dt_qint32, dt_qint8> : HSwishOpBase<dt_qint32, dt_qint8> {
    using HSwishOpBase::HSwishOpBase;
    using HSwishOpBase::operator();
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int32_t);

    void operator()(const GI_INT32_V2_t& vsrc, dt_qint8* dst) const {
        GiStoreLowInt8(reinterpret_cast<int8_t*>(dst), operator()(vsrc));
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc) const {
        auto vitem0 = GiMultiplyFloat32(
                GiCastToFloat32(GiGetSubVectorInt32V2(vsrc, 0)),
                GiFixLenType2GiFloat32Type(this->vscale_src));
        auto vitem1 = GiMultiplyFloat32(
                GiCastToFloat32(GiGetSubVectorInt32V2(vsrc, 1)),
                GiFixLenType2GiFloat32Type(this->vscale_src));

        H_SWISH_KERN_FALLBACK(Float32, vitem0, vitem1);
        vitem0 =
                GiMultiplyFloat32(vitem0, GiFixLenType2GiFloat32Type(this->vscale_dst));
        vitem1 =
                GiMultiplyFloat32(vitem1, GiFixLenType2GiFloat32Type(this->vscale_dst));
        GI_FLOAT32_V2_t tmp;
        GiSetSubVectorFloat32V2(tmp, 0, vitem0);
        GiSetSubVectorFloat32V2(tmp, 1, vitem1);

        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(tmp);
    }
};

}  // namespace fallback
}  // namespace megdnn

#include "src/fallback/elemwise_helper/kimpl/kern_macro_epilogue.h"
// vim: syntax=cpp.doxygen
