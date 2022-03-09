/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/fuse_add_h_swish.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/kern_macro_prologue.h"
#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddHSwishOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        float tmp = src0 + src1;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        return tmp;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddHSwishOp;

#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width)                \
    template <>                                                                       \
    struct FuseAddHSwishOp<_ctype> : FuseAddHSwishOpBase<_ctype> {                    \
        using FuseAddHSwishOpBase::FuseAddHSwishOpBase;                               \
        using FuseAddHSwishOpBase::operator();                                        \
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
            auto val1 = src0.val[0];                                                  \
            auto val2 = src0.val[1];                                                  \
            auto val3 = src1.val[0];                                                  \
            auto val4 = src1.val[1];                                                  \
            val1 = GiAdd##_func_suffix(val1, val3);                                   \
            val2 = GiAdd##_func_suffix(val2, val4);                                   \
            H_SWISH_KERN_FALLBACK(_func_suffix, val1, val2);                          \
            return {{val1, val2}};                                                    \
        }                                                                             \
        void operator()(                                                              \
                const _simd_type& src0, const _simd_type& src1,                       \
                dst_ctype* dst) const {                                               \
            auto vitem = operator()(src0, src1);                                      \
            GiStore##_func_suffix(dst, vitem);                                        \
        }                                                                             \
        _simd_type operator()(const _simd_type& src0, const _simd_type& src1) const { \
            auto val1 = src0;                                                         \
            auto val2 = src1;                                                         \
            val1 = GiAdd##_func_suffix(val1, val2);                                   \
            H_SWISH_KERN_N1_FALLBACK(_func_suffix, val1);                             \
            return val1;                                                              \
        }                                                                             \
    };
OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
#undef OP

template <>
struct FuseAddHSwishOpBase<dt_qint32, dt_qint8> : BinaryOpBase<dt_qint32, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1, dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        float tmp =
                src0.as_int32() * this->scale_src0 + src1.as_int32() * this->scale_src1;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        tmp *= this->scale_dst;
        return QConverter::convert<dt_qint8, float>(tmp);
    }
};

template <>
struct FuseAddHSwishOp<dt_qint32, dt_qint8> : FuseAddHSwishOpBase<dt_qint32, dt_qint8> {
    using FuseAddHSwishOpBase::FuseAddHSwishOpBase;
    using FuseAddHSwishOpBase::operator();
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    void operator()(
            const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1,
            dt_qint8* dst) const {
        GiStoreLowInt8(reinterpret_cast<int8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1) const {
        GI_FLOAT32_t vitem0, vitem1;

        vitem0 = GiAddFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[0]), this->vscale_src0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[0]), this->vscale_src1));
        vitem1 = GiAddFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[1]), this->vscale_src0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[1]), this->vscale_src1));
        H_SWISH_KERN_FALLBACK(Float32, vitem0, vitem1);
        vitem0 = GiMultiplyFloat32(vitem0, this->vscale_dst);
        vitem1 = GiMultiplyFloat32(vitem1, this->vscale_dst);
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>({{vitem0, vitem1}});
    }
};

#include "src/fallback/elemwise_helper/kimpl/kern_macro_epilogue.h"

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
