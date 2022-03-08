/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/fuse_add_relu.h
 */
#pragma once

#include "gi_util_impl_helper.h"
#include "src/fallback/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace fallback {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddReluOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(
            const src_ctype& src0, const src_ctype& src1, dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        auto tmp = src0 + src1;
        return tmp > 0 ? tmp : 0;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddReluOp;

#define OP(_ctype, _simd_type, _simd_type2, _func_suffix, _simd_width)                \
    template <>                                                                       \
    struct FuseAddReluOp<_ctype> : FuseAddReluOpBase<_ctype> {                        \
        using FuseAddReluOpBase::FuseAddReluOpBase;                                   \
        using FuseAddReluOpBase::operator();                                          \
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
            FUSE_ADD_RELU_SIMD_PACK2_FALLBACK(val1, val2, val3, val4, _func_suffix);  \
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
            FUSE_ADD_RELU_SIMD_PACK_FALLBACK(val1, val2, _func_suffix);               \
            return val1;                                                              \
        }                                                                             \
    };
OP(dt_float32, GI_FLOAT32_t, GI_FLOAT32_V2_t, Float32, GI_SIMD_LEN_BYTE / sizeof(float))
OP(dt_int32, GI_INT32_t, GI_INT32_V2_t, Int32, GI_SIMD_LEN_BYTE / sizeof(int32_t))
OP(dt_int8, GI_INT8_t, GI_INT8_V2_t, Int8, GI_SIMD_LEN_BYTE / sizeof(int8_t))
#undef OP

template <typename ctype>
struct FuseAddReluOpCommon;

template <>
struct FuseAddReluOpCommon<float> {
    inline static GI_FLOAT32_t vzero() { return GiBroadcastFloat32(0); }
};

template <>
struct FuseAddReluOpCommon<int> {
    inline static GI_INT32_t vzero() { return GiBroadcastInt32(0); }
};

template <>
struct FuseAddReluOpBase<dt_qint8, dt_qint8> : BinaryOpBase<dt_qint8, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint8& src0, const dt_qint8& src1, dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1) const {
        return QConverter::convert<dt_qint8, float>(std::max<float>(
                src0.as_int8() * this->scale0 + src1.as_int8() * this->scale1, 0.f));
    }
};

template <>
struct FuseAddReluOp<dt_qint8, dt_qint8> : FuseAddReluOpBase<dt_qint8, dt_qint8>,
                                           FuseAddReluOpCommon<float> {
    using FuseAddReluOpBase::FuseAddReluOpBase;
    using FuseAddReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);

    void operator()(
            const GI_INT8_V2_t& vsrc0, const GI_INT8_V2_t& vsrc1, dt_qint8* dst) const {
        OPERATOR_BINARY_QINT8_FALLBACK;
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1) const {
        auto vitem0 = GiAddFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[0]), this->vscale0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[0]), this->vscale1));
        auto vitem1 = GiAddFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[1]), this->vscale0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[1]), this->vscale1));

        vitem0 = GiMaximumFloat32(vitem0, this->vzero());
        vitem1 = GiMaximumFloat32(vitem1, this->vzero());
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>({{vitem0, vitem1}});
    }
};

template <>
struct FuseAddReluOpBase<dt_qint32, dt_qint8> : BinaryOpBase<dt_qint32, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1, dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        return QConverter::convert<dt_qint8, float>(std::max<float>(
                src0.as_int32() * this->scale0 + src1.as_int32() * this->scale1, 0.f));
    }
};

template <>
struct FuseAddReluOp<dt_qint32, dt_qint8> : FuseAddReluOpBase<dt_qint32, dt_qint8>,
                                            FuseAddReluOpCommon<float> {
    using FuseAddReluOpBase::FuseAddReluOpBase;
    using FuseAddReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;
    void operator()(
            const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1,
            dt_qint8* dst) const {
        GiStoreLowInt8(reinterpret_cast<int8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1) const {
        auto vitem0 = GiAddFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[0]), this->vscale0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[0]), this->vscale1));
        auto vitem1 = GiAddFloat32(
                GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[1]), this->vscale0),
                GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[1]), this->vscale1));

        vitem0 = GiMaximumFloat32(vitem0, this->vzero());
        vitem1 = GiMaximumFloat32(vitem1, this->vzero());
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>({{vitem0, vitem1}});
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
