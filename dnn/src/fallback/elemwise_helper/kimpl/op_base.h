/**
 * \file dnn/src/fallback/elemwise_helper/kimpl/op_base.h
 */
#pragma once

#include <cmath>
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"
#include "src/fallback/elemwise/gi_impl/gi_mathfun.h"
#include "src/fallback/quantized_converter.h"

#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/general_intrinsic/gi_int.h"

namespace megdnn {
namespace fallback {

////////////////////////// unary //////////////////////////
template <typename _src_ctype, typename _dst_ctype = _src_ctype>
struct OpBase {
    using src_ctype = _src_ctype;
    using dst_ctype = _dst_ctype;
    OpBase() = default;
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct UnaryOpBase : OpBase<src_ctype, dst_ctype> {
    using OpBase<src_ctype, dst_ctype>::OpBase;
    UnaryOpBase() = default;
    UnaryOpBase(DType /*src_dtype*/, DType /*dst_dtype*/) {}
};

#define OPERATOR_UNARY_QINT8_FALLBACK                                                  \
    GI_INT16_t vsrct0 = GiMoveLowLongInt8(vsrc.val[0]);                                \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst), operator()(                                \
                                                    {{GiMoveLowLongInt16(vsrct0),      \
                                                      GiMoveHighLongInt16(vsrct0)}})); \
    GI_INT16_t vsrct1 = GiMoveHighLongInt8(vsrc.val[0]);                               \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst + 8),                                        \
            operator()({{GiMoveLowLongInt16(vsrct1), GiMoveHighLongInt16(vsrct1)}}));  \
    GI_INT16_t vsrct2 = GiMoveLowLongInt8(vsrc.val[1]);                                \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst + 16),                                       \
            operator()({{GiMoveLowLongInt16(vsrct2), GiMoveHighLongInt16(vsrct2)}}));  \
    GI_INT16_t vsrct3 = GiMoveHighLongInt8(vsrc.val[1]);                               \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst + 24),                                       \
            operator()({{GiMoveLowLongInt16(vsrct3), GiMoveHighLongInt16(vsrct3)}}))

//! scale_src = src.scale; scale_dst = 1.f / dst.scale (div -> mul)
//! scale = src.scale / dst.scale
template <>
struct UnaryOpBase<dt_qint8, dt_qint8> : OpBase<dt_qint8, dt_qint8> {
    using OpBase::OpBase;
    float scale_src, scale_dst;
    GI_FLOAT32_t vscale_src, vscale_dst;
    float scale;
    GI_FLOAT32_t vscale;

    void init(float src_scale, float dst_scale) {
        scale_src = src_scale;
        vscale_src = GiBroadcastFloat32(scale_src);
        scale_dst = 1.f / dst_scale;
        vscale_dst = GiBroadcastFloat32(scale_dst);
        scale = src_scale / dst_scale;
        vscale = GiBroadcastFloat32(scale);
    }

    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src_scale, dst_scale);
    }
    UnaryOpBase(float src_scale, float dst_scale) { init(src_scale, dst_scale); }
};

template <>
struct UnaryOpBase<dt_qint32, dt_qint8> : OpBase<dt_qint32, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_qint8;
    float scale;
    GI_FLOAT32_t vscale;
    float scale_src, scale_dst;
    GI_FLOAT32_t vscale_src, vscale_dst;

    void init(float src_scale, float dst_scale) {
        scale_src = src_scale;
        vscale_src = GiBroadcastFloat32(src_scale);
        scale_dst = 1 / dst_scale;
        vscale_dst = GiBroadcastFloat32(scale_dst);
        scale = src_scale / dst_scale;
        vscale = GiBroadcastFloat32(scale);
    }

    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src_scale, dst_scale);
    }

    UnaryOpBase(float src_scale, float dst_scale) { init(src_scale, dst_scale); }
};

////////////////////////// binary //////////////////////////
template <typename src_ctype, typename dst_ctype = src_ctype>
struct BinaryOpBase : OpBase<src_ctype, dst_ctype> {
    using OpBase<src_ctype, dst_ctype>::OpBase;
    BinaryOpBase() = default;
    BinaryOpBase(DType /*src0_dtype*/, DType /*src1_dtype*/, DType /*dst_dtype*/) {}
};

/* ================= binary op for quantized types ================== */

#define OPERATOR_BINARY_QINT8_FALLBACK                                                 \
    GI_INT16_t vsrct0_0 = GiMoveLowLongInt8(vsrc0.val[0]);                             \
    GI_INT16_t vsrct1_0 = GiMoveLowLongInt8(vsrc1.val[0]);                             \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst),                                            \
            operator()(                                                                \
                    {{GiMoveLowLongInt16(vsrct0_0), GiMoveHighLongInt16(vsrct0_0)}},   \
                    {{GiMoveLowLongInt16(vsrct1_0), GiMoveHighLongInt16(vsrct1_0)}})); \
    GI_INT16_t vsrct0_1 = GiMoveHighLongInt8(vsrc0.val[0]);                            \
    GI_INT16_t vsrct1_1 = GiMoveHighLongInt8(vsrc1.val[0]);                            \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst + 8),                                        \
            operator()(                                                                \
                    {{GiMoveLowLongInt16(vsrct0_1), GiMoveHighLongInt16(vsrct0_1)}},   \
                    {{GiMoveLowLongInt16(vsrct1_1), GiMoveHighLongInt16(vsrct1_1)}})); \
    GI_INT16_t vsrct0_2 = GiMoveLowLongInt8(vsrc0.val[1]);                             \
    GI_INT16_t vsrct1_2 = GiMoveLowLongInt8(vsrc1.val[1]);                             \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst + 16),                                       \
            operator()(                                                                \
                    {{GiMoveLowLongInt16(vsrct0_2), GiMoveHighLongInt16(vsrct0_2)}},   \
                    {{GiMoveLowLongInt16(vsrct1_2), GiMoveHighLongInt16(vsrct1_2)}})); \
    GI_INT16_t vsrct0_3 = GiMoveHighLongInt8(vsrc0.val[1]);                            \
    GI_INT16_t vsrct1_3 = GiMoveHighLongInt8(vsrc1.val[1]);                            \
    GiStoreLowInt8(                                                                    \
            reinterpret_cast<int8_t*>(dst + 24),                                       \
            operator()(                                                                \
                    {{GiMoveLowLongInt16(vsrct0_3), GiMoveHighLongInt16(vsrct0_3)}},   \
                    {{GiMoveLowLongInt16(vsrct1_3), GiMoveHighLongInt16(vsrct1_3)}}))

//! scale_src0 = src0.scale; scale_src1 = src1.scale; scale_dst = 1.f /
//! dst.scale scale0 = src0.scale / dst.scale; scale1 = src1.scale / dst.scale
template <>
struct BinaryOpBase<dt_qint8, dt_qint8> : OpBase<dt_qint8, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint8;
    using dst_ctype = dt_qint8;
    float scale_src0, scale_src1, scale_dst;
    GI_FLOAT32_t vscale_src0, vscale_src1, vscale_dst;
    float scale0, scale1;
    GI_FLOAT32_t vscale0, vscale1;

    void init(float src0_scale, float src1_scale, float dst_scale) {
        scale_src0 = src0_scale;
        vscale_src0 = GiBroadcastFloat32(scale_src0);
        scale_src1 = src1_scale;
        vscale_src1 = GiBroadcastFloat32(scale_src1);
        scale_dst = 1.f / dst_scale;
        vscale_dst = GiBroadcastFloat32(scale_dst);
        scale0 = src0_scale / dst_scale;
        vscale0 = GiBroadcastFloat32(scale0);
        scale1 = src1_scale / dst_scale;
        vscale1 = GiBroadcastFloat32(scale1);
    }

    BinaryOpBase(DType src0_dtype, DType src1_dtype, DType dst_dtype) {
        float src0_scale = src0_dtype.param<dtype::QuantizedS8>().scale;
        float src1_scale = src1_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src0_scale, src1_scale, dst_scale);
    }

    BinaryOpBase(float src0_scale, float src1_scale, float dst_scale) {
        init(src0_scale, src1_scale, dst_scale);
    }
};

template <>
struct BinaryOpBase<dt_qint32, dt_qint8> : OpBase<dt_qint32, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_qint8;
    float scale0, scale1;
    GI_FLOAT32_t vscale0, vscale1;
    float scale_src0, scale_src1, scale_dst;
    GI_FLOAT32_t vscale_src0, vscale_src1, vscale_dst;

    void init(float src0_scale, float src1_scale, float dst_scale) {
        scale_src0 = src0_scale;
        vscale_src0 = GiBroadcastFloat32(src0_scale);
        scale_src1 = src1_scale;
        vscale_src1 = GiBroadcastFloat32(src1_scale);
        scale_dst = 1 / dst_scale;
        vscale_dst = GiBroadcastFloat32(scale_dst);
        scale0 = src0_scale / dst_scale;
        vscale0 = GiBroadcastFloat32(scale0);
        scale1 = src1_scale / dst_scale;
        vscale1 = GiBroadcastFloat32(scale1);
    }

    BinaryOpBase(DType src0_dtype, DType src1_dtype, DType dst_dtype) {
        float src0_scale = src0_dtype.param<dtype::QuantizedS32>().scale;
        float src1_scale = src1_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src0_scale, src1_scale, dst_scale);
    }

    BinaryOpBase(float src0_scale, float src1_scale, float dst_scale) {
        init(src0_scale, src1_scale, dst_scale);
    }
};

////////////////////////// ternary //////////////////////////
template <typename src_ctype, typename dst_ctype = src_ctype>
struct TernaryOpBase : OpBase<src_ctype, dst_ctype> {
    using OpBase<src_ctype, dst_ctype>::OpBase;
    TernaryOpBase() = default;
    TernaryOpBase(
            DType /*src0_dtype*/, DType /*src1_dtype*/, DType /*src2_dtype*/,
            DType /*dst_dtype*/) {}
};

#define OPERATOR_TERNARY_QINT8_FALLBACK                                            \
    GI_INT16_t vsrct0 = GiMoveLowLongInt8(vsrc0.val[0]);                           \
    GI_INT16_t vsrct1 = GiMoveLowLongInt8(vsrc1.val[0]);                           \
    GI_INT16_t vsrct2 = GiMoveLowLongInt8(vsrc2.val[0]);                           \
    GiStoreLowInt8(                                                                \
            reinterpret_cast<int8_t*>(dst),                                        \
            operator()(                                                            \
                    {{GiMoveLowLongInt16(vsrct0), GiMoveHighLongInt16(vsrct0)}},   \
                    {{GiMoveLowLongInt16(vsrct1), GiMoveHighLongInt16(vsrct1)}},   \
                    {{GiMoveLowLongInt16(vsrct2), GiMoveHighLongInt16(vsrct2)}})); \
    vsrct0 = GiMoveHighLongInt8(vsrc0.val[0]);                                     \
    vsrct1 = GiMoveHighLongInt8(vsrc1.val[0]);                                     \
    vsrct2 = GiMoveHighLongInt8(vsrc2.val[0]);                                     \
    GiStoreLowInt8(                                                                \
            reinterpret_cast<int8_t*>(dst + 8),                                    \
            operator()(                                                            \
                    {{GiMoveLowLongInt16(vsrct0), GiMoveHighLongInt16(vsrct0)}},   \
                    {{GiMoveLowLongInt16(vsrct1), GiMoveHighLongInt16(vsrct1)}},   \
                    {{GiMoveLowLongInt16(vsrct2), GiMoveHighLongInt16(vsrct2)}})); \
    vsrct0 = GiMoveLowLongInt8(vsrc0.val[1]);                                      \
    vsrct1 = GiMoveLowLongInt8(vsrc1.val[1]);                                      \
    vsrct2 = GiMoveLowLongInt8(vsrc2.val[1]);                                      \
    GiStoreLowInt8(                                                                \
            reinterpret_cast<int8_t*>(dst + 16),                                   \
            operator()(                                                            \
                    {{GiMoveLowLongInt16(vsrct0), GiMoveHighLongInt16(vsrct0)}},   \
                    {{GiMoveLowLongInt16(vsrct1), GiMoveHighLongInt16(vsrct1)}},   \
                    {{GiMoveLowLongInt16(vsrct2), GiMoveHighLongInt16(vsrct2)}})); \
    vsrct0 = GiMoveHighLongInt8(vsrc0.val[1]);                                     \
    vsrct1 = GiMoveHighLongInt8(vsrc1.val[1]);                                     \
    vsrct2 = GiMoveHighLongInt8(vsrc2.val[1]);                                     \
    GiStoreLowInt8(                                                                \
            reinterpret_cast<int8_t*>(dst + 24),                                   \
            operator()(                                                            \
                    {{GiMoveLowLongInt16(vsrct0), GiMoveHighLongInt16(vsrct0)}},   \
                    {{GiMoveLowLongInt16(vsrct1), GiMoveHighLongInt16(vsrct1)}},   \
                    {{GiMoveLowLongInt16(vsrct2), GiMoveHighLongInt16(vsrct2)}}))

/*========================= ternaty op for quanzited ====================*/
template <>
struct TernaryOpBase<dt_qint8, dt_qint8> : OpBase<dt_qint8, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint8;
    using dst_ctype = dt_qint8;
    float scale_src0, scale_src1, scale_src2, scale_dst;
    GI_FLOAT32_t vscale_src0, vscale_src1, vscale_src2, vscale_dst;
    float scale0, scale1, scale2;
    GI_FLOAT32_t vscale0, vscale1, vscale2;
    void init(float src0_scale, float src1_scale, float src2_scale, float dst_scale) {
        scale_src0 = src0_scale;
        scale_src1 = src1_scale;
        scale_src2 = src2_scale;
        scale_dst = 1.f / dst_scale;
        vscale_src0 = GiBroadcastFloat32(scale_src0);
        vscale_src1 = GiBroadcastFloat32(scale_src1);
        vscale_src2 = GiBroadcastFloat32(scale_src2);
        vscale_dst = GiBroadcastFloat32(scale_dst);
        scale0 = src0_scale / dst_scale;
        scale1 = src1_scale / dst_scale;
        scale2 = src2_scale / dst_scale;
        vscale0 = GiBroadcastFloat32(scale0);
        vscale1 = GiBroadcastFloat32(scale1);
        vscale2 = GiBroadcastFloat32(scale2);
    }
    TernaryOpBase(
            DType src0_dtype, DType src1_dtype, DType src2_dtype, DType dst_dtype) {
        float src0_scale = src0_dtype.param<dtype::QuantizedS8>().scale;
        float src1_scale = src1_dtype.param<dtype::QuantizedS8>().scale;
        float src2_scale = src2_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src0_scale, src1_scale, src2_scale, dst_scale);
    }
    TernaryOpBase(
            float src0_scale, float src1_scale, float src2_scale, float dst_scale) {
        init(src0_scale, src1_scale, src2_scale, dst_scale);
    }
};

////////////////////////// fixup //////////////////////////
struct FixupBase {
    GI_INT32_t vmultiplier, vshift;
    FixupBase(float scale) {
        //! ignore Fixup if scale >= 0.5, using typecvt instead of shift &
        //! multiplier, as it may introduce errors.
        if (scale >= 0.5)
            return;

        int shift = static_cast<int>(::ceilf(::log2f(0.5 / scale)));
        scale *= ::powf(2, shift);
        //! Using double can get full precision here, but it can be ignored.
        vmultiplier = GiBroadcastInt32(
                std::round(static_cast<double>(scale) * ((2LL) << 30)));
        vshift = GiBroadcastInt32(-shift);
    }
};

//////////////////////// quantization common ////////////////////
template <typename src_type, typename dst_type, typename Op>
struct UnaryQuantizationOp;

template <typename Op>
struct UnaryQuantizationOp<dt_qint8, dt_qint8, Op> : UnaryOpBase<dt_qint8, dt_qint8> {
    using UnaryOpBase<dt_qint8, dt_qint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    Op op;

    void operator()(const dt_qint8& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }

    dt_qint8 operator()(const dt_qint8& src) const {
        float fsrc = src.as_int8() * this->scale_src;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }

    void operator()(const GI_INT8_V2_t& vsrc, dt_qint8* dst) const {
        OPERATOR_UNARY_QINT8_FALLBACK;
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc) const {
        auto vitem0 = GiMultiplyFloat32(GiCastToFloat32(vsrc.val[0]), this->vscale_src);
        auto vitem1 = GiMultiplyFloat32(GiCastToFloat32(vsrc.val[1]), this->vscale_src);
        auto val = this->op({{vitem0, vitem1}});
        val.val[0] = GiMultiplyFloat32(val.val[0], this->vscale_dst);
        val.val[1] = GiMultiplyFloat32(val.val[1], this->vscale_dst);
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(val);
    }
};

template <typename src_type, typename dst_type, typename Op>
struct BinaryQuantizationOp;

template <typename Op>
struct BinaryQuantizationOp<dt_qint8, dt_qint8, Op> : BinaryOpBase<dt_qint8, dt_qint8> {
    using BinaryOpBase<dt_qint8, dt_qint8>::BinaryOpBase;
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    Op op;

    void operator()(const dt_qint8& src0, const dt_qint8& src1, dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1) const {
        float fsrc0 = src0.as_int8() * this->scale_src0;
        float fsrc1 = src1.as_int8() * this->scale_src1;
        float fdst = op(fsrc0, fsrc1);
        fdst = fdst * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fdst);
    }

    void operator()(
            const GI_INT8_V2_t& vsrc0, const GI_INT8_V2_t& vsrc1, dt_qint8* dst) const {
        OPERATOR_BINARY_QINT8_FALLBACK;
    }

    GI_INT8_t operator()(const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1) const {
        auto val0 = GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[0]), this->vscale_src0);
        auto val1 = GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[1]), this->vscale_src0);
        auto val2 = GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[0]), this->vscale_src1);
        auto val3 = GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[1]), this->vscale_src1);
        auto val = op({{val0, val1}}, {{val2, val3}});
        val.val[0] = GiMultiplyFloat32(val.val[0], this->vscale_dst);
        val.val[1] = GiMultiplyFloat32(val.val[1], this->vscale_dst);
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(val);
    }
};

template <typename src_type, typename dst_type, typename Op>
struct TernaryQuantizationOp;

template <typename Op>
struct TernaryQuantizationOp<dt_qint8, dt_qint8, Op>
        : TernaryOpBase<dt_qint8, dt_qint8> {
    using TernaryOpBase<dt_qint8, dt_qint8>::TernaryOpBase;
    constexpr static size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    Op op;

    void operator()(
            const dt_qint8& src0, const dt_qint8& src1, const dt_qint8& src2,
            dt_qint8* dst) const {
        *dst = operator()(src0, src1, src2);
    }

    dt_qint8 operator()(
            const dt_qint8& src0, const dt_qint8& src1, const dt_qint8& src2) const {
        float fsrc0 = src0.as_int8() * this->scale_src0;
        float fsrc1 = src1.as_int8() * this->scale_src1;
        float fsrc2 = src2.as_int8() * this->scale_src2;
        float fdst = op(fsrc0, fsrc1, fsrc2);
        fdst = fdst * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fdst);
    }

    void operator()(
            const GI_INT8_V2_t& vsrc0, const GI_INT8_V2_t& vsrc1,
            const GI_INT8_V2_t& vsrc2, dt_qint8* dst) const {
        OPERATOR_TERNARY_QINT8_FALLBACK;
    }

    GI_INT8_t operator()(
            const GI_INT32_V2_t& vsrc0, const GI_INT32_V2_t& vsrc1,
            const GI_INT32_V2_t& vsrc2) const {
        auto val0 = GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[0]), this->vscale_src0);
        auto val1 = GiMultiplyFloat32(GiCastToFloat32(vsrc0.val[1]), this->vscale_src0);
        auto val2 = GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[0]), this->vscale_src1);
        auto val3 = GiMultiplyFloat32(GiCastToFloat32(vsrc1.val[1]), this->vscale_src1);
        auto val4 = GiMultiplyFloat32(GiCastToFloat32(vsrc2.val[0]), this->vscale_src2);
        auto val5 = GiMultiplyFloat32(GiCastToFloat32(vsrc2.val[1]), this->vscale_src2);
        auto val = op({{val0, val1}}, {{val2, val3}}, {{val4, val5}});
        val.val[0] = GiMultiplyFloat32(val.val[0], this->vscale_dst);
        val.val[1] = GiMultiplyFloat32(val.val[1], this->vscale_dst);
        return QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(val);
    }
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
