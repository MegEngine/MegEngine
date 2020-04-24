/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/op_base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cmath>
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/arm_common/elemwise/neon_mathfun.h"
#include "src/arm_common/quantized_converter.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

namespace megdnn {
namespace arm_common {

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

#define OPERATOR_UNARY_QINT8                                              \
    int16x8_t vsrct = vmovl_low_s8(vsrc.val[0]);                          \
    vst1_s8(reinterpret_cast<int8_t*>(dst),                               \
            operator()({{vmovl_low_s16(vsrct), vmovl_high_s16(vsrct)}})); \
                                                                          \
    vsrct = vmovl_high_s8(vsrc.val[0]);                                   \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 8),                           \
            operator()({{vmovl_low_s16(vsrct), vmovl_high_s16(vsrct)}})); \
                                                                          \
    vsrct = vmovl_low_s8(vsrc.val[1]);                                    \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 16),                          \
            operator()({{vmovl_low_s16(vsrct), vmovl_high_s16(vsrct)}})); \
                                                                          \
    vsrct = vmovl_high_s8(vsrc.val[1]);                                   \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 24),                          \
            operator()({{vmovl_low_s16(vsrct), vmovl_high_s16(vsrct)}}));

#define OPERATOR_UNARY_QUINT8                                             \
    uint16x8_t vsrct = vmovl_low_u8(vsrc.val[0]);                         \
    vst1_u8(reinterpret_cast<uint8_t*>(dst),                              \
            operator()({{vmovl_low_u16(vsrct), vmovl_high_u16(vsrct)}})); \
                                                                          \
    vsrct = vmovl_high_u8(vsrc.val[0]);                                   \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 8),                          \
            operator()({{vmovl_low_u16(vsrct), vmovl_high_u16(vsrct)}})); \
                                                                          \
    vsrct = vmovl_low_u8(vsrc.val[1]);                                    \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 16),                         \
            operator()({{vmovl_low_u16(vsrct), vmovl_high_u16(vsrct)}})); \
                                                                          \
    vsrct = vmovl_high_u8(vsrc.val[1]);                                   \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 24),                         \
            operator()({{vmovl_low_u16(vsrct), vmovl_high_u16(vsrct)}}));

//! scale_src = src.scale; scale_dst = 1.f / dst.scale (div -> mul)
//! scale = src.scale / dst.scale
template <>
struct UnaryOpBase<dt_qint8, dt_qint8> : OpBase<dt_qint8, dt_qint8> {
    using OpBase::OpBase;
    float scale_src, scale_dst;
    float32x4_t vscale_src, vscale_dst;
    float scale;
    float32x4_t vscale;

    void init(float src_scale, float dst_scale) {
        scale_src = src_scale;
        vscale_src = vdupq_n_f32(scale_src);
        scale_dst = 1.f / dst_scale;
        vscale_dst = vdupq_n_f32(scale_dst);
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src_scale, dst_scale);
    }
    UnaryOpBase(float src_scale, float dst_scale) {
        init(src_scale, dst_scale);
    }
};


//! scale_src = src.scale; scale_dst = 1.f / dst.scale
//! scale_zp = src.zp * src.scale; dzp = dst.zp
//! scale = src.scale / dst.scale; szp = src.zp * scale
template <>
struct UnaryOpBase<dt_quint8, dt_quint8> : OpBase<dt_quint8, dt_quint8> {
    using OpBase::OpBase;
    float scale_src, scale_dst;
    float32x4_t vscale_src, vscale_dst;
    float scale_zp;
    float32x4_t vscale_zp;
    uint8_t dzp;
    int32x4_t vdzp;
    float scale, szp;
    float32x4_t vscale, vszp;

    void init(float src_scale, float dst_scale, uint8_t src_zp,
              uint8_t dst_zp) {
        scale_src = src_scale;
        scale_dst = 1.f / dst_scale;
        vscale_src = vdupq_n_f32(scale_src);
        vscale_dst = vdupq_n_f32(scale_dst);
        scale_zp = src_zp * src_scale;
        vscale_zp = vdupq_n_f32(scale_zp);
        dzp = dst_zp;
        vdzp = vdupq_n_s32(static_cast<int32_t>(dzp));
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
        szp = src_zp * scale;
        vszp = vdupq_n_f32(szp);
    }
    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::Quantized8Asymm>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        uint8_t src_zp = src_dtype.param<dtype::Quantized8Asymm>().zero_point;
        uint8_t dst_zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        init(src_scale, dst_scale, src_zp, dst_zp);
    }
    UnaryOpBase(float src_scale, float dst_scale, uint8_t src_zp,
                uint8_t dst_zp) {
        init(src_scale, dst_scale, src_zp, dst_zp);
    }
    float32x4x2_t cvt_to_float(const uint32x4x2_t& vsrc) {
        auto vitem0 = vmulq_f32(vcvtq_f32_u32(vsrc.val[0]), this->vscale_src);
        vitem0 = vsubq_f32(vitem0, this->vscale_zp);
        auto vitem1 = vmulq_f32(vcvtq_f32_u32(vsrc.val[1]), this->vscale_src);
        vitem1 = vsubq_f32(vitem1, this->vscale_zp);
        return {{vitem0, vitem1}};
    }
    uint8x8_t cvt_float_to_dst(float32x4x2_t& vsrc) {
        auto vitem0 = vmulq_f32(vsrc.val[0], this->vscale_dst);
        auto vitem1 = vmulq_f32(vsrc.val[1], this->vscale_dst);
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                {{vitem0, vitem1}}, this->vdzp);
    }
    float32x4x2_t cvt_to_fdst(const uint32x4x2_t& vsrc) {
        auto vitem0 = vmulq_f32(vcvtq_f32_u32(vsrc.val[0]), this->vscale);
        vitem0 = vsubq_f32(vitem0, this->vszp);
        auto vitem1 = vmulq_f32(vcvtq_f32_u32(vsrc.val[1]), this->vscale);
        vitem1 = vsubq_f32(vitem1, this->vszp);
        return {{vitem0, vitem1}};
    }
    uint8x8_t cvt_fdst_to_dst(float32x4x2_t& vsrc) {
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                vsrc, this->vdzp);
    }
};

template <>
struct UnaryOpBase<dt_qint32, dt_qint8> : OpBase<dt_qint32, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_qint8;
    float scale;
    float32x4_t vscale;
    float scale_src, scale_dst;
    float32x4_t vscale_src, vscale_dst;

    void init(float src_scale, float dst_scale) {
        scale_src = src_scale;
        vscale_src = vdupq_n_f32(src_scale);
        scale_dst = 1 / dst_scale;
        vscale_dst = vdupq_n_f32(scale_dst);
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src_scale, dst_scale);
    }

    UnaryOpBase(float src_scale, float dst_scale) {
        init(src_scale, dst_scale);
    }
};

template <>
struct UnaryOpBase<dt_qint32, dt_quint8> : OpBase<dt_qint32, dt_quint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_quint8;
    float scale;
    float32x4_t vscale;
    float scale_src, scale_dst;
    float32x4_t vscale_src, vscale_dst;
    uint8_t zp;
    int32x4_t vzp;

    void init(float src_scale, float dst_scale, uint8_t zero_point) {
        scale_src = src_scale;
        vscale_src = vdupq_n_f32(src_scale);
        scale_dst = 1 / dst_scale;
        vscale_dst = vdupq_n_f32(scale_dst);
        zp = zero_point;
        vzp = vdupq_n_s32(static_cast<int>(zp));
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    UnaryOpBase(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        uint8_t zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        init(src_scale, dst_scale, zp);
    }

    UnaryOpBase(float src_scale, float dst_scale, uint8_t zero_point) {
        init(src_scale, dst_scale, zero_point);
    }
};

////////////////////////// binary //////////////////////////
template <typename src_ctype, typename dst_ctype = src_ctype>
struct BinaryOpBase : OpBase<src_ctype, dst_ctype> {
    using OpBase<src_ctype, dst_ctype>::OpBase;
    BinaryOpBase() = default;
    BinaryOpBase(DType /*src0_dtype*/, DType /*src1_dtype*/,
                 DType /*dst_dtype*/) {}
};

#define OPERATOR_BINARY_QINT8                                               \
    int16x8_t vsrct0 = vmovl_low_s8(vsrc0.val[0]);                          \
    int16x8_t vsrct1 = vmovl_low_s8(vsrc1.val[0]);                          \
    vst1_s8(reinterpret_cast<int8_t*>(dst),                                 \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}})); \
                                                                            \
    vsrct0 = vmovl_high_s8(vsrc0.val[0]);                                   \
    vsrct1 = vmovl_high_s8(vsrc1.val[0]);                                   \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 8),                             \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}})); \
                                                                            \
    vsrct0 = vmovl_low_s8(vsrc0.val[1]);                                    \
    vsrct1 = vmovl_low_s8(vsrc1.val[1]);                                    \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 16),                            \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}})); \
                                                                            \
    vsrct0 = vmovl_high_s8(vsrc0.val[1]);                                   \
    vsrct1 = vmovl_high_s8(vsrc1.val[1]);                                   \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 24),                            \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}}))

#define OPERATOR_BINARY_QUINT8                                              \
    uint16x8_t vsrct0 = vmovl_low_u8(vsrc0.val[0]);                         \
    uint16x8_t vsrct1 = vmovl_low_u8(vsrc1.val[0]);                         \
    vst1_u8(reinterpret_cast<uint8_t*>(dst),                                \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}})); \
                                                                            \
    vsrct0 = vmovl_high_u8(vsrc0.val[0]);                                   \
    vsrct1 = vmovl_high_u8(vsrc1.val[0]);                                   \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 8),                            \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}})); \
                                                                            \
    vsrct0 = vmovl_low_u8(vsrc0.val[1]);                                    \
    vsrct1 = vmovl_low_u8(vsrc1.val[1]);                                    \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 16),                           \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}})); \
                                                                            \
    vsrct0 = vmovl_high_u8(vsrc0.val[1]);                                   \
    vsrct1 = vmovl_high_u8(vsrc1.val[1]);                                   \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 24),                           \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}}))

/* ================= binary op for quantized types ================== */

//! scale_src0 = src0.scale; scale_src1 = src1.scale; scale_dst = 1.f /
//! dst.scale scale0 = src0.scale / dst.scale; scale1 = src1.scale / dst.scale
template <>
struct BinaryOpBase<dt_qint8, dt_qint8> : OpBase<dt_qint8, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint8;
    using dst_ctype = dt_qint8;
    float scale_src0, scale_src1, scale_dst;
    float32x4_t vscale_src0, vscale_src1, vscale_dst;
    float scale0, scale1;
    float32x4_t vscale0, vscale1;

    void init(float src0_scale, float src1_scale, float dst_scale) {
        scale_src0 = src0_scale;
        vscale_src0 = vdupq_n_f32(scale_src0);
        scale_src1 = src1_scale;
        vscale_src1 = vdupq_n_f32(scale_src1);
        scale_dst = 1.f / dst_scale;
        vscale_dst = vdupq_n_f32(scale_dst);
        scale0 = src0_scale / dst_scale;
        vscale0 = vdupq_n_f32(scale0);
        scale1 = src1_scale / dst_scale;
        vscale1 = vdupq_n_f32(scale1);
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

//! scale_src0 = src0.scale; scale_src1 = src1.scale; scale_dst = 1.f /
//! dst.scale scale_zp0 = src0.zp * src0.scale; scale_zp1 = src1.zp * src1.scale
//! scale0 = src0.scale / dst.scale; scale1 = src1.scale / dst.scale
//! szp0 = src0.zp * scale0; szp1 = src1.zp * scale1
//! dzp = dst.zp
template <>
struct BinaryOpBase<dt_quint8, dt_quint8> : OpBase<dt_quint8, dt_quint8> {
    using OpBase::OpBase;
    using src_ctype = dt_quint8;
    using dst_ctype = dt_quint8;
    float scale_src0, scale_src1, scale_dst;
    float32x4_t vscale_src0, vscale_src1, vscale_dst;
    float scale_zp0, scale_zp1;
    float32x4_t vscale_zp0, vscale_zp1;
    float scale0, scale1, szp0, szp1;
    float32x4_t vscale0, vscale1, vszp0, vszp1;
    uint8_t dzp;
    int32x4_t vdzp;

    void init(float src0_scale, float src1_scale, float dst_scale,
              uint8_t src0_zp, uint8_t src1_zp, uint8_t dst_zp) {
        scale_src0 = src0_scale;
        vscale_src0 = vdupq_n_f32(scale_src0);
        scale_src1 = src1_scale;
        vscale_src1 = vdupq_n_f32(scale_src1);
        scale_dst = 1.f / dst_scale;
        vscale_dst = vdupq_n_f32(scale_dst);
        scale_zp0 = src0_zp * src0_scale;
        vscale_zp0 = vdupq_n_f32(scale_zp0);
        scale_zp1 = src1_zp * src1_scale;
        vscale_zp1 = vdupq_n_f32(scale_zp1);
        scale0 = src0_scale / dst_scale;
        vscale0 = vdupq_n_f32(scale0);
        scale1 = src1_scale / dst_scale;
        vscale1 = vdupq_n_f32(scale1);
        dzp = dst_zp;
        vdzp = vdupq_n_s32(static_cast<int32_t>(dzp));
        szp0 = src0_zp * scale0;
        szp1 = src1_zp * scale1;
        vszp0 = vdupq_n_f32(szp0);
        vszp1 = vdupq_n_f32(szp1);
    }

    BinaryOpBase(DType src0_dtype, DType src1_dtype, DType dst_dtype) {
        float src0_scale = src0_dtype.param<dtype::Quantized8Asymm>().scale;
        float src1_scale = src1_dtype.param<dtype::Quantized8Asymm>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        uint8_t src0_zp = src0_dtype.param<dtype::Quantized8Asymm>().zero_point;
        uint8_t src1_zp = src1_dtype.param<dtype::Quantized8Asymm>().zero_point;
        uint8_t dst_zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        init(src0_scale, src1_scale, dst_scale, src0_zp, src1_zp, dst_zp);
    }

    BinaryOpBase(float src0_scale, float src1_scale, float dst_scale,
                 uint8_t src0_zp, uint8_t src1_zp, uint8_t dst_zp) {
        init(src0_scale, src1_scale, dst_scale, src0_zp, src1_zp, dst_zp);
    }
};

template <>
struct BinaryOpBase<dt_qint32, dt_qint8> : OpBase<dt_qint32, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_qint8;
    float scale0, scale1;
    float32x4_t vscale0, vscale1;
    float scale_src0, scale_src1, scale_dst;
    float32x4_t vscale_src0, vscale_src1, vscale_dst;

    void init(float src0_scale, float src1_scale, float dst_scale) {
        scale_src0 = src0_scale;
        vscale_src0 = vdupq_n_f32(src0_scale);
        scale_src1 = src1_scale;
        vscale_src1 = vdupq_n_f32(src1_scale);
        scale_dst = 1 / dst_scale;
        vscale_dst = vdupq_n_f32(scale_dst);
        scale0 = src0_scale / dst_scale;
        vscale0 = vdupq_n_f32(scale0);
        scale1 = src1_scale / dst_scale;
        vscale1 = vdupq_n_f32(scale1);
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

template <>
struct BinaryOpBase<dt_qint32, dt_quint8> : OpBase<dt_qint32, dt_quint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint32;
    using dst_ctype = dt_quint8;
    float scale0, scale1;
    float32x4_t vscale0, vscale1;
    uint8_t zp;
    int32x4_t vzp;
    float scale_src0, scale_src1, scale_dst;
    float32x4_t vscale_src0, vscale_src1, vscale_dst;

    void init(float src0_scale, float src1_scale, float dst_scale,
              uint8_t zero_point) {
        scale_src0 = src0_scale;
        vscale_src0 = vdupq_n_f32(src0_scale);
        scale_src1 = src1_scale;
        vscale_src1 = vdupq_n_f32(src1_scale);
        scale_dst = 1 / dst_scale;
        vscale_dst = vdupq_n_f32(scale_dst);
        zp = zero_point;
        vzp = vdupq_n_s32(static_cast<int>(zp));
        scale0 = src0_scale / dst_scale;
        vscale0 = vdupq_n_f32(scale0);
        scale1 = src1_scale / dst_scale;
        vscale1 = vdupq_n_f32(scale1);
    }

    BinaryOpBase(DType src0_dtype, DType src1_dtype, DType dst_dtype) {
        float src0_scale = src0_dtype.param<dtype::QuantizedS32>().scale;
        float src1_scale = src1_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        uint8_t zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        init(src0_scale, src1_scale, dst_scale, zp);
    }

    BinaryOpBase(float src0_scale, float src1_scale, float dst_scale,
                 uint8_t zero_point) {
        init(src0_scale, src1_scale, dst_scale, zero_point);
    }
};

////////////////////////// ternary //////////////////////////
template <typename src_ctype, typename dst_ctype = src_ctype>
struct TernaryOpBase : OpBase<src_ctype, dst_ctype> {
    using OpBase<src_ctype, dst_ctype>::OpBase;
    TernaryOpBase() = default;
    TernaryOpBase(DType /*src0_dtype*/, DType /*src1_dtype*/,
                  DType /*src2_dtype*/, DType /*dst_dtype*/) {}
};

#define OPERATOR_TERNARY_QINT8                                              \
    int16x8_t vsrct0 = vmovl_low_s8(vsrc0.val[0]);                          \
    int16x8_t vsrct1 = vmovl_low_s8(vsrc1.val[0]);                          \
    int16x8_t vsrct2 = vmovl_low_s8(vsrc2.val[0]);                          \
    vst1_s8(reinterpret_cast<int8_t*>(dst),                                 \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}},   \
                       {{vmovl_low_s16(vsrct2), vmovl_high_s16(vsrct2)}})); \
                                                                            \
    vsrct0 = vmovl_high_s8(vsrc0.val[0]);                                   \
    vsrct1 = vmovl_high_s8(vsrc1.val[0]);                                   \
    vsrct2 = vmovl_high_s8(vsrc2.val[0]);                                   \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 8),                             \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}},   \
                       {{vmovl_low_s16(vsrct2), vmovl_high_s16(vsrct2)}})); \
                                                                            \
    vsrct0 = vmovl_low_s8(vsrc0.val[1]);                                    \
    vsrct1 = vmovl_low_s8(vsrc1.val[1]);                                    \
    vsrct2 = vmovl_low_s8(vsrc2.val[1]);                                    \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 16),                            \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}},   \
                       {{vmovl_low_s16(vsrct2), vmovl_high_s16(vsrct2)}})); \
                                                                            \
    vsrct0 = vmovl_high_s8(vsrc0.val[1]);                                   \
    vsrct1 = vmovl_high_s8(vsrc1.val[1]);                                   \
    vsrct2 = vmovl_high_s8(vsrc2.val[1]);                                   \
    vst1_s8(reinterpret_cast<int8_t*>(dst + 24),                            \
            operator()({{vmovl_low_s16(vsrct0), vmovl_high_s16(vsrct0)}},   \
                       {{vmovl_low_s16(vsrct1), vmovl_high_s16(vsrct1)}},   \
                       {{vmovl_low_s16(vsrct2), vmovl_high_s16(vsrct2)}}))

#define OPERATOR_TERNARY_QUINT8                                             \
    uint16x8_t vsrct0 = vmovl_low_u8(vsrc0.val[0]);                         \
    uint16x8_t vsrct1 = vmovl_low_u8(vsrc1.val[0]);                         \
    uint16x8_t vsrct2 = vmovl_low_u8(vsrc2.val[0]);                         \
    vst1_u8(reinterpret_cast<uint8_t*>(dst),                                \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}},   \
                       {{vmovl_low_u16(vsrct2), vmovl_high_u16(vsrct2)}})); \
                                                                            \
    vsrct0 = vmovl_high_u8(vsrc0.val[0]);                                   \
    vsrct1 = vmovl_high_u8(vsrc1.val[0]);                                   \
    vsrct2 = vmovl_high_u8(vsrc2.val[0]);                                   \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 8),                            \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}},   \
                       {{vmovl_low_u16(vsrct2), vmovl_high_u16(vsrct2)}})); \
                                                                            \
    vsrct0 = vmovl_low_u8(vsrc0.val[1]);                                    \
    vsrct1 = vmovl_low_u8(vsrc1.val[1]);                                    \
    vsrct2 = vmovl_low_u8(vsrc2.val[1]);                                    \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 16),                           \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}},   \
                       {{vmovl_low_u16(vsrct2), vmovl_high_u16(vsrct2)}})); \
                                                                            \
    vsrct0 = vmovl_high_u8(vsrc0.val[1]);                                   \
    vsrct1 = vmovl_high_u8(vsrc1.val[1]);                                   \
    vsrct2 = vmovl_high_u8(vsrc2.val[1]);                                   \
    vst1_u8(reinterpret_cast<uint8_t*>(dst + 24),                           \
            operator()({{vmovl_low_u16(vsrct0), vmovl_high_u16(vsrct0)}},   \
                       {{vmovl_low_u16(vsrct1), vmovl_high_u16(vsrct1)}},   \
                       {{vmovl_low_u16(vsrct2), vmovl_high_u16(vsrct2)}}))

/*========================= ternaty op for quanzited ====================*/
template <>
struct TernaryOpBase<dt_qint8, dt_qint8> : OpBase<dt_qint8, dt_qint8> {
    using OpBase::OpBase;
    using src_ctype = dt_qint8;
    using dst_ctype = dt_qint8;
    float scale_src0, scale_src1, scale_src2, scale_dst;
    float32x4_t vscale_src0, vscale_src1, vscale_src2, vscale_dst;
    float scale0, scale1, scale2;
    float32x4_t vscale0, vscale1, vscale2;
    void init(float src0_scale, float src1_scale, float src2_scale,
              float dst_scale) {
        scale_src0 = src0_scale;
        scale_src1 = src1_scale;
        scale_src2 = src2_scale;
        scale_dst = 1.f / dst_scale;
        vscale_src0 = vdupq_n_f32(scale_src0);
        vscale_src1 = vdupq_n_f32(scale_src1);
        vscale_src2 = vdupq_n_f32(scale_src2);
        vscale_dst = vdupq_n_f32(scale_dst);
        scale0 = src0_scale / dst_scale;
        scale1 = src1_scale / dst_scale;
        scale2 = src2_scale / dst_scale;
        vscale0 = vdupq_n_f32(scale0);
        vscale1 = vdupq_n_f32(scale1);
        vscale2 = vdupq_n_f32(scale2);
    }
    TernaryOpBase(DType src0_dtype, DType src1_dtype, DType src2_dtype,
                  DType dst_dtype) {
        float src0_scale = src0_dtype.param<dtype::QuantizedS8>().scale;
        float src1_scale = src1_dtype.param<dtype::QuantizedS8>().scale;
        float src2_scale = src2_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        init(src0_scale, src1_scale, src2_scale, dst_scale);
    }
    TernaryOpBase(float src0_scale, float src1_scale, float src2_scale,
                  float dst_scale) {
        init(src0_scale, src1_scale, src2_scale, dst_scale);
    }
};

template <>
struct TernaryOpBase<dt_quint8, dt_quint8> : OpBase<dt_quint8, dt_quint8> {
    using OpBase::OpBase;
    using src_ctype = dt_quint8;
    using dst_ctype = dt_quint8;
    float scale_src0, scale_src1, scale_src2, scale_dst;
    float32x4_t vscale_src0, vscale_src1, vscale_src2, vscale_dst;
    float scale_zp0, scale_zp1, scale_zp2;
    float32x4_t vscale_zp0, vscale_zp1, vscale_zp2;
    float scale0, scale1, scale2;
    float32x4_t vscale0, vscale1, vscale2;
    uint8_t dzp;
    int32x4_t vdzp;
    void init(float src0_scale, float src1_scale, float src2_scale,
              float dst_scale, uint8_t src0_zp, uint8_t src1_zp,
              uint8_t src2_zp, uint8_t dst_zp) {
        scale_src0 = src0_scale;
        scale_src1 = src1_scale;
        scale_src2 = src2_scale;
        scale_dst = 1.f / dst_scale;
        vscale_src0 = vdupq_n_f32(scale_src0);
        vscale_src1 = vdupq_n_f32(scale_src1);
        vscale_src2 = vdupq_n_f32(scale_src2);
        vscale_dst = vdupq_n_f32(scale_dst);
        scale_zp0 = src0_zp * scale_src0;
        scale_zp1 = src1_zp * scale_src1;
        scale_zp2 = src2_zp * scale_src2;
        vscale_zp0 = vdupq_n_f32(scale_zp0);
        vscale_zp1 = vdupq_n_f32(scale_zp1);
        vscale_zp2 = vdupq_n_f32(scale_zp2);
        scale0 = src0_scale / dst_scale;
        scale1 = src1_scale / dst_scale;
        scale2 = src2_scale / dst_scale;
        vscale0 = vdupq_n_f32(scale0);
        vscale1 = vdupq_n_f32(scale1);
        vscale2 = vdupq_n_f32(scale2);
        dzp = dst_zp;
        vdzp = vdupq_n_s32(static_cast<int32_t>(dzp));
    }
    TernaryOpBase(DType src0_dtype, DType src1_dtype, DType src2_dtype,
                  DType dst_dtype) {
        float src0_scale = src0_dtype.param<dtype::Quantized8Asymm>().scale;
        float src1_scale = src1_dtype.param<dtype::Quantized8Asymm>().scale;
        float src2_scale = src2_dtype.param<dtype::Quantized8Asymm>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        uint8_t src0_zp = src0_dtype.param<dtype::Quantized8Asymm>().zero_point;
        uint8_t src1_zp = src1_dtype.param<dtype::Quantized8Asymm>().zero_point;
        uint8_t src2_zp = src2_dtype.param<dtype::Quantized8Asymm>().zero_point;
        uint8_t dst_zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        init(src0_scale, src1_scale, src2_scale, dst_scale, src0_zp, src1_zp,
             src2_zp, dst_zp);
    }
    TernaryOpBase(float src0_scale, float src1_scale, float src2_scale,
                  float dst_scale, uint8_t src0_zp, uint8_t src1_zp,
                  uint8_t src2_zp, uint8_t dst_zp) {
        init(src0_scale, src1_scale, src2_scale, dst_scale, src0_zp, src1_zp,
             src2_zp, dst_zp);
    }
};

////////////////////////// fixup //////////////////////////
struct FixupBase {
    int32x4_t vmultiplier, vshift;
    FixupBase(float scale) {
        //! ignore Fixup if scale >= 0.5, using typecvt instead of shift &
        //! multiplier, as it may introduce errors.
        if (scale >= 0.5)
            return;

        int shift = static_cast<int>(::ceilf(::log2f(0.5 / scale)));
        scale *= ::powf(2, shift);
        //! Using double can get full precision here, but it can be ignored.
        vmultiplier = vdupq_n_s32(
                std::round(static_cast<double>(scale) * ((2LL) << 30)));
        vshift = vdupq_n_s32(-shift);
    }
};

//////////////////////// quantization common ////////////////////
template <typename src_type, typename dst_type, typename Op>
struct UnaryQuantizationOp;

template <typename Op>
struct UnaryQuantizationOp<dt_qint8, dt_qint8, Op>
        : UnaryOpBase<dt_qint8, dt_qint8> {
    using UnaryOpBase<dt_qint8, dt_qint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
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

    void operator()(const int8x16x2_t& vsrc, dt_qint8* dst) const {
        OPERATOR_UNARY_QINT8;
    }

    int8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale_src);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale_src);
        auto val = this->op({{vitem0, vitem1}});
        val.val[0] = vmulq_f32(val.val[0], this->vscale_dst);
        val.val[1] = vmulq_f32(val.val[1], this->vscale_dst);
        return QConverter::convert<int8x8_t, float32x4x2_t>(val);
    }
};

template <typename Op>
struct UnaryQuantizationOp<dt_quint8, dt_quint8, Op>
        : UnaryOpBase<dt_quint8, dt_quint8> {
    using UnaryOpBase<dt_quint8, dt_quint8>::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_quint8& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }

    dt_quint8 operator()(const dt_quint8& src) const {
        float fsrc = src.as_uint8() * this->scale_src - this->scale_zp;
        fsrc = op(fsrc);
        fsrc = fsrc * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }

    void operator()(const uint8x16x2_t& vsrc, dt_quint8* dst) const {
        OPERATOR_UNARY_QUINT8;
    }

    uint8x8_t operator()(const uint32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_u32(vsrc.val[0]), this->vscale_src);
        vitem0 = vsubq_f32(vitem0, this->vscale_zp);
        auto vitem1 = vmulq_f32(vcvtq_f32_u32(vsrc.val[1]), this->vscale_src);
        vitem1 = vsubq_f32(vitem1, this->vscale_zp);
        auto val = this->op({{vitem0, vitem1}});
        val.val[0] = vmulq_f32(val.val[0], this->vscale_dst);
        val.val[1] = vmulq_f32(val.val[1], this->vscale_dst);
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                val, this->vdzp);
    }
};

template <typename src_type, typename dst_type, typename Op>
struct BinaryQuantizationOp;

template <typename Op>
struct BinaryQuantizationOp<dt_qint8, dt_qint8, Op>
        : BinaryOpBase<dt_qint8, dt_qint8> {
    using BinaryOpBase<dt_qint8, dt_qint8>::BinaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_qint8& src0, const dt_qint8& src1,
                    dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1) const {
        float fsrc0 = src0.as_int8() * this->scale_src0;
        float fsrc1 = src1.as_int8() * this->scale_src1;
        float fdst = op(fsrc0, fsrc1);
        fdst = fdst * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fdst);
    }

    void operator()(const int8x16x2_t& vsrc0, const int8x16x2_t& vsrc1,
                    dt_qint8* dst) const {
        OPERATOR_BINARY_QINT8;
    }

    int8x8_t operator()(const int32x4x2_t& vsrc0,
                        const int32x4x2_t& vsrc1) const {
        auto val0 = vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale_src0);
        auto val1 = vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale_src0);
        auto val2 = vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale_src1);
        auto val3 = vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale_src1);
        auto val = op({{val0, val1}}, {{val2, val3}});
        val.val[0] = vmulq_f32(val.val[0], this->vscale_dst);
        val.val[1] = vmulq_f32(val.val[1], this->vscale_dst);
        return QConverter::convert<int8x8_t, float32x4x2_t>(val);
    }
};

template <typename Op>
struct BinaryQuantizationOp<dt_quint8, dt_quint8, Op>
        : BinaryOpBase<dt_quint8, dt_quint8> {
    using BinaryOpBase<dt_quint8, dt_quint8>::BinaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_quint8& src0, const dt_quint8& src1,
                    dt_quint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_quint8 operator()(const dt_quint8& src0, const dt_quint8& src1) const {
        float fsrc0 = src0.as_uint8() * this->scale_src0 - this->scale_zp0;
        float fsrc1 = src1.as_uint8() * this->scale_src1 - this->scale_zp1;
        float fdst = op(fsrc0, fsrc1);
        fdst = fdst * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fdst, this->dzp);
    }

    void operator()(const uint8x16x2_t& vsrc0, const uint8x16x2_t& vsrc1,
                    dt_quint8* dst) const {
        OPERATOR_BINARY_QUINT8;
    }

    uint8x8_t operator()(const uint32x4x2_t& vsrc0,
                         const uint32x4x2_t& vsrc1) const {
        auto val0 = vmulq_f32(vcvtq_f32_u32(vsrc0.val[0]), this->vscale_src0);
        val0 = vsubq_f32(val0, this->vscale_zp0);
        auto val1 = vmulq_f32(vcvtq_f32_u32(vsrc0.val[1]), this->vscale_src0);
        val1 = vsubq_f32(val1, this->vscale_zp0);
        auto val2 = vmulq_f32(vcvtq_f32_u32(vsrc1.val[0]), this->vscale_src1);
        val2 = vsubq_f32(val2, this->vscale_zp1);
        auto val3 = vmulq_f32(vcvtq_f32_u32(vsrc1.val[1]), this->vscale_src1);
        val3 = vsubq_f32(val3, this->vscale_zp1);
        auto val = op({{val0, val1}}, {{val2, val3}});
        val.val[0] = vmulq_f32(val.val[0], this->vscale_dst);
        val.val[1] = vmulq_f32(val.val[1], this->vscale_dst);
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                val, this->vdzp);
    }
};

template <typename src_type, typename dst_type, typename Op>
struct TernaryQuantizationOp;

template <typename Op>
struct TernaryQuantizationOp<dt_qint8, dt_qint8, Op>
        : TernaryOpBase<dt_qint8, dt_qint8> {
    using TernaryOpBase<dt_qint8, dt_qint8>::TernaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_qint8& src0, const dt_qint8& src1,
                    const dt_qint8& src2, dt_qint8* dst) const {
        *dst = operator()(src0, src1, src2);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1,
                        const dt_qint8& src2) const {
        float fsrc0 = src0.as_int8() * this->scale_src0;
        float fsrc1 = src1.as_int8() * this->scale_src1;
        float fsrc2 = src2.as_int8() * this->scale_src2;
        float fdst = op(fsrc0, fsrc1, fsrc2);
        fdst = fdst * this->scale_dst;
        return QConverter::convert<dt_qint8, float>(fdst);
    }

    void operator()(const int8x16x2_t& vsrc0, const int8x16x2_t& vsrc1,
                    const int8x16x2_t& vsrc2, dt_qint8* dst) const {
        OPERATOR_TERNARY_QINT8;
    }

    int8x8_t operator()(const int32x4x2_t& vsrc0,
                        const int32x4x2_t& vsrc1,
                        const int32x4x2_t& vsrc2) const {
        auto val0 = vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale_src0);
        auto val1 = vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale_src0);
        auto val2 = vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale_src1);
        auto val3 = vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale_src1);
        auto val4 = vmulq_f32(vcvtq_f32_s32(vsrc2.val[0]), this->vscale_src2);
        auto val5 = vmulq_f32(vcvtq_f32_s32(vsrc2.val[1]), this->vscale_src2);
        auto val = op({{val0, val1}}, {{val2, val3}}, {{val4, val5}});
        val.val[0] = vmulq_f32(val.val[0], this->vscale_dst);
        val.val[1] = vmulq_f32(val.val[1], this->vscale_dst);
        return QConverter::convert<int8x8_t, float32x4x2_t>(val);
    }
};

template <typename Op>
struct TernaryQuantizationOp<dt_quint8, dt_quint8, Op>
        : TernaryOpBase<dt_quint8, dt_quint8> {
    using TernaryOpBase<dt_quint8, dt_quint8>::TernaryOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    Op op;

    void operator()(const dt_quint8& src0, const dt_quint8& src1,
                    const dt_quint8& src2, dt_quint8* dst) const {
        *dst = operator()(src0, src1, src2);
    }

    dt_quint8 operator()(const dt_quint8& src0, const dt_quint8& src1,
                         const dt_quint8& src2) const {
        float fsrc0 = src0.as_uint8() * this->scale_src0 - this->scale_zp0;
        float fsrc1 = src1.as_uint8() * this->scale_src1 - this->scale_zp1;
        float fsrc2 = src2.as_uint8() * this->scale_src2 - this->scale_zp2;
        float fdst = op(fsrc0, fsrc1, fsrc2);
        fdst = fdst * this->scale_dst;
        return QConverter::convert<dt_quint8, float, uint8_t>(fdst, this->dzp);
    }

    void operator()(const uint8x16x2_t& vsrc0, const uint8x16x2_t& vsrc1,
                    const uint8x16x2_t& vsrc2, dt_quint8* dst) const {
        OPERATOR_TERNARY_QUINT8;
    }

    uint8x8_t operator()(const uint32x4x2_t& vsrc0, const uint32x4x2_t& vsrc1,
                         const uint32x4x2_t& vsrc2) const {
        auto val0 = vmulq_f32(vcvtq_f32_u32(vsrc0.val[0]), this->vscale_src0);
        val0 = vsubq_f32(val0, this->vscale_zp0);
        auto val1 = vmulq_f32(vcvtq_f32_u32(vsrc0.val[1]), this->vscale_src0);
        val1 = vsubq_f32(val1, this->vscale_zp0);
        auto val2 = vmulq_f32(vcvtq_f32_u32(vsrc1.val[0]), this->vscale_src1);
        val2 = vsubq_f32(val2, this->vscale_zp1);
        auto val3 = vmulq_f32(vcvtq_f32_u32(vsrc1.val[1]), this->vscale_src1);
        val3 = vsubq_f32(val3, this->vscale_zp1);
        auto val4 = vmulq_f32(vcvtq_f32_u32(vsrc2.val[0]), this->vscale_src2);
        val4 = vsubq_f32(val4, this->vscale_zp2);
        auto val5 = vmulq_f32(vcvtq_f32_u32(vsrc2.val[1]), this->vscale_src2);
        val5 = vsubq_f32(val5, this->vscale_zp2);
        auto val = op({{val0, val1}}, {{val2, val3}}, {{val4, val5}});
        val.val[0] = vmulq_f32(val.val[0], this->vscale_dst);
        val.val[1] = vmulq_f32(val.val[1], this->vscale_dst);
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                val, this->vdzp);
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
