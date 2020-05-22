/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/mul.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/arm_common/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct MulOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return src0 * src1;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct MulOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)    \
    template <>                                                           \
    struct MulOp<_ctype> : MulOpBase<_ctype> {                            \
        using MulOpBase::MulOpBase;                                       \
        using MulOpBase::operator();                                      \
        constexpr static size_t SIMD_WIDTH = _simd_width;                 \
        void operator()(const _neon_type2& src0, const _neon_type2& src1, \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem.val[0]);                      \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);         \
        }                                                                 \
        _neon_type2 operator()(const _neon_type2& src0,                   \
                               const _neon_type2& src1) const {           \
            auto vitem0 = vmulq_##_func_suffix(src0.val[0], src1.val[0]); \
            auto vitem1 = vmulq_##_func_suffix(src0.val[1], src1.val[1]); \
            return {{vitem0, vitem1}};                                    \
        }                                                                 \
        void operator()(const _neon_type& src0, const _neon_type& src1,   \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem);                             \
        }                                                                 \
        _neon_type operator()(const _neon_type& src0,                     \
                              const _neon_type& src1) const {             \
            return vmulq_##_func_suffix(src0, src1);                      \
        }                                                                 \
    };
OP(dt_float32, float32x4_t, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8_t, float16x8x2_t, f16, 8)
#endif
OP(dt_int32, int32x4_t, int32x4x2_t, s32, 4)
OP(dt_int16, int16x8_t, int16x8x2_t, s16, 8)
OP(dt_int8, int8x16_t, int8x16x2_t, s8, 16)
#undef OP

template <>
struct MulOpBase<dt_qint8, dt_qint8> : BinaryOpBase<dt_qint8, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint8& src0, const dt_qint8& src1,
                    dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1) const {
        return QConverter::convert<dt_qint8, float>(
                src0.as_int8() * scale_src0 * src1.as_int8() * scale1);
    }
};

template <>
struct MulOpBase<dt_quint8, dt_quint8> : BinaryOpBase<dt_quint8, dt_quint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_quint8& src0, const dt_quint8& src1,
                    dt_quint8* dst) const {
        *dst = operator()(src0, src1);
    }
    dt_quint8 operator()(const dt_quint8& src0, const dt_quint8& src1) const {
        float fsrc0 = src0.as_uint8() * scale_src0 - this->scale_zp0;
        float fsrc1 = src1.as_uint8() * scale_src1 - this->scale_zp1;
        return QConverter::convert<dt_quint8, float, uint8_t>(
                fsrc0 * fsrc1 * scale_dst, this->dzp);
    }
};

template <>
struct MulOp<dt_qint8, dt_qint8> : MulOpBase<dt_qint8, dt_qint8> {
    using MulOpBase::MulOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using MulOpBase::operator();

    void operator()(const int8x16x2_t& vsrc0, const int8x16x2_t& vsrc1,
                    dt_qint8* dst) const {
        OPERATOR_BINARY_QINT8;
    }

    int8x8_t operator()(const int32x4x2_t& vsrc0,
                        const int32x4x2_t& vsrc1) const {
        auto vitem0 = vmulq_f32(
                vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale_src0),
                vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale1));
        auto vitem1 = vmulq_f32(
                vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale_src0),
                vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale1));

        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
};

template <>
struct MulOp<dt_quint8, dt_quint8> : MulOpBase<dt_quint8, dt_quint8> {
    using MulOpBase::MulOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using MulOpBase::operator();

    void operator()(const uint8x16x2_t& vsrc0, const uint8x16x2_t& vsrc1,
                    dt_quint8* dst) const {
        OPERATOR_BINARY_QUINT8;
    }
    uint8x8_t operator()(const uint32x4x2_t& vsrc0,
                         const uint32x4x2_t vsrc1) const {
        auto vfsrc0 = vsubq_f32(
                vmulq_f32(vcvtq_f32_u32(vsrc0.val[0]), this->vscale_src0),
                this->vscale_zp0);
        auto vfsrc1 = vsubq_f32(
                vmulq_f32(vcvtq_f32_u32(vsrc1.val[0]), this->vscale_src1),
                this->vscale_zp1);
        auto vitem0 = vmulq_f32(vmulq_f32(vfsrc0, vfsrc1), this->vscale_dst);
        vfsrc0 = vsubq_f32(
                vmulq_f32(vcvtq_f32_u32(vsrc0.val[1]), this->vscale_src0),
                this->vscale_zp0);
        vfsrc1 = vsubq_f32(
                vmulq_f32(vcvtq_f32_u32(vsrc1.val[1]), this->vscale_src1),
                this->vscale_zp1);
        auto vitem1 = vmulq_f32(vmulq_f32(vfsrc0, vfsrc1), this->vscale_dst);
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                {{vitem0, vitem1}}, this->vdzp);
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
