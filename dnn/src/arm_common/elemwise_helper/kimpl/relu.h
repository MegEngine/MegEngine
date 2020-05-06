/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/relu.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "src/arm_common/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct ReluOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        return src > 0 ? src : 0;
    }
};

template <typename src_ctype, typename dst_type = src_ctype>
struct ReluOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width) \
    template <>                                                        \
    struct ReluOp<_ctype> : ReluOpBase<_ctype> {                       \
        using ReluOpBase::ReluOpBase;                                  \
        using ReluOpBase::operator();                                  \
        constexpr static size_t SIMD_WIDTH = _simd_width;              \
        void operator()(const _neon_type2& src, _ctype* dst) const {   \
            auto vitem = operator()(src);                              \
            vst1q_##_func_suffix(dst, vitem.val[0]);                   \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);      \
        }                                                              \
        _neon_type2 operator()(const _neon_type2& src) const {         \
            auto vzero = vdupq_n_##_func_suffix(0);                    \
            auto vitem0 = vmaxq_##_func_suffix(src.val[0], vzero);     \
            auto vitem1 = vmaxq_##_func_suffix(src.val[1], vzero);     \
            return {{vitem0, vitem1}};                                 \
        }                                                              \
        void operator()(const _neon_type& src, _ctype* dst) const {    \
            auto vitem = operator()(src);                              \
            vst1q_##_func_suffix(dst, vitem);                          \
        }                                                              \
        _neon_type operator()(const _neon_type& src) const {           \
            auto vzero = vdupq_n_##_func_suffix(0);                    \
            return vmaxq_##_func_suffix(src, vzero);                   \
        }                                                              \
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
struct ReluOpBase<dt_qint8, dt_qint8> : UnaryOpBase<dt_qint8, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint8& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }
    dt_qint8 operator()(const dt_qint8& src) const {
        float fsrc = src.as_int8() * this->scale;
        fsrc = std::max<float>(fsrc, 0.f);
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
};

template <>
struct ReluOpBase<dt_quint8, dt_quint8> : UnaryOpBase<dt_quint8, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_quint8& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }
    dt_quint8 operator()(const dt_quint8& src) const {
        float fsrc = src.as_uint8() * this->scale - szp;
        fsrc = std::max<float>(fsrc, 0.f);
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }
};

template <>
struct ReluOp<dt_qint8, dt_qint8> : ReluOpBase<dt_qint8, dt_qint8> {
    using ReluOpBase::ReluOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using ReluOpBase::operator();

    void operator()(const int8x16x2_t& vsrc, dt_qint8* dst) const {
        OPERATOR_UNARY_QINT8;
    }
    int8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vzero = vdupq_n_f32(0.f);
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale);
        vitem0 = vmaxq_f32(vitem0, vzero);
        vitem1 = vmaxq_f32(vitem1, vzero);
        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
};

template <>
struct ReluOp<dt_quint8, dt_quint8> : ReluOpBase<dt_quint8, dt_quint8> {
    using ReluOpBase::ReluOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using ReluOpBase::operator();

    void operator()(const uint8x16x2_t& vsrc, dt_quint8* dst) const {
        OPERATOR_UNARY_QUINT8;
    }
    uint8x8_t operator()(const uint32x4x2_t& vsrc) const {
        auto vzero = vdupq_n_f32(0.f);
        auto vitem0 = vmulq_f32(vcvtq_f32_u32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_u32(vsrc.val[1]), this->vscale);
        vitem0 = vsubq_f32(vitem0, this->vszp);
        vitem1 = vsubq_f32(vitem1, this->vszp);
        vitem0 = vmaxq_f32(vitem0, vzero);
        vitem1 = vmaxq_f32(vitem1, vzero);
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                {{vitem0, vitem1}}, this->vdzp);
    }
};

template <>
struct ReluOpBase<dt_qint32, dt_qint8> : UnaryOpBase<dt_qint32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint32& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }

    dt_qint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale;
        fsrc = std::max<float>(fsrc, 0.f);
        return QConverter::convert<dt_qint8, float>(fsrc);
    }
};

template <>
struct ReluOpBase<dt_qint32, dt_quint8> : UnaryOpBase<dt_qint32, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint32& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }

    dt_quint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale;
        fsrc = std::max<float>(fsrc, 0.f);
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->zp);
    }
};

#if __ARM_ARCH >= 8
template <>
struct ReluOp<dt_qint32, dt_qint8> : ReluOpBase<dt_qint32, dt_qint8> {
    using ReluOpBase::ReluOpBase;
    using ReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc, dt_qint8* dst) const {
        vst1_s8(reinterpret_cast<int8_t*>(dst), operator()(vsrc));
    }
    void operator()(const int32x4_t& src, dt_qint8* dst) const {
        vst1_lane_s32(reinterpret_cast<int32_t*>(dst),
                      (int32x2_t)(operator()(src)), 0);
    }

    int8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale);
        vitem0 = vmaxq_f32(vitem0, QConverterBase::vfzero());
        vitem1 = vmaxq_f32(vitem1, QConverterBase::vfzero());

        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
    int8x8_t operator()(const int32x4_t& src) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(src), this->vscale);
        vitem0 = vmaxq_f32(vitem0, QConverterBase::vfzero());
        return QConverter::convert<int8x8_t, float32x4_t>(vitem0);
    }
    int8x8_t operator()(const float32x4_t& src) const {
        auto vitem0 = vmulq_f32(src, this->vscale);
        vitem0 = vmaxq_f32(vitem0, QConverterBase::vfzero());
        return QConverter::convert<int8x8_t, float32x4_t>(vitem0);
    }
};
#else
template <>
struct ReluOp<dt_qint32, dt_qint8> : ReluOpBase<dt_qint32, dt_qint8>,
                                     FixupBase {
    using ReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    ReluOp(DType src_dtype, DType dst_dtype)
            : ReluOpBase(src_dtype, dst_dtype), FixupBase(scale) {}

    ReluOp(float src_scale, float dst_scale)
            : ReluOpBase(src_scale, dst_scale), FixupBase(scale) {}

    void operator()(const int32x4x2_t& vsrc, dt_qint8* dst) const {
        vst1_s8(reinterpret_cast<int8_t*>(dst), operator()(vsrc));
    }

    int8x8_t operator()(const int32x4x2_t& vsrc) const {
        int32x4_t vitem0 = vqrdmulhq_s32(vsrc.val[0], vmultiplier);
        int32x4_t vitem1 = vqrdmulhq_s32(vsrc.val[1], vmultiplier);
        vitem0 = vmaxq_s32(vitem0, QConverterBase::vzero());
        vitem1 = vmaxq_s32(vitem1, QConverterBase::vzero());
        return vqmovn_s16(vcombine_s16(vqmovn_s32(vrshlq_s32(vitem0, vshift)),
                                       vqmovn_s32(vrshlq_s32(vitem1, vshift))));
    }
    int8x8_t operator()(const float32x4_t& vsrc) const {
        int32x4_t vitem0 = vqrdmulhq_s32(vcvtq_s32_f32(vsrc), vmultiplier);
        vitem0 = vmaxq_s32(vitem0, QConverterBase::vzero());
        vitem0 = vrshlq_s32(vitem0, vshift);
        int16x4_t vitem = vqmovn_s32(vitem0);
        return vqmovn_s16(vcombine_s16(vitem, vitem));
    }
    void operator()(const int32x4_t& src, dt_qint8* dst) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(src), this->vscale);
        vitem0 = vmaxq_f32(vitem0, QConverterBase::vfzero());
        auto result = QConverter::convert<int8x8_t, float32x4_t>(vitem0);
        vst1_lane_s32(reinterpret_cast<int32_t*>(dst), (int32x2_t)result, 0);
    }
    void operator()(const float32x4_t& src, dt_qint8* dst) const {
        auto vitem0 = vmulq_f32(src, this->vscale);
        vitem0 = vmaxq_f32(vitem0, QConverterBase::vfzero());
        auto result = QConverter::convert<int8x8_t, float32x4_t>(vitem0);
        vst1_lane_s32(reinterpret_cast<int32_t*>(dst), (int32x2_t)result, 0);
    }
};
#endif

template <>
struct ReluOp<dt_qint32, dt_quint8> : ReluOpBase<dt_qint32, dt_quint8> {
    using ReluOpBase::ReluOpBase;
    using ReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc, dt_quint8* dst) const {
        vst1_u8(reinterpret_cast<uint8_t*>(dst), operator()(vsrc));
    }

    uint8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale);
        vitem0 = vmaxq_f32(vitem0, QConverterBase::vfzero());
        vitem1 = vmaxq_f32(vitem1, QConverterBase::vfzero());

        return QConverter::convert<uint8x8_t, float32x4x2_t>({{vitem0, vitem1}},
                                                             this->vzp);
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
