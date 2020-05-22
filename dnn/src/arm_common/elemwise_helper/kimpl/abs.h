/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/abs.h
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
struct AbsOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        return src > 0 ? src : (-src);
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype>
struct AbsOp;

#define OP(_ctype, _neon_type, _func_suffix, _simd_width)           \
    template <>                                                     \
    struct AbsOp<_ctype> : AbsOpBase<_ctype> {                      \
        using AbsOpBase::AbsOpBase;                                 \
        using AbsOpBase::operator();                                \
        constexpr static size_t SIMD_WIDTH = _simd_width;           \
        void operator()(const _neon_type& src, _ctype* dst) const { \
            auto vitem = operator()(src);                           \
            vst1q_##_func_suffix(dst, vitem.val[0]);                \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);   \
        }                                                           \
        _neon_type operator()(const _neon_type& src) const {        \
            auto vitem0 = vabsq_##_func_suffix(src.val[0]);         \
            auto vitem1 = vabsq_##_func_suffix(src.val[1]);         \
            return {{vitem0, vitem1}};                              \
        }                                                           \
    };
OP(dt_float32, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8x2_t, f16, 8)
#endif
OP(dt_int32, int32x4x2_t, s32, 4)
OP(dt_int16, int16x8x2_t, s16, 8)
OP(dt_int8, int8x16x2_t, s8, 16)
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
struct AbsOpBase<dt_quint8, dt_quint8> : UnaryOpBase<dt_quint8, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_quint8& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }
    dt_quint8 operator()(const dt_quint8& src) const {
        float fsrc = src.as_uint8() * this->scale - this->szp;
        fsrc = fsrc > 0 ? fsrc : -fsrc;
        return QConverter::convert<dt_quint8, float, uint8_t>(fsrc, this->dzp);
    }
};

template <>
struct AbsOp<dt_qint8, dt_qint8> : AbsOpBase<dt_qint8, dt_qint8> {
    using AbsOpBase::AbsOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using AbsOpBase::operator();
    void operator()(const int8x16x2_t& vsrc, dt_qint8* dst) const {
        OPERATOR_UNARY_QINT8;
    }
    int8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale);
        vitem0 = vabsq_f32(vitem0);
        vitem1 = vabsq_f32(vitem1);
        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
};

template <>
struct AbsOp<dt_quint8, dt_quint8> : AbsOpBase<dt_quint8, dt_quint8> {
    using AbsOpBase::AbsOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using AbsOpBase::operator();
    void operator()(const uint8x16x2_t& vsrc, dt_quint8* dst) const {
        OPERATOR_UNARY_QUINT8;
    }
    uint8x8_t operator()(const uint32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_u32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_u32(vsrc.val[1]), this->vscale);
        vitem0 = vsubq_f32(vitem0, this->vszp);
        vitem1 = vsubq_f32(vitem1, this->vszp);
        vitem0 = vabsq_f32(vitem0);
        vitem1 = vabsq_f32(vitem1);
        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                {{vitem0, vitem1}}, this->vdzp);
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
