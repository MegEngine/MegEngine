/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/add.h
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
struct AddOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }

    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        return src0 + src1;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype,
          bool enable_opt_or_fixup = false>
struct AddOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)    \
    template <>                                                           \
    struct AddOp<_ctype> : AddOpBase<_ctype> {                            \
        using AddOpBase::AddOpBase;                                       \
        using AddOpBase::operator();                                      \
        constexpr static size_t SIMD_WIDTH = _simd_width;                 \
        void operator()(const _neon_type2& src0, const _neon_type2& src1, \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem.val[0]);                      \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);         \
        }                                                                 \
        _neon_type2 operator()(const _neon_type2& src0,                   \
                               const _neon_type2& src1) const {           \
            auto vitem0 = vaddq_##_func_suffix(src0.val[0], src1.val[0]); \
            auto vitem1 = vaddq_##_func_suffix(src0.val[1], src1.val[1]); \
            return {{vitem0, vitem1}};                                    \
        }                                                                 \
        void operator()(const _neon_type& src0, const _neon_type& src1,   \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem);                             \
        }                                                                 \
        _neon_type operator()(const _neon_type& src0,                     \
                              const _neon_type& src1) const {             \
            return vaddq_##_func_suffix(src0, src1);                      \
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
struct AddOpBase<dt_qint8, dt_qint8> : BinaryOpBase<dt_qint8, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint8& src0, const dt_qint8& src1,
                    dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1) const {
        return QConverter::convert<dt_qint8, float>(
                src0.as_int8() * this->scale0 + src1.as_int8() * this->scale1);
    }
};

template <>
struct AddOpBase<dt_quint8, dt_quint8> : BinaryOpBase<dt_quint8, dt_quint8> {
    float szp;
    float32x4_t vszp;

    AddOpBase(DType src0_dtype, DType src1_dtype, DType dst_dtype)
            : BinaryOpBase(src0_dtype, src1_dtype, dst_dtype) {
        szp = this->szp0 + this->szp1;
        vszp = vdupq_n_f32(szp);
    }
    void operator()(const dt_quint8& src0, const dt_quint8& src1,
                    dt_quint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_quint8 operator()(const dt_quint8& src0, const dt_quint8& src1) const {
        return QConverter::convert<dt_quint8, float, uint8_t>(
                src0.as_uint8() * this->scale0 +
                        src1.as_uint8() * this->scale1 - this->szp,
                this->dzp);
    }
};

template <>
struct AddOp<dt_qint8, dt_qint8> : AddOpBase<dt_qint8, dt_qint8> {
    using AddOpBase::AddOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using AddOpBase::operator();

    void operator()(const int8x16x2_t& vsrc0, const int8x16x2_t& vsrc1,
                    dt_qint8* dst) const {
        OPERATOR_BINARY_QINT8;
    }

    int8x8_t operator()(const int32x4x2_t& vsrc0,
                        const int32x4x2_t& vsrc1) const {
        auto vitem0 = vaddq_f32(
                vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale0),
                vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale1));
        auto vitem1 = vaddq_f32(
                vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale0),
                vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale1));

        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
};

template <>
struct AddOp<dt_quint8, dt_quint8> : AddOpBase<dt_quint8, dt_quint8> {
    using AddOpBase::AddOpBase;
    constexpr static size_t SIMD_WIDTH = 16;
    using AddOpBase::operator();

    void operator()(const uint8x16x2_t& vsrc0, const uint8x16x2_t& vsrc1,
                    dt_quint8* dst) const {
        OPERATOR_BINARY_QUINT8;
    }

    uint8x8_t operator()(const uint32x4x2_t& vsrc0,
                         const uint32x4x2_t& vsrc1) const {
        auto vitem0 = vsubq_f32(
                vaddq_f32(
                        vmulq_f32(vcvtq_f32_u32(vsrc0.val[0]), this->vscale0),
                        vmulq_f32(vcvtq_f32_u32(vsrc1.val[0]), this->vscale1)),
                this->vszp);
        auto vitem1 = vsubq_f32(
                vaddq_f32(
                        vmulq_f32(vcvtq_f32_u32(vsrc0.val[1]), this->vscale0),
                        vmulq_f32(vcvtq_f32_u32(vsrc1.val[1]), this->vscale1)),
                this->vszp);

        return QConverter::convert<uint8x8_t, float32x4x2_t, int32x4_t>(
                {{vitem0, vitem1}}, this->vdzp);
    }
};

template <>
struct AddOpBase<dt_qint32, dt_qint8> : BinaryOpBase<dt_qint32, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1,
                    dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        return QConverter::convert<dt_qint8, float>(
                src0.as_int32() * this->scale0 +
                src1.as_int32() * this->scale1);
    }
};

template <>
struct AddOpBase<dt_qint32, dt_quint8> : BinaryOpBase<dt_qint32, dt_quint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1,
                    dt_quint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_quint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        return QConverter::convert<dt_quint8, float>(
                src0.as_int32() * this->scale0 + src1.as_int32() * this->scale1,
                zp);
    }
};

#if MEGDNN_AARCH64
template <bool enable_opt_or_fixup>
struct AddOp<dt_qint32, dt_qint8, enable_opt_or_fixup>
        : AddOpBase<dt_qint32, dt_qint8> {
    using AddOpBase::AddOpBase;
    using AddOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc0, const int32x4x2_t& vsrc1,
                    dt_qint8* dst) const {
        vst1_s8(reinterpret_cast<int8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    int8x8_t operator()(const int32x4x2_t& vsrc0,
                        const int32x4x2_t& vsrc1) const {
        if (enable_opt_or_fixup) {
            auto vitem0 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[0], vsrc1.val[0])),
                    this->vscale0);
            auto vitem1 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[1], vsrc1.val[1])),
                    this->vscale0);
            return QConverter::convert<int8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}});
        } else {
            auto vitem0 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale1));
            auto vitem1 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale1));
            return QConverter::convert<int8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}});
        }
    }
};
#else
template <bool enable_opt_or_fixup>
struct AddOp<dt_qint32, dt_qint8, enable_opt_or_fixup>
        : AddOpBase<dt_qint32, dt_qint8>, FixupBase {
    using AddOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    AddOp(DType src0_dtype, DType src1_dtype, DType dst_dtype)
            : AddOpBase(src0_dtype, src1_dtype, dst_dtype), FixupBase(scale0) {}

    AddOp(float src0_scale, float src1_scale, float dst_scale)
            : AddOpBase(src0_scale, src1_scale, dst_scale), FixupBase(scale0) {}

    void operator()(const int32x4x2_t& vsrc0, const int32x4x2_t& vsrc1,
                    dt_qint8* dst) const {
        vst1_s8(reinterpret_cast<int8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    int8x8_t operator()(const int32x4x2_t& vsrc0,
                        const int32x4x2_t& vsrc1) const {
        if (enable_opt_or_fixup) {
            auto vitem0 = vqrdmulhq_s32(vaddq_s32(vsrc0.val[0], vsrc1.val[0]),
                                        vmultiplier);
            auto vitem1 = vqrdmulhq_s32(vaddq_s32(vsrc0.val[1], vsrc1.val[1]),
                                        vmultiplier);
            // FIXME Theoretically, we should check shift != 0 here.
            auto fixup0 = vshrq_n_s32(vitem0, 31);
            auto fixup1 = vshrq_n_s32(vitem1, 31);
            vitem0 = vqaddq_s32(vitem0, fixup0);
            vitem1 = vqaddq_s32(vitem1, fixup1);
            return vqmovn_s16(
                    vcombine_s16(vqmovn_s32(vrshlq_s32(vitem0, vshift)),
                                 vqmovn_s32(vrshlq_s32(vitem1, vshift))));
        } else {
            auto vitem0 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale1));
            auto vitem1 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale1));
            return QConverter::convert<int8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}});
        }
    }
};
#endif

template <bool enable_opt_or_fixup>
struct AddOp<dt_qint32, dt_quint8, enable_opt_or_fixup>
        : AddOpBase<dt_qint32, dt_quint8> {
    using AddOpBase::AddOpBase;
    constexpr static size_t SIMD_WIDTH = 4;
    using AddOpBase::operator();

    void operator()(const int32x4x2_t& vsrc0, const int32x4x2_t& vsrc1,
                    dt_quint8* dst) const {
        vst1_u8(reinterpret_cast<uint8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    uint8x8_t operator()(const int32x4x2_t& vsrc0,
                         const int32x4x2_t& vsrc1) const {
        if (enable_opt_or_fixup) {
            auto vitem0 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[0], vsrc1.val[0])),
                    this->vscale0);
            auto vitem1 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[1], vsrc1.val[1])),
                    this->vscale0);
            return QConverter::convert<uint8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}}, this->vzp);
        } else {
            auto vitem0 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale1));
            auto vitem1 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale1));
            return QConverter::convert<uint8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}}, this->vzp);
        }
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
