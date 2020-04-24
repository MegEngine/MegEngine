/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/fuse_add_relu.h
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
#include "src/arm_common/elemwise/neon_util_impl_helper.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddReluOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        auto tmp = src0 + src1;
        return tmp > 0 ? tmp : 0;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype,
          bool enable_opt_or_fixup = false>
struct FuseAddReluOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)      \
    template <>                                                             \
    struct FuseAddReluOp<_ctype> : FuseAddReluOpBase<_ctype> {              \
        using FuseAddReluOpBase::FuseAddReluOpBase;                         \
        using FuseAddReluOpBase::operator();                                \
        constexpr static size_t SIMD_WIDTH = _simd_width;                   \
        void operator()(const _neon_type2& src0, const _neon_type2& src1,   \
                        dst_ctype* dst) const {                             \
            auto vitem = operator()(src0, src1);                            \
            vst1q_##_func_suffix(dst, vitem.val[0]);                        \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);           \
        }                                                                   \
        _neon_type2 operator()(const _neon_type2& src0,                     \
                               const _neon_type2& src1) const {             \
            auto val1 = src0.val[0];                                        \
            auto val2 = src0.val[1];                                        \
            auto val3 = src1.val[0];                                        \
            auto val4 = src1.val[1];                                        \
            FUSE_ADD_RELU_NEON_PACK2(val1, val2, val3, val4, _func_suffix); \
            return {{val1, val2}};                                          \
        }                                                                   \
        void operator()(const _neon_type& src0, const _neon_type& src1,     \
                        dst_ctype* dst) const {                             \
            auto vitem = operator()(src0, src1);                            \
            vst1q_##_func_suffix(dst, vitem);                               \
        }                                                                   \
        _neon_type operator()(const _neon_type& src0,                       \
                              const _neon_type& src1) const {               \
            auto val1 = src0;                                               \
            auto val2 = src1;                                               \
            FUSE_ADD_RELU_NEON_PACK(val1, val2, _func_suffix);              \
            return val1;                                                    \
        }                                                                   \
    };
OP(dt_float32, float32x4_t, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8_t, float16x8x2_t, f16, 8)
#endif
OP(dt_int32, int32x4_t, int32x4x2_t, s32, 4)
OP(dt_int16, int16x8_t, int16x8x2_t, s16, 8)
OP(dt_int8, int8x16_t, int8x16x2_t, s8, 16)
#undef OP

template <typename ctype>
struct FuseAddReluOpCommon;

template <>
struct FuseAddReluOpCommon<float> {
    inline static float32x4_t vzero() { return vdupq_n_f32(0); }
};

template <>
struct FuseAddReluOpCommon<int> {
    inline static int32x4_t vzero() { return vdupq_n_s32(0); }
};

template <>
struct FuseAddReluOpBase<dt_qint8, dt_qint8>
        : BinaryOpBase<dt_qint8, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint8& src0, const dt_qint8& src1,
                    dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint8& src0, const dt_qint8& src1) const {
        return QConverter::convert<dt_qint8, float>(std::max<float>(
                src0.as_int8() * this->scale0 + src1.as_int8() * this->scale1,
                0.f));
    }
};

template <>
struct FuseAddReluOpBase<dt_quint8, dt_quint8>
        : BinaryOpBase<dt_quint8, dt_quint8> {
    float szp;
    float32x4_t vszp;

    FuseAddReluOpBase(DType src0_dtype, DType src1_dtype, DType dst_dtype)
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
                std::max<float>(src0.as_uint8() * this->scale0 +
                                        src1.as_uint8() * this->scale1 -
                                        this->szp,
                                0.f),
                this->dzp);
    }
};

template <>
struct FuseAddReluOp<dt_qint8, dt_qint8>
        : FuseAddReluOpBase<dt_qint8, dt_qint8>, FuseAddReluOpCommon<float> {
    using FuseAddReluOpBase::FuseAddReluOpBase;
    using FuseAddReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 16;

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

        vitem0 = vmaxq_f32(vitem0, this->vzero());
        vitem1 = vmaxq_f32(vitem1, this->vzero());
        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
};

template <>
struct FuseAddReluOp<dt_quint8, dt_quint8>
        : FuseAddReluOpBase<dt_quint8, dt_quint8>, FuseAddReluOpCommon<float> {
    using FuseAddReluOpBase::FuseAddReluOpBase;
    using FuseAddReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 16;

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

        vitem0 = vmaxq_f32(vitem0, this->vzero());
        vitem1 = vmaxq_f32(vitem1, this->vzero());
        return QConverter::convert<uint8x8_t, float32x4x2_t>({{vitem0, vitem1}},
                                                             this->vdzp);
    }
};

template <>
struct FuseAddReluOpBase<dt_qint32, dt_qint8>
        : BinaryOpBase<dt_qint32, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1,
                    dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        return QConverter::convert<dt_qint8, float>(std::max<float>(
                src0.as_int32() * this->scale0 + src1.as_int32() * this->scale1,
                0.f));
    }
};

template <>
struct FuseAddReluOpBase<dt_qint32, dt_quint8>
        : BinaryOpBase<dt_qint32, dt_quint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1,
                    dt_quint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_quint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        return QConverter::convert<dt_quint8, float>(
                std::max<float>(src0.as_int32() * this->scale0 +
                                        src1.as_int32() * this->scale1,
                                0.f),
                zp);
    }
};

#if MEGDNN_AARCH64
template <bool enable_opt_or_fixup>
struct FuseAddReluOp<dt_qint32, dt_qint8, enable_opt_or_fixup>
        : FuseAddReluOpBase<dt_qint32, dt_qint8>, FuseAddReluOpCommon<float> {
    using FuseAddReluOpBase::FuseAddReluOpBase;
    using FuseAddReluOpBase::operator();
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
            vitem0 = vmaxq_f32(vitem0, this->vzero());
            vitem1 = vmaxq_f32(vitem1, this->vzero());
            return QConverter::convert<int8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}});

        } else {
            auto vitem0 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale1));
            auto vitem1 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale1));

            vitem0 = vmaxq_f32(vitem0, this->vzero());
            vitem1 = vmaxq_f32(vitem1, this->vzero());
            return QConverter::convert<int8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}});
        }
    }
};
#else
template <bool enable_opt_or_fixup>
struct FuseAddReluOp<dt_qint32, dt_qint8, enable_opt_or_fixup>
        : FuseAddReluOpBase<dt_qint32, dt_qint8>,
          FuseAddReluOpCommon<float>,
          FixupBase {
    using FuseAddReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;
    FuseAddReluOp(DType src0_dtype, DType src1_dtype, DType dst_dtype)
            : FuseAddReluOpBase(src0_dtype, src1_dtype, dst_dtype),
              FixupBase(scale0) {}

    FuseAddReluOp(float src0_scale, float src1_scale, float dst_scale)
            : FuseAddReluOpBase(src0_scale, src1_scale, dst_scale),
              FixupBase(scale0) {}

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
            vitem0 = vmaxq_s32(vitem0, FuseAddReluOpCommon<int>::vzero());
            vitem1 = vmaxq_s32(vitem1, FuseAddReluOpCommon<int>::vzero());
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

            vitem0 = vmaxq_f32(vitem0, this->vzero());
            vitem1 = vmaxq_f32(vitem1, this->vzero());
            return QConverter::convert<int8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}});
        }
    }
};
#endif

template <bool enable_opt_or_fixup>
struct FuseAddReluOp<dt_qint32, dt_quint8, enable_opt_or_fixup>
        : FuseAddReluOpBase<dt_qint32, dt_quint8>, FuseAddReluOpCommon<float> {
    using FuseAddReluOpBase::FuseAddReluOpBase;
    using FuseAddReluOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;
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
            vitem0 = vmaxq_f32(vitem0, this->vzero());
            vitem1 = vmaxq_f32(vitem1, this->vzero());
            return QConverter::convert<uint8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}}, this->vzp);

        } else {
            auto vitem0 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale1));
            auto vitem1 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale1));

            vitem0 = vmaxq_f32(vitem0, this->vzero());
            vitem1 = vmaxq_f32(vitem1, this->vzero());

            return QConverter::convert<uint8x8_t, float32x4x2_t>(
                    {{vitem0, vitem1}}, this->vzp);
        }
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
