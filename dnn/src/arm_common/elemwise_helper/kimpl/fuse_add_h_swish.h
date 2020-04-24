/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/fuse_add_h_swish.h
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
#include "src/arm_common/elemwise_helper/kimpl/kern_macro_prologue.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct FuseAddHSwishOpBase : BinaryOpBase<src_ctype, dst_ctype> {
    using BinaryOpBase<src_ctype, dst_ctype>::BinaryOpBase;
    void operator()(const src_ctype& src0, const src_ctype& src1,
                    dst_ctype* dst) const {
        *dst = operator()(src0, src1);
    }
    dst_ctype operator()(const src_ctype& src0, const src_ctype& src1) const {
        float tmp = src0 + src1;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        return tmp;
    }
};

template <typename src_ctype, typename dst_ctype = src_ctype,
          bool enable_opt_or_fixup = false>
struct FuseAddHSwishOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)    \
    template <>                                                           \
    struct FuseAddHSwishOp<_ctype> : FuseAddHSwishOpBase<_ctype> {        \
        using FuseAddHSwishOpBase::FuseAddHSwishOpBase;                   \
        using FuseAddHSwishOpBase::operator();                            \
        constexpr static size_t SIMD_WIDTH = _simd_width;                 \
        void operator()(const _neon_type2& src0, const _neon_type2& src1, \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem.val[0]);                      \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);         \
        }                                                                 \
        _neon_type2 operator()(const _neon_type2& src0,                   \
                               const _neon_type2& src1) const {           \
            auto val1 = src0.val[0];                                      \
            auto val2 = src0.val[1];                                      \
            auto val3 = src1.val[0];                                      \
            auto val4 = src1.val[1];                                      \
            val1 = vaddq_##_func_suffix(val1, val3);                      \
            val2 = vaddq_##_func_suffix(val2, val4);                      \
            H_SWISH_KERN(_func_suffix, val1, val2);                       \
            return {{val1, val2}};                                        \
        }                                                                 \
        void operator()(const _neon_type& src0, const _neon_type& src1,   \
                        dst_ctype* dst) const {                           \
            auto vitem = operator()(src0, src1);                          \
            vst1q_##_func_suffix(dst, vitem);                             \
        }                                                                 \
        _neon_type operator()(const _neon_type& src0,                     \
                              const _neon_type& src1) const {             \
            auto val1 = src0;                                             \
            auto val2 = src1;                                             \
            val1 = vaddq_##_func_suffix(val1, val2);                      \
            H_SWISH_KERN_N1(_func_suffix, val1);                          \
            return val1;                                                  \
        }                                                                 \
    };
OP(dt_float32, float32x4_t, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8_t, float16x8x2_t, f16, 8)
#endif
#undef OP

template <>
struct FuseAddHSwishOpBase<dt_qint32, dt_qint8>
        : BinaryOpBase<dt_qint32, dt_qint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1,
                    dt_qint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_qint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        float tmp = src0.as_int32() * this->scale_src0 +
                    src1.as_int32() * this->scale_src1;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        tmp *= this->scale_dst;
        return QConverter::convert<dt_qint8, float>(tmp);
    }
};

template <>
struct FuseAddHSwishOpBase<dt_qint32, dt_quint8>
        : BinaryOpBase<dt_qint32, dt_quint8> {
    using BinaryOpBase::BinaryOpBase;
    void operator()(const dt_qint32& src0, const dt_qint32& src1,
                    dt_quint8* dst) const {
        *dst = operator()(src0, src1);
    }

    dt_quint8 operator()(const dt_qint32& src0, const dt_qint32& src1) const {
        float tmp = src0.as_int32() * this->scale_src0 +
                    src1.as_int32() * this->scale_src1;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        tmp *= this->scale_dst;
        return QConverter::convert<dt_quint8, float>(tmp, zp);
    }
};

template <bool enable_opt_or_fixup>
struct FuseAddHSwishOp<dt_qint32, dt_qint8, enable_opt_or_fixup>
        : FuseAddHSwishOpBase<dt_qint32, dt_qint8> {
    using FuseAddHSwishOpBase::FuseAddHSwishOpBase;
    using FuseAddHSwishOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;
    void operator()(const int32x4x2_t& vsrc0, const int32x4x2_t& vsrc1,
                    dt_qint8* dst) const {
        vst1_s8(reinterpret_cast<int8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    int8x8_t operator()(const int32x4x2_t& vsrc0,
                        const int32x4x2_t& vsrc1) const {
        float32x4_t vitem0, vitem1;
        if (enable_opt_or_fixup) {
            vitem0 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[0], vsrc1.val[0])),
                    this->vscale_src0);
            vitem1 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[1], vsrc1.val[1])),
                    this->vscale_src0);

        } else {
            vitem0 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale_src0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale_src1));
            vitem1 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale_src0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale_src1));
        }
        H_SWISH_KERN(f32, vitem0, vitem1);
        vitem0 = vmulq_f32(vitem0, this->vscale_dst);
        vitem1 = vmulq_f32(vitem1, this->vscale_dst);
        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
};

template <bool enable_opt_or_fixup>
struct FuseAddHSwishOp<dt_qint32, dt_quint8, enable_opt_or_fixup>
        : FuseAddHSwishOpBase<dt_qint32, dt_quint8> {
    using FuseAddHSwishOpBase::FuseAddHSwishOpBase;
    using FuseAddHSwishOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;
    void operator()(const int32x4x2_t& vsrc0, const int32x4x2_t& vsrc1,
                    dt_quint8* dst) const {
        vst1_u8(reinterpret_cast<uint8_t*>(dst), operator()(vsrc0, vsrc1));
    }

    uint8x8_t operator()(const int32x4x2_t& vsrc0,
                         const int32x4x2_t& vsrc1) const {
        float32x4_t vitem0, vitem1;
        if (enable_opt_or_fixup) {
            vitem0 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[0], vsrc1.val[0])),
                    this->vscale_src0);
            vitem1 = vmulq_f32(
                    vcvtq_f32_s32(vaddq_s32(vsrc0.val[1], vsrc1.val[1])),
                    this->vscale_src0);

        } else {
            vitem0 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[0]), this->vscale_src0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[0]), this->vscale_src1));
            vitem1 = vaddq_f32(
                    vmulq_f32(vcvtq_f32_s32(vsrc0.val[1]), this->vscale_src0),
                    vmulq_f32(vcvtq_f32_s32(vsrc1.val[1]), this->vscale_src1));
        }

        H_SWISH_KERN(f32, vitem0, vitem1);
        vitem0 = vmulq_f32(vitem0, this->vscale_dst);
        vitem1 = vmulq_f32(vitem1, this->vscale_dst);
        return QConverter::convert<uint8x8_t, float32x4x2_t>({{vitem0, vitem1}},
                                                             this->vzp);
    }
};

#include "src/arm_common/elemwise_helper/kimpl/kern_macro_epilogue.h"

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
