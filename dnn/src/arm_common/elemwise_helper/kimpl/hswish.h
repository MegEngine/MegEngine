/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/hswish.h
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

#include "src/arm_common/elemwise_helper/kimpl/kern_macro_prologue.h"
#include "src/arm_common/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct HSwishOpBase : UnaryOpBase<src_ctype, dst_ctype> {
    using UnaryOpBase<src_ctype, dst_ctype>::UnaryOpBase;
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dst_ctype operator()(const src_ctype& src) const {
        float tmp = src;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        return (tmp);
    }
};

//! h_swish(x) = x * clip(x + 3, 0, 6) / 6
template <typename src_ctype, typename dst_ctype = src_ctype>
struct HSwishOp;

#define OP(_ctype, _neon_type, _neon_type2, _func_suffix, _simd_width)         \
    template <>                                                                \
    struct HSwishOp<_ctype> : HSwishOpBase<_ctype> {                           \
        using HSwishOpBase::HSwishOpBase;                                      \
        using HSwishOpBase::operator();                                        \
        constexpr static size_t SIMD_WIDTH = _simd_width;                      \
        void operator()(const _neon_type2& src, _ctype* dst) const {           \
            auto vitem = operator()(src);                                      \
            vst1q_##_func_suffix(dst, vitem.val[0]);                           \
            vst1q_##_func_suffix(dst + SIMD_WIDTH, vitem.val[1]);              \
        }                                                                      \
        void operator()(const _neon_type& src, _ctype* dst) const {            \
            auto vitem = operator()(src);                                      \
            vst1q_##_func_suffix(dst, vitem);                                  \
        }                                                                      \
        _neon_type2 operator()(const _neon_type2& src) const {                 \
            auto val1 = src.val[0];                                            \
            auto val2 = src.val[1];                                            \
            H_SWISH_KERN(_func_suffix, val1, val2);                            \
            return {{val1, val2}};                                             \
        }                                                                      \
        _neon_type operator()(const _neon_type& src) const {                   \
            auto val_zero = vdupq_n_##_func_suffix(0.f);                       \
            auto val_six = vdupq_n_##_func_suffix(6.f);                        \
            auto val_three = vdupq_n_##_func_suffix(3.f);                      \
            auto val_rec_six = vdupq_n_##_func_suffix(1.f / 6.f);              \
            auto clip1 = vmaxq_##_func_suffix(                                 \
                    vminq_##_func_suffix(vaddq_##_func_suffix(src, val_three), \
                                         val_six),                             \
                    val_zero);                                                 \
            return vmulq_##_func_suffix(vmulq_##_func_suffix(src, clip1),      \
                                        val_rec_six);                          \
        }                                                                      \
    };

OP(dt_float32, float32x4_t, float32x4x2_t, f32, 4)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
OP(__fp16, float16x8_t, float16x8x2_t, f16, 8)
#endif
#undef OP

template <>
struct HSwishOpBase<dt_qint32, dt_qint8> : UnaryOpBase<dt_qint32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint32& src, dt_qint8* dst) const {
        *dst = operator()(src);
    }

    dt_qint8 operator()(const dt_qint32& src) const {
        float tmp = src.as_int32() * this->scale_src;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        tmp *= this->scale_dst;
        return QConverter::convert<dt_qint8, float>(tmp);
    }
};

template <>
struct HSwishOpBase<dt_qint32, dt_quint8> : UnaryOpBase<dt_qint32, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    void operator()(const dt_qint32& src, dt_quint8* dst) const {
        *dst = operator()(src);
    }

    dt_quint8 operator()(const dt_qint32& src) const {
        float tmp = src.as_int32() * this->scale_src;
        tmp = tmp * std::max(std::min(tmp + 3.f, 6.f), 0.f) / 6.f;
        tmp *= this->scale_dst;
        return QConverter::convert<dt_quint8, float>(tmp, zp);
    }
};

template <>
struct HSwishOp<dt_qint32, dt_qint8> : HSwishOpBase<dt_qint32, dt_qint8> {
    using HSwishOpBase::HSwishOpBase;
    using HSwishOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc, dt_qint8* dst) const {
        vst1_s8(reinterpret_cast<int8_t*>(dst), operator()(vsrc));
    }
    void operator()(const int32x4_t& vsrc, dt_qint8* dst) const {
        vst1_lane_s32(reinterpret_cast<int32_t*>(dst),
                      (int32x2_t)(operator()(vsrc)), 0);
    }

    int8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale_src);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale_src);

        H_SWISH_KERN(f32, vitem0, vitem1);
        vitem0 = vmulq_f32(vitem0, this->vscale_dst);
        vitem1 = vmulq_f32(vitem1, this->vscale_dst);

        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
    int8x8_t operator()(const int32x4_t& src) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(src), this->vscale_src);

        H_SWISH_KERN_N1(f32, vitem0);
        vitem0 = vmulq_f32(vitem0, this->vscale_dst);

        return QConverter::convert<int8x8_t, float32x4_t>(vitem0);
    }
};

template <>
struct HSwishOp<dt_qint32, dt_quint8> : HSwishOpBase<dt_qint32, dt_quint8> {
    using HSwishOpBase::HSwishOpBase;
    using HSwishOpBase::operator();
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc, dt_quint8* dst) const {
        vst1_u8(reinterpret_cast<uint8_t*>(dst), operator()(vsrc));
    }

    uint8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale_src);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale_src);
        H_SWISH_KERN(f32, vitem0, vitem1);
        vitem0 = vmulq_f32(vitem0, this->vscale_dst);
        vitem1 = vmulq_f32(vitem1, this->vscale_dst);

        return QConverter::convert<uint8x8_t, float32x4x2_t>({{vitem0, vitem1}},
                                                             this->vzp);
    }
};

}  // namespace arm_common
}  // namespace megdnn

#include "src/arm_common/elemwise_helper/kimpl/kern_macro_epilogue.h"
// vim: syntax=cpp.doxygen
