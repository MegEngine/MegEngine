/**
 * \file dnn/src/arm_common/quantized_converter.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

namespace megdnn {
namespace arm_common {

struct QConverterBase {
    inline static int32x4_t vzero() { return vdupq_n_s32(0); }

    inline static float32x4_t vfzero() { return vdupq_n_f32(0.f); }

    inline static float32x4_t vfhalf() { return vdupq_n_f32(0.5f); }

    inline static float32x4_t vfneg_half() { return vdupq_n_f32(-0.5f); }
};

struct QConverter {
    template <typename dst_type, typename... src_type>
    static inline dst_type convert(const src_type&... src);
};

template <>
inline dt_qint8 QConverter::convert(const float& src) {
    return dt_qint8(saturate<int8_t, float>(std::round(src), -128, 127));
}

template <>
inline dt_quint8 QConverter::convert(const float& src, const uint8_t& zp) {
    return dt_quint8(saturate<uint8_t, float>(std::round(src) + zp, 0, 255));
}

template <>
inline dt_qint32 QConverter::convert(const float& src) {
    return dt_qint32(
            saturate<int32_t, float>(std::round(src), -2147483648, 2147483647));
}

template <>
inline float32x4x2_t QConverter::convert(const int16x8_t& vsrc) {
    int32x4_t vhi = vmovl_s16(vget_high_s16(vsrc));
    int32x4_t vlo = vmovl_s16(vget_low_s16(vsrc));
    return {{vcvtq_f32_s32(vlo), vcvtq_f32_s32(vhi)}};
}

template <>
inline float32x4x2_t QConverter::convert(const uint16x8_t& vsrc) {
    uint32x4_t vhi = vmovl_u16(vget_high_u16(vsrc));
    uint32x4_t vlo = vmovl_u16(vget_low_u16(vsrc));
    return {{vcvtq_f32_u32(vlo), vcvtq_f32_u32(vhi)}};
}

#if __ARM_ARCH >= 8
template <>
inline int8x8_t QConverter::convert(const float32x4x2_t& vsrc) {
    int32x4_t vres0 = vcvtaq_s32_f32(vsrc.val[0]);
    int32x4_t vres1 = vcvtaq_s32_f32(vsrc.val[1]);
    return vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
}
template <>
inline int8x8_t QConverter::convert(const float32x4_t& src) {
    int32x4_t res0 = vcvtaq_s32_f32(src);
    int16x4_t res0_int16 = vqmovn_s32(res0);
    return vqmovn_s16(vcombine_s16(res0_int16, res0_int16));
}

template <>
inline uint8x8_t QConverter::convert(const float32x4x2_t& vsrc, const int32x4_t& vzp) {
    int32x4_t vres0 = vcvtaq_s32_f32(vsrc.val[0]);
    int32x4_t vres1 = vcvtaq_s32_f32(vsrc.val[1]);
    vres0 = vqaddq_s32(vres0, vzp);
    vres1 = vqaddq_s32(vres1, vzp);
    vres0 = vmaxq_s32(vres0, QConverterBase::vzero());
    vres1 = vmaxq_s32(vres1, QConverterBase::vzero());

    return vqmovn_u16(
            vreinterpretq_u16_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1))));
}

template <>
inline int32x4_t QConverter::convert(const float32x4_t& vsrc) {
    return vcvtaq_s32_f32(vsrc);
}

#else
template <>
inline int8x8_t QConverter::convert(const float32x4x2_t& vsrc) {
    float32x4_t vinc0 = vbslq_f32(
            vcgeq_f32(vsrc.val[0], QConverterBase::vfzero()), QConverterBase::vfhalf(),
            QConverterBase::vfneg_half());
    float32x4_t vinc1 = vbslq_f32(
            vcgeq_f32(vsrc.val[1], QConverterBase::vfzero()), QConverterBase::vfhalf(),
            QConverterBase::vfneg_half());

    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(vsrc.val[0], vinc0));
    int32x4_t vres1 = vcvtq_s32_f32(vaddq_f32(vsrc.val[1], vinc1));

    return vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
}

template <>
inline int8x8_t QConverter::convert(const float32x4_t& src) {
    float32x4_t vinc0 = vbslq_f32(
            vcgeq_f32(src, QConverterBase::vfzero()), QConverterBase::vfhalf(),
            QConverterBase::vfneg_half());

    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(src, vinc0));
    int16x4_t vres0_int16 = vqmovn_s32(vres0);
    return vqmovn_s16(vcombine_s16(vres0_int16, vres0_int16));
}

template <>
inline uint8x8_t QConverter::convert(const float32x4x2_t& vsrc, const int32x4_t& vzp) {
    float32x4_t vinc0 = vbslq_f32(
            vcgeq_f32(vsrc.val[0], QConverterBase::vfzero()), QConverterBase::vfhalf(),
            QConverterBase::vfneg_half());
    float32x4_t vinc1 = vbslq_f32(
            vcgeq_f32(vsrc.val[1], QConverterBase::vfzero()), QConverterBase::vfhalf(),
            QConverterBase::vfneg_half());

    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(vsrc.val[0], vinc0));
    int32x4_t vres1 = vcvtq_s32_f32(vaddq_f32(vsrc.val[1], vinc1));
    vres0 = vqaddq_s32(vres0, vzp);
    vres1 = vqaddq_s32(vres1, vzp);
    vres0 = vmaxq_s32(vres0, QConverterBase::vzero());
    vres1 = vmaxq_s32(vres1, QConverterBase::vzero());

    return vqmovn_u16(
            vreinterpretq_u16_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1))));
}

template <>
inline int32x4_t QConverter::convert(const float32x4_t& vsrc) {
    float32x4_t vinc = vbslq_f32(
            vcgeq_f32(vsrc, QConverterBase::vfzero()), QConverterBase::vfhalf(),
            QConverterBase::vfneg_half());
    return vcvtq_s32_f32(vaddq_f32(vsrc, vinc));
}

#endif

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
