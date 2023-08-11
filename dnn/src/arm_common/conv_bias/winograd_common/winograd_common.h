#pragma once

#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace arm_common {
namespace {

template <typename ctype, typename otype>
struct InputGetter;

template <>
struct InputGetter<const int8_t*, int16x4_t> {
    int16x4_t operator()(const int8_t* ptr) {
        auto data = reinterpret_cast<const int32_t*>(ptr);
        int8x8_t v_int8 = vreinterpret_s8_s32(vld1_dup_s32(data));
        return vget_low_s16(vmovl_s8(v_int8));
    }
};

template <>
struct InputGetter<const uint8_t*, uint16x4_t> {
    uint16x4_t zp;
    InputGetter(uint8_t zero_point) {
        zp = vdup_n_u16(static_cast<uint16_t>(zero_point));
    }
    uint16x4_t operator()(const uint8_t* ptr) {
        const uint32_t* data = reinterpret_cast<const uint32_t*>(ptr);
        uint8x8_t v_uint8 = vreinterpret_u8_u32(vld1_dup_u32(data));
        return vget_low_u16(vmovl_u8(v_uint8)) - zp;
    }
};

template <>
struct InputGetter<const int8_t*, float32x4_t> {
    float32x4_t operator()(const int8_t* ptr) {
        const int32_t* data = reinterpret_cast<const int32_t*>(ptr);
        int8x8_t v_int8 = vreinterpret_s8_s32(vld1_dup_s32(data));
        return vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(v_int8))));
    }
};

}  // namespace
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
