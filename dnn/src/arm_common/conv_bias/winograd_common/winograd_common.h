/**
 * \file dnn/src/arm_common/conv_bias/winograd_common/winograd_common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

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
        return vget_low_s16(vmovl_s8(vld1_s8(ptr)));
    }
};

template <>
struct InputGetter<const uint8_t*, uint16x4_t> {
    uint16x4_t zp;
    InputGetter(uint8_t zero_point) {
        zp = vdup_n_u16(static_cast<uint16_t>(zero_point));
    }
    uint16x4_t operator()(const uint8_t* ptr) {
        return vget_low_u16(vmovl_u8(vld1_u8(ptr))) - zp;
    }
};

template <>
struct InputGetter<const int8_t*, float32x4_t> {
    float32x4_t operator()(const int8_t* ptr) {
        return vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(ptr)))));
    }
};
}  // namespace
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
