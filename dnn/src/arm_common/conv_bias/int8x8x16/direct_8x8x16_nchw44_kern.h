/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/direct_nchw44_kern.h
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
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace int8x8x16_direct_nchw44 {

/**
origin src shape <n, ic/4, h, w, 4>
packed src shape <n, ic/4, h, w, 16>
example: (format like <ic>)
origin
<0>  <1>  <2>  <3>
packed
low 64 bit  <0> <0> <0> <0> | <1> <1> <1> <1>
---------------------------------------------------------------------
high 64 bit <2> <2> <2> <2> | <3> <3> <3> <3>
**/
static inline void nchw44_pack_src(const int8_t* src, int8_t* dst, int length) {
    static const uint8_t src_idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                               2, 2, 2, 2, 3, 3, 3, 3};
    constexpr int pack_ic = 4;
    constexpr int simd_len = 16;
    uint8x16_t src_idx = vld1q_u8(src_idx_buffer);
    for (int i = 0; i < length; i++) {
        int8x16_t result = vld_dup_tbl_s32(src + i * pack_ic, src_idx);
        vst1q_s8(dst + i * simd_len, result);
    }
}

template <BiasMode bias_mode, int filter_size, int stride>
struct ConvDirectInt8Nchw44Choose {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int16_t* bias, int16_t* dst, const size_t oc,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t oh, const size_t ow);
};

}  // namespace int8_direct_nchw44
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
