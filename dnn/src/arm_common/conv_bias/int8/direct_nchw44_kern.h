/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_nchw44_kern.h
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
#include "src/arm_common/conv_bias/int8/direct.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace int8_direct_nchw44 {

/**
origin weight shape <oc/4, ic/4, fh, fw, 4, 4>
packed weight shape <oc/4, ic/4, fh, fw, 16>
example: (format like weight<oc, ic>)
origin
<0, 0>  <1, 0>  <2, 0>  <3, 0>
<0, 1>  <1, 1>  <2, 1>  <3, 1>
<0, 2>  <1, 2>  <2, 2>  <3, 2>
<0, 3>  <1, 3>  <2, 3>  <3, 3>
packed
low 64 bit  <0, 0> <0, 1> <1, 2> <1, 3> | <2, 0> <2, 1> <3, 2> <3, 3>
---------------------------------------------------------------------
high 64 bit <0, 3> <0, 2> <1, 1> <1, 0> | <2, 3> <2, 2> <3, 1> <3, 0>
**/
static inline void nchw44_pack_filter(const int8_t* src, int8_t* dst,
                                      int length) {
    static const uint8_t weight_idx_buffer[16] = {0,  4, 9, 13, 2,  6,  11, 15,
                                                  12, 8, 5, 1,  14, 10, 7,  3};
    constexpr int simd_len = 16;
    uint8x16_t weight_idx = vld1q_u8(weight_idx_buffer);
    for (int i = 0; i < length; i++) {
        int8x16_t result = vldq_tbl_s8(src + i * simd_len, weight_idx);
        vst1q_s8(dst + i * simd_len, result);
    }
}
/**
origin src shape <n, ic/4, h, w, 4>
packed src shape <n, ic/4, h, w, 16>
example: (format like <ic>)
origin
<0>  <0>  <0>  <0>
packed
low 64 bit  <0> <1> <2> <3> | <0> <1> <2> <3>
---------------------------------------------------------------------
high 64 bit <3> <2> <1> <0> | <3> <2> <1> <0>
**/
static inline void nchw44_pack_src(const int8_t* src, int8_t* dst, int length) {
    static const uint8_t src_idx_buffer[16] = {0, 1, 2, 3, 0, 1, 2, 3,
                                               3, 2, 1, 0, 3, 2, 1, 0};
    constexpr int pack_ic = 4;
    constexpr int simd_len = 16;
    uint8x16_t src_idx = vld1q_u8(src_idx_buffer);
    for (int i = 0; i < length; i++) {
        int8x16_t result = vld_dup_tbl_s32(src + i * pack_ic, src_idx);
        vst1q_s8(dst + i * simd_len, result);
    }
}

template <BiasMode bias_mode, typename Op, int filter_size, typename DstType,
          int stride>
struct ConvDirectInt8Nchw44Choose {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int32_t* bias, int32_t* temp, DstType* dst,
                     const size_t oc, const size_t ic, const size_t ih,
                     const size_t iw, const size_t oh, const size_t ow,
                     const Op& op);
};

template <BiasMode bias_mode, typename Op, int filter_size, typename DstType,
          int stride>
void conv_direct_int8_nchw44(const int8_t* src, const int8_t* filter,
                             const int32_t* bias, int32_t* temp, DstType* dst,
                             const size_t oc, const size_t ic, const size_t ih,
                             const size_t iw, const size_t oh, const size_t ow,
                             const Op& op) {
    ConvDirectInt8Nchw44Choose<bias_mode, Op, filter_size, DstType,
                               stride>::impl(src, filter, bias, temp, dst, oc,
                                             ic, ih, iw, oh, ow, op);
}

}  // namespace int8_direct_nchw44
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
