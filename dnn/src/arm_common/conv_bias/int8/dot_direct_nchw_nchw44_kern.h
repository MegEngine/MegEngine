/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/dot_direct_nchw_nchw44_kern.h
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
#if __ARM_FEATURE_DOTPROD

#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace dot_direct_nchw_nchw44 {
template <int src_idx, int weight_idx, int c_dim, typename Func, int ow_block,
          int stride, typename T, typename T2, typename T3, typename T4>
struct ShiftCalHelper {
    static void impl(T& c, T2& src, T3& weight);
};

template <int src_idx, int weight_idx, int c_dim, typename FUNC, int ow_block,
          int stride, typename T, typename T2, typename T3>
inline void cal_helper(T& c, T2& src, T3& weight) {
    ShiftCalHelper<src_idx, weight_idx, c_dim, FUNC, ow_block, stride, T, T2,
                   T3, int>::impl(c, src, weight);
};
//! OCHelper is used to trans oc_block to row number of result regs
template <int oc>
struct OCHelper {
public:
    static const int val = -1;
};

template <>
struct OCHelper<4> {
public:
    static const int val = 1;
};
#if MEGDNN_AARCH64
template <>
struct OCHelper<8> {
public:
    static const int val = 2;
};
#endif
/**
 *  oc8_ow8(m = 8, n = 8) and oc4_ow8(m = 4, n = 8) gemm like kernel
 * */
template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int oc_block, int ow_block, int stride>
struct KerNeonDotXXs2Nchw44Int8 {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op);
};

template <int stride>
void pack_src_int8_nchw_nchw44_dot(int8_t* sptr_base, const int8_t* sptr_origin,
                                   const int, const int pw, const int,
                                   const int ih, const int iw, const int iw2,
                                   const int pad_top, const int pad_bottom,
                                   const int ic, const int ic_stride, int8_t*);

static inline void pack_weight_int8_nchw_nchw44_dot(int8_t* dst_ptr,
                                                    const int8_t* src_ptr,
                                                    const int oc, const int ic,
                                                    const int fh, const int fw,
                                                    const int fw2) {
    constexpr int oc_step = 4;
    const int fw_remain = fw2 - fw;
    const int dst_ic_stride = fh * fw2;
    const int oc_step_stride = fh * fw2 * ic * oc_step;
    static const uint8_t transpose_4x4_idx[16] = {0, 4, 8,  12, 1, 5, 9,  13,
                                                  2, 6, 10, 14, 3, 7, 11, 15};
    uint8x16_t tbl_transpose_4x4 = vld1q_u8(&transpose_4x4_idx[0]);
    rep_step(oc_idx, oc, oc_step) {
        int32_t* dst_temp_ptr =
                reinterpret_cast<int32_t*>(dst_ptr + oc_idx * ic * fh * fw2);
        const int32_t* src_temp_ptr = reinterpret_cast<const int32_t*>(
                src_ptr + oc_idx * ic * fh * fw);
        // transpose ic and pad
        rep(fh_idx, fh) {
            rep(fw_idx, fw) {
                rep(ic_idx, ic) {
                    *(dst_temp_ptr + ic_idx * dst_ic_stride) = *src_temp_ptr;
                    src_temp_ptr++;
                }
                dst_temp_ptr++;
            }
            rep(ic_idx, ic) {
                memset(dst_temp_ptr + ic_idx * dst_ic_stride, 0,
                       sizeof(int8_t) * oc_step * fw_remain);
            }
            dst_temp_ptr += fw_remain;
        }
        // transpose fw oc
        int8_t* trans_dst_temp_ptr =
                reinterpret_cast<int8_t*>(dst_ptr + oc_idx * ic * fh * fw2);

        rep_step(idx, oc_step_stride, 16) {
            int8x16_t temp = vld1q_s8(trans_dst_temp_ptr + idx);
            vst1q_s8(trans_dst_temp_ptr + idx,
                     vqtbl1q_s8(temp, tbl_transpose_4x4));
        }
    }
}

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void conv_direct_int8_nchw_nchw44_dot(const int8_t* src, const int8_t* filter,
                                      const int32_t* bias, int32_t* temp,
                                      int8_t* dst, const int oc, const int ic,
                                      const int ih, const int iw, const int oh,
                                      const int oh_block, const int ow,
                                      const Op& op);

}  // namespace dot_direct_nchw_nchw44
}  // namespace arm_common
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
