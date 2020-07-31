/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/direct_nchw_nchw44_kern.h
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
#include "megdnn/arch.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace i8i8i16_direct_nchw_nchw44 {
/**
 * @brief
 * stride2 from [oc / 4, fh, fw, ic, 4] to [oc / 8, ic, fh, fw, 8]
 * stride1 from [oc / 4, fh, fw, ic, 4] to [oc / 8, ic, fh, fw, 16]
 * @param in_ptr
 * @param dst_ptr
 * @param oc
 * @param kh
 * @param kw
 * @param ic
 */
template <int stride>
inline void pack_weight_int8_nchw_nchw44(const int8_t* in_ptr, int8_t* dst_ptr,
                                         const int oc, const int kh,
                                         const int kw, const int ic);
template <>
inline void pack_weight_int8_nchw_nchw44<2>(const int8_t* in_ptr,
                                            int8_t* dst_ptr, const int oc,
                                            const int kh, const int kw,
                                            const int ic) {
    constexpr int in_pack_oc = 4;
    constexpr int out_pack_oc = 8;
    constexpr int out_pair = 2;
    const int filter_size = kh * kw;
    const int in_oc_stride = filter_size * ic;
    const int oc_remain = oc % out_pack_oc;
    const int oc_end = oc - oc_remain;
    int32_t* pack_dst_ptr = (int32_t*)dst_ptr;
    for (int oc_idx = 0; oc_idx < oc_end; oc_idx += out_pack_oc) {
        const int32_t* in_oc0_ptr = (int32_t*)(in_ptr + oc_idx * in_oc_stride);
        const int32_t* in_oc1_ptr =
                (int32_t*)(in_ptr + (oc_idx + in_pack_oc) * in_oc_stride);

        for (int filter_idx = 0; filter_idx < filter_size; ++filter_idx) {
            for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                int32_t temp0 = *in_oc0_ptr++;
                int32_t temp1 = *in_oc1_ptr++;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             0] = temp0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             1] = temp1;
            }
        }
        pack_dst_ptr += ic * filter_size * out_pair;
    }
    if (oc_remain > 0) {
        const int32_t* in_oc0_ptr = (int32_t*)(in_ptr + oc_end * in_oc_stride);

        for (int filter_idx = 0; filter_idx < filter_size; ++filter_idx) {
            for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                int32_t temp0 = *in_oc0_ptr++;
                int32_t temp1 = 0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             0] = temp0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             1] = temp1;
            }
        }
    }
}
template <>
inline void pack_weight_int8_nchw_nchw44<1>(const int8_t* in_ptr,
                                            int8_t* dst_ptr, const int oc,
                                            const int kh, const int kw,
                                            const int ic) {
    constexpr int in_pack_oc = 4;
    constexpr int out_pack_oc = 8;
    constexpr int out_pair = 4;
    const int filter_size = kh * kw;
    const int in_oc_stride = filter_size * ic;
    const int oc_remain = oc % out_pack_oc;
    const int oc_end = oc - oc_remain;
    int32_t* pack_dst_ptr = (int32_t*)dst_ptr;
    for (int oc_idx = 0; oc_idx < oc_end; oc_idx += out_pack_oc) {
        const int32_t* in_oc0_ptr = (int32_t*)(in_ptr + oc_idx * in_oc_stride);
        const int32_t* in_oc1_ptr =
                (int32_t*)(in_ptr + (oc_idx + in_pack_oc) * in_oc_stride);

        for (int filter_idx = 0; filter_idx < filter_size; ++filter_idx) {
            for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                int32_t temp0 = *in_oc0_ptr++;
                int32_t temp1 = *in_oc1_ptr++;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             0] = temp0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             1] = temp1;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             2] = temp0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             3] = temp1;
            }
        }
        pack_dst_ptr += ic * filter_size * out_pair;
    }
    if (oc_remain > 0) {
        const int32_t* in_oc0_ptr = (int32_t*)(in_ptr + oc_end * in_oc_stride);

        for (int filter_idx = 0; filter_idx < filter_size; ++filter_idx) {
            for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                int32_t temp0 = *in_oc0_ptr++;
                int32_t temp1 = 0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             0] = temp0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             1] = temp1;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             2] = temp0;
                pack_dst_ptr[(ic_idx * filter_size + filter_idx) * out_pair +
                             3] = temp1;
            }
        }
    }
}

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void conv_direct_i8i8i16_nchw_nchw44(const int8_t* src, const int8_t* filter,
                                     const int16_t* bias, int8_t*, int16_t* dst,
                                     const int oc, const int ic, const int ih,
                                     const int iw, const int oh,
                                     const int oh_block, const int ow,
                                     const Op& op, const int, const int);

}  // namespace i8i8i16_direct_nchw_nchw44

}  // namespace arm_common
}  // namespace megdnn
// vim: syntax=cpp.doxygen
