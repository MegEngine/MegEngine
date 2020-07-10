/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_dotprod_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#ifdef __ARM_FEATURE_DOTPROD

#include "src/arm_common/elemwise_helper/kimpl/typecvt.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

#include "src/arm_common/conv_bias/int8/direct_dotprod_nchw44.h"

namespace megdnn {
namespace arm_common {
namespace direct_dotprod_nchw44 {

template <>
void copy_packed_src_int8_nchw44<1>(int8_t* dst, const int dst_step,
                                    const int8_t* src, const int src_step,
                                    const int ic, const int ic_step,
                                    const int ih, const int pad_left,
                                    const int pad_right, const int pad_top,
                                    const int pad_bottom) {
    MEGDNN_MARK_USED_VAR(pad_right);
    constexpr int IC_PACK_SIZE = 4;
    rep_step(ic_idx, ic, IC_PACK_SIZE) {
        const int8_t* i_src = src + ic_idx * ic_step;
        //! pad top
        int bytes_pad_top = pad_top * dst_step * IC_PACK_SIZE * sizeof(int8_t);
        memset(dst, 0, bytes_pad_top);
        dst += bytes_pad_top / sizeof(int8_t);
        rep(ih_idx, ih) {
            int bytes_row_in_dst = dst_step * IC_PACK_SIZE * sizeof(int8_t);
            memset(dst, 0, bytes_row_in_dst);

            //! left elements
            int pad_left_elements = pad_left * IC_PACK_SIZE;
            //! copy row [ih_idx, x]
            int bytes_copy = src_step * IC_PACK_SIZE * sizeof(int8_t);
            memcpy(dst + pad_left_elements, i_src, bytes_copy);

            //! dst move to next row
            dst += bytes_row_in_dst / sizeof(int8_t);
            //! src move to next row
            i_src += bytes_copy / sizeof(int8_t);
        }
        //! pad bottom
        int bytes_pad_bottom =
                pad_bottom * dst_step * IC_PACK_SIZE * sizeof(int8_t);
        memset(dst, 0, bytes_pad_bottom);
        dst += bytes_pad_bottom / sizeof(int8_t);
    }
}

template <>
void copy_packed_src_int8_nchw44<2>(int8_t* dst, const int dst_step,
                                    const int8_t* src, const int src_step,
                                    const int ic, const int ic_step,
                                    const int ih, const int pad_left,
                                    const int pad_right, const int pad_top,
                                    const int pad_bottom) {
    MEGDNN_MARK_USED_VAR(pad_right);
    constexpr int IC_PACK_SIZE = 4;
    int odd_start = megdnn::div_ceil(dst_step, 2);
    bool nochange = pad_left % 2 == 0;
    rep_step(ic_idx, ic, IC_PACK_SIZE) {
        const int32_t* i_src =
                reinterpret_cast<const int32_t*>(src + ic_idx * ic_step);
        int bytes_pad_top = pad_top * dst_step * IC_PACK_SIZE * sizeof(int8_t);
        memset(dst, 0, bytes_pad_top);
        dst += bytes_pad_top / sizeof(int8_t);
        rep(ih_idx, ih) {
            int bytes_row_in_dst = dst_step * IC_PACK_SIZE * sizeof(int8_t);
            memset(dst, 0, bytes_row_in_dst);

            int32_t* dst_even = reinterpret_cast<int32_t*>(dst) + pad_left / 2 +
                                pad_left % 2;
            int32_t* dst_odd =
                    reinterpret_cast<int32_t*>(dst) + odd_start + pad_left / 2;
            int i_src_idx = 0;

            if (nochange) {
                for (; i_src_idx + 7 < src_step; i_src_idx += 8) {
                    int32x4x2_t tmp;
                    tmp = vld2q_s32(i_src + i_src_idx);
                    vst1q_s32(dst_even, tmp.val[0]);
                    vst1q_s32(dst_odd, tmp.val[1]);
                    dst_even += 4;
                    dst_odd += 4;
                }
            } else {
                for (; i_src_idx + 7 < src_step; i_src_idx += 8) {
                    int32x4x2_t tmp;
                    tmp = vld2q_s32(i_src + i_src_idx);
                    vst1q_s32(dst_even, tmp.val[1]);
                    vst1q_s32(dst_odd, tmp.val[0]);
                    dst_even += 4;
                    dst_odd += 4;
                }
            }

            for (; i_src_idx < src_step; ++i_src_idx) {
                if (nochange) {
                    if (i_src_idx % 2 == 0) {
                        *dst_even = *(i_src + i_src_idx);
                        dst_even++;
                    } else {
                        *dst_odd = *(i_src + i_src_idx);
                        dst_odd++;
                    }
                } else {
                    if (i_src_idx % 2 == 0) {
                        *dst_odd = *(i_src + i_src_idx);
                        dst_odd++;
                    } else {
                        *dst_even = *(i_src + i_src_idx);
                        dst_even++;
                    }
                }
            }
            //! dst move to next row
            dst += bytes_row_in_dst / sizeof(int8_t);
            //! src move to next row
            i_src += src_step;
        }
        //! pad bottom
        int bytes_pad_bottom =
                pad_bottom * dst_step * IC_PACK_SIZE * sizeof(int8_t);
        memset(dst, 0, bytes_pad_bottom);
        dst += bytes_pad_bottom / sizeof(int8_t);
    }
}

}  // namespace direct_dotprod_nchw44
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
