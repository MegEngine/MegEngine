/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_dotprod_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifdef __ARM_FEATURE_DOTPROD

#include "src/arm_common/elemwise_helper/kimpl/typecvt.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

#include "src/arm_common/conv_bias/int8/direct_dotprod_nchw44.h"
#include "src/arm_common/conv_bias/int8/direct_dotprod_nchw44_kern.h"
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

template <typename dst_type, int stride, BiasMode bias_mode, typename Op,
          int filter_size>
void conv_direct_sdot_int8_nchw44(dst_type* dst, const int oh, const int ow,
                                  const int8_t* src, const int ih, const int iw,
                                  const int8_t* filter, const int32_t* bias,
                                  const int oh_size, const int oc, const int ic,
                                  const Op& op) {
    constexpr int FH = filter_size;
    constexpr int FW = filter_size;
    constexpr int IC_PACK_SIZE = 4;
    constexpr int OC_PACK_SIZE = 4;

#if MEGDNN_AARCH64
    constexpr int OC_BIG_INTERVAL = 12;
    constexpr int OC_MID_INTERVAL = 8;
    constexpr int OC_SMA_INTERVAL = 4;
#else
    constexpr int OC_BIG_INTERVAL = 4;
    constexpr int OC_MID_INTERVAL = 4;
    constexpr int OC_SMA_INTERVAL = 4;
#endif

    constexpr int OW_INTERVAL = 8;
    constexpr int SH = stride;

    const int dst_numbers_per_channel = oh * ow;
    const int ow_remain = ow % OW_INTERVAL;
    const int ow_end_idx = ow - ow_remain;
    const int oc_remain =
            oc % OC_BIG_INTERVAL;  //! NCHW44 means oc_remain = 4 or 8
    const int oc_end_idx = oc - oc_remain;
    const int dst_numbers_4channel_packed =
            dst_numbers_per_channel * OC_PACK_SIZE;

    using remain_fun = std::function<void(
            dst_type * dst, const int dst_step, const int8_t* src, const int ih,
            const int iw, const int8_t* filter, const int32_t* bias,
            const int ic, const Op& op)>;

    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_mid_oc_remain = nullptr;
    remain_fun kern_sma_oc_remain = nullptr;

    switch (ow_remain) {
#define cb(step)                                                          \
    case step:                                                            \
        kern_big_oc_remain =                                              \
                KernNeonSdotNCHW44<dst_type, stride, bias_mode, Op, step, \
                                   filter_size, OC_BIG_INTERVAL,          \
                                   OW_INTERVAL>::impl;                    \
        kern_mid_oc_remain =                                              \
                KernNeonSdotNCHW44<dst_type, stride, bias_mode, Op, step, \
                                   filter_size, OC_MID_INTERVAL,          \
                                   OW_INTERVAL>::impl;                    \
        kern_sma_oc_remain =                                              \
                KernNeonSdotNCHW44<dst_type, stride, bias_mode, Op, step, \
                                   filter_size, OC_SMA_INTERVAL,          \
                                   OW_INTERVAL>::impl;                    \
        break;
        UNROLL_CALL_RAW(8, cb);
#undef cb
        default:
            megdnn_assert(0, "no remain %d for kern", ow_remain);
    }

    //! filter layout is [OC/4, IC/4, FH, FW, 4OC, 4IC]
    //! cut [oc, oh, ow] into [oc/OC_INTERVAL, 1, ow/OW_INTERVAL, OW_INTERVAL,
    //! oh, OC_INTERVAL] to calculate KernNeonSdotNCHW44 calculates
    //! [OW_INTERVAL, 1, OC_INTERVAL] each time
    for (int oc_idx = 0; oc_idx < oc_end_idx; oc_idx += OC_BIG_INTERVAL) {
        const int filter_offset_in_element = oc_idx * ic * FH * FW;
        for (int oh_idx = 0; oh_idx < oh_size; ++oh_idx) {
            for (int ow_idx = 0; ow_idx < ow_end_idx; ow_idx += OW_INTERVAL) {
                const int src_offset_in_element =
                        (oh_idx * SH * iw + ow_idx) * IC_PACK_SIZE;
                const int dst_offset_in_element =
                        oc_idx * dst_numbers_per_channel +
                        (oh_idx * ow + ow_idx) * OC_PACK_SIZE;
                const int bias_offset_in_element = oc_idx;
                KernNeonSdotNCHW44<dst_type, stride, bias_mode, Op, OW_INTERVAL,
                                   filter_size, OC_BIG_INTERVAL, OW_INTERVAL>::
                        impl(dst + dst_offset_in_element,
                             dst_numbers_4channel_packed,
                             src + src_offset_in_element, ih, iw,
                             filter + filter_offset_in_element,
                             bias + bias_offset_in_element, ic, op);
            }
            if (ow_remain) {
                const int src_offset_in_element =
                        (oh_idx * SH * iw + ow_end_idx) * IC_PACK_SIZE;
                const int dst_offset_in_element =
                        oc_idx * dst_numbers_per_channel +
                        (oh_idx * ow + ow_end_idx) * OC_PACK_SIZE;
                const int bias_offset_in_element = oc_idx;
                kern_big_oc_remain(dst + dst_offset_in_element,
                                   dst_numbers_4channel_packed,
                                   src + src_offset_in_element, ih, iw,
                                   filter + filter_offset_in_element,
                                   bias + bias_offset_in_element, ic, op);
            }
        }
    }

#ifdef MEGDNN_AARCH64
    //! oc_remain must be 4 or 8 on aarch64 and must be 0 on aarch32
    if (oc_remain) {
        int oc_idx = oc_end_idx;
        const int filter_offset_in_element = oc_idx * ic * FH * FW;
        for (int oh_idx = 0; oh_idx < oh_size; ++oh_idx) {
            for (int ow_idx = 0; ow_idx < ow_end_idx; ow_idx += OW_INTERVAL) {
                const int src_offset_in_element =
                        (oh_idx * SH * iw + ow_idx) * IC_PACK_SIZE;
                const int dst_offset_in_element =
                        oc_idx * dst_numbers_per_channel +
                        (oh_idx * ow + ow_idx) * OC_PACK_SIZE;
                const int bias_offset_in_element = oc_idx;
                if (oc_remain == 8) {
                    KernNeonSdotNCHW44<
                            dst_type, stride, bias_mode, Op, OW_INTERVAL,
                            filter_size, OC_MID_INTERVAL,
                            OW_INTERVAL>::impl(dst + dst_offset_in_element,
                                               dst_numbers_4channel_packed,
                                               src + src_offset_in_element, ih,
                                               iw,
                                               filter +
                                                       filter_offset_in_element,
                                               bias + bias_offset_in_element,
                                               ic, op);
                } else {
                    KernNeonSdotNCHW44<
                            dst_type, stride, bias_mode, Op, OW_INTERVAL,
                            filter_size, OC_SMA_INTERVAL,
                            OW_INTERVAL>::impl(dst + dst_offset_in_element,
                                               dst_numbers_4channel_packed,
                                               src + src_offset_in_element, ih,
                                               iw,
                                               filter +
                                                       filter_offset_in_element,
                                               bias + bias_offset_in_element,
                                               ic, op);
                }
            }
            if (ow_remain) {
                const int src_offset_in_element =
                        (oh_idx * SH * iw + ow_end_idx) * IC_PACK_SIZE;
                const int dst_offset_in_element =
                        oc_idx * dst_numbers_per_channel +
                        (oh_idx * ow + ow_end_idx) * OC_PACK_SIZE;
                const int bias_offset_in_element = oc_idx;
                if (oc_remain == 8) {
                    kern_mid_oc_remain(dst + dst_offset_in_element,
                                       dst_numbers_4channel_packed,
                                       src + src_offset_in_element, ih, iw,
                                       filter + filter_offset_in_element,
                                       bias + bias_offset_in_element, ic, op);
                } else {
                    kern_sma_oc_remain(dst + dst_offset_in_element,
                                       dst_numbers_4channel_packed,
                                       src + src_offset_in_element, ih, iw,
                                       filter + filter_offset_in_element,
                                       bias + bias_offset_in_element, ic, op);
                }
            }
        }
    }
#endif
}

#define CONSTRUCT_FUNC(filter_size)                                           \
    template <typename dst_type, BiasMode bias_mode, typename Op, int stride> \
    void conv_direct_##filter_size##x##filter_size##_int8_nchw44(             \
            dst_type* dst, const int oh, const int ow, const int8_t* src,     \
            const int ih, const int iw, const int8_t* weight,                 \
            const int32_t* bias, const int oh_size, const int oc,             \
            const int ic, const Op& op) {                                     \
        conv_direct_sdot_int8_nchw44<dst_type, stride, bias_mode, Op,         \
                                     filter_size>(                            \
                dst, oh, ow, src, ih, iw, weight, bias, oh_size, oc, ic, op); \
    }

CONSTRUCT_FUNC(2);
CONSTRUCT_FUNC(3);
CONSTRUCT_FUNC(5);
CONSTRUCT_FUNC(7);
#undef CONSTRUCT_FUNC

#define INSTANTIATION(dst_type, stride, i, bias_mode, Op)                      \
    template void conv_direct_##i##x##i##_int8_nchw44<dst_type, bias_mode, Op, \
                                                      stride>(                 \
            dst_type * dst, const int oh, const int ow, const int8_t* src,     \
            const int ih, const int iw, const int8_t* weight,                  \
            const int32_t* bias, const int oh_size, const int oc,              \
            const int ic, const Op& op);

#define FOR_OP(stride, i, bias_mode)                          \
    INSTANTIATION(dt_int8, stride, i, bias_mode,              \
                  TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
    INSTANTIATION(dt_int32, stride, i, bias_mode,             \
                  NoneOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    INSTANTIATION(dt_int8, stride, i, bias_mode,              \
                  ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    INSTANTIATION(dt_int8, stride, i, bias_mode,              \
                  HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)

#define FOR_BIAS(stride, i)              \
    FOR_OP(stride, i, BiasMode::NO_BIAS) \
    FOR_OP(stride, i, BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_FILTER(stride) \
    FOR_BIAS(stride, 2)    \
    FOR_BIAS(stride, 3)    \
    FOR_BIAS(stride, 5)    \
    FOR_BIAS(stride, 7)

FOR_FILTER(1)
FOR_FILTER(2)

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_IC
#undef FOR_BIAS
#undef FOR_NONLINEAR
#undef FOR_REMAIN
#undef INSTANTIATION

}  // namespace direct_dotprod_nchw44
}  // namespace arm_common
}  // namespace megdnn
#endif

//vim: syntax=cpp.doxygen
