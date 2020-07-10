/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw44_s2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#if __ARM_FEATURE_DOTPROD

#include "src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw44_common.h"
namespace megdnn {
namespace arm_common {
namespace direct_dotprod_nchw44 {
template <typename dst_type, BiasMode bias_mode, typename Op, int ow_remain,
          int filter_size, int oc_interval, int ow_interval>
struct KernNeonSdotNCHW44<dst_type, 2, bias_mode, Op, ow_remain, filter_size,
                          oc_interval, ow_interval> {
    static void impl(dst_type* dst, const int dst_step, const int8_t* src,
                     const int ih, const int iw, const int8_t* filter,
                     const int32_t* bias, const int ic, const Op& op) {
        constexpr int FH = filter_size;
        constexpr int FW = filter_size;
        constexpr int filter_next_row =
                FW * OC_PACK_SIZE *
                IC_PACK_SIZE;  //! [OC/4, IC/4, FH, FW, 4OC, 4IC]

        const int filter_next_4oc =
                FH * FW * ic * OC_PACK_SIZE;  //! [OC/4, IC/4, FH, FW, 4OC, 4IC]
        const int src_next_ic = ih * iw;
        const int src_next_row = iw * IC_PACK_SIZE;

        constexpr int NSRC = (ow_interval * 2 + filter_size - 3) / 8 + 1;
        constexpr int LOOP = oc_interval / 4;

        int32x4_t res[3][ow_interval];
        init_ocx_ow8<LOOP, bias_mode>(res, bias, OC_PACK_SIZE);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += IC_PACK_SIZE) {
            const int8_t* i_src = src + ic_idx * src_next_ic;
            const int8_t* i_filter = filter + ic_idx * FH * FW * OC_PACK_SIZE;
            for (int fh_idx = 0; fh_idx < FH; ++fh_idx) {
                int8x16_t src[2][3];
                int8x16_t weight[3];
                const int offset = megdnn::div_ceil(iw, 2) * IC_PACK_SIZE;

                load_helper<NSRC, 0, SIMD_LEN, 2, Vld1q_s8>(src, i_src, offset);

//! do not use switch order 3,2,1 because it will slow the speed.
#define CALC_PART(step)                                             \
    switch (LOOP) {                                                 \
        case 1:                                                     \
            weight[0] = vld1q_s8(i_filter + filter_next_4oc * 0 +   \
                                 filter_next_col * step);           \
            cal_helper<0, step % 2, step / 2, 0>(res, src, weight); \
            break;                                                  \
        case 2:                                                     \
            weight[0] = vld1q_s8(i_filter + filter_next_4oc * 0 +   \
                                 filter_next_col * step);           \
            cal_helper<0, step % 2, step / 2, 0>(res, src, weight); \
            weight[1] = vld1q_s8(i_filter + filter_next_4oc * 1 +   \
                                 filter_next_col * step);           \
            cal_helper<1, step % 2, step / 2, 1>(res, src, weight); \
            break;                                                  \
        case 3:                                                     \
            weight[0] = vld1q_s8(i_filter + filter_next_4oc * 0 +   \
                                 filter_next_col * step);           \
            cal_helper<0, step % 2, step / 2, 0>(res, src, weight); \
            weight[1] = vld1q_s8(i_filter + filter_next_4oc * 1 +   \
                                 filter_next_col * step);           \
            cal_helper<1, step % 2, step / 2, 1>(res, src, weight); \
            weight[2] = vld1q_s8(i_filter + filter_next_4oc * 2 +   \
                                 filter_next_col * step);           \
            cal_helper<2, step % 2, step / 2, 2>(res, src, weight); \
            break;                                                  \
        default:                                                    \
            break;                                                  \
    }

                switch (filter_size) {
                    case 2:
                        UNROLL_CALL_RAW(2, CALC_PART);
                        break;
                    case 3:
                        UNROLL_CALL_RAW(3, CALC_PART);
                        break;
                    case 5:
                        UNROLL_CALL_RAW(5, CALC_PART);
                        break;
                    case 7:
                        UNROLL_CALL_RAW(7, CALC_PART);
                        break;
                    default:
                        break;
                }
#undef CALC_PART

                i_filter += filter_next_row;
                i_src += src_next_row;
            }
        }
        store_ocx_owx_remain_static<LOOP, ow_remain, Op>(res, op, dst,
                                                         dst_step);
    }
};

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

#define INSTANTIATION(dst_type, stride, filter_size, bias_mode, Op)         \
    template void conv_direct_sdot_int8_nchw44<dst_type, stride, bias_mode, \
                                               Op, filter_size>(            \
            dst_type * dst, const int oh, const int ow, const int8_t* src,  \
            const int ih, const int iw, const int8_t* weight,               \
            const int32_t* bias, const int oh_size, const int oc,           \
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
// vim: syntax=cpp.doxygen