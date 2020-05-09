/**
 * \file dnn/src/arm_common/pooling/do_pooling_2x2_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/pooling/do_pooling_2x2_nchw44.h"
#include "src/arm_common/pooling/algo.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"

namespace megdnn {
namespace arm_common {

void do_max_pooling_2x2_stride1_int8_nchw44_NEON(const int8_t* src, int8_t* dst,
                                                 size_t IH, size_t IW,
                                                 size_t OH, size_t OW,
                                                 size_t PH, size_t PW,
                                                 const WorkspaceBundle& ws) {
    const int8_t* sptr = nullptr;
    size_t IH2, IW2;
    sptr = handle_padding(src, IH, IW, IH2, IW2, PH, PW, ws, true);
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* __restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* __restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src0123 = vld1q_s8(sptr0);
            int8x16_t src1234 = vld1q_s8(sptr0 + 4);
            int8x16_t max0 = vmaxq_s8(src0123, src1234);

            src0123 = vld1q_s8(sptr1);
            src1234 = vld1q_s8(sptr1 + 4);
            int8x16_t max1 = vmaxq_s8(src0123, src1234);

            int8x16_t max_out = vmaxq_s8(max0, max1);

            vst1q_s8(dptr, max_out);

            sptr0 += 16;
            sptr1 += 16;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);

            int8x8_t max_out = vmax_s8(src001, src101);
#define store(i) *(dptr + i) = std::max(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
            sptr0 += 4;
            sptr1 += 4;
            dptr += 4;
        }
    }
}
void do_max_pooling_2x2_stride2_int8_nchw44_NEON(const int8_t* src, int8_t* dst,
                                                 size_t IH, size_t IW,
                                                 size_t OH, size_t OW,
                                                 size_t PH, size_t PW,
                                                 const WorkspaceBundle& ws) {
    const int8_t* sptr = nullptr;
    size_t IH2, IW2;
    sptr = handle_padding(src, IH, IW, IH2, IW2, PH, PW, ws, true);
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* __restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* __restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src04 = vld1q_s8(sptr0 + 4 * 4);
            int32x4x2_t src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00),
                                            vreinterpretq_s32_s8(src04));
            int32x4_t src0246 = src_tmp.val[0];
            int32x4_t src1357 = src_tmp.val[1];
            int8x16_t max0 = vmaxq_s8(vreinterpretq_s8_s32(src0246),
                                      vreinterpretq_s8_s32(src1357));

            src00 = vld1q_s8(sptr1);
            src04 = vld1q_s8(sptr1 + 4 * 4);
            src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00),
                                vreinterpretq_s32_s8(src04));
            src0246 = src_tmp.val[0];
            src1357 = src_tmp.val[1];
            int8x16_t max1 = vmaxq_s8(vreinterpretq_s8_s32(src0246),
                                      vreinterpretq_s8_s32(src1357));

            int8x16_t max_out = vmaxq_s8(max0, max1);

            vst1q_s8(dptr, max_out);

            sptr0 += 32;
            sptr1 += 32;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);

            int8x8_t max_out = vmax_s8(src001, src101);
#define store(i) *(dptr + i) = std::max(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
            sptr0 += 8;
            sptr1 += 8;
            dptr += 4;
        }
    }
}

void do_avg_pooling_2x2_stride1_int8_nchw44_NEON(const int8_t* src, int8_t* dst,
                                                 size_t IH, size_t IW,
                                                 size_t OH, size_t OW,
                                                 size_t PH, size_t PW,
                                                 const WorkspaceBundle& ws) {
    int16_t filter_size = 4;
    const int8_t* sptr = nullptr;
    size_t IH2, IW2;
    sptr = handle_padding(src, IH, IW, IH2, IW2, PH, PW, ws, false);
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* __restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* __restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src0123, src1234;
            int16x8_t src01, src23, src12, src34;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                      \
    src0123 = vld1q_s8(sptr##i);             \
    src1234 = vld1q_s8(sptr##i + 4);         \
    src01 = vmovl_s8(vget_low_s8(src0123));  \
    src23 = vmovl_s8(vget_high_s8(src0123)); \
    src12 = vmovl_s8(vget_low_s8(src1234));  \
    src34 = vmovl_s8(vget_high_s8(src1234)); \
    sum01 = vaddq_s16(sum01, src01);         \
    sum01 = vaddq_s16(sum01, src12);         \
    sum23 = vaddq_s16(sum23, src23);         \
    sum23 = vaddq_s16(sum23, src34);

            UNROLL_CALL_NOWRAPPER(2, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                  \
    sum##i = vgetq_lane_s16(sum01, i) > 0                             \
                     ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / \
                               filter_size                            \
                     : (vgetq_lane_s16(sum01, i) - filter_size / 2) / \
                               filter_size;
#define sum23_avg(i)                                                  \
    sum##i = vgetq_lane_s16(sum23, i) > 0                             \
                     ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / \
                               filter_size                            \
                     : (vgetq_lane_s16(sum23, i) - filter_size / 2) / \
                               filter_size;

#define store_sum01(i) *(dptr + i) = static_cast<int8_t>(sum##i);
#define store_sum23(i) *(dptr + i + 8) = static_cast<int8_t>(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 16;
            sptr1 += 16;
            dptr += 16;
#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);
            int16x8_t src00 = vmovl_s8(src001);
            int16x8_t src10 = vmovl_s8(src101);
            int16x8_t max_tmp = vaddq_s16(src00, src10);
#define do_acc(i)    \
    int16_t sum##i = \
            vgetq_lane_s16(max_tmp, i) + vgetq_lane_s16(max_tmp, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = static_cast<int8_t>(sum##i);
            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
#undef do_avg
#undef do_acc
            sptr0 += 4;
            sptr1 += 4;
            dptr += 4;
        }
    }
}

void do_avg_pooling_2x2_stride2_int8_nchw44_NEON(const int8_t* src, int8_t* dst,
                                                 size_t IH, size_t IW,
                                                 size_t OH, size_t OW,
                                                 size_t PH, size_t PW,
                                                 const WorkspaceBundle& ws) {
    int16_t filter_size = 4;
    const int8_t* sptr = nullptr;
    size_t IH2, IW2;
    sptr = handle_padding(src, IH, IW, IH2, IW2, PH, PW, ws, false);
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* __restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* __restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int32x4x2_t src_tmp;
            int8x16_t src00, src04;
            int32x4_t src0246, src1357;
            int16x8_t src02, src13, src46, src57;
            int16x8_t sum01 = vdupq_n_s16(0);
            int16x8_t sum23 = vdupq_n_s16(0);

#define CACULATE_ROW(i)                                            \
    src00 = vld1q_s8(sptr##i);                                     \
    src04 = vld1q_s8(sptr##i + 4 * 4);                             \
    src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00),               \
                        vreinterpretq_s32_s8(src04));              \
    src0246 = src_tmp.val[0];                                      \
    src1357 = src_tmp.val[1];                                      \
    src02 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src0246)));  \
    src46 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src0246))); \
    src13 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(src1357)));  \
    src57 = vmovl_s8(vget_high_s8(vreinterpretq_s8_s32(src1357))); \
    sum01 = vaddq_s16(sum01, src02);                               \
    sum01 = vaddq_s16(sum01, src13);                               \
    sum23 = vaddq_s16(sum23, src46);                               \
    sum23 = vaddq_s16(sum23, src57);

            UNROLL_CALL_NOWRAPPER(2, CACULATE_ROW)

#define sum_define(i) int16_t sum##i;
            UNROLL_CALL_NOWRAPPER(8, sum_define)

#define sum01_avg(i)                                                  \
    sum##i = vgetq_lane_s16(sum01, i) > 0                             \
                     ? (vgetq_lane_s16(sum01, i) + filter_size / 2) / \
                               filter_size                            \
                     : (vgetq_lane_s16(sum01, i) - filter_size / 2) / \
                               filter_size;
#define sum23_avg(i)                                                  \
    sum##i = vgetq_lane_s16(sum23, i) > 0                             \
                     ? (vgetq_lane_s16(sum23, i) + filter_size / 2) / \
                               filter_size                            \
                     : (vgetq_lane_s16(sum23, i) - filter_size / 2) / \
                               filter_size;
#define store_sum01(i) *(dptr + i) = static_cast<int8_t>(sum##i);
#define store_sum23(i) *(dptr + i + 8) = static_cast<int8_t>(sum##i);

            UNROLL_CALL_NOWRAPPER(8, sum01_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum01)

            UNROLL_CALL_NOWRAPPER(8, sum23_avg)
            UNROLL_CALL_NOWRAPPER(8, store_sum23)

            sptr0 += 32;
            sptr1 += 32;
            dptr += 16;
#undef store_sum01
#undef store_sum23
#undef sum01_avg
#undef sum23_avg
#undef sum_define
#undef CACULATE_ROW
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);
            int16x8_t src00 = vmovl_s8(src001);
            int16x8_t src10 = vmovl_s8(src101);
            int16x8_t max_tmp = vaddq_s16(src00, src10);

#define do_acc(i)    \
    int16_t sum##i = \
            vgetq_lane_s16(max_tmp, i) + vgetq_lane_s16(max_tmp, i + 4);
#define do_avg(i)                                                  \
    sum##i = sum##i > 0 ? (sum##i + filter_size / 2) / filter_size \
                        : (sum##i - filter_size / 2) / filter_size;
#define store(i) *(dptr + i) = static_cast<int8_t>(sum##i);
            UNROLL_CALL_NOWRAPPER(4, do_acc)
            UNROLL_CALL_NOWRAPPER(4, do_avg)
            UNROLL_CALL_NOWRAPPER(4, store)
#undef do_avg
#undef do_acc
#undef store
            sptr0 += 8;
            sptr1 += 8;
            dptr += 4;
        }
    }
}

}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
