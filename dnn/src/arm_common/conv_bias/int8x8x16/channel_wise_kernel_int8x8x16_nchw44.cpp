/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/channel_wise_kernel_int8x8x16_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/int8x8x16/channel_wise_kernel.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;

#define INIT_SUM()                                       \
    int16x8_t init_sum;                                  \
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) { \
        int16x4_t tmpsum = vld1_s16(bptr);               \
        init_sum = vcombine_s16(tmpsum, tmpsum);         \
    } else {                                             \
        init_sum = vdupq_n_s16(0);                       \
    }

#define STORE_1_LINE_RESULT(dst, oh, ow, OW, sum)                        \
    do {                                                                 \
        dt_int16* dptr =                                                 \
                reinterpret_cast<dt_int16*>(dst) + (oh)*OW * 4 + ow * 4; \
        vst1q_s16(dptr, sum[0]);                                         \
        vst1q_s16(dptr + 8, sum[1]);                                     \
        vst1q_s16(dptr + 16, sum[2]);                                    \
        vst1q_s16(dptr + 24, sum[3]);                                    \
    } while (0);

#define STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum)                      \
    do {                                                                 \
        dt_int16* dptr =                                                 \
                reinterpret_cast<dt_int16*>(dst) + (oh)*OW * 4 + ow * 4; \
        vst1q_s16(dptr, sum[0]);                                         \
        vst1q_s16(dptr + 8, sum[1]);                                     \
    } while (0);

#define STORE_REMAIN(dst, oh, ow, OW, sum, remain)                       \
    do {                                                                 \
        dt_int16* dptr =                                                 \
                reinterpret_cast<dt_int16*>(dst) + oh * OW * 4 + ow * 4; \
        if (remain == 1) {                                               \
            vst1_s16(dptr, vget_low_s16(sum[0]));                        \
        } else if (remain == 2) {                                        \
            vst1q_s16(dptr, sum[0]);                                     \
        } else if (remain == 3) {                                        \
            vst1q_s16(dptr, sum[0]);                                     \
            vst1_s16(dptr + 8, vget_low_s16(sum[1]));                    \
        }                                                                \
    } while (0);

template <BiasMode bias_mode>
void channel_wise_nchw44_8x8x16::direct_stride1_2x2_int8x8x16(
        const int8_t* src, const int8_t* filter, const int16_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW) {
    MEGDNN_MARK_USED_VAR(IH);
    const int16_t* __restrict bptr = bias;
    INIT_SUM();
    const int* fptr = reinterpret_cast<const int*>(filter);
    int8x8_t kern[4];
#define cb(i) kern[i] = vreinterpret_s8_s32(vld1_dup_s32(fptr + i));
    UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
#define LOAD_SRC(_sptr, _src)       \
    _src[0] = vld1q_s8(_sptr);      \
    _src[1] = vld1q_s8(_sptr + 16); \
    _src[1] = vextq_s8(_src[0], _src[1], 4);

#define CALC_ONE_LINE_4_RESULT(_sum, _src, _kid0, _kid1)             \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[0]), kern[_kid0]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[0]), kern[_kid0]); \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[1]), kern[_kid1]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[1]), kern[_kid1]);

#define LOAD_SRC_8(_sptr, _src)              \
    _src[0] = vld1q_s8(_sptr);               \
    _src[2] = vld1q_s8(_sptr + 16);          \
    _src[3] = vld1q_s8(_sptr + 32);          \
    _src[1] = vextq_s8(_src[0], _src[2], 4); \
    _src[3] = vextq_s8(_src[2], _src[3], 4);

#define CALC_ONE_LINE_8_RESULT(_sum,_src,_kid0,_kid1)\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[0]),kern[_kid0]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[0]),kern[_kid0]);\
            _sum[2]=vmlal_s8(_sum[2], vget_low_s8(_src[2]),kern[_kid0]);\
            _sum[3]=vmlal_s8(_sum[3],vget_high_s8(_src[2]),kern[_kid0]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[1]),kern[_kid1]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[1]),kern[_kid1]);\
            _sum[2]=vmlal_s8(_sum[2], vget_low_s8(_src[3]),kern[_kid1]);\
            _sum[3]=vmlal_s8(_sum[3],vget_high_s8(_src[3]),kern[_kid1]);

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + (ih + 0) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;

            int16x8_t sum[2][4];
            int8x16_t src[2][4];

#define cb(i)             \
    sum[0][i] = init_sum; \
    sum[1][i] = init_sum;

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_SRC_8(sptr0, src[0]);
            LOAD_SRC_8(sptr1, src[1]);

            CALC_ONE_LINE_8_RESULT(sum[0], src[0], 0, 1);
            LOAD_SRC_8(sptr2, src[0]);
            CALC_ONE_LINE_8_RESULT(sum[0], src[1], 2, 3);
            CALC_ONE_LINE_8_RESULT(sum[1], src[1], 0, 1);
            CALC_ONE_LINE_8_RESULT(sum[1], src[0], 2, 3);

            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int16x8_t sum[2][2];
            int8x16_t src[2][2];

#define cb(i)             \
    sum[0][i] = init_sum; \
    sum[1][i] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
            LOAD_SRC(sptr0, src[0]);
            LOAD_SRC(sptr1, src[1]);

            CALC_ONE_LINE_4_RESULT(sum[0], src[0], 0, 1);
            LOAD_SRC(sptr2, src[0]);
            CALC_ONE_LINE_4_RESULT(sum[0], src[1], 2, 3);
            CALC_ONE_LINE_4_RESULT(sum[1], src[1], 0, 1);
            CALC_ONE_LINE_4_RESULT(sum[1], src[0], 2, 3);

            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_4_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int16x8_t sum[2][2];
            int8x16_t src[2][2];

#define cb(i)             \
    sum[0][i] = init_sum; \
    sum[1][i] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
            LOAD_SRC(sptr0, src[0]);
            LOAD_SRC(sptr1, src[1]);

            CALC_ONE_LINE_4_RESULT(sum[0], src[0], 0, 1);
            LOAD_SRC(sptr2, src[0]);
            CALC_ONE_LINE_4_RESULT(sum[0], src[1], 2, 3);
            CALC_ONE_LINE_4_RESULT(sum[1], src[1], 0, 1);
            CALC_ONE_LINE_4_RESULT(sum[1], src[0], 2, 3);
            STORE_REMAIN(dst, oh, ow, OW, sum[0], remain);
            STORE_REMAIN(dst, (oh + 1), ow, OW, sum[1], remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + (ih + 0) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;

            int16x8_t sum[4];
            int8x16_t src[2][4];
#define cb(i) sum[i] = init_sum;

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

            LOAD_SRC_8(sptr0, src[0]);
            LOAD_SRC_8(sptr1, src[1]);

            CALC_ONE_LINE_8_RESULT(sum, src[0], 0, 1);
            CALC_ONE_LINE_8_RESULT(sum, src[1], 2, 3);
            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum);
        }

        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;

            int16x8_t sum[2];
            int8x16_t src[2][2];
            sum[0] = init_sum;
            sum[1] = init_sum;

            LOAD_SRC(sptr0, src[0]);
            LOAD_SRC(sptr1, src[1]);

            CALC_ONE_LINE_4_RESULT(sum, src[0], 0, 1);
            CALC_ONE_LINE_4_RESULT(sum, src[1], 2, 3);

            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum);

        }

        if (ow < OW) {
            size_t iw = ow;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            int16x8_t sum[2];
            int8x16_t src[2][2];
            sum[0] = init_sum;
            sum[1] = init_sum;

            LOAD_SRC(sptr0, src[0]);
            LOAD_SRC(sptr1, src[1]);

            CALC_ONE_LINE_4_RESULT(sum, src[0], 0, 1);
            CALC_ONE_LINE_4_RESULT(sum, src[1], 2, 3);
            STORE_REMAIN(dst, oh, ow, OW, sum, remain);
        }
    }
}
#undef CALC_ONE_LINE_4_RESULT
#undef CALC_ONE_LINE_8_RESULT
#undef LOAD_SRC
#undef LOAD_SRC_8

template <BiasMode bias_mode>
void channel_wise_nchw44_8x8x16::direct_stride1_3x3_int8x8x16(
        const int8_t* sptr, const int8_t* fptr, const int16_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW) {
    MEGDNN_MARK_USED_VAR(IH);
    const int16_t* __restrict bptr = bias;
    INIT_SUM();
    const int* filter = reinterpret_cast<const int*>(fptr);
    int8x8_t kern[9];
#define cb(i) kern[i] = vreinterpret_s8_s32(vld1_dup_s32(filter + i));
    UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb

#define LOAD_6_SRC(src, sptr0)              \
    src[0] = vld1q_s8(sptr0);               \
    src[1] = vld1q_s8(sptr0 + 16);          \
    tmp_src0 = vld1q_s8(sptr0 + 32);        \
    src[2] = vextq_s8(src[0], src[1], 4);   \
    src[3] = vextq_s8(src[1], tmp_src0, 4); \
    src[4] = vextq_s8(src[0], src[1], 8);   \
    src[5] = vextq_s8(src[1], tmp_src0, 8);

#define LOAD_3_SRC(sptr, src)             \
    src[0] = vld1q_s8(sptr);              \
    src[2] = vld1q_s8(sptr + 16);         \
    src[1] = vextq_s8(src[0], src[2], 4); \
    src[2] = vextq_s8(src[0], src[2], 8);

#define CALC_ONE_LINE(_src, _kern0, _kern1, _kern2, _sum)       \
    _sum[0] = vmlal_s8(_sum[0], _kern0, vget_low_s8(_src[0]));  \
    _sum[1] = vmlal_s8(_sum[1], _kern0, vget_high_s8(_src[0])); \
    _sum[0] = vmlal_s8(_sum[0], _kern1, vget_low_s8(_src[1]));  \
    _sum[1] = vmlal_s8(_sum[1], _kern1, vget_high_s8(_src[1])); \
    _sum[0] = vmlal_s8(_sum[0], _kern2, vget_low_s8(_src[2]));  \
    _sum[1] = vmlal_s8(_sum[1], _kern2, vget_high_s8(_src[2]));

#define CALC_ONE(_src, _i, _j, _kern, _sum)                     \
    _sum[0] = vmlal_s8(_sum[0], _kern, vget_low_s8(_src[_i]));  \
    _sum[1] = vmlal_s8(_sum[1], _kern, vget_high_s8(_src[_i])); \
    _sum[2] = vmlal_s8(_sum[2], _kern, vget_low_s8(_src[_j]));  \
    _sum[3] = vmlal_s8(_sum[3], _kern, vget_high_s8(_src[_j]));

    size_t oh = 0_z;
    for (; oh + 3 <= OH; oh += 3) {
        size_t ih = oh;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = sptr + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = sptr + (ih + 4) * IW * 4 + iw * 4;
            int16x8_t sum0[4], sum1[4], sum2[4];
#define cb(j)           \
    sum0[j] = init_sum; \
    sum1[j] = init_sum; \
    sum2[j] = init_sum;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

            int8x16_t src[2][6];
            int8x16_t tmp_src0;
            LOAD_6_SRC(src[0], sptr0);  //! line0
            LOAD_6_SRC(src[1], sptr1);  //! line1
            CALC_ONE(src[0], 0, 1, kern[0], sum0);
            CALC_ONE(src[0], 2, 3, kern[1], sum0);
            CALC_ONE(src[0], 4, 5, kern[2], sum0);
            CALC_ONE(src[1], 0, 1, kern[3], sum0);
            CALC_ONE(src[1], 2, 3, kern[4], sum0);
            CALC_ONE(src[1], 4, 5, kern[5], sum0);

            LOAD_6_SRC(src[0], sptr2);  //! line2
            CALC_ONE(src[0], 0, 1, kern[6], sum0);
            CALC_ONE(src[0], 2, 3, kern[7], sum0);
            CALC_ONE(src[0], 4, 5, kern[8], sum0);

            CALC_ONE(src[1], 0, 1, kern[0], sum1);
            CALC_ONE(src[1], 2, 3, kern[1], sum1);
            CALC_ONE(src[1], 4, 5, kern[2], sum1);

            CALC_ONE(src[0], 0, 1, kern[3], sum1);
            CALC_ONE(src[0], 2, 3, kern[4], sum1);
            CALC_ONE(src[0], 4, 5, kern[5], sum1);

            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum0)
            LOAD_6_SRC(src[1], sptr3);  //! line3

            CALC_ONE(src[1], 0, 1, kern[6], sum1);
            CALC_ONE(src[1], 2, 3, kern[7], sum1);
            CALC_ONE(src[1], 4, 5, kern[8], sum1);

            CALC_ONE(src[0], 0, 1, kern[0], sum2);
            CALC_ONE(src[0], 2, 3, kern[1], sum2);
            CALC_ONE(src[0], 4, 5, kern[2], sum2);

            CALC_ONE(src[1], 0, 1, kern[3], sum2);
            CALC_ONE(src[1], 2, 3, kern[4], sum2);
            CALC_ONE(src[1], 4, 5, kern[5], sum2);
            LOAD_6_SRC(src[0], sptr4);  //! line4
            STORE_1_LINE_RESULT(dst, (oh + 1), ow, OW, sum1)

            CALC_ONE(src[0], 0, 1, kern[6], sum2);
            CALC_ONE(src[0], 2, 3, kern[7], sum2);
            CALC_ONE(src[0], 4, 5, kern[8], sum2);
            STORE_1_LINE_RESULT(dst, (oh + 2), ow, OW, sum2)
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = sptr + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = sptr + (ih + 4) * IW * 4 + iw * 4;

            int16x8_t sum0[2], sum1[2], sum2[2];
#define cb(j)           \
    sum0[j] = init_sum; \
    sum1[j] = init_sum; \
    sum2[j] = init_sum;
            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

            int8x16_t src[2][3];

            LOAD_3_SRC(sptr0,src[0]);
            LOAD_3_SRC(sptr1,src[1]);

            CALC_ONE_LINE(src[0],kern[0],kern[1],kern[2],sum0);//line0
            CALC_ONE_LINE(src[1],kern[3],kern[4],kern[5],sum0);//line1
            CALC_ONE_LINE(src[1],kern[0],kern[1],kern[2],sum1);//line1

            LOAD_3_SRC(sptr2,src[0]);//line2

            CALC_ONE_LINE(src[0],kern[6],kern[7],kern[8],sum0);//line2
            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum0)

            CALC_ONE_LINE(src[0],kern[3],kern[4],kern[5],sum1);//line2
            CALC_ONE_LINE(src[0],kern[0],kern[1],kern[2],sum2);//line2
            LOAD_3_SRC(sptr3,src[1]);//line3

            CALC_ONE_LINE(src[1],kern[6],kern[7],kern[8],sum1);//line3
            STORE_1_LINE_4_RESULT(dst, (oh+1), ow, OW, sum1)
            CALC_ONE_LINE(src[1],kern[3],kern[4],kern[5],sum2);//line3
            LOAD_3_SRC(sptr4,src[0]);
            CALC_ONE_LINE(src[0],kern[6],kern[7],kern[8],sum2);//line4
            STORE_1_LINE_4_RESULT(dst, (oh+2), ow, OW, sum2)
        }
        if (ow < OW) {
            size_t iw = ow;
            size_t remain = OW - ow;
            
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = sptr + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = sptr + (ih + 4) * IW * 4 + iw * 4;
            int16x8_t sum0[2], sum1[2], sum2[2];
            int8x16_t src[2][3];
#define cb(j)           \
    sum0[j] = init_sum; \
    sum1[j] = init_sum; \
    sum2[j] = init_sum;
            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

            LOAD_3_SRC(sptr0,src[0]);//line2
            LOAD_3_SRC(sptr1,src[1]);//line2
            CALC_ONE_LINE(src[0], kern[0], kern[1], kern[2], sum0);  // line0
            CALC_ONE_LINE(src[1],kern[3],kern[4],kern[5],sum0);//line1
            CALC_ONE_LINE(src[1],kern[0],kern[1],kern[2],sum1);//line1

            LOAD_3_SRC(sptr2,src[0]);//line2

            CALC_ONE_LINE(src[0],kern[6],kern[7],kern[8],sum0);//line2
            STORE_REMAIN(dst, (oh+0), ow, OW, sum0,remain)

            CALC_ONE_LINE(src[0],kern[3],kern[4],kern[5],sum1);//line2
            CALC_ONE_LINE(src[0],kern[0],kern[1],kern[2],sum2);//line2
            LOAD_3_SRC(sptr3,src[1]);//line3

            CALC_ONE_LINE(src[1],kern[6],kern[7],kern[8],sum1);//line3
            STORE_REMAIN(dst, (oh+1), ow, OW, sum1,remain)
            CALC_ONE_LINE(src[1],kern[3],kern[4],kern[5],sum2);//line3
            LOAD_3_SRC(sptr4,src[0]);
            CALC_ONE_LINE(src[0],kern[6],kern[7],kern[8],sum2);//line4
            STORE_REMAIN(dst, (oh+2), ow, OW, sum2, remain)
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            int16x8_t sum0[4];
            int8x16_t src[2][6];
            int8x16_t tmp_src0;

            sum0[0] = init_sum;
            sum0[1] = init_sum;
            sum0[2] = init_sum;
            sum0[3] = init_sum;

            LOAD_6_SRC(src[0], sptr0);  //! line0
            LOAD_6_SRC(src[1], sptr1);  //! line1
            CALC_ONE(src[0], 0, 1, kern[0], sum0);
            CALC_ONE(src[0], 2, 3, kern[1], sum0);
            CALC_ONE(src[0], 4, 5, kern[2], sum0);
            CALC_ONE(src[1], 0, 1, kern[3], sum0);
            CALC_ONE(src[1], 2, 3, kern[4], sum0);
            CALC_ONE(src[1], 4, 5, kern[5], sum0);
            LOAD_6_SRC(src[0], sptr2);  //! line2
            CALC_ONE(src[0], 0, 1, kern[6], sum0);
            CALC_ONE(src[0], 2, 3, kern[7], sum0);
            CALC_ONE(src[0], 4, 5, kern[8], sum0);

            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum0);
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;

            int16x8_t sum00[2];
            int8x16_t src[2][3];

            sum00[0] = init_sum;
            sum00[1] = init_sum;

            LOAD_3_SRC(sptr0, src[0]);
            LOAD_3_SRC(sptr1, src[1]);

            CALC_ONE_LINE(src[0], kern[0], kern[1], kern[2], sum00);  // line0
            CALC_ONE_LINE(src[1], kern[3], kern[4], kern[5], sum00);  // line1

            LOAD_3_SRC(sptr2, src[0]);  // line2

            CALC_ONE_LINE(src[0], kern[6], kern[7], kern[8], sum00);  // line2
            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum00)
        }
        if (ow < OW) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;

            int16x8_t sum00[2];
            int8x16_t src[2][3];

            sum00[0] = init_sum;
            sum00[1] = init_sum;

            LOAD_3_SRC(sptr0, src[0]);
            LOAD_3_SRC(sptr1, src[1]);

            CALC_ONE_LINE(src[0], kern[0], kern[1], kern[2], sum00);  // line0
            CALC_ONE_LINE(src[1], kern[3], kern[4], kern[5], sum00);  // line1

            LOAD_3_SRC(sptr2, src[0]);  // line2

            CALC_ONE_LINE(src[0], kern[6], kern[7], kern[8], sum00);  // line2
            STORE_REMAIN(dst, oh, ow, OW, sum00,(OW-ow))
        }
    }
#undef LOAD_3_SRC
#undef LOAD_6_SRC
#undef CALC_ONE
#undef CALC_ONE_LINE
}

template <BiasMode bias_mode>
void channel_wise_nchw44_8x8x16::direct_stride1_5x5_int8x8x16(
        const int8_t* sptr, const int8_t* fptr, const int16_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW) {
    MEGDNN_MARK_USED_VAR(IH);
    const int16_t* __restrict bptr = bias;
    INIT_SUM();
    const int* filter = reinterpret_cast<const int*>(fptr);
    int8x8_t kern[25];
#define cb(i) kern[i] = vreinterpret_s8_s32(vld1_dup_s32(filter + i));
    UNROLL_CALL_NOWRAPPER(25, cb);
#undef cb
#define LOAD_1_LINE_SRC(sptr, src)        \
    src[0] = vld1q_s8(sptr);              \
    src[4] = vld1q_s8(sptr + 16);         \
    src[1] = vextq_s8(src[0], src[4], 4); \
    src[2] = vextq_s8(src[0], src[4], 8); \
    src[3] = vextq_s8(src[0], src[4], 12);

#define LOAD_1_LINE_10_SRC(sptr, src)      \
    src[0] = vld1q_s8(sptr);               \
    src[4] = vld1q_s8(sptr + 16);          \
    src[8] = vld1q_s8(sptr + 32);          \
    src[1] = vextq_s8(src[0], src[4], 4);  \
    src[2] = vextq_s8(src[0], src[4], 8);  \
    src[3] = vextq_s8(src[0], src[4], 12); \
    src[5] = vextq_s8(src[4], src[8], 4);  \
    src[6] = vextq_s8(src[4], src[8], 8);  \
    src[7] = vextq_s8(src[4], src[8], 12);


#define CALC_ONE_LINE_4_RESULT(_sum,_src,_kid0,_kid1,_kid2,_kid3,_kid4)\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[0]),kern[_kid0]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[0]),kern[_kid0]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[1]),kern[_kid1]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[1]),kern[_kid1]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[2]),kern[_kid2]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[2]),kern[_kid2]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[3]),kern[_kid3]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[3]),kern[_kid3]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[4]),kern[_kid4]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[4]),kern[_kid4]);

#define CALC_ONE_LINE_8_RESULT(_sum,_src,_kid0,_kid1,_kid2,_kid3,_kid4)\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[0]),kern[_kid0]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[0]),kern[_kid0]);\
            _sum[2]=vmlal_s8(_sum[2], vget_low_s8(_src[4]),kern[_kid0]);\
            _sum[3]=vmlal_s8(_sum[3],vget_high_s8(_src[4]),kern[_kid0]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[1]),kern[_kid1]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[1]),kern[_kid1]);\
            _sum[2]=vmlal_s8(_sum[2], vget_low_s8(_src[5]),kern[_kid1]);\
            _sum[3]=vmlal_s8(_sum[3],vget_high_s8(_src[5]),kern[_kid1]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[2]),kern[_kid2]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[2]),kern[_kid2]);\
            _sum[2]=vmlal_s8(_sum[2], vget_low_s8(_src[6]),kern[_kid2]);\
            _sum[3]=vmlal_s8(_sum[3],vget_high_s8(_src[6]),kern[_kid2]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[3]),kern[_kid3]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[3]),kern[_kid3]);\
            _sum[2]=vmlal_s8(_sum[2], vget_low_s8(_src[7]),kern[_kid3]);\
            _sum[3]=vmlal_s8(_sum[3],vget_high_s8(_src[7]),kern[_kid3]);\
            _sum[0]=vmlal_s8(_sum[0], vget_low_s8(_src[4]),kern[_kid4]);\
            _sum[1]=vmlal_s8(_sum[1],vget_high_s8(_src[4]),kern[_kid4]);\
            _sum[2]=vmlal_s8(_sum[2], vget_low_s8(_src[8]),kern[_kid4]);\
            _sum[3]=vmlal_s8(_sum[3],vget_high_s8(_src[8]),kern[_kid4]);

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            const int8_t* __restrict sptr5 = sptr4 + IW * 4;

            int16x8_t sum[2][4];
            int8x16_t src[2][9];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            
            LOAD_1_LINE_10_SRC(sptr0,src[0]);
            LOAD_1_LINE_10_SRC(sptr1,src[1]);

            CALC_ONE_LINE_8_RESULT(sum[0],src[0],0,1,2,3,4);
            LOAD_1_LINE_10_SRC(sptr2,src[0]);//line2
            CALC_ONE_LINE_8_RESULT(sum[0],src[1],5,6,7,8,9);//line1
            CALC_ONE_LINE_8_RESULT(sum[1],src[1],0,1,2,3,4);//line1
            LOAD_1_LINE_10_SRC(sptr3,src[1]);//line3
            CALC_ONE_LINE_8_RESULT(sum[0],src[0],10,11,12,13,14);//line2
            CALC_ONE_LINE_8_RESULT(sum[1],src[0],5,6,7,8,9);//line2
            LOAD_1_LINE_10_SRC(sptr4,src[0]);//line4
            CALC_ONE_LINE_8_RESULT(sum[0],src[1],15,16,17,18,19);//line3
            CALC_ONE_LINE_8_RESULT(sum[1],src[1],10,11,12,13,14);//line3
            LOAD_1_LINE_10_SRC(sptr5,src[1]);//line5
            CALC_ONE_LINE_8_RESULT(sum[0],src[0],20,21,22,23,24);//line4
            CALC_ONE_LINE_8_RESULT(sum[1],src[0],15,16,17,18,19);//line3
            CALC_ONE_LINE_8_RESULT(sum[1],src[1],20,21,22,23,24);//line3

            STORE_1_LINE_RESULT(dst,oh,ow,OW,sum[0]);
            STORE_1_LINE_RESULT(dst,(oh+1),ow,OW,sum[1]);
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            const int8_t* __restrict sptr5 = sptr4 + IW * 4;

            int16x8_t sum[2][2];
            int8x16_t src[2][5];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

            
            LOAD_1_LINE_SRC(sptr0,src[0]);
            LOAD_1_LINE_SRC(sptr1,src[1]);

            CALC_ONE_LINE_4_RESULT(sum[0],src[0],0,1,2,3,4);
            LOAD_1_LINE_SRC(sptr2,src[0]);//line2
            CALC_ONE_LINE_4_RESULT(sum[0],src[1],5,6,7,8,9);//line1
            CALC_ONE_LINE_4_RESULT(sum[1],src[1],0,1,2,3,4);//line1
            LOAD_1_LINE_SRC(sptr3,src[1]);//line3
            CALC_ONE_LINE_4_RESULT(sum[0],src[0],10,11,12,13,14);//line2
            CALC_ONE_LINE_4_RESULT(sum[1],src[0],5,6,7,8,9);//line2
            LOAD_1_LINE_SRC(sptr4,src[0]);//line4
            CALC_ONE_LINE_4_RESULT(sum[0],src[1],15,16,17,18,19);//line3
            CALC_ONE_LINE_4_RESULT(sum[1],src[1],10,11,12,13,14);//line3
            LOAD_1_LINE_SRC(sptr5,src[1]);//line5
            CALC_ONE_LINE_4_RESULT(sum[0],src[0],20,21,22,23,24);//line4
            CALC_ONE_LINE_4_RESULT(sum[1],src[0],15,16,17,18,19);//line3
            CALC_ONE_LINE_4_RESULT(sum[1],src[1],20,21,22,23,24);//line3

            STORE_1_LINE_4_RESULT(dst,oh,ow,OW,sum[0]);
            STORE_1_LINE_4_RESULT(dst,(oh+1),ow,OW,sum[1]);
        }
        if (ow < OW) {
            size_t remain = OW - ow;
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            const int8_t* __restrict sptr5 = sptr4 + IW * 4;
            
            int16x8_t sum[2][2];
            int8x16_t src[2][5];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
            LOAD_1_LINE_SRC(sptr0,src[0]);
            LOAD_1_LINE_SRC(sptr1,src[1]);

            CALC_ONE_LINE_4_RESULT(sum[0],src[0],0,1,2,3,4);
            LOAD_1_LINE_SRC(sptr2,src[0]);//line2
            CALC_ONE_LINE_4_RESULT(sum[0],src[1],5,6,7,8,9);//line1
            CALC_ONE_LINE_4_RESULT(sum[1],src[1],0,1,2,3,4);//line1
            LOAD_1_LINE_SRC(sptr3,src[1]);//line3
            CALC_ONE_LINE_4_RESULT(sum[0],src[0],10,11,12,13,14);//line2
            CALC_ONE_LINE_4_RESULT(sum[1],src[0],5,6,7,8,9);//line2
            LOAD_1_LINE_SRC(sptr4,src[0]);//line4
            CALC_ONE_LINE_4_RESULT(sum[0],src[1],15,16,17,18,19);//line3
            CALC_ONE_LINE_4_RESULT(sum[1],src[1],10,11,12,13,14);//line3
            LOAD_1_LINE_SRC(sptr5,src[1]);//line5
            CALC_ONE_LINE_4_RESULT(sum[0],src[0],20,21,22,23,24);//line4
            CALC_ONE_LINE_4_RESULT(sum[1],src[0],15,16,17,18,19);//line3
            CALC_ONE_LINE_4_RESULT(sum[1],src[1],20,21,22,23,24);//line3

            STORE_REMAIN(dst,oh,ow,OW,sum[0],remain);
            STORE_REMAIN(dst,(oh+1),ow,OW,sum[1],remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;

            int16x8_t sum[4];
            int8x16_t src[2][9];
#define cb(j) sum[j] = init_sum;

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_1_LINE_10_SRC(sptr0,src[0]);
            LOAD_1_LINE_10_SRC(sptr1,src[1]);

            CALC_ONE_LINE_8_RESULT(sum,src[0],0,1,2,3,4);
            LOAD_1_LINE_10_SRC(sptr2,src[0]);//line2
            CALC_ONE_LINE_8_RESULT(sum,src[1],5,6,7,8,9);//line1
            LOAD_1_LINE_10_SRC(sptr3,src[1]);//line3
            CALC_ONE_LINE_8_RESULT(sum,src[0],10,11,12,13,14);//line2
            LOAD_1_LINE_10_SRC(sptr4,src[0]);//line4
            CALC_ONE_LINE_8_RESULT(sum,src[1],15,16,17,18,19);//line3
            CALC_ONE_LINE_8_RESULT(sum,src[0],20,21,22,23,24);//line4

            STORE_1_LINE_RESULT(dst,oh,ow,OW,sum);
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;

            int16x8_t sum[2];
            int8x16_t src[2][5];
            sum[0]=init_sum;
            sum[1]=init_sum;

            
            LOAD_1_LINE_SRC(sptr0,src[0]);
            LOAD_1_LINE_SRC(sptr1,src[1]);

            CALC_ONE_LINE_4_RESULT(sum,src[0],0,1,2,3,4);
            LOAD_1_LINE_SRC(sptr2,src[0]);//line2
            CALC_ONE_LINE_4_RESULT(sum,src[1],5,6,7,8,9);//line1
            LOAD_1_LINE_SRC(sptr3,src[1]);//line3
            CALC_ONE_LINE_4_RESULT(sum,src[0],10,11,12,13,14);//line2
            LOAD_1_LINE_SRC(sptr4,src[0]);//line4
            CALC_ONE_LINE_4_RESULT(sum,src[1],15,16,17,18,19);//line3
            CALC_ONE_LINE_4_RESULT(sum,src[0],20,21,22,23,24);//line4

            STORE_1_LINE_4_RESULT(dst,oh,ow,OW,sum);
            }
        if (ow < OW) {
            size_t remain = OW - ow;
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            int16x8_t sum[2];
            int8x16_t src[2][5];
            sum[0]=init_sum;
            sum[1]=init_sum;
            
            LOAD_1_LINE_SRC(sptr0,src[0]);
            LOAD_1_LINE_SRC(sptr1,src[1]);

            CALC_ONE_LINE_4_RESULT(sum,src[0],0,1,2,3,4);
            LOAD_1_LINE_SRC(sptr2,src[0]);//line2
            CALC_ONE_LINE_4_RESULT(sum,src[1],5,6,7,8,9);//line1
            LOAD_1_LINE_SRC(sptr3,src[1]);//line3
            CALC_ONE_LINE_4_RESULT(sum,src[0],10,11,12,13,14);//line2
            LOAD_1_LINE_SRC(sptr4,src[0]);//line4
            CALC_ONE_LINE_4_RESULT(sum,src[1],15,16,17,18,19);//line3
            CALC_ONE_LINE_4_RESULT(sum,src[0],20,21,22,23,24);//line4
            STORE_REMAIN(dst,oh,ow,OW,sum,remain);
        }
    }
#undef LOAD_1_LINE_SRC
#undef LOAD_1_LINE_10_SRC
#undef CALC_ONE_LINE_4_RESULT
#undef CALC_ONE_LINE_8_RESULT
}

template <BiasMode bias_mode>
void channel_wise_nchw44_8x8x16::direct_stride2_2x2_int8x8x16(
        const int8_t* src, const int8_t* filter, const int16_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW) {
    MEGDNN_MARK_USED_VAR(IH);
    const int16_t* __restrict bptr = bias;
    INIT_SUM();
const int* fptr = reinterpret_cast<const int*>(filter);
    int8x8_t kern[4];
#define cb(i) kern[i] = vreinterpret_s8_s32(vld1_dup_s32(fptr + i));
    UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define CALC_ONE_LINE_8_RESULT(_sum, _rowid, _kid0, _kid1)           \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##0), kern[_kid0]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##0), kern[_kid0]); \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(row##_rowid##2), kern[_kid0]);  \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(row##_rowid##2), kern[_kid0]); \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##1), kern[_kid1]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##1), kern[_kid1]); \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(row##_rowid##3), kern[_kid1]);  \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(row##_rowid##3), kern[_kid1]);

#define CALC_ONE_LINE_4_RESULT(_sum, _rowid, _kid0, _kid1)           \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##0), kern[_kid0]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##0), kern[_kid0]); \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##1), kern[_kid1]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##1), kern[_kid1]); 

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;

            int16x8_t sum[2][4];
#define cb(i)             \
    sum[0][i] = init_sum; \
    sum[1][i] = init_sum;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
#define cb(i)\
const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(4,cb)
#undef cb

#define cb(i)\
           int32x4x2_t tmp_row##i##_00 = vld2q_s32(tmp_sptr##i);\
           int32x4x2_t tmp_row##i##_01 = vld2q_s32(tmp_sptr##i+8);

            UNROLL_CALL_NOWRAPPER(4,cb)
#undef cb

#define cb(i)\
            int8x16_t row##i##0 =vreinterpretq_s8_s32(tmp_row##i##_00.val[0]);\
            int8x16_t row##i##1 =vreinterpretq_s8_s32(tmp_row##i##_00.val[1]);\
            int8x16_t row##i##2 =vreinterpretq_s8_s32(tmp_row##i##_01.val[0]);\
            int8x16_t row##i##3 =vreinterpretq_s8_s32(tmp_row##i##_01.val[1]);

            UNROLL_CALL_NOWRAPPER(4,cb)
#undef cb

            CALC_ONE_LINE_8_RESULT(sum[0],0,0,1);
            CALC_ONE_LINE_8_RESULT(sum[0],1,2,3);
            CALC_ONE_LINE_8_RESULT(sum[1],2,0,1);
            CALC_ONE_LINE_8_RESULT(sum[1],3,2,3);
            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            int16x8_t sum[2][2];
#define cb(i)             \
    sum[0][i] = init_sum; \
    sum[1][i] = init_sum;
            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i)\
            int32x4x2_t tmp_row##i = vld2q_s32(tmp_sptr##i);

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i)\
           int8x16_t row##i##0 =vreinterpretq_s8_s32(tmp_row##i.val[0]);\
           int8x16_t row##i##1 =vreinterpretq_s8_s32(tmp_row##i.val[1]);\

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            CALC_ONE_LINE_4_RESULT(sum[0],0,0,1);
            CALC_ONE_LINE_4_RESULT(sum[0],1,2,3);

            CALC_ONE_LINE_4_RESULT(sum[1],2,0,1);
            CALC_ONE_LINE_4_RESULT(sum[1],3,2,3);
            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_4_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;

            int16x8_t sum[2][2];
#define cb(i)             \
    sum[0][i] = init_sum; \
    sum[1][i] = init_sum;
            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i) int32x4x2_t tmp_row##i = vld2q_s32(tmp_sptr##i);

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i)\
           int8x16_t row##i##0 =vreinterpretq_s8_s32(tmp_row##i.val[0]);\
           int8x16_t row##i##1 =vreinterpretq_s8_s32(tmp_row##i.val[1]);\

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            CALC_ONE_LINE_4_RESULT(sum[0],0,0,1);
            CALC_ONE_LINE_4_RESULT(sum[0],1,2,3);

            CALC_ONE_LINE_4_RESULT(sum[1],2,0,1);
            CALC_ONE_LINE_4_RESULT(sum[1],3,2,3);

            STORE_REMAIN(dst, (oh+0), ow, OW, sum[0], remain);
            STORE_REMAIN(dst, (oh+1), ow, OW, sum[1], remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            int16x8_t sum[4] = {init_sum, init_sum, init_sum, init_sum};
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(2, cb)
#undef cb

#define cb(i)                                             \
    int32x4x2_t tmp_row##i##_00 = vld2q_s32(tmp_sptr##i); \
    int32x4x2_t tmp_row##i##_01 = vld2q_s32(tmp_sptr##i + 8);

            UNROLL_CALL_NOWRAPPER(2, cb)
#undef cb

#define cb(i)                                                           \
    int8x16_t row##i##0 = vreinterpretq_s8_s32(tmp_row##i##_00.val[0]); \
    int8x16_t row##i##1 = vreinterpretq_s8_s32(tmp_row##i##_00.val[1]); \
    int8x16_t row##i##2 = vreinterpretq_s8_s32(tmp_row##i##_01.val[0]); \
    int8x16_t row##i##3 = vreinterpretq_s8_s32(tmp_row##i##_01.val[1]);

            UNROLL_CALL_NOWRAPPER(2, cb)
#undef cb

            CALC_ONE_LINE_8_RESULT(sum, 0, 0, 1);
            CALC_ONE_LINE_8_RESULT(sum, 1, 2, 3);
            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum);
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            int16x8_t sum[2]={init_sum,init_sum};
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

#define cb(i) int32x4x2_t tmp_row##i = vld2q_s32(tmp_sptr##i);

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

#define cb(i)\
           int8x16_t row##i##0 =vreinterpretq_s8_s32(tmp_row##i.val[0]);\
           int8x16_t row##i##1 =vreinterpretq_s8_s32(tmp_row##i.val[1]);

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
            CALC_ONE_LINE_4_RESULT(sum,0,0,1);
            CALC_ONE_LINE_4_RESULT(sum,1,2,3);

            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum);
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            int16x8_t sum[2]={init_sum,init_sum};
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

#define cb(i) int32x4x2_t tmp_row##i = vld2q_s32(tmp_sptr##i);

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

#define cb(i)\
           int8x16_t row##i##0 =vreinterpretq_s8_s32(tmp_row##i.val[0]);\
           int8x16_t row##i##1 =vreinterpretq_s8_s32(tmp_row##i.val[1]);

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
            CALC_ONE_LINE_4_RESULT(sum,0,0,1);
            CALC_ONE_LINE_4_RESULT(sum,1,2,3);
            STORE_REMAIN(dst, oh, ow, OW, sum, remain);
        }
    }
#undef CALC_ONE_LINE_4_RESULT
#undef CALC_ONE_LINE_8_RESULT
}

template <BiasMode bias_mode>
void channel_wise_nchw44_8x8x16::direct_stride2_3x3_int8x8x16(
        const int8_t* src, const int8_t* filter, const int16_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW) {
    MEGDNN_MARK_USED_VAR(IH);

    const int16_t* __restrict bptr = bias;
    INIT_SUM();

    const int* fptr = reinterpret_cast<const int*>(filter);
    int8x8_t kern[9];
#define cb(i) kern[i] = vreinterpret_s8_s32(vld1_dup_s32(fptr + i));
    UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb
#define CALC_ONE_LINE_8_RESULT(_sum, _rowid, _kid0, _kid1, _kid2)           \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##0), kern[_kid0]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##0), kern[_kid0]); \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(row##_rowid##3), kern[_kid0]);  \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(row##_rowid##3), kern[_kid0]); \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##1), kern[_kid1]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##1), kern[_kid1]); \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(row##_rowid##4), kern[_kid1]);  \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(row##_rowid##4), kern[_kid1]); \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##2), kern[_kid2]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##2), kern[_kid2]); \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(row##_rowid##5), kern[_kid2]);  \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(row##_rowid##5), kern[_kid2]);

#define CALC_ONE_LINE_4_RESULT(_sum, _rowid, _kid0, _kid1, _kid2)           \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##0), kern[_kid0]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##0), kern[_kid0]); \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##1), kern[_kid1]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##1), kern[_kid1]); \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(row##_rowid##2), kern[_kid2]);  \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(row##_rowid##2), kern[_kid2]);

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int16x8_t sum[2][4];

#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i)                                                 \
    int32x4x2_t tmp_row##i##_00 = vld2q_s32(tmp_sptr##i);     \
    int32x4x2_t tmp_row##i##_03 = vld2q_s32(tmp_sptr##i + 8); \
    int32x4_t tmp_row##i = vld1q_s32(tmp_sptr##i + 16);

            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i)                                                              \
    int8x16_t row##i##0 = vreinterpretq_s8_s32(tmp_row##i##_00.val[0]);    \
    int8x16_t row##i##1 = vreinterpretq_s8_s32(tmp_row##i##_00.val[1]);    \
    int8x16_t row##i##2 = vreinterpretq_s8_s32(                            \
            vextq_s32(tmp_row##i##_00.val[0], tmp_row##i##_03.val[0], 1)); \
    int8x16_t row##i##3 = vreinterpretq_s8_s32(tmp_row##i##_03.val[0]);    \
    int8x16_t row##i##4 = vreinterpretq_s8_s32(tmp_row##i##_03.val[1]);    \
    int8x16_t row##i##5 = vreinterpretq_s8_s32(                            \
            vextq_s32(tmp_row##i##_03.val[0], tmp_row##i, 1));

            UNROLL_CALL_NOWRAPPER(5, cb)
#undef cb

            CALC_ONE_LINE_8_RESULT(sum[0], 0, 0, 1, 2);
            CALC_ONE_LINE_8_RESULT(sum[0], 1, 3, 4, 5);
            CALC_ONE_LINE_8_RESULT(sum[0], 2, 6, 7, 8);
            CALC_ONE_LINE_8_RESULT(sum[1], 2, 0, 1, 2);
            CALC_ONE_LINE_8_RESULT(sum[1], 3, 3, 4, 5);
            CALC_ONE_LINE_8_RESULT(sum[1], 4, 6, 7, 8);

            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;

            int16x8_t sum[2][2];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i)                                            \
    int32x4x2_t tmp_row##i##_0 = vld2q_s32(tmp_sptr##i); \
    int32x4_t tmp_row##i##_1 = vld1q_s32(tmp_sptr##i + 8);

            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i)                                               \
    int8x16_t row##i##0 = vreinterpretq_s8_s32(tmp_row##i##_0.val[0]); \
    int8x16_t row##i##1 = vreinterpretq_s8_s32(tmp_row##i##_0.val[1]); \
    int8x16_t row##i##2 =                                   \
            vreinterpretq_s8_s32(vextq_s32(tmp_row##i##_0.val[0], tmp_row##i##_1, 1));

            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb
            CALC_ONE_LINE_4_RESULT(sum[0], 0, 0, 1, 2);
            CALC_ONE_LINE_4_RESULT(sum[0], 1, 3, 4, 5);
            CALC_ONE_LINE_4_RESULT(sum[0], 2, 6, 7, 8);
            CALC_ONE_LINE_4_RESULT(sum[1], 2, 0, 1, 2);
            CALC_ONE_LINE_4_RESULT(sum[1], 3, 3, 4, 5);
            CALC_ONE_LINE_4_RESULT(sum[1], 4, 6, 7, 8);
            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_4_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;

            int16x8_t sum[2][2];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i)                                            \
    int32x4x2_t tmp_row##i##_0 = vld2q_s32(tmp_sptr##i); \
    int32x4_t tmp_row##i##_1 = vld1q_s32(tmp_sptr##i + 8);

            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i)                                               \
    int8x16_t row##i##0 = vreinterpretq_s8_s32(tmp_row##i##_0.val[0]); \
    int8x16_t row##i##1 = vreinterpretq_s8_s32(tmp_row##i##_0.val[1]); \
    int8x16_t row##i##2 =                                   \
            vreinterpretq_s8_s32(vextq_s32(tmp_row##i##_0.val[0], tmp_row##i##_1, 1));

            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb
            CALC_ONE_LINE_4_RESULT(sum[0], 0, 0, 1, 2);
            CALC_ONE_LINE_4_RESULT(sum[0], 1, 3, 4, 5);
            CALC_ONE_LINE_4_RESULT(sum[0], 2, 6, 7, 8);
            CALC_ONE_LINE_4_RESULT(sum[1], 2, 0, 1, 2);
            CALC_ONE_LINE_4_RESULT(sum[1], 3, 3, 4, 5);
            CALC_ONE_LINE_4_RESULT(sum[1], 4, 6, 7, 8);

            STORE_REMAIN(dst, (oh + 0), ow, OW, sum[0], remain);
            STORE_REMAIN(dst, (oh + 1), ow, OW, sum[1], remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
#if MEGDNN_AARCH64
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;

            int16x8_t sum[4] = {init_sum, init_sum, init_sum, init_sum};
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(3, cb);
#undef cb

#define cb(i)                                                 \
    int32x4x2_t tmp_row##i##_00 = vld2q_s32(tmp_sptr##i);     \
    int32x4x2_t tmp_row##i##_03 = vld2q_s32(tmp_sptr##i + 8); \
    int32x4_t tmp_row##i = vld1q_s32(tmp_sptr##i + 16);

            UNROLL_CALL_NOWRAPPER(3, cb);
#undef cb

#define cb(i)                                                              \
    int8x16_t row##i##0 = vreinterpretq_s8_s32(tmp_row##i##_00.val[0]);     \
    int8x16_t row##i##1 = vreinterpretq_s8_s32(tmp_row##i##_00.val[1]);     \
    int8x16_t row##i##2 = vreinterpretq_s8_s32(                             \
            vextq_s32(tmp_row##i##_00.val[0], tmp_row##i##_03.val[0], 1)); \
    int8x16_t row##i##3 = vreinterpretq_s8_s32(tmp_row##i##_03.val[0]);    \
    int8x16_t row##i##4 = vreinterpretq_s8_s32(tmp_row##i##_03.val[1]);    \
    int8x16_t row##i##5 = vreinterpretq_s8_s32(                            \
            vextq_s32(tmp_row##i##_03.val[0], tmp_row##i, 1));

            UNROLL_CALL_NOWRAPPER(3, cb)
#undef cb

            CALC_ONE_LINE_8_RESULT(sum, 0, 0, 1, 2);
            CALC_ONE_LINE_8_RESULT(sum, 1, 3, 4, 5);
            CALC_ONE_LINE_8_RESULT(sum, 2, 6, 7, 8);
            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum);
        }
#endif
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int16x8_t sum[2] = {init_sum, init_sum};
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(3, cb)
#undef cb

#define cb(i)                                            \
    int32x4x2_t tmp_row##i##_0 = vld2q_s32(tmp_sptr##i); \
    int32x4_t tmp_row##i##_1 = vld1q_s32(tmp_sptr##i + 8);

            UNROLL_CALL_NOWRAPPER(3, cb)
#undef cb

#define cb(i)                                               \
    int8x16_t row##i##0 = vreinterpretq_s8_s32(tmp_row##i##_0.val[0]); \
    int8x16_t row##i##1 = vreinterpretq_s8_s32(tmp_row##i##_0.val[1]); \
    int8x16_t row##i##2 =                                   \
            vreinterpretq_s8_s32(vextq_s32(tmp_row##i##_0.val[0], tmp_row##i##_1, 1));

            UNROLL_CALL_NOWRAPPER(3, cb)
#undef cb

            CALC_ONE_LINE_4_RESULT(sum, 0, 0, 1, 2);
            CALC_ONE_LINE_4_RESULT(sum, 1, 3, 4, 5);
            CALC_ONE_LINE_4_RESULT(sum, 2, 6, 7, 8);
            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum);
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int16x8_t sum[2] = {init_sum, init_sum};
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);

            UNROLL_CALL_NOWRAPPER(3, cb)
#undef cb

#define cb(i)                                            \
    int32x4x2_t tmp_row##i##_0 = vld2q_s32(tmp_sptr##i); \
    int32x4_t tmp_row##i##_1 = vld1q_s32(tmp_sptr##i + 8);

            UNROLL_CALL_NOWRAPPER(3, cb)
#undef cb

#define cb(i)                                               \
    int8x16_t row##i##0 = vreinterpretq_s8_s32(tmp_row##i##_0.val[0]); \
    int8x16_t row##i##1 = vreinterpretq_s8_s32(tmp_row##i##_0.val[1]); \
    int8x16_t row##i##2 =                                   \
            vreinterpretq_s8_s32(vextq_s32(tmp_row##i##_0.val[0], tmp_row##i##_1, 1));

            UNROLL_CALL_NOWRAPPER(3, cb)
#undef cb

            CALC_ONE_LINE_4_RESULT(sum, 0, 0, 1, 2);
            CALC_ONE_LINE_4_RESULT(sum, 1, 3, 4, 5);
            CALC_ONE_LINE_4_RESULT(sum, 2, 6, 7, 8);

            STORE_REMAIN(dst, oh, ow, OW, sum, remain);
        }
    }
#undef CALC_ONE_LINE_4_RESULT
#undef CALC_ONE_LINE_8_RESULT
#undef LOAD_5_SRC
}

#if MEGDNN_AARCH64

template <BiasMode bias_mode>
void channel_wise_nchw44_8x8x16::direct_stride2_5x5_int8x8x16(
        const int8_t* src, const int8_t* filter, const int16_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW) {
    MEGDNN_MARK_USED_VAR(IH);
    const int16_t* __restrict bptr = bias;
    INIT_SUM();

    const int* fptr = reinterpret_cast<const int*>(filter);
    int8x8_t kern[25];
#define cb(i) kern[i] = vreinterpret_s8_s32(vld1_dup_s32(fptr + i));
    UNROLL_CALL_NOWRAPPER(25, cb);
#undef cb

#define LOAD_5_SRC(_src, _id)                                  \
    do {                                                       \
        int32x4x2_t tmp_row_01 = vld2q_s32(tmp_sptr##_id);     \
        int32x4x2_t tmp_row_23 = vld2q_s32(tmp_sptr##_id + 2); \
        int32x4_t tmp_row = vld1q_s32(tmp_sptr##_id + 10);     \
        _src[0] = vreinterpretq_s8_s32(tmp_row_01.val[0]);     \
        _src[1] = vreinterpretq_s8_s32(tmp_row_01.val[1]);     \
        _src[2] = vreinterpretq_s8_s32(tmp_row_23.val[0]);     \
        _src[3] = vreinterpretq_s8_s32(tmp_row_23.val[1]);     \
        _src[4] = vreinterpretq_s8_s32(                        \
                vextq_s32(tmp_row_23.val[0], tmp_row, 1));     \
    } while (0);

#define CALC_ONE_LINE_4_RESULT(_sum, _src, _kid0, _kid1, _kid2, _kid3,     \
                                   _kid4)                                  \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[0]), kern[_kid0]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[0]), kern[_kid0]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[1]), kern[_kid1]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[1]), kern[_kid1]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[2]), kern[_kid2]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[2]), kern[_kid2]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[3]), kern[_kid3]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[3]), kern[_kid3]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[4]), kern[_kid4]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[4]), kern[_kid4]);

#define LOAD_10_SRC(_src, _id)                                       \
    do {                                                             \
        int32x4x2_t tmp_row_01 = vld2q_s32(tmp_sptr##_id);           \
        int32x4x2_t tmp_row_23 = vld2q_s32(tmp_sptr##_id + 8);       \
        int32x4x2_t tmp_row = vld2q_s32(tmp_sptr##_id + 16);         \
        _src[0] = vreinterpretq_s8_s32(tmp_row_01.val[0]);           \
        _src[1] = vreinterpretq_s8_s32(tmp_row_01.val[1]);           \
        _src[2] = vreinterpretq_s8_s32(                              \
                vextq_s32(tmp_row_01.val[0], tmp_row_23.val[0], 1)); \
        _src[3] = vreinterpretq_s8_s32(                              \
                vextq_s32(tmp_row_01.val[1], tmp_row_23.val[1], 1)); \
        _src[4] = vreinterpretq_s8_s32(                              \
                vextq_s32(tmp_row_01.val[0], tmp_row_23.val[0], 2)); \
        _src[5] = vreinterpretq_s8_s32(tmp_row_23.val[0]);           \
        _src[6] = vreinterpretq_s8_s32(tmp_row_23.val[1]);           \
        _src[7] = vreinterpretq_s8_s32(                              \
                vextq_s32(tmp_row_23.val[0], tmp_row.val[0], 1));    \
        _src[8] = vreinterpretq_s8_s32(                              \
                vextq_s32(tmp_row_23.val[1], tmp_row.val[1], 1));    \
        _src[9] = vreinterpretq_s8_s32(                              \
                vextq_s32(tmp_row_23.val[0], tmp_row.val[0], 2));    \
    } while (0);

#define CALC_ONE_LINE_8_RESULT(_sum, _src, _kid0, _kid1, _kid2, _kid3,     \
                                   _kid4)                                  \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[0]), kern[_kid0]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[0]), kern[_kid0]);       \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(_src[5]), kern[_kid0]);        \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(_src[5]), kern[_kid0]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[1]), kern[_kid1]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[1]), kern[_kid1]);       \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(_src[6]), kern[_kid1]);        \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(_src[6]), kern[_kid1]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[2]), kern[_kid2]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[2]), kern[_kid2]);       \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(_src[7]), kern[_kid2]);        \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(_src[7]), kern[_kid2]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[3]), kern[_kid3]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[3]), kern[_kid3]);       \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(_src[8]), kern[_kid3]);        \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(_src[8]), kern[_kid3]);       \
    _sum[0] = vmlal_s8(_sum[0], vget_low_s8(_src[4]), kern[_kid4]);        \
    _sum[1] = vmlal_s8(_sum[1], vget_high_s8(_src[4]), kern[_kid4]);       \
    _sum[2] = vmlal_s8(_sum[2], vget_low_s8(_src[9]), kern[_kid4]);        \
    _sum[3] = vmlal_s8(_sum[3], vget_high_s8(_src[9]), kern[_kid4]);

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr5 = src + (ih + 5) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr6 = src + (ih + 6) * IW * 4 + iw * 4;
            int16x8_t sum[2][4];
            int8x16_t src[3][10];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(7, cb);
#undef cb

            LOAD_10_SRC(src[0], 0);  // line0
            LOAD_10_SRC(src[1], 1);  // line1
            CALC_ONE_LINE_8_RESULT(sum[0], src[0], 0, 1, 2, 3, 4);
            LOAD_10_SRC(src[2], 2);  // line2
            CALC_ONE_LINE_8_RESULT(sum[0], src[1], 5, 6, 7, 8, 9);
            LOAD_10_SRC(src[0], 3);  // line3
            CALC_ONE_LINE_8_RESULT(sum[0], src[2], 10, 11, 12, 13, 14);
            CALC_ONE_LINE_8_RESULT(sum[1], src[2], 0, 1, 2, 3, 4);
            LOAD_10_SRC(src[1], 4);  // line4
            CALC_ONE_LINE_8_RESULT(sum[0], src[0], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_8_RESULT(sum[0], src[1], 20, 21, 22, 23, 24);
            LOAD_10_SRC(src[2], 5);  // line5
            CALC_ONE_LINE_8_RESULT(sum[1], src[0], 5, 6, 7, 8, 9);
            CALC_ONE_LINE_8_RESULT(sum[1], src[1], 10, 11, 12, 13, 14);
            LOAD_10_SRC(src[0], 6);  // line6
            CALC_ONE_LINE_8_RESULT(sum[1], src[2], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_8_RESULT(sum[1], src[0], 20, 21, 22, 23, 24);

            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr5 = src + (ih + 5) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr6 = src + (ih + 6) * IW * 4 + iw * 4;
            int16x8_t sum[2][2];
            int8x16_t src[3][5];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(7, cb);
#undef cb

            LOAD_5_SRC(src[0], 0);  // line0
            LOAD_5_SRC(src[1], 1);  // line1
            CALC_ONE_LINE_4_RESULT(sum[0], src[0], 0, 1, 2, 3, 4);
            LOAD_5_SRC(src[2], 2);  // line2
            CALC_ONE_LINE_4_RESULT(sum[0], src[1], 5, 6, 7, 8, 9);
            LOAD_5_SRC(src[0], 3);  // line3
            CALC_ONE_LINE_4_RESULT(sum[0], src[2], 10, 11, 12, 13, 14);
            CALC_ONE_LINE_4_RESULT(sum[1], src[2], 0, 1, 2, 3, 4);
            LOAD_5_SRC(src[1], 4);  // line4
            CALC_ONE_LINE_4_RESULT(sum[0], src[0], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_4_RESULT(sum[1], src[0], 5, 6, 7, 8, 9);
            LOAD_5_SRC(src[2], 5);  // line5
            CALC_ONE_LINE_4_RESULT(sum[0], src[1], 20, 21, 22, 23, 24);
            CALC_ONE_LINE_4_RESULT(sum[1], src[1], 10, 11, 12, 13, 14);
            LOAD_5_SRC(src[0], 6);  // line6
            CALC_ONE_LINE_4_RESULT(sum[1], src[2], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_4_RESULT(sum[1], src[0], 20, 21, 22, 23, 24);

            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_4_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr5 = src + (ih + 5) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr6 = src + (ih + 6) * IW * 4 + iw * 4;
            int16x8_t sum[2][2];
            int8x16_t src[3][5];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(7, cb);
#undef cb
            LOAD_5_SRC(src[0], 0);  // line0
            LOAD_5_SRC(src[1], 1);  // line1
            CALC_ONE_LINE_4_RESULT(sum[0], src[0], 0, 1, 2, 3, 4);
            LOAD_5_SRC(src[2], 2);  // line2
            CALC_ONE_LINE_4_RESULT(sum[0], src[1], 5, 6, 7, 8, 9);
            LOAD_5_SRC(src[0], 3);  // line3
            CALC_ONE_LINE_4_RESULT(sum[0], src[2], 10, 11, 12, 13, 14);
            CALC_ONE_LINE_4_RESULT(sum[1], src[2], 0, 1, 2, 3, 4);
            LOAD_5_SRC(src[1], 4);  // line4
            CALC_ONE_LINE_4_RESULT(sum[0], src[0], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_4_RESULT(sum[1], src[0], 5, 6, 7, 8, 9);
            LOAD_5_SRC(src[2], 5);  // line5
            CALC_ONE_LINE_4_RESULT(sum[0], src[1], 20, 21, 22, 23, 24);
            CALC_ONE_LINE_4_RESULT(sum[1], src[1], 10, 11, 12, 13, 14);
            LOAD_5_SRC(src[0], 6);  // line6
            CALC_ONE_LINE_4_RESULT(sum[1], src[2], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_4_RESULT(sum[1], src[0], 20, 21, 22, 23, 24);

            STORE_REMAIN(dst, oh, ow, OW, sum[0], remain);
            STORE_REMAIN(dst, (oh + 1), ow, OW, sum[1], remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 8 <= OW; ow += 8) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int16x8_t sum[4] = {init_sum, init_sum, init_sum, init_sum};
            int8x16_t src[3][10];
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb
            LOAD_10_SRC(src[0], 0);  // line0
            LOAD_10_SRC(src[1], 1);  // line1
            CALC_ONE_LINE_8_RESULT(sum, src[0], 0, 1, 2, 3, 4);
            LOAD_10_SRC(src[2], 2);  // line2
            CALC_ONE_LINE_8_RESULT(sum, src[1], 5, 6, 7, 8, 9);
            LOAD_10_SRC(src[0], 3);  // line3
            CALC_ONE_LINE_8_RESULT(sum, src[2], 10, 11, 12, 13, 14);
            LOAD_10_SRC(src[1], 4);  // line4
            CALC_ONE_LINE_8_RESULT(sum, src[0], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_8_RESULT(sum, src[1], 20, 21, 22, 23, 24);

            STORE_1_LINE_RESULT(dst, oh, ow, OW, sum);
        }
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;

            int16x8_t sum[2] = {init_sum, init_sum};
#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

            int8x16_t src[3][5];
            LOAD_5_SRC(src[0], 0);  // line0
            LOAD_5_SRC(src[1], 1);  // line1
            CALC_ONE_LINE_4_RESULT(sum, src[0], 0, 1, 2, 3, 4);
            LOAD_5_SRC(src[2], 2);  // line2
            CALC_ONE_LINE_4_RESULT(sum, src[1], 5, 6, 7, 8, 9);
            LOAD_5_SRC(src[0], 3);  // line3
            CALC_ONE_LINE_4_RESULT(sum, src[2], 10, 11, 12, 13, 14);
            LOAD_5_SRC(src[1], 4);  // line4
            CALC_ONE_LINE_4_RESULT(sum, src[0], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_4_RESULT(sum, src[1], 20, 21, 22, 23, 24);

            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum);
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int16x8_t sum[2] = {init_sum, init_sum};

#define cb(i) \
    const int32_t* tmp_sptr##i = reinterpret_cast<const int32_t*>(sptr##i);
            UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

            int8x16_t src[3][5];
            LOAD_5_SRC(src[0], 0);  // line0
            LOAD_5_SRC(src[1], 1);  // line1
            CALC_ONE_LINE_4_RESULT(sum, src[0], 0, 1, 2, 3, 4);
            LOAD_5_SRC(src[2], 2);  // line2
            CALC_ONE_LINE_4_RESULT(sum, src[1], 5, 6, 7, 8, 9);
            LOAD_5_SRC(src[0], 3);  // line3
            CALC_ONE_LINE_4_RESULT(sum, src[2], 10, 11, 12, 13, 14);
            LOAD_5_SRC(src[1], 4);  // line4
            CALC_ONE_LINE_4_RESULT(sum, src[0], 15, 16, 17, 18, 19);
            CALC_ONE_LINE_4_RESULT(sum, src[1], 20, 21, 22, 23, 24);

            STORE_REMAIN(dst, oh, ow, OW, sum, remain);
        }
    }
}
#undef CALC_ONE_LINE_8_RESULT
#undef CALC_ONE_LINE_4_RESULT
#undef LOAD_10_SRC
#undef LOAD_5_SRC
#elif MEGDNN_ARMV7
template <BiasMode bias_mode>
void channel_wise_nchw44_8x8x16::direct_stride2_5x5_int8x8x16(
        const int8_t* src, const int8_t* filter, const int16_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW) {
    MEGDNN_MARK_USED_VAR(IH);
    const int16_t* __restrict bptr = bias;
    const int32_t* tmp_filter = reinterpret_cast<const int32_t*>(filter);
    INIT_SUM();
    int8x8_t kern0[3], kern1[3], kern2[3], kern3[3], kern4[3];
    
    int32x2_t tmp_kern = vdup_n_s32(tmp_filter[4]);
    tmp_kern = vset_lane_s32(0,tmp_kern,1);
    kern0[0] = vld1_s8(filter);
    kern0[1] = vld1_s8(filter + 8);
    kern0[2] = vreinterpret_s8_s32(tmp_kern);
    
    tmp_kern = vdup_n_s32(tmp_filter[9]);
    tmp_kern = vset_lane_s32(0,tmp_kern,1);
    kern1[0] = vld1_s8(filter + 20);
    kern1[1] = vld1_s8(filter + 28);
    kern1[2] = vreinterpret_s8_s32(tmp_kern);
    
    tmp_kern = vdup_n_s32(tmp_filter[14]);
    tmp_kern = vset_lane_s32(0,tmp_kern,1);
    kern2[0] = vld1_s8(filter + 40);
    kern2[1] = vld1_s8(filter + 48);
    kern2[2] = vreinterpret_s8_s32(tmp_kern);
    
    tmp_kern = vdup_n_s32(tmp_filter[19]);
    tmp_kern = vset_lane_s32(0,tmp_kern,1);
    kern3[0] = vld1_s8(filter + 60);
    kern3[1] = vld1_s8(filter + 68);
    kern3[2] = vreinterpret_s8_s32(tmp_kern);
    
    tmp_kern = vdup_n_s32(tmp_filter[24]);
    tmp_kern = vset_lane_s32(0,tmp_kern,1);
    kern4[0] = vld1_s8(filter + 80);
    kern4[1] = vld1_s8(filter + 88);
    kern4[2] = vreinterpret_s8_s32(tmp_kern);

#define LOAD_3_SRC_ARRAY(_src,_sptr)\
            _src[0] = vld1q_s8(_sptr);/*0 1 2 3  */\
            _src[1] = vld1q_s8(_sptr + 16);/*4 5 6 7 */\
            _src[2] = vld1q_s8(_sptr + 32);/*8 9 10 11*/

#define CALC_ONE_LINE(_src, _kern, _sum)                                 \
    tmpsum0 = vmull_s8(vget_low_s8(_src[0]), _kern[0]);           /*01*/ \
    tmpsum1 = vmull_s8(vget_high_s8(_src[0]), _kern[0]);          /*23*/ \
    tmpsum0 = vmlal_s8(tmpsum0, vget_high_s8(_src[0]), _kern[1]); /*23*/ \
    tmpsum1 = vmlal_s8(tmpsum1, vget_low_s8(_src[1]), _kern[1]);  /*45*/ \
    tmpsum0 = vmlal_s8(tmpsum0, vget_low_s8(_src[1]), _kern[2]);  /*4*/  \
    tmpsum1 = vmlal_s8(tmpsum1, vget_high_s8(_src[1]), _kern[2]); /*6*/  \
    res0 = vadd_s16(vget_low_s16(tmpsum0), vget_high_s16(tmpsum0));      \
    res1 = vadd_s16(vget_low_s16(tmpsum1), vget_high_s16(tmpsum1));      \
    _sum[0] = vaddq_s16(_sum[0], vcombine_s16(res0, res1));              \
                                                                         \
    tmpsum0 = vmull_s8(vget_low_s8(_src[1]), _kern[0]);           /*45*/ \
    tmpsum1 = vmull_s8(vget_high_s8(_src[1]), _kern[0]);          /*67*/ \
    tmpsum0 = vmlal_s8(tmpsum0, vget_high_s8(_src[1]), _kern[1]);  /*67*/ \
    tmpsum1 = vmlal_s8(tmpsum1, vget_low_s8(_src[2]), _kern[1]);  /*89*/ \
    tmpsum0 = vmlal_s8(tmpsum0, vget_low_s8(_src[2]), _kern[2]);  /*8*/  \
    tmpsum1 = vmlal_s8(tmpsum1, vget_high_s8(_src[2]), _kern[2]); /*10*/ \
    res0 = vadd_s16(vget_low_s16(tmpsum0), vget_high_s16(tmpsum0));      \
    res1 = vadd_s16(vget_low_s16(tmpsum1), vget_high_s16(tmpsum1));      \
    _sum[1] = vaddq_s16(_sum[1], vcombine_s16(res0, res1));

#define CALC_8_RESULT()                 \
    LOAD_3_SRC_ARRAY(src0, sptr0);      \
    LOAD_3_SRC_ARRAY(src1, sptr1);      \
    CALC_ONE_LINE(src0, kern0, sum[0]); \
                                        \
    LOAD_3_SRC_ARRAY(src0, sptr2);      \
    CALC_ONE_LINE(src1, kern1, sum[0]); \
                                        \
    LOAD_3_SRC_ARRAY(src1, sptr3);      \
    CALC_ONE_LINE(src0, kern2, sum[0]); \
    CALC_ONE_LINE(src0, kern0, sum[1]); \
                                        \
    LOAD_3_SRC_ARRAY(src0, sptr4);      \
    CALC_ONE_LINE(src1, kern3, sum[0]); \
    CALC_ONE_LINE(src1, kern1, sum[1]); \
                                        \
    LOAD_3_SRC_ARRAY(src1, sptr5);      \
    CALC_ONE_LINE(src0, kern4, sum[0]); \
    CALC_ONE_LINE(src0, kern2, sum[1]); \
                                        \
    LOAD_3_SRC_ARRAY(src0, sptr6);      \
    CALC_ONE_LINE(src1, kern3, sum[1]); \
    CALC_ONE_LINE(src0, kern4, sum[1]);

#define CALC_4_RESULT()              \
    LOAD_3_SRC_ARRAY(src0, sptr0);   \
    LOAD_3_SRC_ARRAY(src1, sptr1);   \
    CALC_ONE_LINE(src0, kern0, sum); \
                                     \
    LOAD_3_SRC_ARRAY(src0, sptr2);   \
    CALC_ONE_LINE(src1, kern1, sum); \
                                     \
    LOAD_3_SRC_ARRAY(src1, sptr3);   \
    CALC_ONE_LINE(src0, kern2, sum); \
                                     \
    LOAD_3_SRC_ARRAY(src0, sptr4);   \
    CALC_ONE_LINE(src1, kern3, sum); \
    CALC_ONE_LINE(src0, kern4, sum);

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr5 = src + (ih + 5) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr6 = src + (ih + 6) * IW * 4 + iw * 4;
            int16x8_t sum[2][2];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
            int8x16_t src0[3], src1[3];
            int16x8_t tmpsum0, tmpsum1;
            int16x4_t res0, res1;
            CALC_8_RESULT();
            STORE_1_LINE_4_RESULT(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE_4_RESULT(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr5 = src + (ih + 5) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr6 = src + (ih + 6) * IW * 4 + iw * 4;
            int16x8_t sum[2][2];
#define cb(j)             \
    sum[0][j] = init_sum; \
    sum[1][j] = init_sum;

            UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb
            int8x16_t src0[3], src1[3];
            int16x8_t tmpsum0, tmpsum1;
            int16x4_t res0, res1;

            CALC_8_RESULT();
            STORE_REMAIN(dst, oh, ow, OW, sum[0],remain);
            STORE_REMAIN(dst, (oh + 1), ow, OW, sum[1],remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;

            int16x8_t sum[2]={init_sum,init_sum};

            int8x16_t src0[3], src1[3];
            int16x8_t tmpsum0, tmpsum1;
            int16x4_t res0, res1;
            CALC_4_RESULT();
            STORE_1_LINE_4_RESULT(dst, oh,ow, OW, sum);
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int16x8_t sum[2] = {init_sum, init_sum};
            int8x16_t src0[3], src1[3];
            int16x8_t tmpsum0, tmpsum1;
            int16x4_t res0, res1;
            CALC_4_RESULT();
            STORE_REMAIN(dst, oh, ow, OW, sum, remain);
        }
    }
}
#undef CALC_ONE_LINE
#undef CALC_4_RESULT
#undef CALC_8_RESULT
#undef LOAD_3_SRC_ARRAY
#endif

#undef INIT_SUM
#undef STORE_1_LINE_RESULT
#undef STORE_1_LINE_4_RESULT
#undef STORE_REMAIN

#define INSTANTIATION(stride, i, bias)                                   \
    template void channel_wise_nchw44_8x8x16::                           \
            direct_##stride##_##i##x##i##_int8x8x16<bias>(               \
                    const int8_t*, const int8_t*, const int16_t*, void*, \
                    const size_t, const size_t, const size_t, const size_t);

#define FOR_OP(stride, i, bias) INSTANTIATION(stride, i, bias)

#define FOR_BIAS(stride, i)              \
    FOR_OP(stride, i, BiasMode::NO_BIAS) \
    FOR_OP(stride, i, BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_FILTER(stride) \
    FOR_BIAS(stride, 2)    \
    FOR_BIAS(stride, 3)    \
    FOR_BIAS(stride, 5)

#define FOR_STRIDE      \
    FOR_FILTER(stride1) \
    FOR_FILTER(stride2)

FOR_STRIDE

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_BIAS
#undef FOR_OP
#undef INSTANTIATION

// vim: syntax=cpp.doxygen
