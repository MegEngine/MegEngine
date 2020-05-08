/**
 * \file dnn/src/arm_common/conv_bias/f16/direct.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./direct.h"
#include "include/megdnn/oprs.h"
#include "midout.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <cstring>
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/simd_macro/neon_helper.h"

MIDOUT_DECL(megdnn_arm_conv_f16)

using namespace megdnn;
using namespace arm_common;
using namespace fp16;
using namespace conv_bias;
namespace {

#define BLOCK_H 8

#define LOAD_RESULT_VAL                                         \
    if (width < 8) {                                            \
        auto load_less_8 = [](__fp16* dst, float16x8_t& data) { \
            if (width == 1u) {                                  \
                data = vld1q_lane_f16(dst, data, 0);            \
            } else if (width == 2u) {                           \
                data = vld1q_lane_f16(dst + 0, data, 0);        \
                data = vld1q_lane_f16(dst + 1, data, 1);        \
            } else if (width == 3u) {                           \
                data = vld1q_lane_f16(dst + 0, data, 0);        \
                data = vld1q_lane_f16(dst + 1, data, 1);        \
                data = vld1q_lane_f16(dst + 2, data, 2);        \
            } else if (width == 4u) {                           \
                data = vld1q_lane_u64(dst, data, 0);            \
            } else if (width == 5u) {                           \
                data = vld1q_lane_u64(dst, data, 0);            \
                data = vld1q_lane_f16(dst + 4, data, 4);        \
            } else if (width == 6u) {                           \
                data = vld1q_lane_u64(dst, data, 0);            \
                data = vld1q_lane_f16(dst + 4, data, 4);        \
                data = vld1q_lane_f16(dst + 5, data, 5);        \
            } else if (width == 7u) {                           \
                data = vld1q_lane_u64(dst, data, 0);            \
                data = vld1q_lane_f16(dst + 4, data, 4);        \
                data = vld1q_lane_f16(dst + 5, data, 5);        \
                data = vld1q_lane_f16(dst + 6, data, 6);        \
            }                                                   \
        };                                                      \
        if (height >= 1)                                        \
            load_less_8(dst + 0 * OW, out0);                    \
        if (height >= 2)                                        \
            load_less_8(dst + 1 * OW, out1);                    \
        if (height >= 3)                                        \
            load_less_8(dst + 2 * OW, out2);                    \
        if (height >= 4)                                        \
            load_less_8(dst + 3 * OW, out3);                    \
        if (height >= 5)                                        \
            load_less_8(dst + 4 * OW, out4);                    \
        if (height >= 6)                                        \
            load_less_8(dst + 5 * OW, out5);                    \
        if (height >= 7)                                        \
            load_less_8(dst + 6 * OW, out6);                    \
        if (height >= 8)                                        \
            load_less_8(dst + 7 * OW, out7);                    \
    } else {                                                    \
        if (height >= 1)                                        \
            out0 = vld1q_f16(dst + 0 * OW);                     \
        if (height >= 2)                                        \
            out1 = vld1q_f16(dst + 1 * OW);                     \
        if (height >= 3)                                        \
            out2 = vld1q_f16(dst + 2 * OW);                     \
        if (height >= 4)                                        \
            out3 = vld1q_f16(dst + 3 * OW);                     \
        if (height >= 5)                                        \
            out4 = vld1q_f16(dst + 4 * OW);                     \
        if (height >= 6)                                        \
            out5 = vld1q_f16(dst + 5 * OW);                     \
        if (height >= 7)                                        \
            out6 = vld1q_f16(dst + 6 * OW);                     \
        if (height >= 8)                                        \
            out7 = vld1q_f16(dst + 7 * OW);                     \
    }

#define STORE_RESULT_VAL                                         \
    if (width < 8) {                                             \
        auto store_less_8 = [](__fp16* dst, float16x8_t& data) { \
            if (width == 1u) {                                   \
                vst1q_lane_f16(dst, data, 0);                    \
            } else if (width == 2u) {                            \
                vst1q_lane_f16(dst + 0, data, 0);                \
                vst1q_lane_f16(dst + 1, data, 1);                \
            } else if (width == 3u) {                            \
                vst1q_lane_f16(dst + 0, data, 0);                \
                vst1q_lane_f16(dst + 1, data, 1);                \
                vst1q_lane_f16(dst + 2, data, 2);                \
            } else if (width == 4u) {                            \
                vst1_f16(dst, vget_low_f16(data));               \
            } else if (width == 5u) {                            \
                vst1_f16(dst, vget_low_f16(data));               \
                vst1q_lane_f16(dst + 4, data, 4);                \
            } else if (width == 6u) {                            \
                vst1_f16(dst, vget_low_f16(data));               \
                vst1q_lane_f16(dst + 4, data, 4);                \
                vst1q_lane_f16(dst + 5, data, 5);                \
            } else if (width == 7u) {                            \
                vst1_f16(dst, vget_low_f16(data));               \
                vst1q_lane_f16(dst + 4, data, 4);                \
                vst1q_lane_f16(dst + 5, data, 5);                \
                vst1q_lane_f16(dst + 6, data, 6);                \
            }                                                    \
        };                                                       \
        if (height >= 1)                                         \
            store_less_8(dst + 0 * OW, out0);                    \
        if (height >= 2)                                         \
            store_less_8(dst + 1 * OW, out1);                    \
        if (height >= 3)                                         \
            store_less_8(dst + 2 * OW, out2);                    \
        if (height >= 4)                                         \
            store_less_8(dst + 3 * OW, out3);                    \
        if (height >= 5)                                         \
            store_less_8(dst + 4 * OW, out4);                    \
        if (height >= 6)                                         \
            store_less_8(dst + 5 * OW, out5);                    \
        if (height >= 7)                                         \
            store_less_8(dst + 6 * OW, out6);                    \
        if (height >= 8)                                         \
            store_less_8(dst + 7 * OW, out7);                    \
    } else {                                                     \
        if (height >= 1)                                         \
            vst1q_f16(dst + 0 * OW, out0);                       \
        if (height >= 2)                                         \
            vst1q_f16(dst + 1 * OW, out1);                       \
        if (height >= 3)                                         \
            vst1q_f16(dst + 2 * OW, out2);                       \
        if (height >= 4)                                         \
            vst1q_f16(dst + 3 * OW, out3);                       \
        if (height >= 5)                                         \
            vst1q_f16(dst + 4 * OW, out4);                       \
        if (height >= 6)                                         \
            vst1q_f16(dst + 5 * OW, out5);                       \
        if (height >= 7)                                         \
            vst1q_f16(dst + 6 * OW, out6);                       \
        if (height >= 8)                                         \
            vst1q_f16(dst + 7 * OW, out7);                       \
    }

template <int FH, int height, int width>
struct do_pixel_proxy {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow);
};

template <int height, int width>
struct do_pixel_proxy<1, height, width> {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        const int ih = oh, iw = ow;
#define cb(i) float16x8_t out##i{0};
        UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        float16x8_t kr0, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_RESULT_VAL;
        for (int fw = 0; fw < FW; ++fw) {
            const __fp16* src_dd = src + fw;
            kr0 = vdupq_n_f16(filter[0 * FW + fw]);

#define cb(i)                                 \
    if (height > i) {                         \
        inp = vld1q_f16(src_dd + i * IW);     \
        out##i = vmlaq_f16(out##i, inp, kr0); \
    }
            UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);

#undef cb
        }
        STORE_RESULT_VAL;
    }
};

template <int height, int width>
struct do_pixel_proxy<2, height, width> {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        const int ih = oh, iw = ow;
#define cb(i) float16x8_t out##i{0};
        UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        float16x8_t kr0, kr1, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_RESULT_VAL;
        for (int fw = 0; fw < FW; ++fw) {
            const __fp16* src_dd = src + fw;
            kr0 = vdupq_n_f16(filter[0 * FW + fw]);
            kr1 = vdupq_n_f16(filter[1 * FW + fw]);

#define cb(i)                                   \
    if (height > i) {                           \
        inp = vld1q_f16(src_dd + i * IW);       \
        out##i = vmlaq_f16(out##i, inp, kr0);   \
        inp = vld1q_f16(src_dd + (i + 1) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr1);   \
    }
            UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        }
        STORE_RESULT_VAL;
    }
};

template <int height, int width>
struct do_pixel_proxy<3, height, width> {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        const int ih = oh, iw = ow;
#define cb(i) float16x8_t out##i{0};
        UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        float16x8_t kr0, kr1, kr2, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_RESULT_VAL;
        for (int fw = 0; fw < FW; ++fw) {
            const __fp16* src_dd = src + fw;
            kr0 = vdupq_n_f16(filter[0 * FW + fw]);
            kr1 = vdupq_n_f16(filter[1 * FW + fw]);
            kr2 = vdupq_n_f16(filter[2 * FW + fw]);
#define cb(i)                                   \
    if (height > i) {                           \
        inp = vld1q_f16(src_dd + i * IW);       \
        out##i = vmlaq_f16(out##i, inp, kr0);   \
        inp = vld1q_f16(src_dd + (i + 1) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr1);   \
        inp = vld1q_f16(src_dd + (i + 2) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr2);   \
    }
            UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);

#undef cb
        }
        STORE_RESULT_VAL;
    }
};

template <int height, int width>
struct do_pixel_proxy<4, height, width> {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        const int ih = oh, iw = ow;
#define cb(i) float16x8_t out##i{0};
        UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        float16x8_t kr0, kr1, kr2, kr3, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_RESULT_VAL;
        for (int fw = 0; fw < FW; ++fw) {
            const __fp16* src_dd = src + fw;
            kr0 = vdupq_n_f16(filter[0 * FW + fw]);
            kr1 = vdupq_n_f16(filter[1 * FW + fw]);
            kr2 = vdupq_n_f16(filter[2 * FW + fw]);
            kr3 = vdupq_n_f16(filter[3 * FW + fw]);
#define cb(i)                                   \
    if (height > i) {                           \
        inp = vld1q_f16(src_dd + i * IW);       \
        out##i = vmlaq_f16(out##i, inp, kr0);   \
        inp = vld1q_f16(src_dd + (i + 1) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr1);   \
        inp = vld1q_f16(src_dd + (i + 2) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr2);   \
        inp = vld1q_f16(src_dd + (i + 3) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr3);   \
    }
            UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);

#undef cb
        }
        STORE_RESULT_VAL;
    }
};

template <int height, int width>
struct do_pixel_proxy<5, height, width> {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        const int ih = oh, iw = ow;
#define cb(i) float16x8_t out##i{0};
        UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        float16x8_t kr0, kr1, kr2, kr3, kr4, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_RESULT_VAL;
        for (int fw = 0; fw < FW; ++fw) {
            const __fp16* src_dd = src + fw;
            kr0 = vdupq_n_f16(filter[0 * FW + fw]);
            kr1 = vdupq_n_f16(filter[1 * FW + fw]);
            kr2 = vdupq_n_f16(filter[2 * FW + fw]);
            kr3 = vdupq_n_f16(filter[3 * FW + fw]);
            kr4 = vdupq_n_f16(filter[4 * FW + fw]);
#define cb(i)                                   \
    if (height > i) {                           \
        inp = vld1q_f16(src_dd + i * IW);       \
        out##i = vmlaq_f16(out##i, inp, kr0);   \
        inp = vld1q_f16(src_dd + (i + 1) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr1);   \
        inp = vld1q_f16(src_dd + (i + 2) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr2);   \
        inp = vld1q_f16(src_dd + (i + 3) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr3);   \
        inp = vld1q_f16(src_dd + (i + 4) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr4);   \
    }
            UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        }
        STORE_RESULT_VAL;
    }
};

template <int height, int width>
struct do_pixel_proxy<6, height, width> {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        const int ih = oh, iw = ow;
#define cb(i) float16x8_t out##i{0};
        UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        float16x8_t kr0, kr1, kr2, kr3, kr4, kr5, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_RESULT_VAL;
        for (int fw = 0; fw < FW; ++fw) {
            const __fp16* src_dd = src + fw;
            kr0 = vdupq_n_f16(filter[0 * FW + fw]);
            kr1 = vdupq_n_f16(filter[1 * FW + fw]);
            kr2 = vdupq_n_f16(filter[2 * FW + fw]);
            kr3 = vdupq_n_f16(filter[3 * FW + fw]);
            kr4 = vdupq_n_f16(filter[4 * FW + fw]);
            kr5 = vdupq_n_f16(filter[5 * FW + fw]);
#define cb(i)                                   \
    if (height > i) {                           \
        inp = vld1q_f16(src_dd + i * IW);       \
        out##i = vmlaq_f16(out##i, inp, kr0);   \
        inp = vld1q_f16(src_dd + (i + 1) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr1);   \
        inp = vld1q_f16(src_dd + (i + 2) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr2);   \
        inp = vld1q_f16(src_dd + (i + 3) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr3);   \
        inp = vld1q_f16(src_dd + (i + 4) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr4);   \
        inp = vld1q_f16(src_dd + (i + 5) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr5);   \
    }
            UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        }
        STORE_RESULT_VAL;
    }
};

template <int height, int width>
struct do_pixel_proxy<7, height, width> {
    static void exec(const __fp16* src, const __fp16* filter, __fp16* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        const int ih = oh, iw = ow;
#define cb(i) float16x8_t out##i{0};
        UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        float16x8_t kr0, kr1, kr2, kr3, kr4, kr5, kr6, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_RESULT_VAL;
        for (int fw = 0; fw < FW; ++fw) {
            const __fp16* src_dd = src + fw;
            kr0 = vdupq_n_f16(filter[0 * FW + fw]);
            kr1 = vdupq_n_f16(filter[1 * FW + fw]);
            kr2 = vdupq_n_f16(filter[2 * FW + fw]);
            kr3 = vdupq_n_f16(filter[3 * FW + fw]);
            kr4 = vdupq_n_f16(filter[4 * FW + fw]);
            kr5 = vdupq_n_f16(filter[5 * FW + fw]);
            kr6 = vdupq_n_f16(filter[6 * FW + fw]);
#define cb(i)                                   \
    if (height > i) {                           \
        inp = vld1q_f16(src_dd + i * IW);       \
        out##i = vmlaq_f16(out##i, inp, kr0);   \
        inp = vld1q_f16(src_dd + (i + 1) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr1);   \
        inp = vld1q_f16(src_dd + (i + 2) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr2);   \
        inp = vld1q_f16(src_dd + (i + 3) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr3);   \
        inp = vld1q_f16(src_dd + (i + 4) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr4);   \
        inp = vld1q_f16(src_dd + (i + 5) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr5);   \
        inp = vld1q_f16(src_dd + (i + 6) * IW); \
        out##i = vmlaq_f16(out##i, inp, kr6);   \
    }
            UNROLL_CALL_NOWRAPPER(BLOCK_H, cb);
#undef cb
        }
        STORE_RESULT_VAL;
    }
};

#undef STORE_RESULT_VAL
#undef LOAD_RESULT_VAL

template <int FH, int height, int width>
void do_pixel(const __fp16* src, const __fp16* filter, __fp16* dst,
              const int IH, const int IW, const int OH, const int OW,
              const int FW, const int oh, const int ow) {
    do_pixel_proxy<FH, height, width>::exec(src, filter, dst, IH, IW, OH, OW,
                                            FW, oh, ow);
}

template <int FH>
void do_conv_tpl_enable_prefetch(const __fp16* src,
                                 const __fp16* filter, __fp16* dst,
                                 const int IH, const int IW, const int OH,
                                 const int OW, const int FW) {
    const int hbeg = 0, hend = OH;
    const int wbeg = 0, wend = OW;
    int i, j;
    for (i = hbeg; i + BLOCK_H <= hend; i += BLOCK_H) {
        for (j = wbeg; j + 8 <= wend; j += 8) {
            // do prefetch
            const int prefetch_index_input =
                    (j + 16) < wend
                            ? i * IW + j + 16
                            : (i + 8) * IW + (((j + 16 - wend) >> 2) << 2);
            const int prefetch_index_output =
                    (j + 16) < wend
                            ? i * OW + j + 16
                            : (i + 8) * OW + (((j + 16 - wend) >> 2) << 2);
            const __fp16* src_prefetch = src + prefetch_index_input;
            const __fp16* dst_prefetch = dst + prefetch_index_output;
            for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {
                __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);
            }
#define unroll_prefetch_cb(i) __builtin_prefetch(dst_prefetch + i * OW, 1, 3);
            UNROLL_CALL_NOWRAPPER(BLOCK_H, unroll_prefetch_cb);
            do_pixel<FH, BLOCK_H, 8>(src, filter, dst, IH, IW, OH, OW, FW, i,
                                     j);
        }
#define DISPATCH(width)                                                       \
    do {                                                                      \
        const int prefetch_index_input = (i + 8) * IW + 12;                   \
        const int prefetch_index_output = (i + 8) * OW + 12;                  \
        const __fp16* src_prefetch = src + prefetch_index_input;              \
        const __fp16* dst_prefetch = dst + prefetch_index_output;             \
        for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {                        \
            __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);              \
        }                                                                     \
        UNROLL_CALL_NOWRAPPER(BLOCK_H, unroll_prefetch_cb);                   \
        do_pixel<FH, BLOCK_H, width>(src, filter, dst, IH, IW, OH, OW, FW, i, \
                                     j);                                      \
    } while (0)
        switch (wend - j) {
            case 1:
                DISPATCH(1);
                break;
            case 2:
                DISPATCH(2);
                break;
            case 3:
                DISPATCH(3);
                break;
            case 4:
                DISPATCH(4);
                break;
            case 5:
                DISPATCH(5);
                break;
            case 6:
                DISPATCH(6);
                break;
            case 7:
                DISPATCH(7);
                break;
        }
#undef DISPATCH
    }

#define DISPATCH2(height, width)                                             \
    do {                                                                     \
        const int prefetch_index_input = IH * IW + 12;                       \
        const __fp16* src_prefetch = src + prefetch_index_input;             \
        for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {                       \
            __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);             \
        }                                                                    \
        do_pixel<FH, height, width>(src, filter, dst, IH, IW, OH, OW, FW, i, \
                                    j);                                      \
    } while (0)

#define DISPATCH1(height)                                                    \
    do {                                                                     \
        for (j = wbeg; j + 8 <= wend; j += 8) {                              \
            const int prefetch_index_input =                                 \
                    (j + 16) < wend                                          \
                            ? i * IW + j + 16                                \
                            : (i + 8) * IW + (((j + 16 - wend) >> 2) << 2);  \
            const int prefetch_index_output =                                \
                    (j + 16) < wend                                          \
                            ? i * OW + j + 16                                \
                            : (i + 8) * OW + (((j + 16 - wend) >> 2) << 2);  \
            const __fp16* src_prefetch = src + prefetch_index_input;         \
            const __fp16* dst_prefetch = dst + prefetch_index_output;        \
            for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {                   \
                __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);         \
            }                                                                \
            UNROLL_CALL_NOWRAPPER(BLOCK_H, unroll_prefetch_cb);              \
            do_pixel<FH, height, 8>(src, filter, dst, IH, IW, OH, OW, FW, i, \
                                    j);                                      \
        }                                                                    \
        switch (wend - j) {                                                  \
            case 1:                                                          \
                DISPATCH2(height, 1);                                        \
                break;                                                       \
            case 2:                                                          \
                DISPATCH2(height, 2);                                        \
                break;                                                       \
            case 3:                                                          \
                DISPATCH2(height, 3);                                        \
                break;                                                       \
            case 4:                                                          \
                DISPATCH2(height, 4);                                        \
                break;                                                       \
            case 5:                                                          \
                DISPATCH2(height, 5);                                        \
                break;                                                       \
            case 6:                                                          \
                DISPATCH2(height, 6);                                        \
                break;                                                       \
            case 7:                                                          \
                DISPATCH2(height, 7);                                        \
                break;                                                       \
        }                                                                    \
    } while (0)
    switch (hend - i) {
        case 1:
            DISPATCH1(1);
            break;
        case 2:
            DISPATCH1(2);
            break;
        case 3:
            DISPATCH1(3);
            break;
#if BLOCK_H == 8
        case 4:
            DISPATCH1(4);
            break;
        case 5:
            DISPATCH1(5);
            break;
        case 6:
            DISPATCH1(6);
            break;
        case 7:
            DISPATCH1(7);
            break;
#endif
    }
#undef DISPATCH1
#undef DISPATCH2
#undef unroll_prefetch_cb
}
template <int FH>
void do_conv_tpl_disable_prefetch(const __fp16* src,
                                  const __fp16* filter, __fp16* dst,
                                  const int IH, const int IW, const int OH,
                                  const int OW, const int FW) {
    const int hbeg = 0, hend = OH;
    const int wbeg = 0, wend = OW;
    int i, j;
    for (i = hbeg; i + BLOCK_H <= hend; i += BLOCK_H) {
        for (j = wbeg; j + 8 <= wend; j += 8) {
            do_pixel<FH, BLOCK_H, 8>(src, filter, dst, IH, IW, OH, OW, FW, i,
                                     j);
        }
#define DISPATCH(width)                                                       \
    do {                                                                      \
        do_pixel<FH, BLOCK_H, width>(src, filter, dst, IH, IW, OH, OW, FW, i, \
                                     j);                                      \
    } while (0)
        switch (wend - j) {
            case 1:
                DISPATCH(1);
                break;
            case 2:
                DISPATCH(2);
                break;
            case 3:
                DISPATCH(3);
                break;
            case 4:
                DISPATCH(4);
                break;
            case 5:
                DISPATCH(5);
                break;
            case 6:
                DISPATCH(6);
                break;
            case 7:
                DISPATCH(7);
                break;
        }
#undef DISPATCH
    }
#define DISPATCH2(height, width)                                             \
    do {                                                                     \
        do_pixel<FH, height, width>(src, filter, dst, IH, IW, OH, OW, FW, i, \
                                    j);                                      \
    } while (0)
#define DISPATCH1(height)                                                    \
    do {                                                                     \
        for (j = wbeg; j + 8 <= wend; j += 8) {                              \
            do_pixel<FH, height, 8>(src, filter, dst, IH, IW, OH, OW, FW, i, \
                                    j);                                      \
        }                                                                    \
        switch (wend - j) {                                                  \
            case 1:                                                          \
                DISPATCH2(height, 1);                                        \
                break;                                                       \
            case 2:                                                          \
                DISPATCH2(height, 2);                                        \
                break;                                                       \
            case 3:                                                          \
                DISPATCH2(height, 3);                                        \
                break;                                                       \
            case 4:                                                          \
                DISPATCH2(height, 4);                                        \
                break;                                                       \
            case 5:                                                          \
                DISPATCH2(height, 5);                                        \
                break;                                                       \
            case 6:                                                          \
                DISPATCH2(height, 6);                                        \
                break;                                                       \
            case 7:                                                          \
                DISPATCH2(height, 7);                                        \
                break;                                                       \
        }                                                                    \
    } while (0)
    switch (hend - i) {
        case 1:
            DISPATCH1(1);
            break;
        case 2:
            DISPATCH1(2);
            break;
        case 3:
            DISPATCH1(3);
            break;
#if BLOCK_H == 8
        case 4:
            DISPATCH1(4);
            break;
        case 5:
            DISPATCH1(5);
            break;
        case 6:
            DISPATCH1(6);
            break;
        case 7:
            DISPATCH1(7);
            break;
#endif
    }
#undef DISPATCH1
#undef DISPATCH2
}
}  // anonymous namespace

void conv_bias::kern_direct_f16(const __fp16* src,
                                  const __fp16* filter, __fp16* dst,
                                  const int IH, const int IW, const int OH,
                                  const int OW, const int FH, const int FW) {
    megdnn_assert_internal(FH <= 7);
    if (IH > 100 && IW > 100) {
#define GAO(FH)                                                              \
    do {                                                                     \
        return do_conv_tpl_enable_prefetch<FH>(src, filter, dst, IH, IW, OH, \
                                               OW, FW);                      \
    } while (0)
        switch (FH) {
            case 1:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(0)) { GAO(1); }
                MIDOUT_END();
                break;
            case 2:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(1)) { GAO(2); }
                MIDOUT_END();
                break;
            case 3:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(2)) { GAO(3); }
                MIDOUT_END();
                break;
            case 4:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(3)) { GAO(4); }
                MIDOUT_END();
                break;
            case 5:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(4)) { GAO(5); }
                MIDOUT_END();
                break;
            case 6:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(5)) { GAO(6); }
                MIDOUT_END();
                break;
            case 7:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(6)) { GAO(7); }
                MIDOUT_END();
                break;
        }
#undef GAO
    } else {
#define GAO(FH)                                                               \
    do {                                                                      \
        return do_conv_tpl_disable_prefetch<FH>(src, filter, dst, IH, IW, OH, \
                                                OW, FW);                      \
    } while (0)
        switch (FH) {
            case 1:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(0)) { GAO(1); }
                MIDOUT_END();
                break;
            case 2:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(1)) { GAO(2); }
                MIDOUT_END();
                break;
            case 3:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(2)) { GAO(3); }
                MIDOUT_END();
                break;
            case 4:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(3)) { GAO(4); }
                MIDOUT_END();
                break;
            case 5:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(4)) { GAO(5); }
                MIDOUT_END();
                break;
            case 6:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(5)) { GAO(6); }
                MIDOUT_END();
                break;
            case 7:
                MIDOUT_BEGIN(megdnn_arm_conv_f16, midout_iv(6)) { GAO(7); }
                MIDOUT_END();
                break;
        }
#undef GAO
    }
    megdnn_assert_internal(0);
}
#endif

// vim: syntax=cpp.doxygen
