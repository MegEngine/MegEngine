/**
 * \file dnn/src/arm_common/conv_bias/fp32/direct.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cstring>
#include "include/megdnn/oprs.h"
#include "midout.h"
#include "src/arm_common/conv_bias/fp32/direct.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/common/unroll_macro.h"
MIDOUT_DECL(megdnn_arm_conv_f32)

using namespace megdnn;
using namespace arm_common;
using namespace fp32;
using namespace conv_bias;

namespace {

template <int FH, int height, int width>
struct do_pixel_proxy {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow);
};

#define cb_load(i) data = vld1q_lane_f32(dst + i, data, i);
#define LOAD_OUT                                               \
    if (width < 4) {                                           \
        auto load_less_4 = [](float* dst, float32x4_t& data) { \
            if (width == 1u) {                                 \
                UNROLL_CALL_NOWRAPPER(1, cb_load);             \
            } else if (width == 2u) {                          \
                UNROLL_CALL_NOWRAPPER(2, cb_load);             \
            } else if (width == 3u) {                          \
                UNROLL_CALL_NOWRAPPER(3, cb_load);             \
            }                                                  \
        };                                                     \
        if (height >= 1)                                       \
            load_less_4(dst + 0 * OW, out0);                   \
        if (height >= 2)                                       \
            load_less_4(dst + 1 * OW, out1);                   \
        if (height >= 3)                                       \
            load_less_4(dst + 2 * OW, out2);                   \
        if (height >= 4)                                       \
            load_less_4(dst + 3 * OW, out3);                   \
    } else {                                                   \
        if (height > 0)                                        \
            out0 = vld1q_f32(dst + 0 * OW);                    \
        if (height > 1)                                        \
            out1 = vld1q_f32(dst + 1 * OW);                    \
        if (height > 2)                                        \
            out2 = vld1q_f32(dst + 2 * OW);                    \
        if (height > 3)                                        \
            out3 = vld1q_f32(dst + 3 * OW);                    \
    }
#define cb_store(i) vst1q_lane_f32(dst + i, data, i);
#define STORE_OUT                                               \
    if (width < 4) {                                            \
        auto store_less_4 = [](float* dst, float32x4_t& data) { \
            if (width == 1u) {                                  \
                UNROLL_CALL_NOWRAPPER(1, cb_store);             \
            } else if (width == 2u) {                           \
                UNROLL_CALL_NOWRAPPER(2, cb_store);             \
            } else if (width == 3u) {                           \
                UNROLL_CALL_NOWRAPPER(3, cb_store);             \
            }                                                   \
        };                                                      \
        if (height >= 1)                                        \
            store_less_4(dst + 0 * OW, out0);                   \
        if (height >= 2)                                        \
            store_less_4(dst + 1 * OW, out1);                   \
        if (height >= 3)                                        \
            store_less_4(dst + 2 * OW, out2);                   \
        if (height >= 4)                                        \
            store_less_4(dst + 3 * OW, out3);                   \
    } else {                                                    \
        if (height >= 1)                                        \
            vst1q_f32(dst + 0 * OW, out0);                      \
        if (height >= 2)                                        \
            vst1q_f32(dst + 1 * OW, out1);                      \
        if (height >= 3)                                        \
            vst1q_f32(dst + 2 * OW, out2);                      \
        if (height >= 4)                                        \
            vst1q_f32(dst + 3 * OW, out3);                      \
    }

template <int height, int width>
struct do_pixel_proxy<1, height, width> {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        (void)IH;
        (void)OH;
        const int ih = oh, iw = ow;
        float32x4_t out0{0}, out1{0}, out2{0}, out3{0}, kr0, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_OUT;
        for (int fw = 0; fw < FW; ++fw) {
            const float* src_dd = src + fw;
            kr0 = vdupq_n_f32(filter[0 * FW + fw]);

            if (height > 0)
                inp = vld1q_f32(src_dd + 0 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr0);

            if (height > 1)
                inp = vld1q_f32(src_dd + 1 * IW);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr0);

            if (height > 2)
                inp = vld1q_f32(src_dd + 2 * IW);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr0);

            if (height > 3)
                inp = vld1q_f32(src_dd + 3 * IW);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr0);
        }
        STORE_OUT;
    }
};

template <int height, int width>
struct do_pixel_proxy<2, height, width> {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        (void)IH;
        (void)OH;
        const int ih = oh, iw = ow;
        float32x4_t out0{0}, out1{0}, out2{0}, out3{0}, kr0, kr1, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_OUT;
        for (int fw = 0; fw < FW; ++fw) {
            const float* src_dd = src + fw;
            kr0 = vdupq_n_f32(filter[0 * FW + fw]);
            kr1 = vdupq_n_f32(filter[1 * FW + fw]);

            if (height > 0)
                inp = vld1q_f32(src_dd + 0 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 1 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr1);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr0);

            if (height > 1)
                inp = vld1q_f32(src_dd + 2 * IW);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr1);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr0);

            if (height > 2)
                inp = vld1q_f32(src_dd + 3 * IW);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr1);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr0);

            if (height > 3)
                inp = vld1q_f32(src_dd + 4 * IW);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr1);
        }
        STORE_OUT;
    }
};

template <int height, int width>
struct do_pixel_proxy<3, height, width> {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        (void)IH;
        (void)OH;
        const int ih = oh, iw = ow;
        float32x4_t out0{0}, out1{0}, out2{0}, out3{0}, kr0, kr1, kr2, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_OUT;
        for (int fw = 0; fw < FW; ++fw) {
            const float* src_dd = src + fw;
            kr0 = vdupq_n_f32(filter[0 * FW + fw]);
            kr1 = vdupq_n_f32(filter[1 * FW + fw]);
            kr2 = vdupq_n_f32(filter[2 * FW + fw]);

            if (height > 0)
                inp = vld1q_f32(src_dd + 0 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 1 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr1);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 2 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr2);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr1);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr0);

            if (height > 1)
                inp = vld1q_f32(src_dd + 3 * IW);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr2);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr1);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr0);

            if (height > 2)
                inp = vld1q_f32(src_dd + 4 * IW);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr2);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr1);

            if (height > 3)
                inp = vld1q_f32(src_dd + 5 * IW);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr2);
        }
        STORE_OUT;
    }
};

template <int height, int width>
struct do_pixel_proxy<4, height, width> {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        (void)IH;
        (void)OH;
        const int ih = oh, iw = ow;
        float32x4_t out0{0}, out1{0}, out2{0}, out3{0}, kr0, kr1, kr2, kr3, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_OUT;
        for (int fw = 0; fw < FW; ++fw) {
            const float* src_dd = src + fw;
            kr0 = vdupq_n_f32(filter[0 * FW + fw]);
            kr1 = vdupq_n_f32(filter[1 * FW + fw]);
            kr2 = vdupq_n_f32(filter[2 * FW + fw]);
            kr3 = vdupq_n_f32(filter[3 * FW + fw]);

            if (height > 0)
                inp = vld1q_f32(src_dd + 0 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 1 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr1);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 2 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr2);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr1);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 3 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr3);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr2);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr1);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr0);

            if (height > 1)
                inp = vld1q_f32(src_dd + 4 * IW);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr3);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr2);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr1);

            if (height > 2)
                inp = vld1q_f32(src_dd + 5 * IW);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr3);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr2);

            if (height > 3)
                inp = vld1q_f32(src_dd + 6 * IW);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr3);
        }
        STORE_OUT;
    }
};

template <int height, int width>
struct do_pixel_proxy<5, height, width> {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        (void)IH;
        (void)OH;
        const int ih = oh, iw = ow;
        float32x4_t out0{0}, out1{0}, out2{0}, out3{0}, kr0, kr1, kr2, kr3, kr4,
                inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_OUT;
        for (int fw = 0; fw < FW; ++fw) {
            const float* src_dd = src + fw;
            kr0 = vdupq_n_f32(filter[0 * FW + fw]);
            kr1 = vdupq_n_f32(filter[1 * FW + fw]);
            kr2 = vdupq_n_f32(filter[2 * FW + fw]);
            kr3 = vdupq_n_f32(filter[3 * FW + fw]);
            kr4 = vdupq_n_f32(filter[4 * FW + fw]);

            if (height > 0)
                inp = vld1q_f32(src_dd + 0 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 1 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr1);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 2 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr2);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr1);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 3 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr3);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr2);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr1);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 4 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr4);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr3);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr2);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr1);

            if (height > 1)
                inp = vld1q_f32(src_dd + 5 * IW);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr4);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr3);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr2);

            if (height > 2)
                inp = vld1q_f32(src_dd + 6 * IW);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr4);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr3);

            if (height > 3)
                inp = vld1q_f32(src_dd + 7 * IW);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr4);
        }
        STORE_OUT;
    }
};

template <int height, int width>
struct do_pixel_proxy<6, height, width> {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        (void)IH;
        (void)OH;
        const int ih = oh, iw = ow;
        float32x4_t out0{0}, out1{0}, out2{0}, out3{0}, kr0, kr1, kr2, kr3, kr4,
                kr5, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_OUT;
        for (int fw = 0; fw < FW; ++fw) {
            const float* src_dd = src + fw;
            kr0 = vdupq_n_f32(filter[0 * FW + fw]);
            kr1 = vdupq_n_f32(filter[1 * FW + fw]);
            kr2 = vdupq_n_f32(filter[2 * FW + fw]);
            kr3 = vdupq_n_f32(filter[3 * FW + fw]);
            kr4 = vdupq_n_f32(filter[4 * FW + fw]);
            kr5 = vdupq_n_f32(filter[5 * FW + fw]);

            if (height > 0)
                inp = vld1q_f32(src_dd + 0 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 1 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr1);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 2 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr2);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr1);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 3 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr3);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr2);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr1);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 4 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr4);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr3);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr2);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr1);

            if (height > 0)
                inp = vld1q_f32(src_dd + 5 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr5);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr4);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr3);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr2);

            if (height > 1)
                inp = vld1q_f32(src_dd + 6 * IW);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr5);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr4);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr3);

            if (height > 2)
                inp = vld1q_f32(src_dd + 7 * IW);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr5);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr4);

            if (height > 3)
                inp = vld1q_f32(src_dd + 8 * IW);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr5);
        }
        STORE_OUT;
    }
};

template <int height, int width>
struct do_pixel_proxy<7, height, width> {
    static void exec(const float* src, const float* filter, float* dst,
                     const int IH, const int IW, const int OH, const int OW,
                     const int FW, const int oh, const int ow) {
        (void)IH;
        (void)OH;
        const int ih = oh, iw = ow;
        float32x4_t out0{0}, out1{0}, out2{0}, out3{0}, kr0, kr1, kr2, kr3, kr4,
                kr5, kr6, inp;
        src += ih * IW + iw;
        dst += oh * OW + ow;
        LOAD_OUT;
        for (int fw = 0; fw < FW; ++fw) {
            const float* src_dd = src + fw;
            kr0 = vdupq_n_f32(filter[0 * FW + fw]);
            kr1 = vdupq_n_f32(filter[1 * FW + fw]);
            kr2 = vdupq_n_f32(filter[2 * FW + fw]);
            kr3 = vdupq_n_f32(filter[3 * FW + fw]);
            kr4 = vdupq_n_f32(filter[4 * FW + fw]);
            kr5 = vdupq_n_f32(filter[5 * FW + fw]);
            kr6 = vdupq_n_f32(filter[6 * FW + fw]);

            if (height > 0)
                inp = vld1q_f32(src_dd + 0 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 1 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr1);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 2 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr2);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr1);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 3 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr3);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr2);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr1);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr0);

            if (height > 0)
                inp = vld1q_f32(src_dd + 4 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr4);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr3);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr2);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr1);

            if (height > 0)
                inp = vld1q_f32(src_dd + 5 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr5);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr4);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr3);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr2);

            if (height > 0)
                inp = vld1q_f32(src_dd + 6 * IW);
            if (height > 0)
                out0 = vmlaq_f32(out0, inp, kr6);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr5);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr4);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr3);

            if (height > 1)
                inp = vld1q_f32(src_dd + 7 * IW);
            if (height > 1)
                out1 = vmlaq_f32(out1, inp, kr6);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr5);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr4);

            if (height > 2)
                inp = vld1q_f32(src_dd + 8 * IW);
            if (height > 2)
                out2 = vmlaq_f32(out2, inp, kr6);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr5);

            if (height > 3)
                inp = vld1q_f32(src_dd + 9 * IW);
            if (height > 3)
                out3 = vmlaq_f32(out3, inp, kr6);
        }
        STORE_OUT;
    }
};
#undef cb_load
#undef cb_load
#undef LOAD_OUT
#undef STORE_OUT

template <int FH, int height, int width>
void do_pixel(const float* src, const float* filter, float* dst, const int IH,
              const int IW, const int OH, const int OW, const int FW,
              const int oh, const int ow) {
    do_pixel_proxy<FH, height, width>::exec(src, filter, dst, IH, IW, OH, OW,
                                            FW, oh, ow);
}

template <int FH>
void do_conv_tpl_enable_prefetch(const float* src, const float* filter,
                                 float* dst, const int IH, const int IW,
                                 const int OH, const int OW, const int FW) {
    const int hbeg = 0, hend = OH;
    const int wbeg = 0, wend = OW;
    int i, j;
    for (i = hbeg; i + 4 <= hend; i += 4) {
        for (j = wbeg; j + 4 <= wend; j += 4) {
            // do prefetch
            const int prefetch_index_input =
                    (j + 16) < wend
                            ? i * IW + j + 16
                            : (i + 4) * IW + (((j + 16 - wend) >> 2) << 2);
            const int prefetch_index_output =
                    (j + 16) < wend
                            ? i * OW + j + 16
                            : (i + 4) * OW + (((j + 16 - wend) >> 2) << 2);
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {
                __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);
            }
            __builtin_prefetch(dst_prefetch + 0 * OW, 1, 3);
            __builtin_prefetch(dst_prefetch + 1 * OW, 1, 3);
            __builtin_prefetch(dst_prefetch + 2 * OW, 1, 3);
            __builtin_prefetch(dst_prefetch + 3 * OW, 1, 3);
            do_pixel<FH, 4, 4>(src, filter, dst, IH, IW, OH, OW, FW, i, j);
        }
#define DISPATCH(width)                                                     \
    do {                                                                    \
        const int prefetch_index_input = (i + 4) * IW + 12;                 \
        const int prefetch_index_output = (i + 4) * OW + 12;                \
        const float* src_prefetch = src + prefetch_index_input;             \
        const float* dst_prefetch = dst + prefetch_index_output;            \
        for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {                      \
            __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);            \
        }                                                                   \
        __builtin_prefetch(dst_prefetch + 0 * OW, 1, 3);                    \
        __builtin_prefetch(dst_prefetch + 1 * OW, 1, 3);                    \
        __builtin_prefetch(dst_prefetch + 2 * OW, 1, 3);                    \
        __builtin_prefetch(dst_prefetch + 3 * OW, 1, 3);                    \
        do_pixel<FH, 4, width>(src, filter, dst, IH, IW, OH, OW, FW, i, j); \
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
        }
#undef DISPATCH
    }

#define DISPATCH2(height, width)                                             \
    do {                                                                     \
        const int prefetch_index_input = IH * IW + 12;                       \
        const float* src_prefetch = src + prefetch_index_input;              \
        for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {                       \
            __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);             \
        }                                                                    \
        do_pixel<FH, height, width>(src, filter, dst, IH, IW, OH, OW, FW, i, \
                                    j);                                      \
    } while (0)

#define DISPATCH1(height)                                                    \
    do {                                                                     \
        for (j = wbeg; j + 4 <= wend; j += 4) {                              \
            const int prefetch_index_input =                                 \
                    (j + 16) < wend                                          \
                            ? i * IW + j + 16                                \
                            : (i + 4) * IW + (((j + 16 - wend) >> 2) << 2);  \
            const int prefetch_index_output =                                \
                    (j + 16) < wend                                          \
                            ? i * OW + j + 16                                \
                            : (i + 4) * OW + (((j + 16 - wend) >> 2) << 2);  \
            const float* src_prefetch = src + prefetch_index_input;          \
            const float* dst_prefetch = dst + prefetch_index_output;         \
            for (int iw_id = 0; iw_id < FH + 3; ++iw_id) {                   \
                __builtin_prefetch(src_prefetch + iw_id * IW, 0, 3);         \
            }                                                                \
            __builtin_prefetch(dst_prefetch + 0 * OW, 1, 3);                 \
            __builtin_prefetch(dst_prefetch + 1 * OW, 1, 3);                 \
            __builtin_prefetch(dst_prefetch + 2 * OW, 1, 3);                 \
            __builtin_prefetch(dst_prefetch + 3 * OW, 1, 3);                 \
            do_pixel<FH, height, 4>(src, filter, dst, IH, IW, OH, OW, FW, i, \
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
    }
#undef DISPATCH1
#undef DISPATCH2
}
template <int FH>
void do_conv_tpl_disable_prefetch(const float* src, const float* filter,
                                  float* dst, const int IH, const int IW,
                                  const int OH, const int OW, const int FW) {
    const int hbeg = 0, hend = OH;
    const int wbeg = 0, wend = OW;
    int i, j;
    for (i = hbeg; i + 4 <= hend; i += 4) {
        for (j = wbeg; j + 4 <= wend; j += 4) {
            do_pixel<FH, 4, 4>(src, filter, dst, IH, IW, OH, OW, FW, i, j);
        }
#define DISPATCH(width)                                                     \
    do {                                                                    \
        do_pixel<FH, 4, width>(src, filter, dst, IH, IW, OH, OW, FW, i, j); \
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
        for (j = wbeg; j + 4 <= wend; j += 4) {                              \
            do_pixel<FH, height, 4>(src, filter, dst, IH, IW, OH, OW, FW, i, \
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
    }
#undef DISPATCH1
#undef DISPATCH2
}
}  // anonymous namespace

void conv_bias::kern_direct(const float* src, const float* filter, float* dst,
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
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(0)) { GAO(1); }
                MIDOUT_END();
                break;
            case 2:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(1)) { GAO(2); }
                MIDOUT_END();
                break;
            case 3:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(2)) { GAO(3); }
                MIDOUT_END();
                break;
            case 4:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(3)) { GAO(4); }
                MIDOUT_END();
                break;
            case 5:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(4)) { GAO(5); }
                MIDOUT_END();
                break;
            case 6:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(5)) { GAO(6); }
                MIDOUT_END();
                break;
            case 7:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(6)) { GAO(7); }
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
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(0)) { GAO(1); }
                MIDOUT_END();
                break;
            case 2:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(1)) { GAO(2); }
                MIDOUT_END();
                break;
            case 3:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(2)) { GAO(3); }
                MIDOUT_END();
                break;
            case 4:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(3)) { GAO(4); }
                MIDOUT_END();
                break;
            case 5:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(4)) { GAO(5); }
                MIDOUT_END();
                break;
            case 6:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(5)) { GAO(6); }
                MIDOUT_END();
                break;
            case 7:
                MIDOUT_BEGIN(megdnn_arm_conv_f32, midout_iv(6)) { GAO(7); }
                MIDOUT_END();
                break;
        }
#undef GAO
    }
    megdnn_assert_internal(0);
}

// vim: syntax=cpp.doxygen
