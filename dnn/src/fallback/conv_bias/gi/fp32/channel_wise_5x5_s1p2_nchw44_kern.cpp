/**
 * \file dnn/src/fallback/conv_bias/gi/fp32/channel_wise_5x5_s1p2_nchw44_kern.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/fallback/conv_bias/gi/fp32/channel_wise_5x5_s1p2_nchw44_kern.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/gi/utils.h"
#include "src/fallback/elemwise_helper/elemwise_op.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"

using namespace megdnn;
using namespace fallback;

namespace {

template <int shift>
static inline void shift_src(GI_FLOAT32_t rsrc[6]) {
    GI_FLOAT32_t t[6];

    t[0] = rsrc[(shift + 0) % 6];
    t[1] = rsrc[(shift + 1) % 6];
    t[2] = rsrc[(shift + 2) % 6];
    t[3] = rsrc[(shift + 3) % 6];
    t[4] = rsrc[(shift + 4) % 6];
    t[5] = rsrc[(shift + 5) % 6];
    rsrc[0] = t[0];
    rsrc[1] = t[1];
    rsrc[2] = t[2];
    rsrc[3] = t[3];
    rsrc[4] = t[4];
    rsrc[5] = t[5];
}

static inline void load_filter(const float* filter, GI_FLOAT32_t rfilter[5]) {
    rfilter[0] = GiLoadFloat32(filter + 0);
    rfilter[1] = GiLoadFloat32(filter + 4);
    rfilter[2] = GiLoadFloat32(filter + 8);
    rfilter[3] = GiLoadFloat32(filter + 12);
    rfilter[4] = GiLoadFloat32(filter + 16);
}

template <BiasMode bias_mode>
static inline GI_FLOAT32_t load_bias(const float* bias, const GI_FLOAT32_t& init) {
    if (bias_mode == BiasMode::BIAS) {
        return GiLoadFloat32(bias);
    } else {
        return init;
    }
}

template <int BW, int bw, BiasMode bias_mode, bool need_load_bias, bool need_do_op>
struct compute_element {
    template <typename Op>
    static inline void call(
            const float*& src, float*& dst, const float*& bias,
            const GI_FLOAT32_t& init, GI_FLOAT32_t rsrc[6], GI_FLOAT32_t rfilter[5],
            const Op& op) {
#define RSRC(i) rsrc[((i) + bw) % 6]
        GI_FLOAT32_t rdst;
        if (need_load_bias) {
            rdst = load_bias<bias_mode>(bias, init);
        } else {
            rdst = GiLoadFloat32(dst);
        }
        RSRC(5) = GiLoadFloat32(src + 12);

        rdst = GiMlaqFloat32(rdst, RSRC(0), rfilter[0]);
        rdst = GiMlaqFloat32(rdst, RSRC(1), rfilter[1]);
        rdst = GiMlaqFloat32(rdst, RSRC(2), rfilter[2]);
        rdst = GiMlaqFloat32(rdst, RSRC(3), rfilter[3]);
        rdst = GiMlaqFloat32(rdst, RSRC(4), rfilter[4]);

        if (need_do_op) {
            rdst = op(rdst);
        }
        GiStoreFloat32(dst, rdst);

        src += 4;
        dst += 4;
        bias += 4;
        compute_element<BW, bw + 1, bias_mode, need_load_bias, need_do_op>::call(
                src, dst, bias, init, rsrc, rfilter, op);
#undef RSRC
    }
};

template <int BW, BiasMode bias_mode, bool need_load_bias, bool need_do_op>
struct compute_element<BW, BW, bias_mode, need_load_bias, need_do_op> {
    template <typename... Types>
    static inline void call(Types... args) {}
};

template <size_t padding, BiasMode bias_mode, bool need_load_bias, bool need_do_op>
struct compute_element_right {
    template <typename Op>
    static inline void call(
            float*& dst, const float*& bias, const GI_FLOAT32_t& init,
            GI_FLOAT32_t rsrc[6], GI_FLOAT32_t rfilter[5], const Op& op) {
        GI_FLOAT32_t rdst;
        if (need_load_bias) {
            rdst = load_bias<bias_mode>(bias, init);
        } else {
            rdst = GiLoadFloat32(dst);
        }

        rdst = GiMlaqFloat32(rdst, rsrc[0 + padding], rfilter[0]);
        rdst = GiMlaqFloat32(rdst, rsrc[1 + padding], rfilter[1]);
        rdst = GiMlaqFloat32(rdst, rsrc[2 + padding], rfilter[2]);
        if (padding < 2) {
            rdst = GiMlaqFloat32(rdst, rsrc[3 + padding], rfilter[3]);
        }
        if (padding < 1) {
            rdst = GiMlaqFloat32(rdst, rsrc[4 + padding], rfilter[4]);
        }

        if (need_do_op) {
            rdst = op(rdst);
        }
        GiStoreFloat32(dst, rdst);

        dst += 4;
        bias += 4;
    }
};

template <BiasMode bias_mode, bool need_load_bias, bool need_do_op>
struct compute_row_src_1x5 {
    template <typename Op>
    static inline void call(
            const float* src, float* dst, const float* bias, const GI_FLOAT32_t& init,
            GI_FLOAT32_t rsrc[6], GI_FLOAT32_t rfilter[5], int W, const Op& op) {
        rsrc[0] = GiZeroFloat32();
        rsrc[1] = GiZeroFloat32();
        rsrc[2] = GiLoadFloat32(src + 0);
        rsrc[3] = GiLoadFloat32(src + 4);
        rsrc[4] = GiLoadFloat32(src + 8);

        int w = 0;

        for (; w + 5 < W - 3; w += 6) {
            compute_element<6, 0, bias_mode, need_load_bias, need_do_op>::call(
                    src, dst, bias, init, rsrc, rfilter, op);
        }
        if (w + 3 < W - 3) {
            compute_element<4, 0, bias_mode, need_load_bias, need_do_op>::call(
                    src, dst, bias, init, rsrc, rfilter, op);
            shift_src<4>(rsrc);
            w += 4;
        }
        if (w + 1 < W - 3) {
            compute_element<2, 0, bias_mode, need_load_bias, need_do_op>::call(
                    src, dst, bias, init, rsrc, rfilter, op);
            shift_src<2>(rsrc);
            w += 2;
        }
        if (w < W - 3) {
            compute_element<1, 0, bias_mode, need_load_bias, need_do_op>::call(
                    src, dst, bias, init, rsrc, rfilter, op);
            shift_src<1>(rsrc);
            w += 1;
        }
        // compute rightmost 3 elements seperately
        compute_element_right<0, bias_mode, need_load_bias, need_do_op>::call(
                dst, bias, init, rsrc, rfilter, op);
        compute_element_right<1, bias_mode, need_load_bias, need_do_op>::call(
                dst, bias, init, rsrc, rfilter, op);
        compute_element_right<2, bias_mode, need_load_bias, need_do_op>::call(
                dst, bias, init, rsrc, rfilter, op);
    }
};

template <size_t top_padding, size_t bottom_padding, BiasMode bias_mode>
struct compute_row {
    template <typename Op>
    static inline void call(
            const float*& src, float*& dst, const float* filter, const float*& bias,
            const GI_FLOAT32_t& init, GI_FLOAT32_t rsrc[6], GI_FLOAT32_t rfilter[5],
            int W, const Op& op) {
        if (top_padding < 1) {
            load_filter(filter + 0, rfilter);
            compute_row_src_1x5<bias_mode, top_padding == 0, false>::call(
                    src - W * 8, dst, bias, init, rsrc, rfilter, W, op);
        }

        if (top_padding < 2) {
            load_filter(filter + 20, rfilter);
            compute_row_src_1x5<bias_mode, top_padding == 1, false>::call(
                    src - W * 4, dst, bias, init, rsrc, rfilter, W, op);
        }

        {
            load_filter(filter + 40, rfilter);
            compute_row_src_1x5<bias_mode, top_padding == 2, bottom_padding == 2>::call(
                    src, dst, bias, init, rsrc, rfilter, W, op);
        }

        if (bottom_padding < 2) {
            load_filter(filter + 60, rfilter);
            compute_row_src_1x5<bias_mode, false, bottom_padding == 1>::call(
                    src + W * 4, dst, bias, init, rsrc, rfilter, W, op);
        }

        if (bottom_padding < 1) {
            load_filter(filter + 80, rfilter);
            compute_row_src_1x5<bias_mode, false, bottom_padding == 0>::call(
                    src + W * 8, dst, bias, init, rsrc, rfilter, W, op);
        }
        src += W * 4;
        dst += W * 4;
        bias += W * 4;
    }
};

}  // namespace

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw44_float::do_conv_kern_5x5_stride1_padding2(
        const float* src, float* dst, const float* filter, const float* bias, int H,
        int W) {
    Op op;

    GI_FLOAT32_t init = GiZeroFloat32();
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = GiLoadFloat32(bias);
    }

    GI_FLOAT32_t rsrc[6];
    GI_FLOAT32_t rfilter[5];

    compute_row<2, 0, bias_mode>::call(
            src, dst, filter, bias, init, rsrc, rfilter, W, op);
    compute_row<1, 0, bias_mode>::call(
            src, dst, filter, bias, init, rsrc, rfilter, W, op);
    for (int h = 2; h < H - 2; h += 1) {
        compute_row<0, 0, bias_mode>::call(
                src, dst, filter, bias, init, rsrc, rfilter, W, op);
    }
    compute_row<0, 1, bias_mode>::call(
            src, dst, filter, bias, init, rsrc, rfilter, W, op);
    compute_row<0, 2, bias_mode>::call(
            src, dst, filter, bias, init, rsrc, rfilter, W, op);
}

#define INSTANTIATION(bias, Op)                                             \
    template void                                                           \
    channel_wise_nchw44_float::do_conv_kern_5x5_stride1_padding2<bias, Op>( \
            const float*, float*, const float*, const float*, int, int);

#define FOR_OP(bias)                           \
    INSTANTIATION(bias, SigmoidOp<dt_float32>) \
    INSTANTIATION(bias, ReluOp<dt_float32>)    \
    INSTANTIATION(bias, HSwishOp<dt_float32>)  \
    INSTANTIATION(bias, NoneOp<dt_float32>)

#define FOR_BIAS                             \
    FOR_OP(BiasMode::NO_BIAS)                \
    FOR_OP(BiasMode::BROADCAST_CHANNEL_BIAS) \
    FOR_OP(BiasMode::BIAS)

FOR_BIAS

#undef FOR_BIAS
#undef FOR_OP
#undef INSTANTIATION

// vim: syntax=cpp.doxygen
