/**
 * \file
 * dnn/src/arm_common/conv_bias/f16/direct_nchw88_kern.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/f16/direct_nchw88_kern.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/fallback/conv_bias/common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

using namespace megdnn;
using namespace arm_common;

template <int PC, int BW, int pc, int bw>
struct compute_fma {
    static inline void call(const float16x8_t* ri, const float16x8_t* rf,
                            float16x8_t* rdst) {
#if defined(__aarch64__)
        rdst[bw] = vfmaq_laneq_f16(rdst[bw], rf[pc], ri[bw], pc);
#else
        rdst[bw] = vfmaq_f16(rdst[bw], rf[pc],
                             vdupq_n_f16(vgetq_lane_f16(ri[bw], pc)));
#endif
        compute_fma<PC, BW, pc, bw + 1>::call(ri, rf, rdst);
    }
};

template <int PC, int BW, int pc>
struct compute_fma<PC, BW, pc, BW> {
    static inline void call(const float16x8_t* ri, const float16x8_t* rf,
                            float16x8_t* rdst) {
        compute_fma<PC, BW, pc + 1, 0>::call(ri, rf, rdst);
    }
};

template <int PC, int BW>
struct compute_fma<PC, BW, PC, 0> {
    static inline void call(const float16x8_t* ri, const float16x8_t* rf,
                            float16x8_t* rdst) {}
};

template <int PC, int BW, int bw>
struct load_dst {
    static inline void call(float16x8_t* rdst, const float16_t* dst_ptr) {
        rdst[bw] = vld1q_f16(dst_ptr + bw * PC);
        load_dst<PC, BW, bw + 1>::call(rdst, dst_ptr);
    }
};

template <int PC, int BW>
struct load_dst<PC, BW, BW> {
    static inline void call(float16x8_t* rdst, const float16_t* dst_ptr) {}
};

template <int PC, int SW, int BW, int bw>
struct load_src {
    static inline void call(float16x8_t* ri, const float16_t* src_ptr) {
        ri[bw] = vld1q_f16(src_ptr + bw * SW * PC);
        load_src<PC, SW, BW, bw + 1>::call(ri, src_ptr);
    }
};

template <int PC, int SW, int BW>
struct load_src<PC, SW, BW, BW> {
    static inline void call(float16x8_t* ri, const float16_t* src_ptr) {}
};

template <int PC, int pc>
struct load_filter {
    static inline void call(float16x8_t* rf, const float16_t* filter_ptr) {
        rf[pc] = vld1q_f16(filter_ptr + pc * PC);
        load_filter<PC, pc + 1>::call(rf, filter_ptr);
    }
};

template <int PC>
struct load_filter<PC, PC> {
    static inline void call(float16x8_t* rf, const float16_t* filter_ptr) {}
};

template <int PC, int BW, int bw>
struct store_dst {
    static inline void call(const float16x8_t* rdst, float16_t* dst_ptr) {
        vst1q_f16(dst_ptr + bw * PC, rdst[bw]);
        store_dst<PC, BW, bw + 1>::call(rdst, dst_ptr);
    }
};

template <int PC, int BW>
struct store_dst<PC, BW, BW> {
    static inline void call(const float16x8_t* rdst, float16_t* dst_ptr) {}
};

template <int FH, int SH, int BW>
static inline void do_conv_kern_1xBW(const float16_t*& src, float16_t*& dst,
                                     const float16_t* filter, int IW, int OW,
                                     int& ow) {
    constexpr int PC = 8;
    constexpr int FW = FH;
    constexpr int SW = SH;

    float16x8_t rf[PC];
    if (FH == 1 && FW == 1) {
        load_filter<PC, 0>::call(rf, filter);
    }

    for (; ow + BW - 1 < OW; ow += BW) {
        float16x8_t rdst[BW];
        load_dst<PC, BW, 0>::call(rdst, dst);

        for (int fh = 0; fh < FH; ++fh) {
            for (int fw = 0; fw < FW; ++fw) {
                float16x8_t ri[BW];
                load_src<PC, SW, BW, 0>::call(ri, src + (fh * IW + fw) * PC);

                if (FH > 1 || FW > 1) {
                    load_filter<PC, 0>::call(rf,
                                             filter + (fh * FW + fw) * PC * PC);
                }

                compute_fma<PC, BW, 0, 0>::call(ri, rf, rdst);
            }
        }

        store_dst<PC, BW, 0>::call(rdst, dst);

        src += SW * BW * PC;
        dst += BW * PC;
    }
}

template <BiasMode bias_mode>
static void do_load_bias_kern(float16_t* dst, const float16_t* bias, int OH,
                              int OW) {
    constexpr int PC = 8;

    if (bias_mode == BiasMode::NO_BIAS) {
        memset(dst, 0, OH * OW * PC * sizeof(float16_t));
    } else if (bias_mode == BiasMode::BIAS) {
        memcpy(dst, bias, OH * OW * PC * sizeof(float16_t));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        float16x8_t bias_v = vld1q_f16(bias);
        int i = 0;
        for (; i + 3 < OH * OW; i += 4) {
            vst1q_f16(dst + PC * 0, bias_v);
            vst1q_f16(dst + PC * 1, bias_v);
            vst1q_f16(dst + PC * 2, bias_v);
            vst1q_f16(dst + PC * 3, bias_v);
            dst += PC * 4;
        }
        for (; i < OH * OW; i += 1) {
            vst1q_f16(dst, bias_v);
            dst += PC;
        }
    }
}

template <typename Op>
static void do_op_kern(float16_t* dst, int OH, int OW) {
    constexpr int PC = 8;

    Op op;

    int i = 0;
    for (; i + 3 < OH * OW; i += 4) {
        float16x8_t dst0 = vld1q_f16(dst + PC * 0);
        float16x8_t dst1 = vld1q_f16(dst + PC * 1);
        float16x8_t dst2 = vld1q_f16(dst + PC * 2);
        float16x8_t dst3 = vld1q_f16(dst + PC * 3);

        dst0 = op(dst0);
        dst1 = op(dst1);
        dst2 = op(dst2);
        dst3 = op(dst3);

        vst1q_f16(dst + PC * 0, dst0);
        vst1q_f16(dst + PC * 1, dst1);
        vst1q_f16(dst + PC * 2, dst2);
        vst1q_f16(dst + PC * 3, dst3);

        dst += PC * 4;
    }
    for (; i < OH * OW; i += 1) {
        vst1q_f16(dst, op(vld1q_f16(dst)));
        dst += PC;
    }
}

template <int FH, int SH>
static void do_conv_kern(const float16_t* src, float16_t* dst,
                         const float16_t* filter, int IC, int IH, int IW,
                         int OH, int OW) {
    constexpr int PC = 8;
    constexpr int FW = FH;

    for (int ic = 0; ic < IC; ic += 1) {
        const float16_t* src_ptr_h = src;
        float16_t* dst_ptr_h = dst;

        for (int oh = 0; oh < OH; oh += 1) {
            const float16_t* src_ptr_w = src_ptr_h;
            float16_t* dst_ptr_w = dst_ptr_h;

            int ow = 0;
            do_conv_kern_1xBW<FH, SH, 4>(src_ptr_w, dst_ptr_w, filter, IW, OW,
                                         ow);
            if (OW & 3) {
                do_conv_kern_1xBW<FH, SH, 2>(src_ptr_w, dst_ptr_w, filter, IW,
                                             OW, ow);
                do_conv_kern_1xBW<FH, SH, 1>(src_ptr_w, dst_ptr_w, filter, IW,
                                             OW, ow);
            }
            src_ptr_h += SH * IW * PC;
            dst_ptr_h += OW * PC;
        }
        src += IH * IW * PC;
        filter += FH * FW * PC * PC;
    }
}

static void do_conv_kern_1x1(const float16_t* src, float16_t* dst,
                             const float16_t* filter, int IC, int OH, int OW) {
    constexpr int PC = 8;
    const int IH = OH;
    const int IW = OW;
    const int IHW = IH * IW;
    const int OHW = OH * OW;

    for (int ic = 0; ic < IC; ic += 1) {
        const float16_t* src_ptr_hw = src;
        float16_t* dst_ptr_hw = dst;

        int ohw = 0;
        do_conv_kern_1xBW<1, 1, 8>(src_ptr_hw, dst_ptr_hw, filter, IHW, OHW,
                                   ohw);
        do_conv_kern_1xBW<1, 1, 4>(src_ptr_hw, dst_ptr_hw, filter, IHW, OHW,
                                   ohw);
        do_conv_kern_1xBW<1, 1, 1>(src_ptr_hw, dst_ptr_hw, filter, IHW, OHW,
                                   ohw);
        src += IHW * PC;
        filter += PC * PC;
    }
}

template <size_t FH, size_t SH, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_fp16_nchw88(const __fp16* src, const __fp16* filter,
                                        const __fp16* bias, __fp16* dst, int IC,
                                        int IH, int IW, int OH, int OW) {
    do_load_bias_kern<bias_mode>(dst, bias, OH, OW);
    if (FH == 1 && SH == 1 && IH == OH && IW == OW) {
        do_conv_kern_1x1(src, dst, filter, IC, OH, OW);
    } else {
        do_conv_kern<FH, SH>(src, dst, filter, IC, IH, IW, OH, OW);
    }
    do_op_kern<Op>(dst, OH, OW);
}

#define INSTANTIATION(stride, filter, bias, Op)                             \
    template void                                                           \
    conv_bias::conv_direct_fp16_nchw88<filter, stride, bias, Op>(           \
            const __fp16*, const __fp16*, const __fp16*, __fp16*, int, int, \
            int, int, int);

#define FOR_OP(stride, filter, bias)                       \
    INSTANTIATION(stride, filter, bias, SigmoidOp<__fp16>) \
    INSTANTIATION(stride, filter, bias, ReluOp<__fp16>)    \
    INSTANTIATION(stride, filter, bias, HSwishOp<__fp16>)  \
    INSTANTIATION(stride, filter, bias, NoneOp<__fp16>)

#define FOR_BIAS(stride, filter)                             \
    FOR_OP(stride, filter, BiasMode::NO_BIAS)                \
    FOR_OP(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
    FOR_OP(stride, filter, BiasMode::BIAS)

#define FOR_FILTER(stride) \
    FOR_BIAS(stride, 1)    \
    FOR_BIAS(stride, 2)    \
    FOR_BIAS(stride, 3)    \
    FOR_BIAS(stride, 5)    \
    FOR_BIAS(stride, 7)

#define FOR_STRIDE \
    FOR_FILTER(1)  \
    FOR_FILTER(2)

FOR_STRIDE

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_BIAS
#undef FOR_OP
#undef INSTANTIATION

#endif

// vim: syntax=cpp.doxygen
