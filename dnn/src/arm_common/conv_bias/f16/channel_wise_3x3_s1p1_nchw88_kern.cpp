#include "src/arm_common/conv_bias/f16/channel_wise_3x3_s1p1_nchw88_kern.h"
#include "src/arm_common/elemwise_helper/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#pragma GCC diagnostic ignored "-Wunused-parameter"

using namespace megdnn;
using namespace arm_common;
using namespace fp16;

namespace {

template <int shift>
static inline void shift_src(float16x8_t rsrc[3][4]) {
    float16x8_t t[4];

    t[0] = rsrc[0][(shift + 0) % 4];
    t[1] = rsrc[0][(shift + 1) % 4];
    t[2] = rsrc[0][(shift + 2) % 4];
    t[3] = rsrc[0][(shift + 3) % 4];
    rsrc[0][0] = t[0];
    rsrc[0][1] = t[1];
    rsrc[0][2] = t[2];
    rsrc[0][3] = t[3];

    t[0] = rsrc[1][(shift + 0) % 4];
    t[1] = rsrc[1][(shift + 1) % 4];
    t[2] = rsrc[1][(shift + 2) % 4];
    t[3] = rsrc[1][(shift + 3) % 4];
    rsrc[1][0] = t[0];
    rsrc[1][1] = t[1];
    rsrc[1][2] = t[2];
    rsrc[1][3] = t[3];

    t[0] = rsrc[2][(shift + 0) % 4];
    t[1] = rsrc[2][(shift + 1) % 4];
    t[2] = rsrc[2][(shift + 2) % 4];
    t[3] = rsrc[2][(shift + 3) % 4];
    rsrc[2][0] = t[0];
    rsrc[2][1] = t[1];
    rsrc[2][2] = t[2];
    rsrc[2][3] = t[3];
}

template <BiasMode bias_mode>
static inline float16x8_t load_bias(const float16_t* bias, const float16x8_t& init) {
    if (bias_mode == BiasMode::BIAS) {
        return vld1q_f16(bias);
    } else {
        return init;
    }
}

template <int BW, int bw, bool has_top, bool has_bottom, BiasMode bias_mode>
struct compute_element {
    template <typename Op>
    static inline void call(
            const float16_t*& src0, const float16_t*& src1, const float16_t*& src2,
            float16_t*& dst, const float16_t*& bias, const float16x8_t& init,
            float16x8_t rsrc[3][4], float16x8_t rfilter[3][3], const Op& op) {
#define RSRC(i, j) rsrc[i][((j) + bw) % 4]
        float16x8_t rdst = load_bias<bias_mode>(bias, init);
        if (has_top) {
            RSRC(0, 3) = vld1q_f16(src0 + 16);
        }
        { RSRC(1, 3) = vld1q_f16(src1 + 16); }
        if (has_bottom) {
            RSRC(2, 3) = vld1q_f16(src2 + 16);
        }

        if (has_top) {
            rdst = vfmaq_f16(rdst, RSRC(0, 0), rfilter[0][0]);
            rdst = vfmaq_f16(rdst, RSRC(0, 1), rfilter[0][1]);
            rdst = vfmaq_f16(rdst, RSRC(0, 2), rfilter[0][2]);
        }
        {
            rdst = vfmaq_f16(rdst, RSRC(1, 0), rfilter[1][0]);
            rdst = vfmaq_f16(rdst, RSRC(1, 1), rfilter[1][1]);
            rdst = vfmaq_f16(rdst, RSRC(1, 2), rfilter[1][2]);
        }
        if (has_bottom) {
            rdst = vfmaq_f16(rdst, RSRC(2, 0), rfilter[2][0]);
            rdst = vfmaq_f16(rdst, RSRC(2, 1), rfilter[2][1]);
            rdst = vfmaq_f16(rdst, RSRC(2, 2), rfilter[2][2]);
        }

        vst1q_f16(dst, op(rdst));

        if (has_top) {
            src0 += 8;
        }
        { src1 += 8; }
        if (has_bottom) {
            src2 += 8;
        }
        dst += 8;
        bias += 8;
        compute_element<BW, bw + 1, has_top, has_bottom, bias_mode>::call(
                src0, src1, src2, dst, bias, init, rsrc, rfilter, op);
#undef RSRC
    }
};

template <int BW, bool has_top, bool has_bottom, BiasMode bias_mode>
struct compute_element<BW, BW, has_top, has_bottom, bias_mode> {
    template <typename Op>
    static inline void call(
            const float16_t*& src0, const float16_t*& src1, const float16_t*& src2,
            float16_t*& dst, const float16_t*& bias, const float16x8_t& init,
            float16x8_t rsrc[3][4], float16x8_t rfilter[3][3], const Op& op) {}
};

template <bool has_top, bool has_bottom, BiasMode bias_mode>
struct compute_element_right {
    template <typename Op>
    static inline void call(
            float16_t*& dst, const float16_t*& bias, const float16x8_t& init,
            float16x8_t rsrc[3][4], float16x8_t rfilter[3][3], const Op& op) {
        float16x8_t rdst = load_bias<bias_mode>(bias, init);

        if (has_top) {
            rdst = vfmaq_f16(rdst, rsrc[0][0], rfilter[0][0]);
            rdst = vfmaq_f16(rdst, rsrc[0][1], rfilter[0][1]);
            rdst = vfmaq_f16(rdst, rsrc[0][2], rfilter[0][2]);
        }
        {
            rdst = vfmaq_f16(rdst, rsrc[1][0], rfilter[1][0]);
            rdst = vfmaq_f16(rdst, rsrc[1][1], rfilter[1][1]);
            rdst = vfmaq_f16(rdst, rsrc[1][2], rfilter[1][2]);
        }
        if (has_bottom) {
            rdst = vfmaq_f16(rdst, rsrc[2][0], rfilter[2][0]);
            rdst = vfmaq_f16(rdst, rsrc[2][1], rfilter[2][1]);
            rdst = vfmaq_f16(rdst, rsrc[2][2], rfilter[2][2]);
        }

        vst1q_f16(dst, op(rdst));

        dst += 8;
        bias += 8;
    }
};

template <bool has_top, bool has_bottom, BiasMode bias_mode>
struct compute_element_right_pad {
    template <typename Op>
    static inline void call(
            float16_t*& dst, const float16_t*& bias, const float16x8_t& init,
            float16x8_t rsrc[3][4], float16x8_t rfilter[3][3], const Op& op) {
        float16x8_t rdst = load_bias<bias_mode>(bias, init);

        if (has_top) {
            rdst = vfmaq_f16(rdst, rsrc[0][1], rfilter[0][0]);
            rdst = vfmaq_f16(rdst, rsrc[0][2], rfilter[0][1]);
        }
        {
            rdst = vfmaq_f16(rdst, rsrc[1][1], rfilter[1][0]);
            rdst = vfmaq_f16(rdst, rsrc[1][2], rfilter[1][1]);
        }
        if (has_bottom) {
            rdst = vfmaq_f16(rdst, rsrc[2][1], rfilter[2][0]);
            rdst = vfmaq_f16(rdst, rsrc[2][2], rfilter[2][1]);
        }

        vst1q_f16(dst, op(rdst));
        dst += 8;
        bias += 8;
    }
};

template <bool has_top, bool has_bottom, BiasMode bias_mode>
struct compute_row {
    template <typename Op>
    static inline void call(
            const float16_t*& src0, const float16_t*& src1, const float16_t*& src2,
            float16_t*& dst, const float16_t*& bias, const float16x8_t& init,
            float16x8_t rsrc[3][4], float16x8_t rfilter[3][3], int W, const Op& op) {
        if (has_top) {
            rsrc[0][0] = vdupq_n_f16(0);
            rsrc[0][1] = vld1q_f16(src0 + 0);
            rsrc[0][2] = vld1q_f16(src0 + 8);
        }
        {
            rsrc[1][0] = vdupq_n_f16(0);
            rsrc[1][1] = vld1q_f16(src1 + 0);
            rsrc[1][2] = vld1q_f16(src1 + 8);
        }
        if (has_bottom) {
            rsrc[2][0] = vdupq_n_f16(0);
            rsrc[2][1] = vld1q_f16(src2 + 0);
            rsrc[2][2] = vld1q_f16(src2 + 8);
        }

        int w = 0;
        const float16_t* src0_ptr = src0;
        const float16_t* src1_ptr = src1;
        const float16_t* src2_ptr = src2;
        float16_t* dst_ptr = dst;
        const float16_t* bias_ptr = bias;

        for (; w + 3 < W - 2; w += 4) {
            compute_element<4, 0, has_top, has_bottom, bias_mode>::call(
                    src0_ptr, src1_ptr, src2_ptr, dst_ptr, bias_ptr, init, rsrc,
                    rfilter, op);
        }
        if (w + 1 < W - 2) {
            compute_element<2, 0, has_top, has_bottom, bias_mode>::call(
                    src0_ptr, src1_ptr, src2_ptr, dst_ptr, bias_ptr, init, rsrc,
                    rfilter, op);
            shift_src<2>(rsrc);
            w += 2;
        }
        if (w < W - 2) {
            compute_element<1, 0, has_top, has_bottom, bias_mode>::call(
                    src0_ptr, src1_ptr, src2_ptr, dst_ptr, bias_ptr, init, rsrc,
                    rfilter, op);
            shift_src<1>(rsrc);
            w += 1;
        }
        compute_element_right<has_top, has_bottom, bias_mode>::call(
                dst_ptr, bias_ptr, init, rsrc, rfilter, op);
        compute_element_right_pad<has_top, has_bottom, bias_mode>::call(
                dst_ptr, bias_ptr, init, rsrc, rfilter, op);

        src0 += W * 8;
        src1 += W * 8;
        src2 += W * 8;
        dst += W * 8;
        bias += W * 8;
    }
};

}  // namespace

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw88::do_conv_kern_3x3_stride1_padding1(
        const float16_t* src, float16_t* dst, const float16_t* filter,
        const float16_t* bias, int H, int W) {
    Op op;

    float16x8_t init;
    if (bias_mode == BiasMode::NO_BIAS) {
        init = vdupq_n_f16(__fp16(0.f));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = vld1q_f16(bias);
    }

    const float16_t* src0 = src - W * 8;
    const float16_t* src1 = src;
    const float16_t* src2 = src + W * 8;

    float16x8_t rfilter[3][3];
    rfilter[0][0] = vld1q_f16(filter + 0);
    rfilter[0][1] = vld1q_f16(filter + 8);
    rfilter[0][2] = vld1q_f16(filter + 16);
    rfilter[1][0] = vld1q_f16(filter + 24);
    rfilter[1][1] = vld1q_f16(filter + 32);
    rfilter[1][2] = vld1q_f16(filter + 40);
    rfilter[2][0] = vld1q_f16(filter + 48);
    rfilter[2][1] = vld1q_f16(filter + 56);
    rfilter[2][2] = vld1q_f16(filter + 64);

    float16x8_t rsrc[3][4];

    compute_row<false, true, bias_mode>::call(
            src0, src1, src2, dst, bias, init, rsrc, rfilter, W, op);

    for (int h = 1; h < H - 1; h += 1) {
        compute_row<true, true, bias_mode>::call(
                src0, src1, src2, dst, bias, init, rsrc, rfilter, W, op);
    }

    compute_row<true, false, bias_mode>::call(
            src0, src1, src2, dst, bias, init, rsrc, rfilter, W, op);
}

#define INSTANTIATION(bias, Op)                                                     \
    template void channel_wise_nchw88::do_conv_kern_3x3_stride1_padding1<bias, Op>( \
            const float16_t*, float16_t*, const float16_t*, const float16_t*, int,  \
            int);

#define FOR_OP(bias)                       \
    INSTANTIATION(bias, SigmoidOp<__fp16>) \
    INSTANTIATION(bias, ReluOp<__fp16>)    \
    INSTANTIATION(bias, HSwishOp<__fp16>)  \
    INSTANTIATION(bias, NoneOp<__fp16>)

#define FOR_BIAS                             \
    FOR_OP(BiasMode::NO_BIAS)                \
    FOR_OP(BiasMode::BROADCAST_CHANNEL_BIAS) \
    FOR_OP(BiasMode::BIAS)

FOR_BIAS

#undef FOR_BIAS
#undef FOR_OP
#undef INSTANTIATION

#endif
// vim: syntax=cpp.doxygen
