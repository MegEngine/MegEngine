#include "src/arm_common/conv_bias/f16/channel_wise_nchw88_kern.h"
#include "src/arm_common/conv_bias/f16/channel_wise_3x3_s1p1_nchw88_kern.h"
#include "src/arm_common/elemwise_helper/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

using namespace megdnn;
using namespace arm_common;
using namespace fp16;

namespace {

template <int size>
void load_vec(float16x8_t* dst, const __fp16* src);

#define cb(i) dst[i] = vld1q_f16(src + i * 8);
#define LOAD_MACRO(n)                                               \
    template <>                                                     \
    inline void load_vec<n>(float16x8_t * dst, const __fp16* src) { \
        UNROLL_CALL_NOWRAPPER(n, cb);                               \
    }
LOAD_MACRO(2);
LOAD_MACRO(3);
LOAD_MACRO(4);
LOAD_MACRO(5);
LOAD_MACRO(6);
LOAD_MACRO(7);
LOAD_MACRO(8);
LOAD_MACRO(9);
LOAD_MACRO(25);
#undef cb
#undef LOAD_MACRO

template <int size>
void compute_vec(float16x8_t& dst, float16x8_t* src, float16x8_t* filter);

#define cb(i) dst = vfmaq_f16(dst, src[i], filter[i]);
#define COMPUTE_MACRO(n)                                                  \
    template <>                                                           \
    inline void compute_vec<n>(                                           \
            float16x8_t & dst, float16x8_t * src, float16x8_t * filter) { \
        UNROLL_CALL_NOWRAPPER(n, cb);                                     \
    }
COMPUTE_MACRO(2);
COMPUTE_MACRO(3);
COMPUTE_MACRO(5);
#undef cb
#undef COMPUTE_MACRO

template <BiasMode bias_mode, int size>
struct load_bias_vec;

#define cb_bias(i) dst[i] = vld1q_f16((bptr) + i * 8);
#define cb_init(i) dst[i] = init;

#define INIT_BIAS_MACRO(n)                                                       \
    template <BiasMode bias_mode>                                                \
    struct load_bias_vec<bias_mode, n> {                                         \
        static void impl(                                                        \
                float16x8_t* dst, const float16x8_t& init, const __fp16* bptr) { \
            if (bias_mode == BiasMode::BIAS) {                                   \
                UNROLL_CALL_NOWRAPPER(n, cb_bias);                               \
            } else {                                                             \
                UNROLL_CALL_NOWRAPPER(n, cb_init);                               \
            }                                                                    \
        }                                                                        \
    };

INIT_BIAS_MACRO(1);
INIT_BIAS_MACRO(2);
INIT_BIAS_MACRO(4);
#undef cb_bias
#undef cb_init
#undef INIT_BIAS_MACRO
}  // namespace

#define COMPUTE_PADDING_KERNEL(oh)                                                     \
    do {                                                                               \
        int iw = ow * stride - PW;                                                     \
        float16x8_t result;                                                            \
        load_bias_vec<bias_mode, 1>::impl(&result, init, bias + (oh)*OW * 8 + ow * 8); \
        for (int kh = 0; kh < fh; kh++) {                                              \
            if (kh + ih < 0 || kh + ih >= static_cast<int>(IH))                        \
                continue;                                                              \
            for (int kw = 0; kw < fh; kw++) {                                          \
                if (kw + iw < 0 || kw + iw >= static_cast<int>(IW))                    \
                    continue;                                                          \
                const __fp16* sptr = src + (kh + ih) * IW * 8 + (kw + iw) * 8;         \
                result = vfmaq_f16(result, kernel[kh * fh + kw], vld1q_f16(sptr));     \
            }                                                                          \
        }                                                                              \
        __fp16* output = dst + (oh)*OW * 8 + ow * 8;                                   \
        op(result, output);                                                            \
    } while (0)

#define COMPUTE_PADDING_TOP()                         \
    do {                                              \
        size_t oh_start = (PH + stride - 1) / stride; \
        for (size_t oh = 0; oh < oh_start; oh++) {    \
            int ih = oh * stride - PH;                \
            for (size_t ow = 0; ow < OW; ow++) {      \
                COMPUTE_PADDING_KERNEL(oh);           \
            }                                         \
        }                                             \
    } while (0)

#define COMPUTE_PADDING_LEFT(n)                           \
    do {                                                  \
        for (int i = 0; i < n; ++i) {                     \
            size_t ow_start = (PW + stride - 1) / stride; \
            int ih = (oh + i) * stride - PH;              \
            for (size_t ow = 0; ow < ow_start; ow++) {    \
                COMPUTE_PADDING_KERNEL(oh + i);           \
            }                                             \
        }                                                 \
    } while (0)

#define COMPUTE_PADDING_RIGHT(n)                         \
    do {                                                 \
        for (int i = 0; i < n; ++i) {                    \
            size_t ow_end = (IW + PW - fh) / stride + 1; \
            int ih = (oh + i) * stride - PH;             \
            for (size_t ow = ow_end; ow < OW; ow++) {    \
                COMPUTE_PADDING_KERNEL(oh + i);          \
            }                                            \
        }                                                \
    } while (0)

#define COMPUTE_PADDING_BOTTOM()                     \
    do {                                             \
        size_t oh_end = (IH + PH - fh) / stride + 1; \
        for (size_t oh = oh_end; oh < OH; oh++) {    \
            int ih = oh * stride - PH;               \
            for (size_t ow = 0; ow < OW; ow++) {     \
                COMPUTE_PADDING_KERNEL(oh);          \
            }                                        \
        }                                            \
    } while (0)

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw88::do_conv_kern_stride1_2x2(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, const size_t PW) {
    float16x8_t kernel[4];
    load_vec<4>(kernel, filter);
    Op op;
    float16x8_t init;
    if (bias_mode == BiasMode::NO_BIAS) {
        init = vdupq_n_f16(__fp16(0.f));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = vld1q_f16(bias);
    }
    constexpr int fh = 2;
    constexpr int stride = 1;
    size_t oh_start = PH;
    size_t ow_start = PW;
    size_t oh_end = IH + PH - 1;
    size_t ow_end = IW + PW - 1;
#define COMPUTE_2X2(dst, src, kernel)        \
    compute_vec<2>(dst[0], &src[0], kernel); \
    compute_vec<2>(dst[1], &src[1], kernel); \
    compute_vec<2>(dst[2], &src[2], kernel); \
    compute_vec<2>(dst[3], &src[3], kernel)

    size_t oh = oh_start;
    COMPUTE_PADDING_TOP();
    for (; oh + 1 < oh_end; oh += 2) {
        COMPUTE_PADDING_LEFT(2);

        size_t ih = oh - oh_start;
        size_t ow = ow_start;
        for (; ow + 3 < ow_end; ow += 4) {
            size_t iw = ow - ow_start;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2][4];
            load_bias_vec<bias_mode, 4>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 4>::impl(
                    dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t src_v[3][5];
            load_vec<5>(src_v[0], input);
            COMPUTE_2X2(dst_v[0], src_v[0], &kernel[0]);
            load_vec<5>(src_v[1], input + IW * 8);
            COMPUTE_2X2(dst_v[0], src_v[1], &kernel[2]);
            COMPUTE_2X2(dst_v[1], src_v[1], &kernel[0]);
            load_vec<5>(src_v[2], input + 2 * IW * 8);
            COMPUTE_2X2(dst_v[1], src_v[2], &kernel[2]);

            op({{dst_v[0][0], dst_v[0][1]}}, output);
            op({{dst_v[0][2], dst_v[0][3]}}, output + 16);
            op({{dst_v[1][0], dst_v[1][1]}}, output + OW * 8);
            op({{dst_v[1][2], dst_v[1][3]}}, output + OW * 8 + 16);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow - ow_start;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2];
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t src_v[3][2];
            load_vec<2>(src_v[0], input);
            compute_vec<2>(dst_v[0], &src_v[0][0], &kernel[0]);
            load_vec<2>(src_v[1], input + IW * 8);
            compute_vec<2>(dst_v[0], &src_v[1][0], &kernel[2]);
            compute_vec<2>(dst_v[1], &src_v[1][0], &kernel[0]);
            load_vec<2>(src_v[2], input + 2 * IW * 8);
            compute_vec<2>(dst_v[1], &src_v[2][0], &kernel[2]);

            op(dst_v[0], output);
            op(dst_v[1], output + OW * 8);
        }

        COMPUTE_PADDING_RIGHT(2);
    }
    for (; oh < oh_end; oh++) {
        COMPUTE_PADDING_LEFT(1);

        size_t ih = oh - oh_start;
        size_t ow = ow_start;
        for (; ow + 3 < ow_end; ow += 4) {
            size_t iw = ow - ow_start;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[1][4];
            load_bias_vec<bias_mode, 4>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[2][5];
            load_vec<5>(src_v[0], input);
            COMPUTE_2X2(dst_v[0], src_v[0], &kernel[0]);
            load_vec<5>(src_v[1], input + IW * 8);
            COMPUTE_2X2(dst_v[0], src_v[1], &kernel[2]);

            op({{dst_v[0][0], dst_v[0][1]}}, output);
            op({{dst_v[0][2], dst_v[0][3]}}, output + 16);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow - ow_start;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v;
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v, init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[2][2];
            load_vec<2>(src_v[0], input);
            compute_vec<2>(dst_v, &src_v[0][0], &kernel[0]);
            load_vec<2>(src_v[1], input + IW * 8);
            compute_vec<2>(dst_v, &src_v[1][0], &kernel[2]);

            op(dst_v, output);
        }
        COMPUTE_PADDING_RIGHT(1);
    }
    COMPUTE_PADDING_BOTTOM();

#undef COMPUTE_2X2
}

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw88::do_conv_kern_stride1_3x3(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, const size_t PW) {
    if (IH == OH && IW == OW && IH >= 3 && IW >= 3 && PH == 1 && PW == 1) {
        do_conv_kern_3x3_stride1_padding1<bias_mode, Op>(
                src, dst, filter, bias, OH, OW);
        return;
    }

    float16x8_t kernel[9];
    load_vec<9>(kernel, filter);
    Op op;
    float16x8_t init;
    if (bias_mode == BiasMode::NO_BIAS) {
        init = vdupq_n_f16(__fp16(0.f));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = vld1q_f16(bias);
    }
    constexpr int fh = 3;
    constexpr int stride = 1;
    size_t oh_start = PH;
    size_t ow_start = PW;
    size_t oh_end = IH + PH - 2;
    size_t ow_end = IW + PW - 2;

    size_t oh = oh_start;
    COMPUTE_PADDING_TOP();
    for (; oh < oh_end; oh += 1) {
        COMPUTE_PADDING_LEFT(1);

        size_t ih = oh - PH;
        size_t ow = ow_start;
        for (; ow + 1 < ow_end; ow += 2) {
            size_t iw = ow - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[1][2];
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[3][4];
            load_vec<4>(src_v[0], input);
            load_vec<4>(src_v[1], input + IW * 8);
            load_vec<4>(src_v[2], input + 2 * IW * 8);
            compute_vec<3>(dst_v[0][0], &src_v[0][0], &kernel[0]);
            compute_vec<3>(dst_v[0][1], &src_v[0][1], &kernel[0]);
            compute_vec<3>(dst_v[0][0], &src_v[1][0], &kernel[3]);
            compute_vec<3>(dst_v[0][1], &src_v[1][1], &kernel[3]);
            compute_vec<3>(dst_v[0][0], &src_v[2][0], &kernel[6]);
            compute_vec<3>(dst_v[0][1], &src_v[2][1], &kernel[6]);

            op({{dst_v[0][0], dst_v[0][1]}}, output);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[1];
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[3][3];
            load_vec<3>(src_v[0], input);
            load_vec<3>(src_v[1], input + IW * 8);
            load_vec<3>(src_v[2], input + 2 * IW * 8);
            compute_vec<3>(dst_v[0], &src_v[0][0], &kernel[0]);
            compute_vec<3>(dst_v[0], &src_v[1][0], &kernel[3]);
            compute_vec<3>(dst_v[0], &src_v[2][0], &kernel[6]);

            op(dst_v[0], output);
        }

        COMPUTE_PADDING_RIGHT(1);
    }
    COMPUTE_PADDING_BOTTOM();
}

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw88::do_conv_kern_stride1_5x5(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, const size_t PW) {
    float16x8_t kernel[25];
    load_vec<25>(kernel, filter);
    Op op;
    float16x8_t init;
    if (bias_mode == BiasMode::NO_BIAS) {
        init = vdupq_n_f16(__fp16(0.f));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = vld1q_f16(bias);
    }
    constexpr int fh = 5;
    constexpr int stride = 1;
    size_t oh_start = PH;
    size_t ow_start = PW;
    size_t oh_end = IH + PH - 4;
    size_t ow_end = IW + PW - 4;

    size_t oh = oh_start;

    COMPUTE_PADDING_TOP();
    for (; oh + 1 < oh_end; oh += 2) {
        COMPUTE_PADDING_LEFT(2);

        size_t ih = oh - PH;
        size_t ow = ow_start;
        for (; ow + 1 < ow_end; ow += 2) {
            size_t iw = ow - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2][2];
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t kernel[2][5];
            float16x8_t src_v[2][6];
#define COMPUTE_5X5_4(i, dst, src, kernel0, kernel1) \
    load_vec<5>(kernel0, filter + i * 5 * 8);        \
    load_vec<6>(src, input + i * IW * 8);            \
    compute_vec<5>(dst[0][0], &src[0], kernel0);     \
    compute_vec<5>(dst[0][1], &src[1], kernel0);     \
    compute_vec<5>(dst[1][0], &src[0], kernel1);     \
    compute_vec<5>(dst[1][1], &src[1], kernel1)
            // line 0
            load_vec<5>(kernel[0], filter);
            load_vec<6>(src_v[0], input);
            compute_vec<5>(dst_v[0][0], &src_v[0][0], kernel[0]);
            compute_vec<5>(dst_v[0][1], &src_v[0][1], kernel[0]);
            // line 1
            COMPUTE_5X5_4(1, dst_v, src_v[1], kernel[1], kernel[0]);
            // line 2
            COMPUTE_5X5_4(2, dst_v, src_v[0], kernel[0], kernel[1]);
            // line 3
            COMPUTE_5X5_4(3, dst_v, src_v[1], kernel[1], kernel[0]);
            // line 4
            COMPUTE_5X5_4(4, dst_v, src_v[0], kernel[0], kernel[1]);
            // line 5
            load_vec<6>(src_v[1], input + 5 * IW * 8);
            compute_vec<5>(dst_v[1][0], &src_v[1][0], kernel[0]);
            compute_vec<5>(dst_v[1][1], &src_v[1][1], kernel[0]);
#undef COMPUTE_5X5_4
            op({{dst_v[0][0], dst_v[0][1]}}, output);
            op({{dst_v[1][0], dst_v[1][1]}}, output + OW * 8);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2][1];
            load_bias_vec<bias_mode, 1>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 1>::impl(
                    dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t kernel[2][5];
            float16x8_t src_v[2][5];
#define COMPUTE_5X5_2(i, dst, src, kernel0, kernel1) \
    load_vec<5>(kernel0, filter + i * 5 * 8);        \
    load_vec<6>(src, input + i * IW * 8);            \
    compute_vec<5>(dst[0][0], &src[0], kernel0);     \
    compute_vec<5>(dst[1][0], &src[0], kernel1);
            // line 0
            load_vec<5>(kernel[0], filter);
            load_vec<5>(src_v[0], input);
            compute_vec<5>(dst_v[0][0], &src_v[0][0], kernel[0]);
            // line 1
            COMPUTE_5X5_2(1, dst_v, src_v[1], kernel[1], kernel[0]);
            // line 2
            COMPUTE_5X5_2(2, dst_v, src_v[0], kernel[0], kernel[1]);
            // line 3
            COMPUTE_5X5_2(3, dst_v, src_v[1], kernel[1], kernel[0]);
            // line 4
            COMPUTE_5X5_2(4, dst_v, src_v[0], kernel[0], kernel[1]);
            // line 5
            load_vec<5>(src_v[1], input + 5 * IW * 8);
            compute_vec<5>(dst_v[1][0], &src_v[1][0], kernel[0]);
#undef COMPUTE_5X5_2
            op(dst_v[0][0], output);
            op(dst_v[1][0], output + OW * 8);
        }
        COMPUTE_PADDING_RIGHT(2);
    }
    for (; oh < oh_end; oh++) {
        COMPUTE_PADDING_LEFT(1);

        size_t ih = oh - PH;
        size_t ow = ow_start;
        for (; ow + 1 < ow_end; ow += 2) {
            size_t iw = ow - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[1][2];
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            float16x8_t kernel[2][5];
            float16x8_t src_v[2][6];
#define COMPUTE_5X5_2(i, dst, src, kernel)      \
    load_vec<5>(kernel, filter + i * 5 * 8);    \
    load_vec<6>(src, input + i * IW * 8);       \
    compute_vec<5>(dst[0][0], &src[0], kernel); \
    compute_vec<5>(dst[0][1], &src[1], kernel)
            // line 0
            COMPUTE_5X5_2(0, dst_v, src_v[0], kernel[0]);
            // line 1
            COMPUTE_5X5_2(1, dst_v, src_v[1], kernel[1]);
            // line 2
            COMPUTE_5X5_2(2, dst_v, src_v[0], kernel[0]);
            // line 3
            COMPUTE_5X5_2(3, dst_v, src_v[1], kernel[1]);
            // line 4
            COMPUTE_5X5_2(4, dst_v, src_v[0], kernel[0]);
#undef COMPUTE_5X5_2
            op({{dst_v[0][0], dst_v[0][1]}}, output);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v;
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v, init, bias + oh * OW * 8 + ow * 8);
            float16x8_t kernel[2][5];
            float16x8_t src_v[2][5];
#define COMPUTE_5X5_1(i, dst, src, kernel)   \
    load_vec<5>(kernel, filter + i * 5 * 8); \
    load_vec<6>(src, input + i * IW * 8);    \
    compute_vec<5>(dst, &src[0], kernel)
            // line 0
            COMPUTE_5X5_1(0, dst_v, src_v[0], kernel[0]);
            // line 1
            COMPUTE_5X5_1(1, dst_v, src_v[1], kernel[1]);
            // line 2
            COMPUTE_5X5_1(2, dst_v, src_v[0], kernel[0]);
            // line 3
            COMPUTE_5X5_1(3, dst_v, src_v[1], kernel[1]);
            // line 4
            COMPUTE_5X5_1(4, dst_v, src_v[0], kernel[0]);
#undef COMPUTE_5X5_1
            op(dst_v, output);
        }
        COMPUTE_PADDING_RIGHT(1);
    }
    COMPUTE_PADDING_BOTTOM();
}

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw88::do_conv_kern_stride2_2x2(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, const size_t PW) {
    float16x8_t kernel[4];
    load_vec<4>(kernel, filter);
    Op op;
    float16x8_t init;
    if (bias_mode == BiasMode::NO_BIAS) {
        init = vdupq_n_f16(__fp16(0.f));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = vld1q_f16(bias);
    }
    constexpr int fh = 2;
    constexpr int stride = 2;
    size_t oh_start = (PH + 1) / 2;
    size_t ow_start = (PW + 1) / 2;
    size_t oh_end = (IH + PH) / 2;
    size_t ow_end = (IW + PW) / 2;

#define COMPUTE_2X2(dst, src, kernel)        \
    compute_vec<2>(dst[0], &src[0], kernel); \
    compute_vec<2>(dst[1], &src[2], kernel); \
    compute_vec<2>(dst[2], &src[4], kernel); \
    compute_vec<2>(dst[3], &src[6], kernel)
    size_t oh = oh_start;
    COMPUTE_PADDING_TOP();
    for (; oh < oh_end; oh++) {
        COMPUTE_PADDING_LEFT(1);
        size_t ih = oh * 2 - PH;
        size_t ow = ow_start;
        for (; ow + 3 < ow_end; ow += 4) {
            size_t iw = ow * 2 - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[4];
            load_bias_vec<bias_mode, 4>::impl(
                    &dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[2][8];
            load_vec<8>(src_v[0], input);
            COMPUTE_2X2(dst_v, src_v[0], &kernel[0]);
            load_vec<8>(src_v[1], input + IW * 8);
            COMPUTE_2X2(dst_v, src_v[1], &kernel[2]);
#undef COMPUTE_2X2
            op({{dst_v[0], dst_v[1]}}, output);
            op({{dst_v[2], dst_v[3]}}, output + 16);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow * 2 - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v;
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v, init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[2][2];
            load_vec<2>(src_v[0], input);
            compute_vec<2>(dst_v, &src_v[0][0], &kernel[0]);
            load_vec<2>(src_v[1], input + IW * 8);
            compute_vec<2>(dst_v, &src_v[1][0], &kernel[2]);

            op(dst_v, output);
        }
        COMPUTE_PADDING_RIGHT(1);
    }
    COMPUTE_PADDING_BOTTOM();
#undef COMPUTE_2X2
}

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw88::do_conv_kern_stride2_3x3(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, const size_t PW) {
    float16x8_t kernel[9];
    load_vec<9>(kernel, filter);
    Op op;
    float16x8_t init;
    if (bias_mode == BiasMode::NO_BIAS) {
        init = vdupq_n_f16(__fp16(0.f));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = vld1q_f16(bias);
    }
    constexpr int fh = 3;
    constexpr int stride = 2;
    size_t oh_start = (PH + 1) / 2;
    size_t ow_start = (PW + 1) / 2;
    size_t oh_end = (IH + PH - 3) / 2 + 1;
    size_t ow_end = (IW + PW - 3) / 2 + 1;

    size_t oh = oh_start;
    COMPUTE_PADDING_TOP();
    for (; oh + 1 < oh_end; oh += 2) {
        COMPUTE_PADDING_LEFT(2);
        size_t ih = oh * 2 - PH;
        size_t ow = ow_start;
        for (; ow + 1 < ow_end; ow += 2) {
            size_t iw = ow * 2 - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2][2];
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t src_v[2][5];
            load_vec<5>(src_v[0], input);
            compute_vec<3>(dst_v[0][0], &src_v[0][0], &kernel[0]);
            compute_vec<3>(dst_v[0][1], &src_v[0][2], &kernel[0]);
            load_vec<5>(src_v[1], input + IW * 8);
            compute_vec<3>(dst_v[0][0], &src_v[1][0], &kernel[3]);
            compute_vec<3>(dst_v[0][1], &src_v[1][2], &kernel[3]);
            load_vec<5>(src_v[0], input + 2 * IW * 8);
            compute_vec<3>(dst_v[0][0], &src_v[0][0], &kernel[6]);
            compute_vec<3>(dst_v[0][1], &src_v[0][2], &kernel[6]);
            compute_vec<3>(dst_v[1][0], &src_v[0][0], &kernel[0]);
            compute_vec<3>(dst_v[1][1], &src_v[0][2], &kernel[0]);
            load_vec<5>(src_v[1], input + 3 * IW * 8);
            compute_vec<3>(dst_v[1][0], &src_v[1][0], &kernel[3]);
            compute_vec<3>(dst_v[1][1], &src_v[1][2], &kernel[3]);
            load_vec<5>(src_v[0], input + 4 * IW * 8);
            compute_vec<3>(dst_v[1][0], &src_v[0][0], &kernel[6]);
            compute_vec<3>(dst_v[1][1], &src_v[0][2], &kernel[6]);

            op({{dst_v[0][0], dst_v[0][1]}}, output);
            op({{dst_v[1][0], dst_v[1][1]}}, output + OW * 8);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow * 2 - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2];
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t src_v[2][3];
            load_vec<3>(src_v[0], input);
            compute_vec<3>(dst_v[0], &src_v[0][0], &kernel[0]);
            load_vec<3>(src_v[1], input + IW * 8);
            compute_vec<3>(dst_v[0], &src_v[1][0], &kernel[3]);
            load_vec<3>(src_v[0], input + 2 * IW * 8);
            compute_vec<3>(dst_v[0], &src_v[0][0], &kernel[6]);
            compute_vec<3>(dst_v[1], &src_v[0][0], &kernel[0]);
            load_vec<3>(src_v[1], input + 3 * IW * 8);
            compute_vec<3>(dst_v[1], &src_v[1][0], &kernel[3]);
            load_vec<3>(src_v[0], input + 4 * IW * 8);
            compute_vec<3>(dst_v[1], &src_v[0][0], &kernel[6]);

            op(dst_v[0], output);
            op(dst_v[1], output + OW * 8);
        }
        COMPUTE_PADDING_RIGHT(2);
    }
    for (; oh < oh_end; oh++) {
        COMPUTE_PADDING_LEFT(1);
        size_t ih = oh * 2 - PH;
        size_t ow = ow_start;
        for (; ow + 1 < ow_end; ow += 2) {
            size_t iw = ow * 2 - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2];
            load_bias_vec<bias_mode, 2>::impl(
                    &dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[3][5];
            load_vec<5>(src_v[0], input);
            compute_vec<3>(dst_v[0], &src_v[0][0], &kernel[0]);
            compute_vec<3>(dst_v[1], &src_v[0][2], &kernel[0]);
            load_vec<5>(src_v[1], input + IW * 8);
            compute_vec<3>(dst_v[0], &src_v[1][0], &kernel[3]);
            compute_vec<3>(dst_v[1], &src_v[1][2], &kernel[3]);
            load_vec<5>(src_v[2], input + 2 * IW * 8);
            compute_vec<3>(dst_v[0], &src_v[2][0], &kernel[6]);
            compute_vec<3>(dst_v[1], &src_v[2][2], &kernel[6]);
            op({{dst_v[0], dst_v[1]}}, output);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow * 2 - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v;
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v, init, bias + oh * OW * 8 + ow * 8);
            float16x8_t src_v[3][3];
            load_vec<3>(src_v[0], input);
            compute_vec<3>(dst_v, &src_v[0][0], &kernel[0]);
            load_vec<3>(src_v[1], input + IW * 8);
            compute_vec<3>(dst_v, &src_v[1][0], &kernel[3]);
            load_vec<3>(src_v[2], input + 2 * IW * 8);
            compute_vec<3>(dst_v, &src_v[2][0], &kernel[6]);
            op(dst_v, output);
        }
        COMPUTE_PADDING_RIGHT(1);
    }
    COMPUTE_PADDING_BOTTOM();
}

template <BiasMode bias_mode, typename Op>
void channel_wise_nchw88::do_conv_kern_stride2_5x5(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, const size_t PW) {
    float16x8_t kernel[25];
    load_vec<25>(kernel, filter);
    Op op;
    float16x8_t init;
    if (bias_mode == BiasMode::NO_BIAS) {
        init = vdupq_n_f16(__fp16(0.f));
    } else if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init = vld1q_f16(bias);
    }
    constexpr int fh = 5;
    constexpr int stride = 2;
    size_t oh_start = (PH + stride - 1) / stride;
    size_t ow_start = (PW + stride - 1) / stride;
    size_t oh_end = (IH + PH - 5) / stride + 1;
    size_t ow_end = (IW + PW - 5) / stride + 1;

    size_t oh = oh_start;
    COMPUTE_PADDING_TOP();
    for (; oh + 1 < oh_end; oh += 2) {
        COMPUTE_PADDING_LEFT(2);
        size_t ih = oh * stride - PH;
        size_t ow = ow_start;
        for (; ow + 1 < ow_end; ow += 2) {
            size_t iw = ow * stride - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2][2];
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 2>::impl(
                    dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t kernel[3][5];
            float16x8_t src_v[2][7];
#define COMPUTE_5X5_4(i, dst, src, kernel0, kernel1) \
    load_vec<5>(kernel0, filter + i * 5 * 8);        \
    load_vec<7>(src, input + i * IW * 8);            \
    compute_vec<5>(dst[0][0], &src[0], kernel0);     \
    compute_vec<5>(dst[0][1], &src[2], kernel0);     \
    compute_vec<5>(dst[1][0], &src[0], kernel1);     \
    compute_vec<5>(dst[1][1], &src[2], kernel1)

#define COMPUTE_5X5_2(i, dst, src, kernel)   \
    load_vec<7>(src, input + i * IW * 8);    \
    compute_vec<5>(dst[0], &src[0], kernel); \
    compute_vec<5>(dst[1], &src[2], kernel)
            // line 0
            load_vec<5>(kernel[0], filter);
            COMPUTE_5X5_2(0, dst_v[0], src_v[0], kernel[0]);
            // line 1
            load_vec<5>(kernel[1], filter + 5 * 8);
            COMPUTE_5X5_2(1, dst_v[0], src_v[1], kernel[1]);
            // line 2
            COMPUTE_5X5_4(2, dst_v, src_v[0], kernel[2], kernel[0]);
            // line 3
            COMPUTE_5X5_4(3, dst_v, src_v[1], kernel[0], kernel[1]);
            // line 4
            COMPUTE_5X5_4(4, dst_v, src_v[0], kernel[1], kernel[2]);
            // line 5
            COMPUTE_5X5_2(5, dst_v[1], src_v[1], kernel[0]);
            // line 6
            COMPUTE_5X5_2(6, dst_v[1], src_v[0], kernel[1]);
#undef COMPUTE_5X5_4
#undef COMPUTE_5X5_2
            op({{dst_v[0][0], dst_v[0][1]}}, output);
            op({{dst_v[1][0], dst_v[1][1]}}, output + OW * 8);
        }
        for (; ow < ow_end; ow++) {
            size_t iw = ow * stride - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v[2];
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v[0], init, bias + oh * OW * 8 + ow * 8);
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v[1], init, bias + (oh + 1) * OW * 8 + ow * 8);
            float16x8_t kernel[3][5];
            float16x8_t src_v[2][5];
#define COMPUTE_5X5_2(i, dst, src, kernel0, kernel1) \
    load_vec<5>(kernel0, filter + i * 5 * 8);        \
    load_vec<5>(src, input + i * IW * 8);            \
    compute_vec<5>(dst[0], &src[0], kernel0);        \
    compute_vec<5>(dst[1], &src[0], kernel1);

#define COMPUTE_5X5_1(i, dst, src, kernel) \
    load_vec<5>(src, input + i * IW * 8);  \
    compute_vec<5>(dst, &src[0], kernel);  \
            // line 0
            load_vec<5>(kernel[0], filter);
            COMPUTE_5X5_1(0, dst_v[0], src_v[0], kernel[0]);
            // line 1
            load_vec<5>(kernel[1], filter + 5 * 8);
            COMPUTE_5X5_1(1, dst_v[0], src_v[1], kernel[1]);
            // line 2
            COMPUTE_5X5_2(2, dst_v, src_v[0], kernel[2], kernel[0]);
            // line 3
            COMPUTE_5X5_2(3, dst_v, src_v[1], kernel[0], kernel[1]);
            // line 4
            COMPUTE_5X5_2(4, dst_v, src_v[0], kernel[1], kernel[2]);
            // line 5
            COMPUTE_5X5_1(5, dst_v[1], src_v[1], kernel[0]);
            // line 6
            COMPUTE_5X5_1(6, dst_v[1], src_v[0], kernel[1]);
#undef COMPUTE_5X5_2
#undef COMPUTE_5X5_1
            op(dst_v[0], output);
            op(dst_v[1], output + OW * 8);
        }
        COMPUTE_PADDING_RIGHT(2);
    }
    for (; oh < oh_end; oh++) {
        COMPUTE_PADDING_LEFT(1);
        size_t ih = oh * stride - PH;
        size_t ow = ow_start;
        for (; ow < ow_end; ow++) {
            size_t iw = ow * stride - PW;
            const __fp16* input = src + ih * IW * 8 + iw * 8;
            __fp16* output = dst + oh * OW * 8 + ow * 8;
            float16x8_t dst_v;
            load_bias_vec<bias_mode, 1>::impl(
                    &dst_v, init, bias + oh * OW * 8 + ow * 8);
            float16x8_t kernel[2][5];
            float16x8_t src_v[2][5];
#define COMPUTE_5X5_1(i, dst, src, kernel)   \
    load_vec<5>(kernel, filter + i * 5 * 8); \
    load_vec<6>(src, input + i * IW * 8);    \
    compute_vec<5>(dst, &src[0], kernel)
            // line 0
            COMPUTE_5X5_1(0, dst_v, src_v[0], kernel[0]);
            // line 1
            COMPUTE_5X5_1(1, dst_v, src_v[1], kernel[1]);
            // line 2
            COMPUTE_5X5_1(2, dst_v, src_v[0], kernel[0]);
            // line 3
            COMPUTE_5X5_1(3, dst_v, src_v[1], kernel[1]);
            // line 4
            COMPUTE_5X5_1(4, dst_v, src_v[0], kernel[0]);
#undef COMPUTE_5X5_1
            op(dst_v, output);
        }
        COMPUTE_PADDING_RIGHT(1);
    }
    COMPUTE_PADDING_BOTTOM();
}

#define INSTANTIATION(stride, i, bias, Op)                                          \
    template void channel_wise_nchw88::do_conv_kern_##stride##_##i##x##i<bias, Op>( \
            const __fp16*, const __fp16*, const __fp16*, __fp16*, const size_t,     \
            const size_t, const size_t, const size_t, const size_t, const size_t);

#define FOR_OP(stride, i, bias)                       \
    INSTANTIATION(stride, i, bias, SigmoidOp<__fp16>) \
    INSTANTIATION(stride, i, bias, ReluOp<__fp16>)    \
    INSTANTIATION(stride, i, bias, HSwishOp<__fp16>)  \
    INSTANTIATION(stride, i, bias, NoneOp<__fp16>)

#define FOR_BIAS(stride, i)                             \
    FOR_OP(stride, i, BiasMode::NO_BIAS)                \
    FOR_OP(stride, i, BiasMode::BROADCAST_CHANNEL_BIAS) \
    FOR_OP(stride, i, BiasMode::BIAS)

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

#endif

// vim: syntax=cpp.doxygen
