#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/gi/fp32/filter_transform.h"
#include "src/fallback/conv_bias/gi/fp32/helper.h"
#include "src/fallback/conv_bias/gi/fp32/strategy.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/fallback/elemwise_helper/op_unary.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_winograd_fp32_F43)

using namespace megdnn;
using namespace fallback;
namespace {

/**
 * input transform
 *
 * wd0 = 4 * (d0 - d2) - (d2 - d4)
 * wd1 = -4 * (d1 + d2) + (d3 + d4)
 * wd2 = 4 * (d1 - d2) + (d4 - d3)
 * wd3 = 2 * (d3 - d1) - (d2 - d4)
 * wd4 = -2 * (d3 - d1) - (d2 - d4)
 * wd5 = -4 * (d3 - d1) + (d5 - d3)
 */

#define INPUT_TRANSFORM(d, wd, i)                                                     \
    do {                                                                              \
        auto tmp0 = SUBF(d##2##i, d##4##i);                                           \
        auto tmp1 = SUBF(d##3##i, d##1##i);                                           \
        wd##0##i = SUBF(MULSF(SUBF(d##0##i, d##2##i), 4.0f), tmp0);                   \
        wd##1##i = SUBF(ADDF(d##3##i, d##4##i), MULSF(ADDF(d##1##i, d##2##i), 4.0f)); \
        wd##2##i = ADDF(MULSF(SUBF(d##1##i, d##2##i), 4.0f), SUBF(d##4##i, d##3##i)); \
        wd##3##i = SUBF(MULSF(tmp1, 2.0f), tmp0);                                     \
        wd##4##i = SUBF(MULSF(tmp1, -2.0f), tmp0);                                    \
        wd##5##i = SUBF(SUBF(d##5##i, d##3##i), MULSF(tmp1, 4.0f));                   \
    } while (0);

#define INPUT_TRANSFORM_V2(d, wd) \
    INPUT_TRANSFORM(d, wd, 0);    \
    INPUT_TRANSFORM(d, wd, 1);

#define GET_VECTOR_HIGH_ELEM(s, i, idx) GiExtractLane##idx##Float32(s##i##1)
#define GET_VECTOR_LOW_ELEM(s, i, idx)  GiExtractLane##idx##Float32(s##i##0)

struct InputTransform4X3 {
    template <bool inner>
    static void transform(
            const float* input, float* input_transform_buf, float* transform_mid_buf,
            int ih_start, int iw_start, size_t ic, size_t IH, size_t IW, size_t IC,
            size_t unit_idx, size_t nr_units_in_tile) {
        constexpr size_t alpha = 4 + 3 - 1;
        if (!inner) {
            memset(transform_mid_buf, 0, sizeof(float) * alpha * alpha);
        }

#define cb(i, j) GI_FLOAT32_t d##i##j;
        UNROLL_CALL_NOWRAPPER_D2(6, 2, cb);
#undef cb
        if (inner) {
            const float* input_ptr = input + ic * IH * IW + ih_start * IW + iw_start;
#define cb(i, j) d##i##j = GiLoadFloat32(input_ptr + IW * i + 4 * j);
            UNROLL_CALL_NOWRAPPER_D2(5, 2, cb);
#undef cb
            d50 = GiLoadFloat32(input_ptr + IW * 5);
            d51 = GiLoadFloat32LowHalf(input_ptr + IW * 5 + 4);
        } else {
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    transform_mid_buf[iho * alpha + iwo] =
                            input[ic * IH * IW + ih * IW + iw];
                }
            }
#define cb(i, j) d##i##j = GiLoadFloat32(transform_mid_buf + alpha * i + 4 * j);
            UNROLL_CALL_NOWRAPPER_D2(5, 2, cb);
#undef cb
            d50 = GiLoadFloat32(transform_mid_buf + alpha * 5);
            d51 = GiLoadFloat32LowHalf(transform_mid_buf + alpha * 5 + 4);
        }

#define cb(i, j) GI_FLOAT32_t wd##i##j;
        UNROLL_CALL_NOWRAPPER_D2(6, 2, cb);
#undef cb

        INPUT_TRANSFORM_V2(d, wd);

#if MEGDNN_AARCH64
#define cb(i, j) GI_FLOAT32_t ret##i##j;
        UNROLL_CALL_NOWRAPPER_D2(6, 2, cb);
#undef cb
        TRANSPOSE_6x6(wd, d);
        INPUT_TRANSFORM_V2(d, ret);

#define cb(i, j) GiStoreFloat32(transform_mid_buf + i * alpha + j * 4, ret##i##j);
        UNROLL_CALL_NOWRAPPER_D2(5, 2, cb);
#undef cb

        GiStoreFloat32(transform_mid_buf + 5 * alpha, ret50);
        float tmp[4];
        GiStoreFloat32(tmp, ret51);
        memcpy(transform_mid_buf + 5 * alpha + 4, tmp, sizeof(float) * 2);

        rep(i, alpha) rep(j, alpha) {
            input_transform_buf
                    [(i * alpha + j) * nr_units_in_tile * IC + unit_idx * IC + ic] =
                            transform_mid_buf[j * alpha + i];
        }
#else
        //!     4     0     0     0     0    0
        //!     0    -4     4    -2     2    4
        //!    -5    -4    -4    -1    -1    0
        //!     0     1    -1     2    -2   -5
        //!     1     1     1     1     1    0
        //!     0     0     0     0     0    1
#define cb(i)                                                                       \
    do {                                                                            \
        auto tmp0 = GET_VECTOR_LOW_ELEM(wd, i, 2) - GET_VECTOR_HIGH_ELEM(wd, i, 0); \
        auto tmp1 = GET_VECTOR_LOW_ELEM(wd, i, 3) - GET_VECTOR_LOW_ELEM(wd, i, 1);  \
        mid_buf1[0] =                                                               \
                (GET_VECTOR_LOW_ELEM(wd, i, 0) - GET_VECTOR_LOW_ELEM(wd, i, 2)) *   \
                        4.0f -                                                      \
                tmp0;                                                               \
        mid_buf1[1] =                                                               \
                (GET_VECTOR_LOW_ELEM(wd, i, 1) + GET_VECTOR_LOW_ELEM(wd, i, 2)) *   \
                        -4.0f +                                                     \
                (GET_VECTOR_LOW_ELEM(wd, i, 3) + GET_VECTOR_HIGH_ELEM(wd, i, 0));   \
        mid_buf1[2] =                                                               \
                (GET_VECTOR_LOW_ELEM(wd, i, 1) - GET_VECTOR_LOW_ELEM(wd, i, 2)) *   \
                        4.0f +                                                      \
                (GET_VECTOR_HIGH_ELEM(wd, i, 0) - GET_VECTOR_LOW_ELEM(wd, i, 3));   \
        mid_buf1[3] = 2.0f * tmp1 - tmp0;                                           \
        mid_buf1[4] = -2.0f * tmp1 - tmp0;                                          \
        mid_buf1[5] = -4.0f * tmp1 + (GET_VECTOR_HIGH_ELEM(wd, i, 1) -              \
                                      GET_VECTOR_LOW_ELEM(wd, i, 3));               \
        mid_buf1 += 6;                                                              \
    } while (0);

        float* mid_buf1 = transform_mid_buf;
        UNROLL_CALL_NOWRAPPER(6, cb);
        mid_buf1 = transform_mid_buf;

#undef cb
        rep(i, alpha) rep(j, alpha) {
            input_transform_buf
                    [(i * alpha + j) * nr_units_in_tile * IC + unit_idx * IC + ic] =
                            transform_mid_buf[i * alpha + j];
        }
#endif
    }
};

#undef INPUT_TRANSFORM_V2
#undef INPUT_TRANSFORM

/**
 * Output Transform: use fma
 *
 * s0 = m0 + (m1 + m2) +     (m3 + m4)
 * s1 =      (m1 - m2) + 2 * (m3 - m4)
 * s2 =      (m1 + m2) + 4 * (m3 + m4)
 * s3 =      (m1 - m2) + 8 * (m3 - m4) + m5
 */
#define OUTPUT_TRANSFORM(m, s, i)                                     \
    do {                                                              \
        auto m1addm2 = ADDF(m##1##i, m##2##i);                        \
        auto m1subm2 = SUBF(m##1##i, m##2##i);                        \
        auto m3addm4 = ADDF(m##3##i, m##4##i);                        \
        auto m3subm4 = SUBF(m##3##i, m##4##i);                        \
        s##0##i = m##0##i;                                            \
        s##0##i = ADDF(s##0##i, m1addm2);                             \
        s##0##i = ADDF(s##0##i, m3addm4);                             \
        s##1##i = m1subm2;                                            \
        s##1##i = GiMultiplyAddScalarFloat32(s##1##i, m3subm4, 2.0f); \
        s##2##i = m1addm2;                                            \
        s##2##i = GiMultiplyAddScalarFloat32(s##2##i, m3addm4, 4.0f); \
        s##3##i = m1subm2;                                            \
        s##3##i = GiMultiplyAddScalarFloat32(s##3##i, m3subm4, 8.0f); \
        s##3##i = ADDF(s##3##i, m##5##i);                             \
    } while (0);

#define OUTPUT_TRANSFORM_V2(m, s) \
    OUTPUT_TRANSFORM(m, s, 0);    \
    OUTPUT_TRANSFORM(m, s, 1);

template <BiasMode bmode, typename Op>
struct OutputTransform4X3 {
    static void transform(
            const float* output_transform_buf, const float* bias, float* output,
            float* transform_mid_buf, size_t oh_start, size_t ow_start, size_t OH,
            size_t OW, size_t oc_start, size_t oc_end, size_t oc_index, size_t unit_idx,
            size_t nr_units_in_tile, const DType& src_dtype, const DType& dst_dtype) {
        constexpr size_t alpha = 4 + 3 - 1;
        Op op(src_dtype, dst_dtype);
        float* mid_buf1 = transform_mid_buf;

        //! AT * m * A
        size_t OC = oc_end - oc_start;
        size_t oc = oc_start + oc_index;

#define cb(m, n)                                            \
    transform_mid_buf[m * alpha + n] = output_transform_buf \
            [(m * alpha + n) * nr_units_in_tile * OC + unit_idx * OC + oc_index];
        UNROLL_CALL_NOWRAPPER_D2(6, 6, cb);
#undef cb

#define cb(i, j) auto m##i##j = GiLoadFloat32(transform_mid_buf + alpha * i + 4 * j);
        UNROLL_CALL_NOWRAPPER_D2(5, 2, cb);
#undef cb
        GI_FLOAT32_t m50, m51;
        m50 = GiLoadFloat32(transform_mid_buf + alpha * 5);
        m51 = GiLoadFloat32LowHalf(transform_mid_buf + alpha * 5 + 4);
#define cb(i, j) GI_FLOAT32_t s##i##j;
        UNROLL_CALL_NOWRAPPER_D2(4, 2, cb);
#undef cb

        OUTPUT_TRANSFORM_V2(m, s);
        /**
         * Output transform: s * A
         *
         * 1    0   0   0
         * 1    1   1   1
         * 1    -1  1   -1
         * 1    2   4   8
         * 1    -2  4   -8
         * 0    0   0   1
         */
#define cb(i)                                                                        \
    do {                                                                             \
        auto m1addm2 = GET_VECTOR_LOW_ELEM(s, i, 1) + GET_VECTOR_LOW_ELEM(s, i, 2);  \
        auto m1subm2 = GET_VECTOR_LOW_ELEM(s, i, 1) - GET_VECTOR_LOW_ELEM(s, i, 2);  \
        auto m3addm4 = GET_VECTOR_LOW_ELEM(s, i, 3) + GET_VECTOR_HIGH_ELEM(s, i, 0); \
        auto m3subm4 = GET_VECTOR_LOW_ELEM(s, i, 3) - GET_VECTOR_HIGH_ELEM(s, i, 0); \
        mid_buf1[0] = GET_VECTOR_LOW_ELEM(s, i, 0) + m1addm2 + m3addm4;              \
        mid_buf1[1] = m1subm2 + 2.f * m3subm4;                                       \
        mid_buf1[2] = m1addm2 + 4.f * m3addm4;                                       \
        mid_buf1[3] = m1subm2 + 8.f * m3subm4 + GET_VECTOR_HIGH_ELEM(s, i, 1);       \
        mid_buf1 += 4;                                                               \
    } while (0);

        mid_buf1 = transform_mid_buf;
        UNROLL_CALL_NOWRAPPER(4, cb);
        mid_buf1 = transform_mid_buf;
#undef cb

        if (oh_start + 4 <= OH && ow_start + 4 <= OW) {
            GI_FLOAT32_t bias0;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias0 = GiBroadcastFloat32(bias[oc]);
            }
            rep(i, 4) {
                size_t oh = oh_start + i;
                GI_FLOAT32_t item0 = GiLoadFloat32(mid_buf1);

                if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                    item0 = GiAddFloat32(item0, bias0);
                } else if (bmode == BiasMode::BIAS) {
                    bias0 = GiLoadFloat32(bias + oc * OH * OW + oh * OW + ow_start);
                    item0 = GiAddFloat32(item0, bias0);
                }
                item0 = op(item0);
                GiStoreFloat32(output + oc * OH * OW + oh * OW + ow_start, item0);

                mid_buf1 += 4;
            }
        } else {
            for (size_t oho = 0; oho < 4 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 4 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    float res = mid_buf1[oho * 4 + owo];
                    if (bmode == BiasMode::BIAS) {
                        res += bias[oc * OH * OW + oh * OW + ow];
                    } else if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        res += bias[oc];
                    }
                    res = op(res);
                    output[oc * OH * OW + oh * OW + ow] = res;
                }
            }
        }
    }
};

#undef GET_VECTOR_HIGH_ELEM
#undef GET_VECTOR_LOW_ELEM
#undef OUTPUT_TRANSFORM_V2
#undef OUTPUT_TRANSFORM

}  // namespace

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_4x3_1x1_f)

void winograd_4x3_1x1_f::filter(
        const float* filter, float* filter_transform_buf, float* transform_mid_buf,
        size_t OC, size_t IC, size_t oc_start, size_t oc_end) {
    FilterTransform4X3<param::MatrixMul::Format::DEFAULT>::transform(
            filter, filter_transform_buf, transform_mid_buf, OC, IC, oc_start, oc_end);
}

void winograd_4x3_1x1_f::input(
        const float* input, float* input_transform_buf, float* transform_mid_buf,
        size_t IH, size_t IW, size_t IC, size_t PH, size_t PW, size_t unit_start_idx,
        size_t nr_units_in_tile) {
    constexpr int alpha = 3 + 4 - 1;

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    rep(ic, IC) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransform4X3::transform<true>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);

            } else {
                InputTransform4X3::transform<false>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);
            }
        }
    }
}

void winograd_4x3_1x1_f::output(
        const float* output_transform_buf, const float* bias, float* output,
        float* transform_mid_buf, BiasMode bmode, NonlineMode nonline_mode, size_t OH,
        size_t OW, size_t oc_start, size_t oc_end, size_t unit_start_idx,
        size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...) \
    OutputTransform4X3<_bmode MEGDNN_COMMA _nonline_op>::transform(__VA_ARGS__);

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);

    for (size_t oc = oc_start; oc < oc_end; oc++) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            GI_DISPATCH_CONV_WINOGRAD_BIAS(
                    megdnn_fallback_winograd_fp32_F43, cb, float, float, bmode,
                    nonline_mode, output_transform_buf, bias, output, transform_mid_buf,
                    oh_start, ow_start, OH, OW, oc_start, oc_end, oc_index, unit_idx,
                    nr_units_in_tile, src_dtype, dst_dtype);
        }
    }
#undef cb
}

}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
