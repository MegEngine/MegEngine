/**
 * \file dnn/src/x86/conv_bias/f32/strategy_6x3_8x8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/x86/conv_bias/f32/strategy.h"
#include "src/x86/elemwise_helper/op_unary.h"
#include "src/x86/avx_helper.h"

#include <x86intrin.h>
#ifdef WIN32
#include <avx2intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <smmintrin.h>
#endif

#include "midout.h"
MIDOUT_DECL(megdnn_x86_winograd_nchw88_fp32_F63_8x8)

using namespace megdnn;
using namespace x86;

namespace {
constexpr size_t alpha = 6 + 3 - 1;
struct InputTransform6X3_NCHW88 {
    template <bool inner>
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    static void prepare(const float* input, float* patch, float* patchT,
                        int ih_start, int iw_start, size_t IH, size_t IW,
                        size_t ic, size_t IC) {
        MEGDNN_MARK_USED_VAR(patch);
        size_t IW8 = IW * 8;              //! For nchw88 mode
        size_t iw8_start = iw_start * 8;  //! For nchw88 mode
        size_t icb = ic / 8;
        if (!(inner && ic + 8 < IC)) {
            memset(patchT, 0, sizeof(float) * 8 * alpha * alpha);
        }
        if (inner) {
            //! Copy to continue memory patchT,
            //! TODO:can be optimized
            const float* input_ptr =
                    input + icb * IH * IW8 + ih_start * IW8 + iw8_start;
            for (size_t ih = 0; ih < alpha; ih++) {
#define cb(i) auto v##i = _mm256_loadu_ps(input_ptr + 8 * i);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) _mm256_storeu_ps(patchT + ih * 8 * alpha + i * 8, v##i);

                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                input_ptr += IW8;
            }
        } else {
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            const float* input_ptr = input + icb * IH * IW8;
            // partial copy
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    auto src = _mm256_loadu_ps(input_ptr + ih * IW8 + iw * 8);
                    _mm256_storeu_ps(patchT + iho * 8 * alpha + iwo * 8, src);
                }
            }
        }
    }

    MEGDNN_ATTRIBUTE_TARGET("avx2")
    static void transform(const float* patchT, float* input_transform_buf,
                          size_t unit_idx, size_t nr_units_in_tile, size_t ic,
                          size_t IC) {
        // BT * d * B
#define cb(m, n)               \
    Vector<float, 8> d##m##n = \
            Vector<float, 8>::load(patchT + m * 8 * 8 + n * 8);
        UNROLL_CALL_NOWRAPPER_D2(8, 8, cb);
#undef cb

        //! B
        //!     1     0     0     0     0    0    0     0
        //!     0     1    -1   0.5  -0.5    2   -2    -1
        //! -5.25     1     1  0.25  0.25    4    4     0
        //!     0 -4.25  4.25  -2.5   2.5 -2.5  2.5  5.25
        //!  5.25 -4.25 -4.25 -1.25 -1.25   -5   -5     0
        //!     0     1    -1     2    -2  0.5 -0.5 -5.25
        //!    -1     1     1     1     1    1    1     0
        //!     0     0     0     0     0    0    0     1
#define cb(m)                                                                  \
    auto t0##m = d0##m + (d4##m - d2##m) * 5.25f - d6##m;                      \
    auto t1##m = d1##m + d2##m + d5##m + d6##m - (d3##m + d4##m) * 4.25f;      \
    auto t2##m = d2##m + d6##m - (d1##m + d5##m) + (d3##m - d4##m) * 4.25f;    \
    auto t3##m = d1##m * 0.5f + d2##m * 0.25f - d3##m * 2.5f - d4##m * 1.25f + \
                 d5##m * 2.f + d6##m;                                          \
    auto t4##m = d1##m * (-0.5f) + d2##m * 0.25f + d3##m * 2.5f -              \
                 d4##m * 1.25f - d5##m * 2.f + d6##m;                          \
    auto t5##m = d1##m * 2.f + d2##m * 4.f - d3##m * 2.5f - d4##m * 5.f +      \
                 d5##m * 0.5f + d6##m;                                         \
    auto t6##m = d1##m * (-2.f) + d2##m * 4.f + d3##m * 2.5f - d4##m * 5.f -   \
                 d5##m * 0.5f + d6##m;                                         \
    auto t7##m = (d7##m - d1##m) + (d3##m - d5##m) * 5.25f;

        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(m)                                                                  \
    d##m##0 = t##m##0 + (t##m##4 - t##m##2) * 5.25f - t##m##6;                 \
    d##m##1 = t##m##1 + t##m##2 + t##m##5 + t##m##6 -                          \
              (t##m##3 + t##m##4) * 4.25f;                                     \
    d##m##2 = t##m##2 + t##m##6 - (t##m##1 + t##m##5) +                        \
              (t##m##3 - t##m##4) * 4.25f;                                     \
    d##m##3 = t##m##1 * 0.5f + t##m##2 * 0.25f - t##m##3 * 2.5f -              \
              t##m##4 * 1.25f + t##m##5 * 2.f + t##m##6;                       \
    d##m##4 = t##m##1 * (-0.5f) + t##m##2 * 0.25f + t##m##3 * 2.5f -           \
              t##m##4 * 1.25f - t##m##5 * 2.f + t##m##6;                       \
    d##m##5 = t##m##1 * 2.f + t##m##2 * 4.f - t##m##3 * 2.5f - t##m##4 * 5.f + \
              t##m##5 * 0.5f + t##m##6;                                        \
    d##m##6 = t##m##1 * (-2.f) + t##m##2 * 4.f + t##m##3 * 2.5f -              \
              t##m##4 * 5.f - t##m##5 * 0.5f + t##m##6;                        \
    d##m##7 = (t##m##7 - t##m##1) + (t##m##3 - t##m##5) * 5.25f;

        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        size_t ICB = IC / 8;
        size_t icb = ic / 8;
#define cb(m, n)                                                \
    d##m##n.save(input_transform_buf +                          \
                 (m * alpha + n) * ICB * nr_units_in_tile * 8 + \
                 icb * nr_units_in_tile * 8 + unit_idx * 8);
        UNROLL_CALL_NOWRAPPER_D2(8, 8, cb)
#undef cb
    }
};

struct FilterTransform6X3_MCHW88 {
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    static void transform(const float* filter, float* filter_transform_buf,
                          float* transform_mid_buf, size_t OC, size_t IC,
                          size_t oc_start, size_t oc_end) {
        // Gg * GT
        // G
        // 1.0000000       0.0000000       0.0000000
        // -0.2222222      -0.2222222      -0.2222222
        // -0.2222222      0.2222222       -0.2222222
        // 0.0111111       0.0222222       0.0444444
        // 0.0111111       -0.0222222      0.0444444
        // 0.7111111       0.3555556       0.1777778
        // 0.7111111       -0.3555556      0.1777778
        // 0.0000000       0.0000000       1.0000000
        MEGDNN_MARK_USED_VAR(transform_mid_buf);
        megdnn_assert(
                (oc_end - oc_start) % 8 == 0 && oc_start % 8 == 0 &&
                        oc_end % 8 == 0 && IC % 8 == 0 && OC % 8 == 0,
                "Winograd filter transform input param is not times of 8!");
        size_t OCB = OC / 8;
        size_t ICB = IC / 8;

        for (size_t ocb = oc_start / 8; ocb < oc_end / 8; ocb++) {
            for (size_t icb = 0; icb < ICB; icb++) {
                for (size_t ic_inner = 0; ic_inner < 8; ic_inner++) {
                    const float* fptr = filter +
                                        (ocb * ICB + icb) * 3 * 3 * 8 * 8 +
                                        ic_inner * 8;

#define cb(m, n)               \
    Vector<float, 8> g##m##n = \
            Vector<float, 8>::load(fptr + (m * 3 + n) * 8 * 8);
                    UNROLL_CALL_NOWRAPPER_D2(3, 3, cb)
#undef cb

#define FILTER_TRANSFORM(n, wd, g)                      \
    auto wd##n##0 = g##0##n;                            \
    tmp0 = (g##0##n + g##2##n) * -0.2222222f;           \
    tmp1 = g##1##n * -0.2222222f;                       \
    auto wd##n##1 = tmp0 + tmp1;                        \
    auto wd##n##2 = tmp0 - tmp1;                        \
    tmp0 = g##0##n * 0.0111111f + g##2##n * 0.0444444f; \
    tmp1 = g##1##n * 0.0222222f;                        \
    auto wd##n##3 = tmp0 + tmp1;                        \
    auto wd##n##4 = tmp0 - tmp1;                        \
    tmp0 = g##0##n * 0.7111111f + g##2##n * 0.1777778f; \
    tmp1 = g##1##n * 0.3555556f;                        \
    auto wd##n##5 = tmp0 + tmp1;                        \
    auto wd##n##6 = tmp0 - tmp1;                        \
    auto wd##n##7 = g##2##n;
                    Vector<float, 8> tmp0, tmp1;
                    UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                    UNROLL_CALL_RAW(8, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM
#define cb_save(m, n)                                                        \
    ret##m##n.save(filter_transform_buf +                                    \
                   (m * alpha + n) * OCB * ICB * 8 * 8 + ocb * ICB * 8 * 8 + \
                   icb * 8 * 8 + ic_inner * 8);
                    UNROLL_CALL_NOWRAPPER_D2(8, 8, cb_save)
#undef cb_save
                }
            }
        }
    }
};
#define CONCAT(a, idx) a##idx
template <BiasMode bmode, typename Op>
struct OutputTransform6X3_NCHW88 {
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    static void transform(const float* output_transform_buf, const float* bias,
                          float* output, float* transform_mid_buf,
                          size_t oh_start, size_t ow_start, size_t OH,
                          size_t OW, size_t oc_start, size_t oc_end,
                          size_t oc_index, size_t unit_idx,
                          size_t nr_units_in_tile, const DType& src_dtype,
                          const DType& dst_dtype) {
        MEGDNN_MARK_USED_VAR(transform_mid_buf);

        Op op(src_dtype, dst_dtype);
        //! AT * m * A
        size_t OCB = (oc_end - oc_start) / 8;
        size_t oc = oc_start + oc_index;
        size_t ocb = oc_index / 8;

#define cb(m, n)                                           \
    auto v##m##n = Vector<float, 8>::load(                 \
            output_transform_buf +                         \
            (m * alpha + n) * OCB * nr_units_in_tile * 8 + \
            ocb * nr_units_in_tile * 8 + unit_idx * 8);
        UNROLL_CALL_NOWRAPPER_D2(8, 8, cb);
#undef cb

        /**
         * A
         *
         * 1    0    0      0       0         0
         * 1    1    1      1       1         1
         * 1   -1    1     -1       1        -1
         * 1    2    4      8      16        32
         * 1   -2    4     -8      16       -32
         * 1  0.5 0.25  0.125  0.0625   0.03125
         * 1 -0.5 0.25 -0.125  0.0625  -0.03125
         * 0  0.0    0      0       0         1
         */

        Vector<float, 8> v1addv2, v1subv2, v3addv4, v3subv4, v5addv6, v5subv6;
#define cb(m)                                                  \
    v1addv2 = v1##m + v2##m;                                   \
    v1subv2 = v1##m - v2##m;                                   \
    v3addv4 = v3##m + v4##m;                                   \
    v3subv4 = v3##m - v4##m;                                   \
    v5addv6 = v5##m + v6##m;                                   \
    v5subv6 = v5##m - v6##m;                                   \
    auto t0##m = v0##m + v1addv2 + v3addv4 + v5addv6;          \
    auto t1##m = v1subv2 + v3subv4 * 2.f + v5subv6 * 0.5f;     \
    auto t2##m = v1addv2 + v3addv4 * 4.f + v5addv6 * 0.25f;    \
    auto t3##m = v1subv2 + v3subv4 * 8.f + v5subv6 * 0.125f;   \
    auto t4##m = v1addv2 + v3addv4 * 16.f + v5addv6 * 0.0625f; \
    auto t5##m = v1subv2 + v3subv4 * 32.f + v5subv6 * 0.03125f + v7##m;

        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(m)                                               \
    v1addv2 = t##m##1 + t##m##2;                            \
    v1subv2 = t##m##1 - t##m##2;                            \
    v3addv4 = t##m##3 + t##m##4;                            \
    v3subv4 = t##m##3 - t##m##4;                            \
    v5addv6 = t##m##5 + t##m##6;                            \
    v5subv6 = t##m##5 - t##m##6;                            \
    v##m##0 = t##m##0 + v1addv2 + v3addv4 + v5addv6;        \
    v##m##1 = v1subv2 + v3subv4 * 2.f + v5subv6 * 0.5f;     \
    v##m##2 = v1addv2 + v3addv4 * 4.f + v5addv6 * 0.25f;    \
    v##m##3 = v1subv2 + v3subv4 * 8.f + v5subv6 * 0.125f;   \
    v##m##4 = v1addv2 + v3addv4 * 16.f + v5addv6 * 0.0625f; \
    v##m##5 = v1subv2 + v3subv4 * 32.f + v5subv6 * 0.03125f + t##m##7;

        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        Vector<float, 8> vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = Vector<float, 8>::load(bias + oc);

#define cb(m, n) v##m##n += vbias;
            UNROLL_CALL_RAW_D2(6, 6, cb);
#undef cb
        }
        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(CONCAT(v##m, n).value);
            UNROLL_CALL_RAW_D2(6, 6, cb);
#undef cb
        }
#define out_save(oho, owo)                                                   \
    do {                                                                     \
        size_t oh = oh_start + oho;                                          \
        size_t ow = ow_start + owo;                                          \
        if (oh < OH && ow < OW) {                                            \
            if (bmode == BiasMode::BIAS) {                                   \
                v##oho##owo += Vector<float, 8>::load(                       \
                        bias + oc / 8 * OH * OW * 8 + oh * OW * 8 + ow * 8); \
                v##oho##owo = op(v##oho##owo.value);                         \
            }                                                                \
            v##oho##owo.save(output + oc / 8 * OH * OW * 8 + oh * OW * 8 +   \
                             ow * 8);                                        \
        }                                                                    \
    } while (0);
        UNROLL_CALL_RAW_D2(6, 6, out_save);
    }
};
#undef CONCAT
}  // namespace

namespace megdnn {
namespace x86 {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_nchw88_6x3_8x8_f)

void winograd_nchw88_6x3_8x8_f::filter(const float* filter,
                                       float* filter_transform_buf,
                                       float* transform_mid_buf, size_t OC,
                                       size_t IC, size_t oc_start,
                                       size_t oc_end) {
    FilterTransform6X3_MCHW88::transform(filter, filter_transform_buf,
                                         transform_mid_buf, OC, IC, oc_start,
                                         oc_end);
}

void winograd_nchw88_6x3_8x8_f::input(const float* input,
                                      float* input_transform_buf,
                                      float* transform_mid_buf, size_t IH,
                                      size_t IW, size_t IC, size_t PH,
                                      size_t PW, size_t unit_start_idx,
                                      size_t nr_units_in_tile) {
    megdnn_assert(IC % 8 == 0);

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w =
            div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    float* patch = transform_mid_buf;
    float* patchT = transform_mid_buf + 8 * alpha * alpha;

    for (size_t ic = 0; ic < IC; ic += 8) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<size_t>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<size_t>(IW)) {
                InputTransform6X3_NCHW88::prepare<true>(input, patch, patchT,
                                                        ih_start, iw_start, IH,
                                                        IW, ic, IC);
                InputTransform6X3_NCHW88::transform(patchT, input_transform_buf,
                                                    unit_idx, nr_units_in_tile,
                                                    ic, IC);
            } else {
                InputTransform6X3_NCHW88::prepare<false>(input, patch, patchT,
                                                         ih_start, iw_start, IH,
                                                         IW, ic, IC);
                InputTransform6X3_NCHW88::transform(patchT, input_transform_buf,
                                                    unit_idx, nr_units_in_tile,
                                                    ic, IC);
            }
        }
    }
}

void winograd_nchw88_6x3_8x8_f::output(const float* output_transform_buf,
                                       const float* bias, float* output,
                                       float* transform_mid_buf, BiasMode bmode,
                                       NonlineMode nonline_mode, size_t OH,
                                       size_t OW, size_t oc_start,
                                       size_t oc_end, size_t unit_start_idx,
                                       size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...)                                       \
    OutputTransform6X3_NCHW88<_bmode MEGDNN_COMMA _nonline_op>::transform( \
            __VA_ARGS__);

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    size_t OC = oc_end - oc_start;

    megdnn_assert(OC % 8 == 0 && oc_start % 8 == 0 && oc_end % 8 == 0,
                  "Winograd output transform input param is not times of 8!");

    for (size_t oc = oc_start; oc + 8 <= oc_end; oc += 8) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;

            DISPATCH_CONV_WINOGRAD_BIAS(
                    megdnn_x86_winograd_nchw88_fp32_F63_8x8, cb, SIMDType::AVX2,
                    float, float, bmode, nonline_mode, output_transform_buf,
                    bias, output, transform_mid_buf, oh_start, ow_start, OH, OW,
                    oc_start, oc_end, oc_index, unit_idx, nr_units_in_tile,
                    src_dtype, dst_dtype);
        }
    }
#undef cb
}

}  // namespace winograd
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
