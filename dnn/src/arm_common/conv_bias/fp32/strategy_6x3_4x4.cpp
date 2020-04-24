/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy_6x3_4x4.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/fp32/filter_transform.h"
#include "src/arm_common/conv_bias/fp32/helper.h"
#include "src/arm_common/conv_bias/fp32/strategy.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_fp32_F63_4x4)

using namespace megdnn;
using namespace arm_common;

namespace {

struct InputTransform6X3 {
    template <bool inner>
    static void prepare(const float* input, float* patch, float* patchT,
                        int ih_start, int iw_start, size_t IH, size_t IW,
                        size_t ic, size_t IC) {
        constexpr size_t alpha = 6 + 3 - 1;
        if (!(inner && ic + 4 < IC)) {
            memset(patch, 0, sizeof(float) * 4 * alpha * alpha);
        }
        if (inner) {
            const float* input_ptr =
                    input + ic * IH * IW + ih_start * IW + iw_start;
            for (size_t ico = 0; ico < 4; ++ico) {
                if (ic + ico < IC) {
#define cb(i)                                     \
    auto v##i##0 = vld1q_f32(input_ptr + IW * i); \
    auto v##i##1 = vld1q_f32(input_ptr + IW * i + 4);

                    UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i)                                            \
    vst1q_f32(patch + ico * 8 * alpha + i * 8, v##i##0); \
    vst1q_f32(patch + ico * 8 * alpha + i * 8 + 4, v##i##1);

                    UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                    input_ptr += IH * IW;
                }
            }
        } else {
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            // partial copy
            for (size_t ico = 0; ico < 4; ++ico) {
                if (ic + ico < IC) {
                    for (int ih = ih0_act; ih < ih1_act; ++ih) {
                        for (int iw = iw0_act; iw < iw1_act; ++iw) {
                            size_t iho = ih - ih_start, iwo = iw - iw_start;
                            patch[ico * alpha * 8 + iho * 8 + iwo] =
                                    input[(ic + ico) * IH * IW + ih * IW + iw];
                        }
                    }
                }
            }
        }

#define cb(i)                                                    \
    transpose_4x4(patch + 8 * i + 0, patchT + 8 * i * 4, 64, 4); \
    transpose_4x4(patch + 8 * i + 4, patchT + 8 * i * 4 + 4 * 4, 64, 4);
        UNROLL_CALL_NOWRAPPER(8, cb)
#undef cb
    }

    static void transform(const float* patchT, float* input_transform_buf,
                          size_t unit_idx, size_t nr_units_in_tile, size_t ic,
                          size_t IC) {
        constexpr size_t alpha = 6 + 3 - 1;
        // BT * d * B
#define cb(m, n) \
        Vector<float, 4> d##m##n = \
                Vector<float, 4>::load(patchT + m * 8 * 4 + n * 4);

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

        size_t ICB = IC / 4;
        size_t icb = ic / 4;
#define cb(m, n)                                                \
    d##m##n.save(input_transform_buf +                          \
                 (m * alpha + n) * ICB * nr_units_in_tile * 4 + \
                 icb * nr_units_in_tile * 4 + unit_idx * 4);
        UNROLL_CALL_NOWRAPPER_D2(8, 8, cb)
#undef cb
    }
};

template <BiasMode bmode, typename Op>
struct OutputTransform6X3 {
    static void transform(const float* output_transform_buf, const float* bias,
                          float* output, float* transform_mid_buf,
                          size_t oh_start, size_t ow_start, size_t OH,
                          size_t OW, size_t oc_start, size_t oc_end,
                          size_t oc_index, size_t unit_idx,
                          size_t nr_units_in_tile, const DType& src_dtype,
                          const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        //! AT * m * A
        constexpr size_t alpha = 6 + 3 - 1;

        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / 4;
        size_t ocb = oc_index / 4;

#define cb(m, n)                                           \
    auto v##m##n = Vector<float, 4>::load(                 \
            output_transform_buf +                         \
            (m * alpha + n) * OCB * nr_units_in_tile * 4 + \
            ocb * nr_units_in_tile * 4 + unit_idx * 4);
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

        Vector<float, 4> v1addv2, v1subv2, v3addv4, v3subv4, v5addv6, v5subv6;
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

        Vector<float, 4> vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = Vector<float, 4>::load(bias + oc);

#define cb(m, n) v##m##n += vbias;
            UNROLL_CALL_RAW_D2(6, 6, cb);
#undef cb
        }
        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(CONCAT(v##m, n).value);
            UNROLL_CALL_RAW_D2(6, 6, cb);
#undef cb
        }

#define cb(m, n) CONCAT(v##m, n).save(transform_mid_buf + (m * 6 + n) * 4);
        UNROLL_CALL_RAW_D2(6, 6, cb);
#undef cb

        for (size_t oco = 0; oco < 4 && oc + oco < oc_end; ++oco) {
            for (size_t oho = 0; oho < 6 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 6 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    float res = transform_mid_buf[oho * 6 * 4 + owo * 4 + oco];
                    if (bmode == BiasMode::BIAS) {
                        res += bias[(oc + oco) * OH * OW + oh * OW + ow];
                        res = op(res);
                    }
                    output[(oc + oco) * OH * OW + oh * OW + ow] = res;
                }
            }
        }
    }
};
}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_6x3_4x4_f)

void winograd_6x3_4x4_f::filter(const float* filter,
                                float* filter_transform_buf,
                                float* transform_mid_buf, size_t OC, size_t IC,
                                size_t oc_start, size_t oc_end) {
    FilterTransform6X3<param::MatrixMul::Format::MK4>::transform(
            filter, filter_transform_buf, transform_mid_buf, OC, IC, oc_start,
            oc_end);
}

void winograd_6x3_4x4_f::input(const float* input, float* input_transform_buf,
                               float* transform_mid_buf, size_t IH, size_t IW,
                               size_t IC, size_t PH, size_t PW,
                               size_t unit_start_idx, size_t nr_units_in_tile) {
    megdnn_assert(IC % 4 == 0);
    constexpr int alpha = 3 + 6 - 1;

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    float* patch = transform_mid_buf;
    float* patchT = transform_mid_buf + 4 * alpha * alpha;

    for (size_t ic = 0; ic < IC; ic += 4) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransform6X3::prepare<true>(input, patch, patchT, ih_start,
                                                 iw_start, IH, IW, ic, IC);
                InputTransform6X3::transform(patchT, input_transform_buf,
                                             unit_idx, nr_units_in_tile, ic,
                                             IC);

            } else {
                InputTransform6X3::prepare<false>(input, patch, patchT,
                                                  ih_start, iw_start, IH, IW,
                                                  ic, IC);
                InputTransform6X3::transform(patchT, input_transform_buf,
                                             unit_idx, nr_units_in_tile, ic,
                                             IC);
            }
        }
    }
}

void winograd_6x3_4x4_f::output(const float* output_transform_buf,
                                const float* bias, float* output,
                                float* transform_mid_buf, BiasMode bmode,
                                NonlineMode nonline_mode, size_t OH, size_t OW,
                                size_t oc_start, size_t oc_end, size_t unit_start_idx,
                                size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...) \
    OutputTransform6X3<_bmode MEGDNN_COMMA _nonline_op>::transform(__VA_ARGS__);

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);

    for (size_t oc = oc_start; oc < oc_end; oc += 4) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            DISPATCH_CONV_WINOGRAD_BIAS(
                    megdnn_arm_common_winograd_fp32_F63_4x4, cb, float, float, bmode,
                    nonline_mode, output_transform_buf, bias, output, transform_mid_buf,
                    oh_start, ow_start, OH, OW, oc_start, oc_end, oc_index, unit_idx,
                    nr_units_in_tile, src_dtype, dst_dtype);
        }
    }
#undef cb
}

}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
