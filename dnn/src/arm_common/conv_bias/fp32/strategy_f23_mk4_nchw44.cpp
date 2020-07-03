/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy_f23_mk4_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/fp32/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

#include "src/naive/matrix_mul/matrix_mul_helper.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/arm_common/conv_bias/fp32/helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_nchw44_fp32_F23_mk4)

using namespace megdnn;
using namespace arm_common;
namespace {

constexpr size_t alpha = 2 + 3 - 1;
constexpr size_t pack_size = 4;

struct InputTransformF23_NCHW44 {
    template <bool inner>
    static void transform(float* patchT, const float* input,
                          float* input_transform_buf, size_t ih_start,
                          size_t iw_start, size_t IH, size_t IW,
                          size_t unit_idx, size_t nr_units_in_tile, size_t ic,
                          size_t IC) {
        size_t IW4 = IW * pack_size;
        size_t icb = ic / pack_size;
        size_t iw4_start = iw_start * pack_size;
        size_t ICB = IC / pack_size;

#define cb(m, n) Vector<float, 4> d##m##n;
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb

        if (!(inner && ic + pack_size < IC)) {
            memset(patchT, 0, sizeof(float) * pack_size * alpha * alpha);
        }
        if (inner) {
            MEGDNN_MARK_USED_VAR(patchT);
            const float* input_ptr =
                    input + icb * IH * IW4 + ih_start * IW4 + iw4_start;
#define cb(n, m) d##m##n = Vector<float, 4>::load(input_ptr + pack_size * n);

            UNROLL_CALL_RAW(4, cb, 0);
            input_ptr += IW4;
            UNROLL_CALL_RAW(4, cb, 1);
            input_ptr += IW4;
            UNROLL_CALL_RAW(4, cb, 2);
            input_ptr += IW4;
            UNROLL_CALL_RAW(4, cb, 3);
#undef cb
        } else {
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            const float* input_ptr = input + icb * IH * IW4;
            // partial copy
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    auto src = vld1q_f32(input_ptr + ih * IW4 + iw * pack_size);
                    vst1q_f32(
                            patchT + iho * alpha * pack_size + iwo * pack_size,
                            src);
                }
            }
#define cb(m, n)                                                      \
    d##m##n = Vector<float, 4>::load(patchT + m * alpha * pack_size + \
                                     n * pack_size);
            UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb
        }
        //! 1   0 -1 0    d00 d01 d02 d03     1 0  0  0
        //! 0   1  1 0    d10 d11 d12 d13     0 1 -1 -1
        //! 0  -1  1 0    d20 d21 d22 d23    -1 1  1  0
        //! 0  -1  0 1    d30 d31 d32 d33     0 0  0  1
#define cb(m)                   \
    auto t0##m = d0##m - d2##m; \
    auto t1##m = d1##m + d2##m; \
    auto t2##m = d2##m - d1##m; \
    auto t3##m = d3##m - d1##m;

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m)                    \
    d##m##0 = t##m##0 - t##m##2; \
    d##m##1 = t##m##1 + t##m##2; \
    d##m##2 = t##m##2 - t##m##1; \
    d##m##3 = t##m##3 - t##m##1;

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m, n)                                                        \
    d##m##n.save(input_transform_buf +                                  \
                 (m * alpha + n) * ICB * nr_units_in_tile * pack_size + \
                 icb * nr_units_in_tile * pack_size + unit_idx * pack_size);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
    }
};

#define CONCAT(a, idx) a##idx
template <BiasMode bmode, typename Op>
struct OutputTransformF23_NCHW44 {
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
        size_t OCB = (oc_end - oc_start) / pack_size;
        size_t oc = oc_start + oc_index;
        size_t ocb = oc_index / pack_size;

#define cb(m, n)                                                   \
    auto v##m##n = Vector<float, 4>::load(                         \
            output_transform_buf +                                 \
            (m * alpha + n) * OCB * nr_units_in_tile * pack_size + \
            ocb * nr_units_in_tile * pack_size + unit_idx * pack_size);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb

        //! 1  1  1 0  v00 v01 v02 v03    1  0
        //! 0  1 -1 1  v10 v11 v12 v13    1  1
        //!            v20 v21 v22 v23    1 -1
        //!            v30 v31 v32 v33    0  1

#define cb(m)                           \
    auto t0##m = v0##m + v1##m + v2##m; \
    auto t1##m = v1##m - v2##m + v3##m;

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m)                              \
    v##m##0 = t##m##0 + t##m##1 + t##m##2; \
    v##m##1 = t##m##1 - t##m##2 + t##m##3;

        UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

        Vector<float, 4> vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = Vector<float, 4>::load(bias + oc);

#define cb(m, n) v##m##n += vbias;
            UNROLL_CALL_RAW_D2(2, 2, cb);
#undef cb
        }
        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(CONCAT(v##m, n).value);
            UNROLL_CALL_RAW_D2(2, 2, cb);
#undef cb
        }
#define out_save(oho, owo)                                                  \
    do {                                                                    \
        size_t oh = oh_start + oho;                                         \
        size_t ow = ow_start + owo;                                         \
        if (oh < OH && ow < OW) {                                           \
            if (bmode == BiasMode::BIAS) {                                  \
                v##oho##owo += Vector<float, 4>::load(bias + oc * OH * OW + \
                                                      oh * OW * pack_size + \
                                                      ow * pack_size);      \
                v##oho##owo = op(v##oho##owo.value);                        \
            }                                                               \
            v##oho##owo.save(output + oc * OH * OW + oh * OW * pack_size +  \
                             ow * pack_size);                               \
        }                                                                   \
    } while (0);
        UNROLL_CALL_RAW_D2(2, 2, out_save);
#undef out_save
    }
};
#undef CONCAT
}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_F23_mk4_f_nchw44)
void winograd_F23_mk4_f_nchw44::filter(const float* filter,
                                       float* filter_transform_buf,
                                       float* transform_mid_buf, size_t OC,
                                       size_t IC, size_t oc_start,
                                       size_t oc_end) {
    //! 1      0    0    v00 v01 v02   1 0.5  0.5 0
    //! 0.5  0.5  0.5    v10 v11 v12   0 0.5 -0.5 0
    //! 0.5 -0.5  0.5    v20 v21 v22   0 0.5  0.5 1
    //! 0      0    1

    constexpr size_t pack_size = 4;

    MEGDNN_MARK_USED_VAR(transform_mid_buf);
    megdnn_assert((oc_end - oc_start) % pack_size == 0 &&
                          oc_start % pack_size == 0 &&
                          oc_end % pack_size == 0 && IC % pack_size == 0 &&
                          OC % pack_size == 0,
                  "NCHW44 Winograd filter transform requires both OC and IC "
                  "are times of 4");
    size_t OCB = OC / pack_size;
    size_t ICB = IC / pack_size;

    for (size_t ocb = oc_start / pack_size; ocb < oc_end / pack_size; ocb++) {
        for (size_t icb = 0; icb < ICB; icb++) {
            for (size_t ic_inner = 0; ic_inner < pack_size; ic_inner++) {
                const float* fptr = filter +
                                    (ocb * ICB + icb) * KERNEL_SIZE *
                                            KERNEL_SIZE * pack_size *
                                            pack_size +
                                    ic_inner * pack_size;

#define cb(m, n)                                       \
    Vector<float, 4> g##m##n = Vector<float, 4>::load( \
            fptr + (m * KERNEL_SIZE + n) * pack_size * pack_size);
                UNROLL_CALL_NOWRAPPER_D2(3, 3, cb)
#undef cb

#define FILTER_TRANSFORM(n, wd, g)    \
    auto wd##n##0 = g##0##n;          \
    tmp0 = (g##0##n + g##2##n) * 0.5; \
    tmp1 = g##1##n * 0.5;             \
    auto wd##n##1 = tmp0 + tmp1;      \
    auto wd##n##2 = tmp0 - tmp1;      \
    auto wd##n##3 = g##2##n;
                Vector<float, 4> tmp0, tmp1;
                UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                UNROLL_CALL_RAW(4, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM
#define cb_save(m, n)                                                    \
    ret##m##n.save(filter_transform_buf +                                \
                   (m * ALPHA + n) * OCB * ICB * pack_size * pack_size + \
                   ocb * ICB * pack_size * pack_size +                   \
                   icb * pack_size * pack_size + ic_inner * pack_size);
                UNROLL_CALL_NOWRAPPER_D2(4, 4, cb_save)
#undef cb_save
            }
        }
    }
}

void winograd_F23_mk4_f_nchw44::input(const float* input,
                                      float* input_transform_buf,
                                      float* transform_mid_buf, size_t IH,
                                      size_t IW, size_t IC, size_t PH,
                                      size_t PW, size_t unit_start_idx,
                                      size_t nr_units_in_tile) {
    megdnn_assert(IC % 4 == 0);
    constexpr int alpha = 3 + 2 - 1;

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w =
            div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
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
                InputTransformF23_NCHW44::transform<true>(
                        patchT, input, input_transform_buf, ih_start, iw_start,
                        IH, IW, unit_idx, nr_units_in_tile, ic, IC);
            } else {
                InputTransformF23_NCHW44::transform<false>(
                        patchT, input, input_transform_buf, ih_start, iw_start,
                        IH, IW, unit_idx, nr_units_in_tile, ic, IC);
            }
        }
    }
}

void winograd_F23_mk4_f_nchw44::output(const float* output_transform_buf,
                                       const float* bias, float* output,
                                       float* transform_mid_buf, BiasMode bmode,
                                       NonlineMode nonline_mode, size_t OH,
                                       size_t OW, size_t oc_start,
                                       size_t oc_end, size_t unit_start_idx,
                                       size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...)                                        \
    for (size_t oc = oc_start; oc < oc_end; oc += 4) {                      \
        size_t oc_index = oc - oc_start;                                    \
        rep(unit_idx, nr_units_in_tile) {                                   \
            size_t index = unit_start_idx + unit_idx;                       \
            auto nh = index / units_w;                                      \
            auto nw = index % units_w;                                      \
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;                       \
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;                       \
            OutputTransformF23_NCHW44<_bmode, _nonline_op>::transform(      \
                    output_transform_buf, bias, output, transform_mid_buf,  \
                    oh_start, ow_start, OH, OW, oc_start, oc_end, oc_index, \
                    unit_idx, nr_units_in_tile, src_dtype, dst_dtype);      \
        }                                                                   \
    }

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    constexpr size_t pack_size = 4;

    size_t OC = oc_end - oc_start;
    megdnn_assert(OC % pack_size == 0 && oc_start % pack_size == 0 &&
                          oc_end % pack_size == 0,
                  "NCHW44 Winograd filter transform requires OC is times of 4");

    DISPATCH_CONV_WINOGRAD_BIAS(megdnn_arm_common_winograd_nchw44_fp32_F23_mk4,
                                cb, float, float, bmode, nonline_mode);
#undef cb
}

}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
