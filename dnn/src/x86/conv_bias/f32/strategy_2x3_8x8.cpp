/**
 * \file dnn/src/x86/conv_bias/f32/strategy_2x3_8x8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
#include <avxintrin.h>
#include <smmintrin.h>
#include <avx2intrin.h>
#include <fmaintrin.h>
#endif

#include "midout.h"
MIDOUT_DECL(megdnn_x86_winograd_nchw88_fp32_F23_8x8)

using namespace megdnn;
using namespace x86;

namespace {
constexpr size_t alpha = 2 + 3 - 1;
struct InputTransform2X3_NCHW88 {
    template <bool inner>
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    static void prepare(const float* input, float* patch, float* patchT,
                        int ih_start, int iw_start, size_t IH, size_t IW,
                        size_t ic, size_t IC) {
        MEGDNN_MARK_USED_VAR(patch);
        size_t IW8 = IW * 8;  //! For nchw88 mode
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
                UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i) _mm256_storeu_ps(patchT + ih * alpha * 8 + i * 8, v##i);

                UNROLL_CALL_NOWRAPPER(4, cb);
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
                    _mm256_storeu_ps(patchT + iho * alpha * 8 + iwo * 8, src);
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
            Vector<float, 8>::load(patchT + m * alpha * 8 + n * 8);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb
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
        size_t ICB = IC / 8;
        size_t icb = ic / 8;
#define cb(m, n)                                                \
    d##m##n.save(input_transform_buf +                          \
                 (m * alpha + n) * ICB * nr_units_in_tile * 8 + \
                 icb * nr_units_in_tile * 8 + unit_idx * 8);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
    }
};

struct FilterTransform2X3_MCHW88 {
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    static void transform(const float* filter, float* filter_transform_buf,
                          float* transform_mid_buf, size_t OC, size_t IC,
                          size_t oc_start, size_t oc_end) {
        /**
         * origin: (4x3) * (3 x 3) * (3 x 4)
         */
        //! 1      0    0    v00 v01 v02   1 0.5  0.5 0
        //! 0.5  0.5  0.5    v10 v11 v12   0 0.5 -0.5 0
        //! 0.5 -0.5  0.5    v20 v21 v22   0 0.5  0.5 1
        //! 0      0    1

        MEGDNN_MARK_USED_VAR(transform_mid_buf);
        megdnn_assert(
                (oc_end - oc_start) % 8 == 0 && oc_start % 8 == 0 &&
                        oc_end % 8 == 0 && IC % 8 == 0 && OC % 8 == 0,
                "Winograd filter transform input param is not times of 8!");
        size_t OCB = OC / 8;
        size_t ICB = IC / 8;

        for (size_t ocb = oc_start / 8; ocb < oc_end / 8; ocb++) {
            for (size_t icb = 0; icb < ICB; icb++) {
                for (size_t ic_inner = 0; ic_inner < 8; ic_inner++){
                    const float* fptr = filter +
                                        (ocb * ICB + icb) * 3 * 3 * 8 * 8 +
                                        ic_inner * 8;

#define cb(m, n)               \
    Vector<float, 8> g##m##n = \
            Vector<float, 8>::load(fptr + (m * 3 + n) * 8 * 8);
                    UNROLL_CALL_NOWRAPPER_D2(3, 3, cb)
#undef cb

#define FILTER_TRANSFORM(n, wd, g)    \
    auto wd##n##0 = g##0##n;          \
    tmp0 = (g##0##n + g##2##n) * 0.5; \
    tmp1 = g##1##n * 0.5;             \
    auto wd##n##1 = tmp0 + tmp1;      \
    auto wd##n##2 = tmp0 - tmp1;      \
    auto wd##n##3 = g##2##n;
                    Vector<float, 8> tmp0, tmp1;
                    UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                    UNROLL_CALL_RAW(4, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM
#define cb_save(m, n)                                                        \
    ret##m##n.save(filter_transform_buf +                                    \
                   (m * alpha + n) * OCB * ICB * 8 * 8 + ocb * ICB * 8 * 8 + \
                   icb * 8 * 8 + ic_inner * 8);
                    UNROLL_CALL_NOWRAPPER_D2(4, 4, cb_save)
#undef cb_save
                }
            }
        }
    }
};
#define CONCAT(a, idx) a##idx
template <BiasMode bmode, typename Op>
struct OutputTransform2X3_NCHW88 {
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

        Vector<float, 8> vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = Vector<float, 8>::load(bias + oc);

#define cb(m, n) v##m##n += vbias;
            UNROLL_CALL_RAW_D2(2, 2, cb);
#undef cb
        }
        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(CONCAT(v##m, n).value);
            UNROLL_CALL_RAW_D2(2, 2, cb);
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
        UNROLL_CALL_RAW_D2(2, 2, out_save);
    }
};
#undef CONCAT
}  // namespace

namespace megdnn {
namespace x86 {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_nchw88_2x3_8x8_f)

void winograd_nchw88_2x3_8x8_f::filter(const float* filter,
                                       float* filter_transform_buf,
                                       float* transform_mid_buf, size_t OC,
                                       size_t IC, size_t oc_start,
                                       size_t oc_end) {
    FilterTransform2X3_MCHW88::transform(filter, filter_transform_buf,
                                         transform_mid_buf, OC, IC, oc_start,
                                         oc_end);
}

void winograd_nchw88_2x3_8x8_f::input(const float* input,
                                      float* input_transform_buf,
                                      float* transform_mid_buf, size_t IH,
                                      size_t IW, size_t IC, size_t PH,
                                      size_t PW, size_t unit_start_idx,
                                      size_t nr_units_in_tile) {
    megdnn_assert(IC % 8 == 0);

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
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
                InputTransform2X3_NCHW88::prepare<true>(input, patch, patchT,
                                                        ih_start, iw_start, IH,
                                                        IW, ic, IC);
                InputTransform2X3_NCHW88::transform(patchT, input_transform_buf,
                                                    unit_idx, nr_units_in_tile,
                                                    ic, IC);
            } else {
                InputTransform2X3_NCHW88::prepare<false>(input, patch, patchT,
                                                         ih_start, iw_start, IH,
                                                         IW, ic, IC);
                InputTransform2X3_NCHW88::transform(patchT, input_transform_buf,
                                                    unit_idx, nr_units_in_tile,
                                                    ic, IC);
            }
        }
    }
}

void winograd_nchw88_2x3_8x8_f::output(const float* output_transform_buf,
                                       const float* bias, float* output,
                                       float* transform_mid_buf, BiasMode bmode,
                                       NonlineMode nonline_mode, size_t OH,
                                       size_t OW, size_t oc_start,
                                       size_t oc_end, size_t unit_start_idx,
                                       size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...)                                       \
    OutputTransform2X3_NCHW88<_bmode MEGDNN_COMMA _nonline_op>::transform( \
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
                    megdnn_x86_winograd_nchw88_fp32_F23_8x8, cb, SIMDType::AVX2,
                    float, float, bmode, nonline_mode, output_transform_buf,
                    bias, output, transform_mid_buf, oh_start, ow_start, OH, OW,
                    oc_start, oc_end, oc_index, unit_idx, nr_units_in_tile, src_dtype,
                    dst_dtype);
        }
    }
#undef cb
}

}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
