/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy_2x3_4x4.cpp
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
MIDOUT_DECL(megdnn_arm_common_winograd_fp32_F23)

using namespace megdnn;
using namespace arm_common;
namespace {

struct InputTransform2X3 {
    template <bool inner>
    static void prepare(const float* input, float* patch, float* patchT,
                        int ih_start, int iw_start, size_t IH, size_t IW,
                        size_t ic, size_t IC) {
        constexpr size_t alpha = 2 + 3 - 1;
        if (!(inner && ic + 4 < IC)) {
            memset(patch, 0, sizeof(float) * 4 * alpha * alpha);
        }
        if (inner) {
            const float* input_ptr =
                    input + ic * IH * IW + ih_start * IW + iw_start;
            for (size_t ico = 0; ico < 4; ++ico) {
                if (ic + ico < IC) {
                    auto v0 = vld1q_f32(input_ptr);
                    auto v1 = vld1q_f32(input_ptr + IW);
                    auto v2 = vld1q_f32(input_ptr + IW * 2);
                    auto v3 = vld1q_f32(input_ptr + IW * 3);

                    vst1q_f32(patch + ico * 4 * alpha + 0 * 4, v0);
                    vst1q_f32(patch + ico * 4 * alpha + 1 * 4, v1);
                    vst1q_f32(patch + ico * 4 * alpha + 2 * 4, v2);
                    vst1q_f32(patch + ico * 4 * alpha + 3 * 4, v3);
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
                            patch[ico * alpha * 4 + iho * 4 + iwo] =
                                    input[(ic + ico) * IH * IW + ih * IW + iw];
                        }
                    }
                }
            }
        }

        transpose_4x4(patch + 0 * 1, patchT + 0 * 4, 16, 4);
        transpose_4x4(patch + 4 * 1, patchT + 4 * 4, 16, 4);
        transpose_4x4(patch + 8 * 1, patchT + 8 * 4, 16, 4);
        transpose_4x4(patch + 12 * 1, patchT + 12 * 4, 16, 4);
    }

    static void transform(const float* patchT, float* input_transform_buf,
                          size_t unit_idx, size_t nr_units_in_tile, size_t ic,
                          size_t IC) {
        constexpr size_t alpha = 2 + 3 - 1;
        // BT * d * B
#define cb(m, n) \
        Vector<float, 4> d##m##n = \
                Vector<float, 4>::load(patchT + m * 4 * 4 + n * 4);

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

#define cb(m)                     \
    d##m##0 = t##m##0 - t##m##2;  \
    d##m##1 = t##m##1 + t##m##2;  \
    d##m##2 = t##m##2 - t##m##1; \
    d##m##3 = t##m##3 - t##m##1;

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

        size_t ICB = IC / 4;
        size_t icb = ic / 4;
#define cb(m, n)                                                \
    d##m##n.save(input_transform_buf +                          \
                 (m * alpha + n) * ICB * nr_units_in_tile * 4 + \
                 icb * nr_units_in_tile * 4 + unit_idx * 4);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
    }
};

template <BiasMode bmode, typename Op>
struct OutputTransform2X3 {
    static void transform(const float* output_transform_buf, const float* bias,
                          float* output, float* transform_mid_buf,
                          size_t oh_start, size_t ow_start, size_t OH,
                          size_t OW, size_t oc_start, size_t oc_end,
                          size_t oc_index, size_t unit_idx,
                          size_t nr_units_in_tile, const DType& src_dtype,
                          const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        //! AT * m * A
        constexpr size_t alpha = 2 + 3 - 1;

        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / 4;
        size_t ocb = oc_index / 4;
        
#define cb(m, n)                                           \
    auto v##m##n = Vector<float, 4>::load(                 \
            output_transform_buf +                         \
            (m * alpha + n) * OCB * nr_units_in_tile * 4 + \
            ocb * nr_units_in_tile * 4 + unit_idx * 4);
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
        v00 = t00 + t01 + t02;
        v10 = t10 + t11 + t12;
        v01 = t01 - t02 + t03;
        v11 = t11 - t12 + t13;

        Vector<float, 4> vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = Vector<float, 4>::load(bias + oc);

            v00 += vbias;
            v10 += vbias;
            v01 += vbias;
            v11 += vbias;
        }
        if (bmode != BiasMode::BIAS) {
            v00 = op(v00.value);
            v01 = op(v01.value);
            v10 = op(v10.value);
            v11 = op(v11.value);
        }

        v00.save(transform_mid_buf + (0 * 2 + 0) * 4);
        v10.save(transform_mid_buf + (1 * 2 + 0) * 4);
        v01.save(transform_mid_buf + (0 * 2 + 1) * 4);
        v11.save(transform_mid_buf + (1 * 2 + 1) * 4);

        for (size_t oco = 0; oco < 4 && oc + oco < oc_end; ++oco) {
            for (size_t oho = 0; oho < 2 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 2 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    float res = transform_mid_buf[oho * 2 * 4 + owo * 4 + oco];
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

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_4x4_f)
void winograd_2x3_4x4_f::filter(const float* filter,
                                float* filter_transform_buf,
                                float* transform_mid_buf, size_t OC, size_t IC,
                                size_t oc_start, size_t oc_end) {
    constexpr int alpha = 2 + 3 - 1;
    //! G * g * GT
    float32x4_t g0{1.f, 0, 0, 0}, g1{0.5, 0.5, 0.5, 0}, g2{0.5, -0.5, 0.5, 0},
            g3{0, 0, 1, 0};
    float32x4_t gt0{1, 0.5, 0.5, 0}, gt1{0, 0.5, -0.5, 0}, gt2{0, 0.5, 0.5, 1},
            gt3{0, 0, 0, 0};
    size_t OCB = OC / 4;
    size_t ICB = IC / 4;

    for (size_t oc = oc_start; oc < oc_end; oc++)
        rep(ic, IC) {
            const float* filter_ptr = filter + (oc * IC + ic) * 3 * 3;
            /**
             * origin: (4x3) * (3 x 3) * (3 x 4)
             * pack to G and g to times of 4
             * now: (4x4) * (4 x 4) * (4 x 4)
             */
            //! 1      0    0 0   v00 v01 v02 0  1 0.5  0.5 0
            //! 0.5  0.5  0.5 0   v10 v11 v12 0  0 0.5 -0.5 0
            //! 0.5 -0.5  0.5 0   v20 v21 v22 0  0 0.5  0.5 1
            //! 0      0    1 0   0   0   0   0  0   0    0 0
            float32x4_t vf0 = vld1q_f32(filter_ptr);
            float32x4_t vf1 = vld1q_f32(filter_ptr + 4);
            float32x4_t vf2 = vdupq_n_f32(filter_ptr[8]);

            float32x4_t v3(vdupq_n_f32(0));
            auto vtmp = vextq_f32(vf1, vf2, 2);
            vtmp = vsetq_lane_f32(0, vtmp, 3);
            float32x4_t v2(vtmp);
            vtmp = vextq_f32(vf0, vf1, 3);
            vtmp = vsetq_lane_f32(0, vtmp, 3);
            float32x4_t v1(vtmp);
            vtmp = vsetq_lane_f32(0, vf0, 3);
            float32x4_t v0(vtmp);

            float32x4_t vsum0 = vdupq_n_f32(0), vsum1 = vdupq_n_f32(0),
                        vsum2 = vdupq_n_f32(0), vsum3 = vdupq_n_f32(0);

            MATRIX_MUL4x4(vsum, g, v);

            float32x4_t vres0 = vdupq_n_f32(0), vres1 = vdupq_n_f32(0),
                        vres2 = vdupq_n_f32(0), vres3 = vdupq_n_f32(0);
            MATRIX_MUL4x4(vres, vsum, gt);

            vst1q_f32(transform_mid_buf, vres0);
            vst1q_f32(transform_mid_buf + 4, vres1);
            vst1q_f32(transform_mid_buf + 8, vres2);
            vst1q_f32(transform_mid_buf + 12, vres3);

            size_t ocb = oc / 4;
            size_t oc4 = oc % 4;
            size_t icb = ic / 4;
            size_t ic4 = ic % 4;
            rep(i, alpha) rep(j, alpha) {
                filter_transform_buf[(i * alpha + j) * OCB * ICB * 4 * 4 +
                                     ocb * ICB * 4 * 4 + icb * 4 * 4 + ic4 * 4 +
                                     oc4] = transform_mid_buf[i * alpha + j];
            }
        }
}

void winograd_2x3_4x4_f::input(const float* input, float* input_transform_buf,
                               float* transform_mid_buf, size_t IH, size_t IW,
                               size_t IC, size_t PH, size_t PW,
                               size_t unit_start_idx, size_t nr_units_in_tile) {
    megdnn_assert(IC % 4 == 0);
    constexpr int alpha = 3 + 2 - 1;

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
                InputTransform2X3::prepare<true>(input, patch, patchT, ih_start,
                                                 iw_start, IH, IW, ic, IC);
                InputTransform2X3::transform(patchT, input_transform_buf,
                                             unit_idx, nr_units_in_tile, ic,
                                             IC);

            } else {
                InputTransform2X3::prepare<false>(input, patch, patchT,
                                                  ih_start, iw_start, IH, IW,
                                                  ic, IC);
                InputTransform2X3::transform(patchT, input_transform_buf,
                                             unit_idx, nr_units_in_tile, ic,
                                             IC);
            }
        }
    }
}

void winograd_2x3_4x4_f::output(const float* output_transform_buf,
                                const float* bias, float* output,
                                float* transform_mid_buf, BiasMode bmode,
                                NonlineMode nonline_mode, size_t OH, size_t OW,
                                size_t oc_start, size_t oc_end, size_t unit_start_idx,
                                size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...) \
    OutputTransform2X3<_bmode MEGDNN_COMMA _nonline_op>::transform(__VA_ARGS__);

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
                    megdnn_arm_common_winograd_fp32_F23, cb, float, float, bmode,
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
