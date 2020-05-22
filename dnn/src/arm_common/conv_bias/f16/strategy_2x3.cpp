/**
 * \file dnn/src/arm_common/conv_bias/f16/strategy_2x3.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/f16/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

#include "src/arm_common/conv_bias/f16/helper.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_fp16_F23)

using namespace megdnn;
using namespace arm_common;

namespace {
void transpose_4x4(const __fp16* src, __fp16* dst, int lda, int ldb) {
    float16x4x2_t a0, a1;
    a0.val[0] = vld1_f16(src + 0 * lda);                // a0a1a2a3
    a0.val[1] = vld1_f16(src + 1 * lda);                // b0b1b2b3
    a1.val[0] = vld1_f16(src + 2 * lda);                // c0c1c2c3
    a1.val[1] = vld1_f16(src + 3 * lda);                // d0d1d2d3
    float16x4x2_t b0 = vzip_f16(a0.val[0], a1.val[0]);  // a0c0a1c1a2c2a3c3
    float16x4x2_t b1 = vzip_f16(a0.val[1], a1.val[1]);  // b0d0b1d1b2d2b3d3
    float16x4x2_t c0 = vzip_f16(b0.val[0], b1.val[0]);  // a0b0c0d0a1b1c1d1
    float16x4x2_t c1 = vzip_f16(b0.val[1], b1.val[1]);  // a2b2c2d2a3b3c3d3
    vst1_f16(dst + 0 * ldb, c0.val[0]);
    vst1_f16(dst + 1 * ldb, c0.val[1]);
    vst1_f16(dst + 2 * ldb, c1.val[0]);
    vst1_f16(dst + 3 * ldb, c1.val[1]);
}

struct InputTransform2X3 {
    template <bool inner>
    static void prepare(const __fp16* input, __fp16* patch, __fp16* patchT,
                        int ih_start, int iw_start, size_t IH, size_t IW,
                        size_t ic, size_t IC) {
        constexpr size_t alpha = 2 + 3 - 1;
        constexpr size_t alpha4 = alpha * 4;
        if (!(inner && ic + 4 < IC)) {
            memset(patch, 0, sizeof(__fp16) * 4 * alpha * alpha);
        }
        if (inner) {
            const __fp16* input_ptr =
                    input + ic * IH * IW + ih_start * IW + iw_start;
            for (size_t ico = 0; ico < 4; ++ico) {
                if (ic + ico < IC) {
                    auto v0 = vld1_f16(input_ptr);
                    auto v1 = vld1_f16(input_ptr + IW);
                    auto v2 = vld1_f16(input_ptr + IW * 2);
                    auto v3 = vld1_f16(input_ptr + IW * 3);

                    vst1_f16(patch + ico * alpha4 + 0 * 4, v0);
                    vst1_f16(patch + ico * alpha4 + 1 * 4, v1);
                    vst1_f16(patch + ico * alpha4 + 2 * 4, v2);
                    vst1_f16(patch + ico * alpha4 + 3 * 4, v3);
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
                            patch[ico * alpha4 + iho * 4 + iwo] =
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

    static void transform(const __fp16* patchT, __fp16* input_transform_buf,
                          size_t unit_idx, size_t nr_units_in_tile, size_t ic,
                          size_t IC) {
        constexpr size_t alpha = 2 + 3 - 1;
        // BT * d * B
#define cb(m, n)                \
    Vector<__fp16, 4> d##m##n = \
            Vector<__fp16, 4>::load(patchT + m * 4 * 4 + n * 4);

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

#define cb(m, n)                                                           \
    d##m##n.save(input_transform_buf +                                     \
                 (m * alpha + n) * nr_units_in_tile * IC + unit_idx * IC + \
                 ic);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
    }
};

template <BiasMode bmode, typename Op>
struct OutputTransform2X3 {
    static void transform(const dt_float16* output_transform_buf,
                          const dt_float16* bias, dt_float16* output,
                          dt_float16* transform_mid_buf, size_t oh_start,
                          size_t ow_start, size_t OH, size_t OW,
                          size_t oc_start, size_t oc_end, size_t oc_index,
                          size_t unit_idx, size_t nr_units_in_tile,
                          const DType& src_dtype, const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        const __fp16* output_transform_ptr =
                reinterpret_cast<const __fp16*>(output_transform_buf);
        const __fp16* bias_ptr = reinterpret_cast<const __fp16*>(bias);
        __fp16* output_ptr = reinterpret_cast<__fp16*>(output);
        __fp16* transform_mid_ptr =
                reinterpret_cast<__fp16*>(transform_mid_buf);

        //! AT * m * A
        constexpr size_t alpha = 2 + 3 - 1;
        size_t OC = oc_end - oc_start;
        size_t oc = oc_start + oc_index;

#define cb(m, n)                                                             \
    auto v##m##n = Vector<__fp16, 4>::load(                                  \
            output_transform_ptr + (m * alpha + n) * nr_units_in_tile * OC + \
            unit_idx * OC + oc_index);
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

        Vector<__fp16, 4> vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = Vector<__fp16, 4>::load(bias_ptr + oc);

            v00 += vbias;
            v10 += vbias;
            v01 += vbias;
            v11 += vbias;
        }
        float16x8_t result01, result23;
        result01 = vcombine_f16(v00.value, v01.value);
        result23 = vcombine_f16(v10.value, v11.value);
        if (bmode != BiasMode::BIAS) {
            result01 = op(result01);
            result23 = op(result23);
        }
        vst1q_f16(transform_mid_ptr, result01);
        vst1q_f16(transform_mid_ptr + 8, result23);

        for (size_t oco = 0; oco < 4 && oc + oco < oc_end; ++oco) {
            for (size_t oho = 0; oho < 2 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 2 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    __fp16 res = transform_mid_ptr[oho * 2 * 4 + owo * 4 + oco];
                    if (bmode == BiasMode::BIAS) {
                        res += bias_ptr[(oc + oco) * OH * OW + oh * OW + ow];
                        res = op(res);
                    }
                    output_ptr[(oc + oco) * OH * OW + oh * OW + ow] = res;
                }
            }
        }
    }
};
}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_4x4_f16)

void winograd_2x3_4x4_f16::filter(const dt_float16* filter,
                                  dt_float16* filter_transform_buf,
                                  dt_float16* transform_mid_buf, size_t OC,
                                  size_t IC, size_t oc_start, size_t oc_end) {
    constexpr int alpha = 2 + 3 - 1;
    //! G * g * GT
    __fp16* filter_transbuf_ptr =
            reinterpret_cast<__fp16*>(filter_transform_buf);
    __fp16* filter_transmid_ptr = reinterpret_cast<__fp16*>(transform_mid_buf);

    for (size_t oc = oc_start; oc < oc_end; oc++) {
        rep(ic, IC) {
            const __fp16* filter_ptr = reinterpret_cast<const __fp16*>(filter) +
                                       (oc * IC + ic) * 3 * 3;
            /**
             * origin: (4x3) * (3 x 3) * (3 x 4)
             * pack to G and g to times of 4
             * now: (4x4) * (4 x 4) * (4 x 4)
             */
            //! 1      0    0 0   v00 v01 v02 0  1 0.5  0.5 0
            //! 0.5  0.5  0.5 0   v10 v11 v12 0  0 0.5 -0.5 0
            //! 0.5 -0.5  0.5 0   v20 v21 v22 0  0 0.5  0.5 1
            //! 0      0    1 0   0   0   0   0  0   0    0 0
            float16x4_t v0 = vld1_f16(filter_ptr);      // 0 1 2 3
            float16x4_t v1 = vld1_f16(filter_ptr + 3);  // 3 4 5 6
            float16x4_t v2 = vld1_f16(filter_ptr + 5);  // 5678
            float16x4_t v3 = vdup_n_f16(0);
            v2 = vext_f16(v2, v3, 1);
            v0 = vset_lane_f16(0, v0, 3);
            v1 = vset_lane_f16(0, v1, 3);
#define cb(i) float16x4_t vsum##i;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            vsum0 = v0;
            float16x4_t v0addv2 = vadd_f16(v0, v2);
            float16x4_t v02addv1 = vadd_f16(v0addv2, v1);
            float16x4_t v02subv1 = vsub_f16(v0addv2, v1);
            vsum1 = vmul_n_f16(v02addv1, 0.5);
            vsum2 = vmul_n_f16(v02subv1, 0.5);
            vsum3 = v2;

#define cb(i)                                                                \
    do {                                                                     \
        mid_buf1[0] = vget_lane_f16(vsum##i, 0);                             \
        __fp16 a0a2 = vget_lane_f16(vsum##i, 0) + vget_lane_f16(vsum##i, 2); \
        __fp16 a0a2adda1 = a0a2 + vget_lane_f16(vsum##i, 1);                 \
        __fp16 a0a2suba1 = a0a2 - vget_lane_f16(vsum##i, 1);                 \
        mid_buf1[1] = a0a2adda1 * 0.5;                                       \
        mid_buf1[2] = a0a2suba1 * 0.5;                                       \
        mid_buf1[3] = vget_lane_f16(vsum##i, 2);                             \
        mid_buf1 += 4;                                                       \
    } while (0);

            __fp16* mid_buf1 = filter_transmid_ptr;
            UNROLL_CALL_NOWRAPPER(4, cb);
            mid_buf1 = filter_transmid_ptr;
#undef cb
            rep(i, alpha) rep(j, alpha) {
                filter_transbuf_ptr[(i * alpha + j) * OC * IC + ic * OC + oc] =
                        filter_transmid_ptr[i * alpha + j];
            }
        }
    }
}

void winograd_2x3_4x4_f16::input(const dt_float16* input,
                                 dt_float16* input_transform_buf,
                                 dt_float16* transform_mid_buf, size_t IH,
                                 size_t IW, size_t IC, size_t PH, size_t PW,
                                 size_t unit_start_idx,
                                 size_t nr_units_in_tile) {
    megdnn_assert(IC % 4 == 0);
    constexpr int alpha = 3 + 2 - 1;

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    dt_float16* patch = transform_mid_buf;
    dt_float16* patchT = transform_mid_buf + 4 * alpha * alpha;

    for (size_t ic = 0; ic < IC; ic += 4) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransform2X3::prepare<true>(
                        reinterpret_cast<const __fp16*>(input),
                        reinterpret_cast<__fp16*>(patch),
                        reinterpret_cast<__fp16*>(patchT), ih_start, iw_start,
                        IH, IW, ic, IC);
                InputTransform2X3::transform(
                        reinterpret_cast<const __fp16*>(patchT),
                        reinterpret_cast<__fp16*>(input_transform_buf),
                        unit_idx, nr_units_in_tile, ic, IC);
            } else {
                InputTransform2X3::prepare<false>(
                        reinterpret_cast<const __fp16*>(input),
                        reinterpret_cast<__fp16*>(patch),
                        reinterpret_cast<__fp16*>(patchT), ih_start, iw_start,
                        IH, IW, ic, IC);
                InputTransform2X3::transform(
                        reinterpret_cast<const __fp16*>(patchT),
                        reinterpret_cast<__fp16*>(input_transform_buf),
                        unit_idx, nr_units_in_tile, ic, IC);
            }
        }
    }
}

void winograd_2x3_4x4_f16::output(const dt_float16* output_transform_buf,
                                  const dt_float16* bias, dt_float16* output,
                                  dt_float16* transform_mid_buf, BiasMode bmode,
                                  NonlineMode nonline_mode, size_t OH, size_t OW,
                                  size_t oc_start, size_t oc_end,
                                  size_t unit_start_idx, size_t nr_units_in_tile) {
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
                    megdnn_arm_common_winograd_fp16_F23, cb, __fp16, __fp16, bmode,
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
#endif
// vim: syntax=cpp.doxygen
