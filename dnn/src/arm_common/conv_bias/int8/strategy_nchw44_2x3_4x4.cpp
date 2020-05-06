/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy_nchw44_2x3_4x4.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/int8/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

#include "src/arm_common/conv_bias/winograd_common/winograd_common.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/arm_common/conv_bias/fp32/helper.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_winograd_nchw44_s8_comp_fp32_f23)

using namespace megdnn;
using namespace arm_common;
namespace {
struct InputTransform2X3 {
    template <bool inner>
    static void prepare(const int8_t* input, float* patch, float* patchT,
                        int ih_start, int iw_start, size_t IH, size_t IW,
                        size_t ic, size_t IC, size_t PH, size_t PW) {
        megdnn_assert(
                ic % 4 == 0 && IC % 4 == 0,
                "Winograd input prepare param is not times of 4!");
        constexpr size_t alpha = 2 + 3 - 1;
        MEGDNN_MARK_USED_VAR(patch);
        if (inner) {
            const int8_t* input_ptr =
                    input + ic * IH * IW + ih_start * IW * 4 + iw_start * 4;
            for (size_t ico = 0; ico < 4; ++ico) {
                int8x16_t v_input = vld1q_s8(input_ptr);
                int16x8_t v_low = vmovl_s8(vget_low_s8(v_input));
                int16x8_t v_high = vmovl_s8(vget_high_s8(v_input));
                int32x4_t v_0 = vmovl_s16(vget_low_s16(v_low));
                int32x4_t v_1 = vmovl_s16(vget_high_s16(v_low));
                int32x4_t v_2 = vmovl_s16(vget_low_s16(v_high));
                int32x4_t v_3 = vmovl_s16(vget_high_s16(v_high));

                vst1q_f32(patchT + ico * 4 * alpha + 0 * 4,
                          vcvtq_f32_s32(v_0));
                vst1q_f32(patchT + ico * 4 * alpha + 1 * 4,
                          vcvtq_f32_s32(v_1));
                vst1q_f32(patchT + ico * 4 * alpha + 2 * 4,
                          vcvtq_f32_s32(v_2));
                vst1q_f32(patchT + ico * 4 * alpha + 3 * 4,
                          vcvtq_f32_s32(v_3));
                input_ptr += IW * 4;
            }
        } else {
            if (PH > 0 || PW > 0) {
                memset(patchT, 0, sizeof(float) * 4 * alpha * alpha);
            }
            InputGetter<const int8_t*, float32x4_t> getter;
            const int8_t* input_ptr = input + ic * IH * IW;
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            // partial copy
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    vst1q_f32(patchT + iho * alpha * 4 + iwo * 4,
                              getter(input_ptr + ih * IW * 4 + iw * 4));
                }
            }
        }
    }

    static void transform(const float* patchT, float* input_transform_buf,
                          size_t unit_idx, size_t nr_units_in_tile, size_t ic,
                          size_t IC) {
        constexpr size_t alpha = 2 + 3 - 1;
        // BT * d * B
#define cb(m, n)               \
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

#define cb(m)                    \
    d##m##0 = t##m##0 - t##m##2; \
    d##m##1 = t##m##1 + t##m##2; \
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
                          int8_t* output, float* transform_mid_buf,
                          size_t oh_start, size_t ow_start, size_t OH,
                          size_t OW, size_t oc_start, size_t oc_end,
                          size_t oc_index, size_t unit_idx,
                          size_t nr_units_in_tile, const DType& src_dtype,
                          const DType& filter_dtype, const DType& dst_dtype) {
        float scale_filter = 0.f;
        MEGDNN_MARK_USED_VAR(transform_mid_buf);
        if (filter_dtype.enumv() == DTypeEnum::QuantizedS8) {
            scale_filter = filter_dtype.param<dtype::QuantizedS8>().scale;
        } else if (filter_dtype.enumv() == DTypeEnum::QuantizedS32) {
            megdnn_assert(filter_dtype.enumv() == DTypeEnum::QuantizedS32);
            scale_filter = filter_dtype.param<dtype::QuantizedS32>().scale;
        }
        float input_filter_scale =
                src_dtype.param<dtype::QuantizedS8>().scale * scale_filter;
        DType buffer_dtype = dtype::QuantizedS32(input_filter_scale);
        Op op(buffer_dtype, dst_dtype);

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

        Vector<float, 4> result[2][2];

        result[0][0] = t00 + t01 + t02;
        result[1][0] = t10 + t11 + t12;
        result[0][1] = t01 - t02 + t03;
        result[1][1] = t11 - t12 + t13;

        const int32_t* tmp_bias =
                static_cast<const int32_t*>(static_cast<const void*>(bias));
        Vector<float, 4> vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            const float32x4_t vvbias = vcvtq_f32_s32(vld1q_s32(tmp_bias + oc));
            vbias = Vector<float, 4>(vvbias);

          result[0][0] += vbias;
          result[0][1] += vbias;
          result[1][0] += vbias;
          result[1][1] += vbias;
        }

#undef cb

#if MEGDNN_AARCH64
        int32_t* tmp_ouput = static_cast<int32_t*>(static_cast<void*>(output));
#endif
        for (size_t oho = 0; oho < 2 && oh_start + oho < OH; ++oho) {
            for (size_t owo = 0; owo < 2 && ow_start + owo < OW; ++owo) {
                size_t oh = oh_start + oho;
                size_t ow = ow_start + owo;

                Vector<float, 4> res;
                res = result[oho][owo];
                if (bmode == BiasMode::BIAS) {
                    const float32x4_t vvbias = vcvtq_f32_s32(vld1q_s32(
                            tmp_bias + oc * OH * OW + oh * OW * 4 + ow * 4));
                    res += Vector<float, 4>(vvbias);
                }
#if MEGDNN_AARCH64
                int8x8_t v_res = op(res.value);
                tmp_ouput[oc * OH * OW / 4 + oh * OW + ow] =
                        vget_lane_s32(vreinterpret_s32_s8(v_res), 0);
#else
                //! armv7 using neon there is some error ,so using scalar
                //! compute
                dt_qint8 res_int8 = dt_qint8(0);
#define cb(i)                                               \
    res_int8 = op(dt_qint32(vgetq_lane_f32(res.value, i))); \
    output[oc * OH * OW + oh * OW * 4 + ow * 4 + i] = res_int8.as_int8();
                UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
#endif
            }
        }
    }
};
}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_4x4_s8_f32_nchw44)
void winograd_2x3_4x4_s8_f32_nchw44::filter(const int8_t* filter,
                                 float* filter_transform_buf,
                                 float* transform_mid_buf, size_t OC, size_t IC,
                                 size_t oc_start, size_t oc_end) {
    constexpr int alpha = 2 + 3 - 1;

    /**
     * origin: (4x3) * (3 x 3) * (3 x 4)
     */
    //! 1      0    0    v00 v01 v02   1 0.5  0.5 0
    //! 0.5  0.5  0.5    v10 v11 v12   0 0.5 -0.5 0
    //! 0.5 -0.5  0.5    v20 v21 v22   0 0.5  0.5 1
    //! 0      0    1

    InputGetter<const int8_t*, float32x4_t> getter;
    MEGDNN_MARK_USED_VAR(transform_mid_buf);
    megdnn_assert((oc_end - oc_start) % 4 == 0 && oc_start % 4 == 0 &&
                          oc_end % 4 == 0 && IC % 4 == 0 && OC % 4 == 0,
                  "Winograd filter transform input param is not times of 4!");
    size_t OCB = OC / 4;
    size_t ICB = IC / 4;

    for (size_t ocb = oc_start / 4; ocb < oc_end / 4; ocb++) {
        for (size_t icb = 0; icb < ICB; icb++) {
            for (size_t ic_inner = 0; ic_inner < 4; ic_inner++) {
                const int8_t* fptr = filter + (ocb * ICB + icb) * 3 * 3 * 4 * 4 +
                                    ic_inner * 4;

#define cb(m, n)               \
    Vector<float, 4> g##m##n = \
            Vector<float, 4>(getter(fptr + (m * 3 + n) * 4 * 4));

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


#define cb(m, n)                                                             \
    ret##m##n.save(filter_transform_buf +                                    \
                   (m * alpha + n) * OCB * ICB * 4 * 4 + ocb * ICB * 4 * 4 + \
                   icb * 4 * 4 + ic_inner * 4);
                UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
            }
        }
    }
}

void winograd_2x3_4x4_s8_f32_nchw44::input(const int8_t* input, float* input_transform_buf,
                                float* transform_mid_buf, size_t IH, size_t IW,
                                size_t IC, size_t PH, size_t PW,
                                size_t unit_start_idx,
                                size_t nr_units_in_tile) {
    megdnn_assert(IC % 4 == 0);
    constexpr int alpha = 3 + 2 - 1;

    auto units_w =
            div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
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
                                                 iw_start, IH, IW, ic, IC,PH,PW);
                InputTransform2X3::transform(patchT, input_transform_buf,
                                             unit_idx, nr_units_in_tile, ic,
                                             IC);

            } else {
                InputTransform2X3::prepare<false>(input, patch, patchT,
                                                  ih_start, iw_start, IH, IW,
                                                  ic, IC,PH,PW);
                InputTransform2X3::transform(patchT, input_transform_buf,
                                             unit_idx, nr_units_in_tile, ic,
                                             IC);
            }
        }
    }
}

void winograd_2x3_4x4_s8_f32_nchw44::output(const float* output_transform_buf,
                                 const float* bias, int8_t* output,
                                 float* transform_mid_buf, BiasMode bmode,
                                 NonlineMode nonline_mode, size_t OH, size_t OW,
                                 size_t oc_start, size_t oc_end,
                                 size_t unit_start_idx,
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
            DISPATCH_CONV_WINOGRAD_BIAS_QUANTIZED(
                    megdnn_arm_common_winograd_nchw44_s8_comp_fp32_f23, cb,
                    dt_qint32, dt_qint8, bmode, nonline_mode,
                    output_transform_buf, bias, output, transform_mid_buf,
                    oh_start, ow_start, OH, OW, oc_start, oc_end, oc_index,
                    unit_idx, nr_units_in_tile, src_dtype, filter_dtype,
                    dst_dtype);
        }
    }
#undef cb
}

}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
