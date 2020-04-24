/**
 * \file dnn/src/arm_common/conv_bias/int8/strategy_2x3_8x8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "src/arm_common/conv_bias/winograd_common/winograd_common.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/arm_common/conv_bias/int8/strategy.h"
#include "src/arm_common/conv_bias/int8/helper.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"

#include "src/common/winograd/winograd_generator.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_s8_F23_8x8)

using namespace megdnn;
using namespace arm_common;

namespace {
void transpose_8x4(const int16_t* src, int16_t* dst, int lda, int ldb) {
    int16x4x2_t a0, a1, a2, a3;
    a0.val[0] = vld1_s16(src + 0 * lda);
    a0.val[1] = vld1_s16(src + 1 * lda);
    a1.val[0] = vld1_s16(src + 2 * lda);
    a1.val[1] = vld1_s16(src + 3 * lda);
    a2.val[0] = vld1_s16(src + 4 * lda);
    a2.val[1] = vld1_s16(src + 5 * lda);
    a3.val[0] = vld1_s16(src + 6 * lda);
    a3.val[1] = vld1_s16(src + 7 * lda);
    int16x4x2_t b0 = vzip_s16(a0.val[0], a1.val[0]);
    int16x4x2_t b1 = vzip_s16(a0.val[1], a1.val[1]);
    int16x4x2_t b2 = vzip_s16(a2.val[0], a3.val[0]);
    int16x4x2_t b3 = vzip_s16(a2.val[1], a3.val[1]);

    int16x4x2_t c0 = vzip_s16(b0.val[0], b1.val[0]);
    int16x4x2_t c1 = vzip_s16(b0.val[1], b1.val[1]);
    int16x4x2_t c2 = vzip_s16(b2.val[0], b3.val[0]);
    int16x4x2_t c3 = vzip_s16(b2.val[1], b3.val[1]);

    vst1_s16(dst + 0 * ldb, c0.val[0]);
    vst1_s16(dst + 1 * ldb, c2.val[0]);
    vst1_s16(dst + 2 * ldb, c0.val[1]);
    vst1_s16(dst + 3 * ldb, c2.val[1]);
    vst1_s16(dst + 4 * ldb, c1.val[0]);
    vst1_s16(dst + 5 * ldb, c3.val[0]);
    vst1_s16(dst + 6 * ldb, c1.val[1]);
    vst1_s16(dst + 7 * ldb, c3.val[1]);
}

struct FilterTransform2X3_qs8 {
    static void transform(const int8_t* filter_ptr, int16_t* filter_transform_buf,
                          int16_t* transform_mid_buf, size_t OC, size_t IC,
                          size_t oc_start, size_t oc_end) {
        constexpr int alpha = 2 + 3 - 1;
        //! G * g * GT
        int16x4_t g0{2, 0, 0, 0}, g1{1, 1, 1, 0}, g2{1, -1, 1, 0},
                g3{0, 0, 2, 0};
        int16x4_t gt0{2, 1, 1, 0}, gt1{0, 1, -1, 0}, gt2{0, 1, 1, 2},
                gt3{0, 0, 0, 0};

        size_t OCB = OC / 8;
        size_t ICB = IC / 8;

#define get_v_general                               \
    InputGetter<const int8_t*, int16x4_t> getter;   \
    int16x4_t v0 = getter(filter);                  \
    int16x4_t v1 = getter(filter + 3);              \
    int16x4_t v2 = getter(filter + 6);              \
    int16x4_t v3 = vdup_n_s16(0);                   \
    /*To avoid the unaligned opcode error on tx1.*/ \
    vset_lane_s16_fix_tx1(0, v0, 3);                \
    vset_lane_s16_fix_tx1(0, v1, 3);                \
    vset_lane_s16_fix_tx1(0, v2, 3);

#define get_v_searal                                                \
    /* To avoid the bus error on armv7(mi9).*/                      \
    int8x8_t s0 = {filter[0], filter[1], filter[2], 0, 0, 0, 0, 0}; \
    int8x8_t s1 = {filter[3], filter[4], filter[5], 0, 0, 0, 0, 0}; \
    int8x8_t s2 = {filter[6], filter[7], filter[8], 0, 0, 0, 0, 0}; \
    int16x4_t v0 = vget_low_s16(vmovl_s8(s0));                      \
    int16x4_t v1 = vget_low_s16(vmovl_s8(s1));                      \
    int16x4_t v2 = vget_low_s16(vmovl_s8(s2));                      \
    int16x4_t v3 = vdup_n_s16(0);

#define cb(oc, ic, get_v)                                                \
    get_v int16x4_t vsum0 = vdup_n_s16(0), vsum1 = vdup_n_s16(0),        \
                    vsum2 = vdup_n_s16(0), vsum3 = vdup_n_s16(0);        \
    MATRIX_MUL4x4(vsum, g, v);                                           \
    int16x4_t vres0 = vdup_n_s16(0), vres1 = vdup_n_s16(0),              \
              vres2 = vdup_n_s16(0), vres3 = vdup_n_s16(0);              \
    MATRIX_MUL4x4(vres, vsum, gt);                                       \
    vst1_s16(transform_mid_buf, vres0);                                  \
    vst1_s16(transform_mid_buf + 4, vres1);                              \
    vst1_s16(transform_mid_buf + 8, vres2);                              \
    vst1_s16(transform_mid_buf + 12, vres3);                             \
    size_t ocb = (oc) / 8;                                               \
    size_t oc8 = (oc) % 8;                                               \
    size_t icb = (ic) / 8;                                               \
    size_t ic8 = (ic) % 8;                                               \
    rep(i, alpha) rep(j, alpha) {                                        \
        filter_transform_buf[(i * alpha + j) * OCB * ICB * 8 * 8 +       \
                             ocb * ICB * 8 * 8 + icb * 8 * 8 + ic8 * 8 + \
                             oc8] = transform_mid_buf[i * alpha + j];    \
    }                                                                    \
    filter += 9;

        for (size_t oc = oc_start; oc < oc_end; oc++) {
            const int8_t* filter = filter_ptr + oc * IC * 3 * 3;
            if (oc != OC - 1) {
                rep(ic, IC) {
                    /**
                     * origin: (4x3) * (3 x 3) * (3 x 4)
                     * pack to G and g to times of 4
                     * now: (4x4) * (4 x 4) * (4 x 4)
                     */
                    //! 1      0    0 0   v00 v01 v02 0  1 0.5  0.5 0
                    //! 0.5  0.5  0.5 0   v10 v11 v12 0  0 0.5 -0.5 0
                    //! 0.5 -0.5  0.5 0   v20 v21 v22 0  0 0.5  0.5 1
                    //! 0      0    1 0   0   0   0   0  0   0    0 0
                    cb(oc, ic, get_v_general);
                }
            } else {
                rep(ic, IC - 1) {
                    cb(OC - 1, ic, get_v_general);
                }
                cb(OC - 1, IC - 1, get_v_searal);
            }
        }
#undef cb
#undef get_v_general
#undef get_v_searal
    }
};

struct InputTransform2X3_qs8 {
    template <bool inner>
    static void prepare(const int8_t* input, int16_t* patch, int16_t* patchT,
                        int ih_start, int iw_start, size_t IH, size_t IW,
                        size_t ic, size_t IC) {
        constexpr size_t alpha = 2 + 3 - 1;
        if (!(inner && ic + 8 < IC)) {
            memset(patch, 0, sizeof(int16_t) * 8 * alpha * alpha);
        }
        if (inner) {
            const int8_t* input_ptr =
                    input + ic * IH * IW + ih_start * IW + iw_start;
            InputGetter<const int8_t*, int16x4_t> getter;
            for (size_t ico = 0; ico < 8; ++ico) {
                if (ic + ico < IC) {
                    int16x4_t v0 = getter(input_ptr);
                    int16x4_t v1 = getter(input_ptr + IW);
                    int16x4_t v2 = getter(input_ptr + IW * 2);
                    int16x4_t v3 = getter(input_ptr + IW * 3);

                    vst1_s16(patch + (ico * 4 * alpha + 0 * 4), v0);
                    vst1_s16(patch + (ico * 4 * alpha + 1 * 4), v1);
                    vst1_s16(patch + (ico * 4 * alpha + 2 * 4), v2);
                    vst1_s16(patch + (ico * 4 * alpha + 3 * 4), v3);
                    input_ptr += IH * IW;
                }
            }
        } else {
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            // partial copy
            for (size_t ico = 0; ico < 8; ++ico) {
                if (ic + ico < IC) {
                    for (int ih = ih0_act; ih < ih1_act; ++ih) {
                        for (int iw = iw0_act; iw < iw1_act; ++iw) {
                            size_t iho = ih - ih_start, iwo = iw - iw_start;
                            patch[ico * alpha * alpha + iho * alpha + iwo] =
                                    static_cast<int16_t>(
                                            input[(ic + ico) * IH * IW +
                                                  ih * IW + iw]);
                        }
                    }
                }
            }
        }
        transpose_8x4(patch + 4 * 0, patchT + 32 * 0, 16, 4);
        transpose_8x4(patch + 4 * 1, patchT + 32 * 1, 16, 4);
        transpose_8x4(patch + 4 * 2, patchT + 32 * 2, 16, 4);
        transpose_8x4(patch + 4 * 3, patchT + 32 * 3, 16, 4);
    }

    static void transform(const int16_t* patchT, int16_t* input_transform_buf,
                          size_t unit_idx, size_t nr_units_in_tile, size_t ic,
                          size_t IC) {
        constexpr size_t alpha = 2 + 3 - 1;
        // BT * d * B
#define cb(m, n)                 \
    Vector<int16_t, 8> d##m##n = \
            Vector<int16_t, 8>::load(patchT + 8 * (m * 4 + n));

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
                 (m * alpha + n) * nr_units_in_tile * ICB * 8 + \
                 icb * nr_units_in_tile * 8 + unit_idx * 8);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
    }
};

template <BiasMode bmode, typename Op>
struct OutputTransform2X3_qs8 {
    static void transform(const int32_t* output_transform_buf,
                          const int32_t* bias, int8_t* output,
                          int32_t* transform_mid_buf, size_t oh_start,
                          size_t ow_start, size_t OH, size_t OW,
                          size_t oc_start, size_t oc_end, size_t oc_index,
                          size_t unit_idx, size_t nr_units_in_tile,
                          const DType& src_dtype, const DType& filter_dtype,
                          const DType& dst_dtype) {
        float scale_filter = 0.f;
        if (filter_dtype.enumv() == DTypeEnum::QuantizedS8) {
            scale_filter = filter_dtype.param<dtype::QuantizedS8>().scale;
        } else {
            megdnn_assert(filter_dtype.enumv() == DTypeEnum::QuantizedS16);
            scale_filter = filter_dtype.param<dtype::QuantizedS16>().scale;
        }
        float input_filter_scale =
                src_dtype.param<dtype::QuantizedS8>().scale * scale_filter;
        DType buffer_dtype = dtype::QuantizedS32(input_filter_scale * 0.5f *
                                                 0.5f * 1.0f * 1.0f);
        Op op(buffer_dtype, dst_dtype);
        //! AT * m * A
        constexpr size_t alpha = 2 + 3 - 1;

        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / 8;
        size_t ocb = oc_index / 8;

#define cb(m, n)                                           \
    auto v##m##n = Vector<int32_t, 8>::load(               \
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
        v00 = t00 + t01 + t02;
        v10 = t10 + t11 + t12;
        v01 = t01 - t02 + t03;
        v11 = t11 - t12 + t13;

        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            Vector<int32_t, 8> vbias;
            vbias = Vector<int32_t, 8>::load(bias + oc) * (2 * 2);

            v00 += vbias;
            v10 += vbias;
            v01 += vbias;
            v11 += vbias;
        }

        v00.save(transform_mid_buf + (0 * 2 + 0) * 8);
        v10.save(transform_mid_buf + (1 * 2 + 0) * 8);
        v01.save(transform_mid_buf + (0 * 2 + 1) * 8);
        v11.save(transform_mid_buf + (1 * 2 + 1) * 8);

        for (size_t oco = 0; oco < 8 && oc + oco < oc_end; ++oco) {
            for (size_t oho = 0; oho < 2 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 2 && ow_start + owo < OW; ++owo) {
                    dt_qint8 res_int8 = dt_qint8(0);
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    int32_t res =
                            transform_mid_buf[oho * 2 * 8 + owo * 8 + oco];
                    if (bmode == BiasMode::BIAS) {
                        res += bias[(oc + oco) * OH * OW + oh * OW + ow] * 2 *
                               2;
                    }
                    res_int8 = op(dt_qint32(res));
                    output[(oc + oco) * OH * OW + oh * OW + ow] =
                            res_int8.as_int8();
                }
            }
        }
    }
};
}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_8x8_s8)

void winograd_2x3_8x8_s8::filter(const int8_t* filter,
                                 int16_t* filter_transform_buf,
                                 int16_t* transform_mid_buf, size_t OC,
                                 size_t IC, size_t oc_start, size_t oc_end) {
    FilterTransform2X3_qs8::transform(filter, filter_transform_buf,
                                      transform_mid_buf, OC, IC, oc_start,
                                      oc_end);
}

void winograd_2x3_8x8_s8::input(const int8_t* input,
                                int16_t* input_transform_buf,
                                int16_t* transform_mid_buf, size_t IH,
                                size_t IW, size_t IC, size_t PH, size_t PW,
                                size_t unit_start_idx,
                                size_t nr_units_in_tile) {
    megdnn_assert(IC % 8 == 0);
    constexpr int alpha = 3 + 2 - 1;
    
    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    int16_t* patch = transform_mid_buf;
    int16_t* patchT = transform_mid_buf + 8 * alpha * alpha;

    for (size_t ic = 0; ic < IC; ic += 8) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransform2X3_qs8::prepare<true>(input, patch, patchT,
                                                     ih_start, iw_start, IH, IW,
                                                     ic, IC);
                InputTransform2X3_qs8::transform(patchT, input_transform_buf,
                                                 unit_idx, nr_units_in_tile, ic,
                                                 IC);

            } else {
                InputTransform2X3_qs8::prepare<false>(input, patch, patchT,
                                                      ih_start, iw_start, IH,
                                                      IW, ic, IC);
                InputTransform2X3_qs8::transform(patchT, input_transform_buf,
                                                 unit_idx, nr_units_in_tile, ic,
                                                 IC);
            }
        }
    }
}

void winograd_2x3_8x8_s8::output(const int* output_transform_buf,
                                 const int* bias, int8_t* output,
                                 int* transform_mid_buf, BiasMode bmode,
                                 NonlineMode nonline_mode, size_t OH, size_t OW,
                                 size_t oc_start, size_t oc_end,
                                 size_t unit_start_idx, 
                                 size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...)                                    \
    OutputTransform2X3_qs8<_bmode MEGDNN_COMMA _nonline_op>::transform( \
            __VA_ARGS__);

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);

    for (size_t oc = oc_start; oc < oc_end; oc += 8) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            DISPATCH_CONV_WINOGRAD_BIAS_QUANTIZED(
                    megdnn_arm_common_winograd_s8_F23_8x8, cb, dt_qint32, dt_qint8,
                    bmode, nonline_mode, output_transform_buf, bias, output,
                    transform_mid_buf, oh_start, ow_start, OH, OW, oc_start, oc_end, oc_index,
                    unit_idx, nr_units_in_tile, src_dtype, filter_dtype, dst_dtype);
        }
    }
#undef cb
}

}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
