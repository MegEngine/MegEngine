#include "src/fallback/conv_bias/gi/fp16/strategy.h"

#if defined(GI_SUPPORT_F16)

#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/gi/fp16/helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

#include "src/fallback/elemwise_helper/op_unary.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_winograd_nchw88_fp16_F23_mk8)

using namespace megdnn;
using namespace fallback;
namespace {

constexpr size_t alpha = 2 + 3 - 1;
constexpr size_t pack_size = 8;

struct InputTransformF23_NCHW88 {
    template <bool inner>
    static void transform(
            gi_float16_t* patchT, const gi_float16_t* input,
            gi_float16_t* input_transform_buf, size_t ih_start, size_t iw_start,
            size_t IH, size_t IW, size_t unit_idx, size_t nr_units_in_tile, size_t ic,
            size_t IC) {
        size_t IW8 = IW * pack_size;
        size_t icb = ic / pack_size;
        size_t iw8_start = iw_start * pack_size;
        size_t ICB = IC / pack_size;

#define cb(m, n) GI_FLOAT16_t d##m##n;
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb

        if (!(inner && ic + pack_size < IC)) {
            memset(patchT, 0, sizeof(gi_float16_t) * pack_size * alpha * alpha);
        }
        if (inner) {
            MEGDNN_MARK_USED_VAR(patchT);
            const gi_float16_t* input_ptr =
                    input + icb * IH * IW8 + ih_start * IW8 + iw8_start;
#define cb(n, m) d##m##n = GiLoadFloat16(input_ptr + pack_size * n);

            UNROLL_CALL_RAW(4, cb, 0);
            input_ptr += IW8;
            UNROLL_CALL_RAW(4, cb, 1);
            input_ptr += IW8;
            UNROLL_CALL_RAW(4, cb, 2);
            input_ptr += IW8;
            UNROLL_CALL_RAW(4, cb, 3);
#undef cb
        } else {
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            const gi_float16_t* input_ptr = input + icb * IH * IW8;
            // partial copy
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    auto src = GiLoadFloat16(input_ptr + ih * IW8 + iw * pack_size);
                    GiStoreFloat16(
                            patchT + iho * alpha * pack_size + iwo * pack_size, src);
                }
            }
#define cb(m, n) \
    d##m##n = GiLoadFloat16(patchT + m * alpha * pack_size + n * pack_size);
            UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb
        }
        //! 1   0 -1 0    d00 d01 d02 d03     1 0  0  0
        //! 0   1  1 0    d10 d11 d12 d13     0 1 -1 -1
        //! 0  -1  1 0    d20 d21 d22 d23    -1 1  1  0
        //! 0  -1  0 1    d30 d31 d32 d33     0 0  0  1
#define cb(m)                          \
    auto t0##m = SUBF16(d0##m, d2##m); \
    auto t1##m = ADDF16(d1##m, d2##m); \
    auto t2##m = SUBF16(d2##m, d1##m); \
    auto t3##m = SUBF16(d3##m, d1##m);

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m)                           \
    d##m##0 = SUBF16(t##m##0, t##m##2); \
    d##m##1 = ADDF16(t##m##1, t##m##2); \
    d##m##2 = SUBF16(t##m##2, t##m##1); \
    d##m##3 = SUBF16(t##m##3, t##m##1);

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m, n)                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (m * alpha + n) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d##m##n);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb)
#undef cb
    }
};

#define CONCAT(a, idx) a##idx
template <BiasMode bmode, typename Op>
struct OutputTransformF23_NCHW88 {
    static void transform(
            const gi_float16_t* output_transform_buf, const gi_float16_t* bias,
            gi_float16_t* output, gi_float16_t* transform_mid_buf, size_t oh_start,
            size_t ow_start, size_t OH, size_t OW, size_t oc_start, size_t oc_end,
            size_t oc_index, size_t unit_idx, size_t nr_units_in_tile,
            const DType& src_dtype, const DType& dst_dtype) {
        MEGDNN_MARK_USED_VAR(transform_mid_buf);
        Op op(src_dtype, dst_dtype);
        //! AT * m * A
        size_t OCB = (oc_end - oc_start) / pack_size;
        size_t oc = oc_start + oc_index;
        size_t ocb = oc_index / pack_size;

#define cb(m, n)                                                   \
    auto v##m##n = GiLoadFloat16(                                  \
            output_transform_buf +                                 \
            (m * alpha + n) * OCB * nr_units_in_tile * pack_size + \
            ocb * nr_units_in_tile * pack_size + unit_idx * pack_size);
        UNROLL_CALL_NOWRAPPER_D2(4, 4, cb);
#undef cb

        //! 1  1  1 0  v00 v01 v02 v03    1  0
        //! 0  1 -1 1  v10 v11 v12 v13    1  1
        //!            v20 v21 v22 v23    1 -1
        //!            v30 v31 v32 v33    0  1

#define cb(m)                                         \
    auto t0##m = ADDF16(ADDF16(v0##m, v1##m), v2##m); \
    auto t1##m = ADDF16(SUBF16(v1##m, v2##m), v3##m);

        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(m)                                            \
    v##m##0 = ADDF16(ADDF16(t##m##0, t##m##1), t##m##2); \
    v##m##1 = ADDF16(SUBF16(t##m##1, t##m##2), t##m##3);

        UNROLL_CALL_NOWRAPPER(2, cb);
#undef cb

        GI_FLOAT16_t vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = GiLoadFloat16(bias + oc);

#define cb(m, n) v##m##n = ADDF16(v##m##n, vbias);
            UNROLL_CALL_RAW_D2(2, 2, cb);
#undef cb
        }
        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(CONCAT(v##m, n));
            UNROLL_CALL_RAW_D2(2, 2, cb);
#undef cb
        }
#define out_save(oho, owo)                                                           \
    do {                                                                             \
        size_t oh = oh_start + oho;                                                  \
        size_t ow = ow_start + owo;                                                  \
        if (oh < OH && ow < OW) {                                                    \
            if (bmode == BiasMode::BIAS) {                                           \
                v##oho##owo = ADDF16(                                                \
                        v##oho##owo, GiLoadFloat16(                                  \
                                             bias + oc * OH * OW +                   \
                                             oh * OW * pack_size + ow * pack_size)); \
                v##oho##owo = op(v##oho##owo);                                       \
            }                                                                        \
            GiStoreFloat16(                                                          \
                    output + oc * OH * OW + oh * OW * pack_size + ow * pack_size,    \
                    v##oho##owo);                                                    \
        }                                                                            \
    } while (0);
        UNROLL_CALL_RAW_D2(2, 2, out_save);
#undef out_save
    }
};
#undef CONCAT
}  // namespace

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_F23_mk8_f16_nchw88)
void winograd_F23_mk8_f16_nchw88::filter(
        const dt_float16* filter, dt_float16* filter_transform_buf,
        dt_float16* transform_mid_buf, size_t OC, size_t IC, size_t oc_start,
        size_t oc_end) {
    //! 1      0    0    v00 v01 v02   1 0.5  0.5 0
    //! 0.5  0.5  0.5    v10 v11 v12   0 0.5 -0.5 0
    //! 0.5 -0.5  0.5    v20 v21 v22   0 0.5  0.5 1
    //! 0      0    1

    constexpr size_t pack_size = 8;

    MEGDNN_MARK_USED_VAR(transform_mid_buf);
    megdnn_assert(
            (oc_end - oc_start) % pack_size == 0 && oc_start % pack_size == 0 &&
                    oc_end % pack_size == 0 && IC % pack_size == 0 &&
                    OC % pack_size == 0,
            "NCHW88 Winograd filter transform requires both OC and IC "
            "are times of 8");
    size_t OCB = OC / pack_size;
    size_t ICB = IC / pack_size;

    for (size_t ocb = oc_start / pack_size; ocb < oc_end / pack_size; ocb++) {
        for (size_t icb = 0; icb < ICB; icb++) {
            for (size_t ic_inner = 0; ic_inner < pack_size; ic_inner++) {
                const gi_float16_t* fptr =
                        reinterpret_cast<const gi_float16_t*>(filter) +
                        (ocb * ICB + icb) * KERNEL_SIZE * KERNEL_SIZE * pack_size *
                                pack_size +
                        ic_inner * pack_size;

#define cb(m, n)           \
    GI_FLOAT16_t g##m##n = \
            GiLoadFloat16(fptr + (m * KERNEL_SIZE + n) * pack_size * pack_size);
                UNROLL_CALL_NOWRAPPER_D2(3, 3, cb)
#undef cb

#define FILTER_TRANSFORM(n, wd, g)      \
    auto wd##n##0 = g##0##n;            \
    tmp00 = ADDF16(g##0##n, g##2##n);   \
    tmp0 = MULSF16(tmp00, 0.5);         \
    tmp1 = MULSF16(g##1##n, 0.5);       \
    auto wd##n##1 = ADDF16(tmp0, tmp1); \
    auto wd##n##2 = SUBF16(tmp0, tmp1); \
    auto wd##n##3 = g##2##n;
                GI_FLOAT16_t tmp0, tmp1, tmp00;
                UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                UNROLL_CALL_RAW(4, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM
#define cb_save(m, n)                                                                 \
    GiStoreFloat16(                                                                   \
            reinterpret_cast<gi_float16_t*>(filter_transform_buf) +                   \
                    (m * ALPHA + n) * OCB * ICB * pack_size * pack_size +             \
                    ocb * ICB * pack_size * pack_size + icb * pack_size * pack_size + \
                    ic_inner * pack_size,                                             \
            ret##m##n);
                UNROLL_CALL_NOWRAPPER_D2(4, 4, cb_save)
#undef cb_save
            }
        }
    }
}

void winograd_F23_mk8_f16_nchw88::input(
        const dt_float16* input, dt_float16* input_transform_buf,
        dt_float16* transform_mid_buf, size_t IH, size_t IW, size_t IC, size_t PH,
        size_t PW, size_t unit_start_idx, size_t nr_units_in_tile) {
    megdnn_assert(IC % IC_BLOCK_SIZE == 0);
    constexpr int alpha = 3 + 2 - 1;

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    gi_float16_t* patchT = reinterpret_cast<gi_float16_t*>(transform_mid_buf) +
                           IC_BLOCK_SIZE * alpha * alpha;

    for (size_t ic = 0; ic < IC; ic += IC_BLOCK_SIZE) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransformF23_NCHW88::transform<true>(
                        patchT, reinterpret_cast<const gi_float16_t*>(input),
                        reinterpret_cast<gi_float16_t*>(input_transform_buf), ih_start,
                        iw_start, IH, IW, unit_idx, nr_units_in_tile, ic, IC);
            } else {
                InputTransformF23_NCHW88::transform<false>(
                        patchT, reinterpret_cast<const gi_float16_t*>(input),
                        reinterpret_cast<gi_float16_t*>(input_transform_buf), ih_start,
                        iw_start, IH, IW, unit_idx, nr_units_in_tile, ic, IC);
            }
        }
    }
}

void winograd_F23_mk8_f16_nchw88::output(
        const dt_float16* output_transform_buf, const dt_float16* bias,
        dt_float16* output, dt_float16* transform_mid_buf, BiasMode bmode,
        NonlineMode nonline_mode, size_t OH, size_t OW, size_t oc_start, size_t oc_end,
        size_t unit_start_idx, size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...)                                              \
    for (size_t oc = oc_start; oc < oc_end; oc += OC_BLOCK_SIZE) {                \
        size_t oc_index = oc - oc_start;                                          \
        rep(unit_idx, nr_units_in_tile) {                                         \
            size_t index = unit_start_idx + unit_idx;                             \
            auto nh = index / units_w;                                            \
            auto nw = index % units_w;                                            \
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;                             \
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;                             \
            OutputTransformF23_NCHW88<_bmode, _nonline_op>::transform(            \
                    reinterpret_cast<const gi_float16_t*>(output_transform_buf),  \
                    reinterpret_cast<const gi_float16_t*>(bias),                  \
                    reinterpret_cast<gi_float16_t*>(output),                      \
                    reinterpret_cast<gi_float16_t*>(transform_mid_buf), oh_start, \
                    ow_start, OH, OW, oc_start, oc_end, oc_index, unit_idx,       \
                    nr_units_in_tile, src_dtype, dst_dtype);                      \
        }                                                                         \
    }

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    constexpr size_t pack_size = 8;

    size_t OC = oc_end - oc_start;
    megdnn_assert(
            OC % pack_size == 0 && oc_start % pack_size == 0 && oc_end % pack_size == 0,
            "NCHW88 Winograd filter transform requires OC is times of 8");

    GI_DISPATCH_CONV_WINOGRAD_BIAS(
            megdnn_fallback_winograd_nchw88_fp16_F23_mk8, cb, gi_float16_t,
            gi_float16_t, bmode, nonline_mode);
#undef cb
}

}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
