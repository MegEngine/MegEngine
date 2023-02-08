#include "src/fallback/conv_bias/gi/fp16/strategy.h"
#if defined(GI_SUPPORT_F16)
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/fallback/conv_bias/gi/fp16/helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/fallback/elemwise_helper/op_unary.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_winograd_fp16_F63_mk8)

using namespace megdnn;
using namespace fallback;

namespace {

constexpr size_t alpha = 6 + 3 - 1;
constexpr size_t pack_size = 8;
constexpr gi_float16_t input_parameters[16] = {5.25f, 4.25f, 0.5f, 0.25f, 2.5f, 1.25f,
                                               2.0f,  4.0f,  5.0f, 0.0f,  0.0f, 0.0f,
                                               0.0f,  0.0f,  0.0f, 0.0f};

struct InputTransformF63_NCHW88 {
    template <bool inner>
    static void prepare(
            const gi_float16_t* input, gi_float16_t* patch, gi_float16_t* patchT,
            int ih_start, int iw_start, size_t IH, size_t IW, size_t ic, size_t IC) {
        MEGDNN_MARK_USED_VAR(patch);
        size_t IW8 = IW * pack_size;
        size_t iw8_start = iw_start * pack_size;
        size_t icb = ic / pack_size;
        if (!(inner && ic + pack_size < IC)) {
            memset(patchT, 0, sizeof(gi_float16_t) * pack_size * alpha * alpha);
        }
        if (inner) {
            const gi_float16_t* input_ptr =
                    input + icb * IH * IW8 + ih_start * IW8 + iw8_start;
            for (size_t ih = 0; ih < alpha; ih++) {
#define cb(i) auto v##i = GiLoadFloat16(input_ptr + pack_size * i);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) GiStoreFloat16(patchT + ih * pack_size * alpha + i * pack_size, v##i);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                input_ptr += IW8;
            }
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
                            patchT + iho * pack_size * alpha + iwo * pack_size, src);
                }
            }
        }
    }

    static void transform(
            const gi_float16_t* patchT, gi_float16_t* input_transform_buf,
            size_t unit_idx, size_t nr_units_in_tile, size_t ic, size_t IC) {
        // BT * d * B

        size_t ICB = IC / pack_size;
        size_t icb = ic / pack_size;

        GI_FLOAT16_t d0, d1, d2, d3, d4, d5, d6, d7;
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MADD(a, b, c, d) GiMultiplyAddScalarFloat16(a, b, *(c + d))
#define MSUB(a, b, c, d) GiMultiplySubScalarFloat16(a, b, *(c + d))
        const gi_float16_t* v0 = input_parameters + 0;
        const gi_float16_t* v1 = input_parameters + 8;
        // const float* v2 = input_parameters + 8;
#else
#define MADD(a, b, c, d) GiSimdFmaLaneFloat16(a, b, c, d)
#define MSUB(a, b, c, d) GiFmsqLaneQFloat16(a, b, c, d)
        GI_FLOAT16_t v0 = GiLoadFloat16(input_parameters + 0);
        GI_FLOAT16_t v1 = GiLoadFloat16(input_parameters + 8);
        // GI_FLOAT32_t v2 = GiLoadFloat32(input_parameters + 8);
#endif

        //! B
        //!     1     0     0     0     0    0    0     0
        //!     0     1    -1   0.5  -0.5    2   -2    -1
        //! -5.25     1     1  0.25  0.25    4    4     0
        //!     0 -4.25  4.25  -2.5   2.5 -2.5  2.5  5.25
        //!  5.25 -4.25 -4.25 -1.25 -1.25   -5   -5     0
        //!     0     1    -1     2    -2  0.5 -0.5 -5.25
        //!    -1     1     1     1     1    1    1     0
        //!     0     0     0     0     0    0    0     1

#define cb(i)                                                                     \
    d1 = GiLoadFloat16(patchT + i * alpha * pack_size + 1 * pack_size);           \
    d2 = GiLoadFloat16(patchT + i * alpha * pack_size + 2 * pack_size);           \
    d3 = GiLoadFloat16(patchT + i * alpha * pack_size + 3 * pack_size);           \
    d4 = GiLoadFloat16(patchT + i * alpha * pack_size + 4 * pack_size);           \
    d5 = GiLoadFloat16(patchT + i * alpha * pack_size + 5 * pack_size);           \
    d6 = GiLoadFloat16(patchT + i * alpha * pack_size + 6 * pack_size);           \
    auto t##i##0 = GiLoadFloat16(patchT + i * alpha * pack_size + 0 * pack_size); \
    auto t##i##7 = GiLoadFloat16(patchT + i * alpha * pack_size + 7 * pack_size); \
    auto t##i##1 = d6;                                                            \
    auto t##i##2 = d6;                                                            \
    auto t##i##3 = d6;                                                            \
    auto t##i##4 = d6;                                                            \
    auto t##i##5 = d6;                                                            \
    auto t##i##6 = d6;                                                            \
    t##i##0 = SUBF16(t##i##0, d6);                                                \
    t##i##1 = ADDF16(t##i##1, d1);                                                \
    t##i##2 = SUBF16(t##i##2, d1);                                                \
    t##i##3 = MADD(t##i##3, d1, v0, 2);                                           \
    t##i##4 = MSUB(t##i##4, d1, v0, 2);                                           \
    t##i##5 = MADD(t##i##5, d1, v0, 6);                                           \
    t##i##6 = MSUB(t##i##6, d1, v0, 6);                                           \
    t##i##7 = SUBF16(t##i##7, d1);                                                \
    t##i##0 = MSUB(t##i##0, d2, v0, 0);                                           \
    t##i##1 = ADDF16(t##i##1, d2);                                                \
    t##i##2 = ADDF16(t##i##2, d2);                                                \
    t##i##3 = MADD(t##i##3, d2, v0, 3);                                           \
    t##i##4 = MADD(t##i##4, d2, v0, 3);                                           \
    t##i##5 = MADD(t##i##5, d2, v0, 7);                                           \
    t##i##6 = MADD(t##i##6, d2, v0, 7);                                           \
    t##i##1 = MSUB(t##i##1, d3, v0, 1);                                           \
    t##i##2 = MADD(t##i##2, d3, v0, 1);                                           \
    t##i##3 = MSUB(t##i##3, d3, v0, 4);                                           \
    t##i##4 = MADD(t##i##4, d3, v0, 4);                                           \
    t##i##5 = MSUB(t##i##5, d3, v0, 4);                                           \
    t##i##6 = MADD(t##i##6, d3, v0, 4);                                           \
    t##i##7 = MADD(t##i##7, d3, v0, 0);                                           \
    t##i##0 = MADD(t##i##0, d4, v0, 0);                                           \
    t##i##1 = MSUB(t##i##1, d4, v0, 1);                                           \
    t##i##2 = MSUB(t##i##2, d4, v0, 1);                                           \
    t##i##3 = MSUB(t##i##3, d4, v0, 5);                                           \
    t##i##4 = MSUB(t##i##4, d4, v0, 5);                                           \
    t##i##5 = MSUB(t##i##5, d4, v1, 0);                                           \
    t##i##6 = MSUB(t##i##6, d4, v1, 0);                                           \
    t##i##1 = ADDF16(t##i##1, d5);                                                \
    t##i##2 = SUBF16(t##i##2, d5);                                                \
    t##i##3 = MADD(t##i##3, d5, v0, 6);                                           \
    t##i##4 = MSUB(t##i##4, d5, v0, 6);                                           \
    t##i##5 = MADD(t##i##5, d5, v0, 2);                                           \
    t##i##6 = MSUB(t##i##6, d5, v0, 2);                                           \
    t##i##7 = MSUB(t##i##7, d5, v0, 0);
        UNROLL_CALL_RAW(8, cb);
#undef cb

#define cb(i)                                                                  \
    d0 = t0##i;                                                                \
    d1 = t6##i;                                                                \
    d2 = t6##i;                                                                \
    d3 = t6##i;                                                                \
    d4 = t6##i;                                                                \
    d5 = t6##i;                                                                \
    d6 = t6##i;                                                                \
    d7 = t7##i;                                                                \
    d0 = SUBF16(d0, t6##i);                                                    \
    d1 = ADDF16(d1, t1##i);                                                    \
    d2 = SUBF16(d2, t1##i);                                                    \
    d3 = MADD(d3, t1##i, v0, 2);                                               \
    d4 = MSUB(d4, t1##i, v0, 2);                                               \
    d5 = MADD(d5, t1##i, v0, 6);                                               \
    d6 = MSUB(d6, t1##i, v0, 6);                                               \
    d7 = SUBF16(d7, t1##i);                                                    \
    d0 = MSUB(d0, t2##i, v0, 0);                                               \
    d1 = ADDF16(d1, t2##i);                                                    \
    d2 = ADDF16(d2, t2##i);                                                    \
    d3 = MADD(d3, t2##i, v0, 3);                                               \
    d4 = MADD(d4, t2##i, v0, 3);                                               \
    d5 = MADD(d5, t2##i, v0, 7);                                               \
    d6 = MADD(d6, t2##i, v0, 7);                                               \
    d1 = MSUB(d1, t3##i, v0, 1);                                               \
    d2 = MADD(d2, t3##i, v0, 1);                                               \
    d3 = MSUB(d3, t3##i, v0, 4);                                               \
    d4 = MADD(d4, t3##i, v0, 4);                                               \
    d5 = MSUB(d5, t3##i, v0, 4);                                               \
    d6 = MADD(d6, t3##i, v0, 4);                                               \
    d7 = MADD(d7, t3##i, v0, 0);                                               \
    d0 = MADD(d0, t4##i, v0, 0);                                               \
    d1 = MSUB(d1, t4##i, v0, 1);                                               \
    d2 = MSUB(d2, t4##i, v0, 1);                                               \
    d3 = MSUB(d3, t4##i, v0, 5);                                               \
    d4 = MSUB(d4, t4##i, v0, 5);                                               \
    d5 = MSUB(d5, t4##i, v1, 0);                                               \
    d6 = MSUB(d6, t4##i, v1, 0);                                               \
    d1 = ADDF16(d1, t5##i);                                                    \
    d2 = SUBF16(d2, t5##i);                                                    \
    d3 = MADD(d3, t5##i, v0, 6);                                               \
    d4 = MSUB(d4, t5##i, v0, 6);                                               \
    d5 = MADD(d5, t5##i, v0, 2);                                               \
    d6 = MSUB(d6, t5##i, v0, 2);                                               \
    d7 = MSUB(d7, t5##i, v0, 0);                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (0 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d0);                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (1 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d1);                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (2 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d2);                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (3 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d3);                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (4 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d4);                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (5 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d5);                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (6 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d6);                                                               \
    GiStoreFloat16(                                                            \
            input_transform_buf +                                              \
                    (7 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d7);
        UNROLL_CALL_RAW(8, cb);
#undef cb
#undef MADD
#undef MSUB
    }
};

template <BiasMode bmode, typename Op>
struct OutputTransformF63_NCHW88 {
    static void transform(
            const gi_float16_t* output_transform_buf, const gi_float16_t* bias,
            gi_float16_t* output, gi_float16_t* transform_mid_buf, size_t oh_start,
            size_t ow_start, size_t OH, size_t OW, size_t oc_start, size_t oc_end,
            size_t oc_index, size_t unit_idx, size_t nr_units_in_tile,
            const DType& src_dtype, const DType& dst_dtype) {
        MEGDNN_MARK_USED_VAR(transform_mid_buf);
        Op op(src_dtype, dst_dtype);
        //! AT * m * A

        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / pack_size;
        size_t ocb = oc_index / pack_size;

#define cb(m, n)                                                   \
    auto v##m##n = GiLoadFloat16(                                  \
            output_transform_buf +                                 \
            (m * alpha + n) * OCB * nr_units_in_tile * pack_size + \
            ocb * nr_units_in_tile * pack_size + unit_idx * pack_size);
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
         * 0    0    0      0       0         1
         */

        /*
         * v1addv2 = v1##m + v2##m;
         * v1subv2 = v1##m - v2##m;
         * v3addv4 = v3##m + v4##m;
         * v3subv4 = v3##m - v4##m;
         * v5addv6 = v5##m + v6##m;
         * v5subv6 = v5##m - v6##m;
         * auto t0##m = v0##m + v1addv2 + v3addv4 + v5addv6;
         * auto t1##m = v1subv2 + v3subv4 * 2.f + v5subv6 * 0.5f;
         * auto t2##m = v1addv2 + v3addv4 * 4.f + v5addv6 * 0.25f;
         * auto t3##m = v1subv2 + v3subv4 * 8.f + v5subv6 * 0.125f;
         * auto t4##m = v1addv2 + v3addv4 * 16.f + v5addv6 * 0.0625f;
         * auto t5##m = v1subv2 + v3subv4 * 32.f + v5subv6 * 0.03125f + v7##m;
         */
        GI_FLOAT16_t v1addv2, v1subv2, v3addv4, v3subv4, v5addv6, v5subv6;
#define cb(m)                                                                         \
    v1addv2 = ADDF16(v1##m, v2##m);                                                   \
    v1subv2 = SUBF16(v1##m, v2##m);                                                   \
    v3addv4 = ADDF16(v3##m, v4##m);                                                   \
    v3subv4 = SUBF16(v3##m, v4##m);                                                   \
    v5addv6 = ADDF16(v5##m, v6##m);                                                   \
    v5subv6 = SUBF16(v5##m, v6##m);                                                   \
    auto t0##m = ADDF16(ADDF16(ADDF16(v0##m, v1addv2), v3addv4), v5addv6);            \
    auto t1##m =                                                                      \
            ADDF16(ADDF16(v1subv2, MULSF16(v3subv4, 2.f)), MULSF16(v5subv6, 0.5f));   \
    auto t2##m =                                                                      \
            ADDF16(ADDF16(v1addv2, MULSF16(v3addv4, 4.f)), MULSF16(v5addv6, 0.25f));  \
    auto t3##m =                                                                      \
            ADDF16(ADDF16(v1subv2, MULSF16(v3subv4, 8.f)), MULSF16(v5subv6, 0.125f)); \
    auto t4##m = ADDF16(                                                              \
            ADDF16(v1addv2, MULSF16(v3addv4, 16.f)), MULSF16(v5addv6, 0.0625f));      \
    auto t5##m =                                                                      \
            ADDF16(ADDF16(ADDF16(v1subv2, MULSF16(v3subv4, 32.f)),                    \
                          MULSF16(v5subv6, 0.03125f)),                                \
                   v7##m);

        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        /*
         * v1addv2 = t##m##1 + t##m##2;
         * v1subv2 = t##m##1 - t##m##2;
         * v3addv4 = t##m##3 + t##m##4;
         * v3subv4 = t##m##3 - t##m##4;
         * v5addv6 = t##m##5 + t##m##6;
         * v5subv6 = t##m##5 - t##m##6;
         * v##m##0 = t##m##0 + v1addv2 + v3addv4 + v5addv6;
         * v##m##1 = v1subv2 + v3subv4 * 2.f + v5subv6 * 0.5f;
         * v##m##2 = v1addv2 + v3addv4 * 4.f + v5addv6 * 0.25f;
         * v##m##3 = v1subv2 + v3subv4 * 8.f + v5subv6 * 0.125f;
         * v##m##4 = v1addv2 + v3addv4 * 16.f + v5addv6 * 0.0625f;
         * v##m##5 = v1subv2 + v3subv4 * 32.f + v5subv6 * 0.03125f + t##m##7;
         */
#define cb(m)                                                                          \
    v1addv2 = ADDF16(t##m##1, t##m##2);                                                \
    v1subv2 = SUBF16(t##m##1, t##m##2);                                                \
    v3addv4 = ADDF16(t##m##3, t##m##4);                                                \
    v3subv4 = SUBF16(t##m##3, t##m##4);                                                \
    v5addv6 = ADDF16(t##m##5, t##m##6);                                                \
    v5subv6 = SUBF16(t##m##5, t##m##6);                                                \
    v##m##0 = ADDF16(ADDF16(ADDF16(t##m##0, v1addv2), v3addv4), v5addv6);              \
    v##m##1 = ADDF16(ADDF16(v1subv2, MULSF16(v3subv4, 2.f)), MULSF16(v5subv6, 0.5f));  \
    v##m##2 = ADDF16(ADDF16(v1addv2, MULSF16(v3addv4, 4.f)), MULSF16(v5addv6, 0.25f)); \
    v##m##3 =                                                                          \
            ADDF16(ADDF16(v1subv2, MULSF16(v3subv4, 8.f)), MULSF16(v5subv6, 0.125f));  \
    v##m##4 = ADDF16(                                                                  \
            ADDF16(v1addv2, MULSF16(v3addv4, 16.f)), MULSF16(v5addv6, 0.0625f));       \
    v##m##5 =                                                                          \
            ADDF16(ADDF16(ADDF16(v1subv2, MULSF16(v3subv4, 32.f)),                     \
                          MULSF16(v5subv6, 0.03125f)),                                 \
                   t##m##7);

        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

        GI_FLOAT16_t vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = GiLoadFloat16(bias + oc);

#define cb(m, n) v##m##n = ADDF16(v##m##n, vbias);
            UNROLL_CALL_RAW_D2(6, 6, cb);
#undef cb
        }
        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(CONCAT(v##m, n));
            UNROLL_CALL_RAW_D2(6, 6, cb);
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
        UNROLL_CALL_RAW_D2(6, 6, out_save);
    }
#undef out_save
};
}  // namespace

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_F63_mk8_f16_nchw88)

void winograd_F63_mk8_f16_nchw88::filter(
        const dt_float16* filter, dt_float16* filter_transform_buf,
        dt_float16* transform_mid_buf, size_t OC, size_t IC, size_t oc_start,
        size_t oc_end) {
    constexpr size_t pack_size = 8;
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
            (oc_end - oc_start) % pack_size == 0 && oc_start % pack_size == 0 &&
                    oc_end % pack_size == 0 && IC % pack_size == 0 &&
                    OC % pack_size == 0,
            "NCHW88 Winograd filter transform requires both OC and IC "
            "are times of 8");

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

                /*
                 * auto wd##n##0 = g##0##n;
                 * tmp0 = (g##0##n + g##2##n) * -0.2222222f;
                 * tmp1 = g##1##n * -0.2222222f;
                 * auto wd##n##1 = tmp0 + tmp1;
                 * auto wd##n##2 = tmp0 - tmp1;
                 * tmp0 = g##0##n * 0.0111111f + g##2##n * 0.0444444f;
                 * tmp1 = g##1##n * 0.0222222f;
                 * auto wd##n##3 = tmp0 + tmp1;
                 * auto wd##n##4 = tmp0 - tmp1;
                 * tmp0 = g##0##n * 0.7111111f + g##2##n * 0.1777778f;
                 * tmp1 = g##1##n * 0.3555556f;
                 * auto wd##n##5 = tmp0 + tmp1;
                 * auto wd##n##6 = tmp0 - tmp1;
                 * auto wd##n##7 = g##2##n;
                 */
#define FILTER_TRANSFORM(n, wd, g)                                             \
    auto wd##n##0 = g##0##n;                                                   \
    tmp0 = MULSF16(ADDF16(g##0##n, g##2##n), -2.0f / 9);                       \
    tmp1 = MULSF16(g##1##n, -2.0f / 9);                                        \
    auto wd##n##1 = ADDF16(tmp0, tmp1);                                        \
    auto wd##n##2 = SUBF16(tmp0, tmp1);                                        \
    tmp0 = ADDF16(MULSF16(g##0##n, 1.0f / 90), MULSF16(g##2##n, 2.0f / 45));   \
    tmp1 = MULSF16(g##1##n, 1.0f / 45);                                        \
    auto wd##n##3 = ADDF16(tmp0, tmp1);                                        \
    auto wd##n##4 = SUBF16(tmp0, tmp1);                                        \
    tmp0 = ADDF16(MULSF16(g##0##n, 0.7111111f), MULSF16(g##2##n, 0.1777778f)); \
    tmp1 = MULSF16(g##1##n, 0.3555556f);                                       \
    auto wd##n##5 = ADDF16(tmp0, tmp1);                                        \
    auto wd##n##6 = SUBF16(tmp0, tmp1);                                        \
    auto wd##n##7 = g##2##n;
                GI_FLOAT16_t tmp0, tmp1;
                UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                UNROLL_CALL_RAW(8, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM
#define cb_save(m, n)                                                   \
    GiStoreFloat16(                                                     \
            reinterpret_cast<gi_float16_t*>(filter_transform_buf) +     \
                    (m * alpha + n) * OC * IC + ocb * IC * pack_size +  \
                    icb * pack_size * pack_size + ic_inner * pack_size, \
            ret##m##n);
                UNROLL_CALL_NOWRAPPER_D2(8, 8, cb_save)
#undef cb_save
            }
        }
    }
}

void winograd_F63_mk8_f16_nchw88::input(
        const dt_float16* input, dt_float16* input_transform_buf,
        dt_float16* transform_mid_buf, size_t IH, size_t IW, size_t IC, size_t PH,
        size_t PW, size_t unit_start_idx, size_t nr_units_in_tile) {
    constexpr size_t pack_size = 8;
    megdnn_assert(IC % pack_size == 0);
    constexpr int alpha = 3 + 6 - 1;

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    gi_float16_t* patch = reinterpret_cast<gi_float16_t*>(transform_mid_buf);
    gi_float16_t* patchT = reinterpret_cast<gi_float16_t*>(transform_mid_buf) +
                           pack_size * alpha * alpha;

    for (size_t ic = 0; ic < IC; ic += pack_size) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransformF63_NCHW88::prepare<true>(
                        reinterpret_cast<const gi_float16_t*>(input), patch, patchT,
                        ih_start, iw_start, IH, IW, ic, IC);
                InputTransformF63_NCHW88::transform(
                        patchT, reinterpret_cast<gi_float16_t*>(input_transform_buf),
                        unit_idx, nr_units_in_tile, ic, IC);

            } else {
                InputTransformF63_NCHW88::prepare<false>(
                        reinterpret_cast<const gi_float16_t*>(input), patch, patchT,
                        ih_start, iw_start, IH, IW, ic, IC);
                InputTransformF63_NCHW88::transform(
                        patchT, reinterpret_cast<gi_float16_t*>(input_transform_buf),
                        unit_idx, nr_units_in_tile, ic, IC);
            }
        }
    }
}

void winograd_F63_mk8_f16_nchw88::output(
        const dt_float16* output_transform_buf, const dt_float16* bias,
        dt_float16* output, dt_float16* transform_mid_buf, BiasMode bmode,
        NonlineMode nonline_mode, size_t OH, size_t OW, size_t oc_start, size_t oc_end,
        size_t unit_start_idx, size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...)                                               \
    for (size_t oc = oc_start; oc < oc_end; oc += pack_size) {                     \
        size_t oc_index = oc - oc_start;                                           \
        rep(unit_idx, nr_units_in_tile) {                                          \
            size_t index = unit_start_idx + unit_idx;                              \
            auto nh = index / units_w;                                             \
            auto nw = index % units_w;                                             \
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;                              \
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;                              \
            OutputTransformF63_NCHW88<_bmode MEGDNN_COMMA _nonline_op>::transform( \
                    reinterpret_cast<const gi_float16_t*>(output_transform_buf),   \
                    reinterpret_cast<const gi_float16_t*>(bias),                   \
                    reinterpret_cast<gi_float16_t*>(output),                       \
                    reinterpret_cast<gi_float16_t*>(transform_mid_buf), oh_start,  \
                    ow_start, OH, OW, oc_start, oc_end, oc_index, unit_idx,        \
                    nr_units_in_tile, src_dtype, dst_dtype);                       \
        }                                                                          \
    }

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    constexpr size_t pack_size = 8;

    size_t OC = oc_end - oc_start;
    megdnn_assert(
            OC % pack_size == 0 && oc_start % pack_size == 0 && oc_end % pack_size == 0,
            "NCHW88 Winograd filter transform requires OC is times of 8");

    GI_DISPATCH_CONV_WINOGRAD_BIAS(
            megdnn_fallback_winograd_fp16_F63_mk8, cb, gi_float16_t, gi_float16_t,
            bmode, nonline_mode);
#undef cb
}

}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
