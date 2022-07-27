#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/fallback/conv_bias/gi/fp32/filter_transform.h"
#include "src/fallback/conv_bias/gi/fp32/helper.h"
#include "src/fallback/conv_bias/gi/fp32/strategy.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/fallback/elemwise_helper/op_unary.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_winograd_fp32_F73_mk4)

using namespace megdnn;
using namespace fallback;

namespace {

constexpr size_t alpha = 7 + 3 - 1;
constexpr size_t pack_size = 4;
constexpr float input_parameters[28] = {
        1.5f,  0.75f,  3.0f,   7.875f, 0.5f,  2.5f,  0.125f, 0.875f,  4.0f,   8.0f,
        5.25f, 7.375f, 5.375f, 3.5f,   7.75f, 0.25f, 2.125f, 10.625f, 0.625f, 4.375f,
        5.0f,  10.0f,  5.75f,  2.75f,  4.25f, 1.75f, 2.0f,   0.0f};

struct InputTransformF73_NCHW44 {
    template <bool inner>
    static void prepare(
            const float* input, float* patch, float* patchT, int ih_start, int iw_start,
            size_t IH, size_t IW, size_t ic, size_t IC) {
        MEGDNN_MARK_USED_VAR(patch);
        size_t IW4 = IW * pack_size;
        size_t iw4_start = iw_start * pack_size;
        size_t icb = ic / pack_size;
        if (!(inner && ic + pack_size < IC)) {
            memset(patchT, 0, sizeof(float) * pack_size * alpha * alpha);
        }
        if (inner) {
            const float* input_ptr =
                    input + icb * IH * IW4 + ih_start * IW4 + iw4_start;
            for (size_t ih = 0; ih < alpha; ih++) {
#define cb(i) auto v##i = GiLoadFloat32(input_ptr + pack_size * i);
                UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb

#define cb(i) GiStoreFloat32(patchT + ih * pack_size * alpha + i * pack_size, v##i);
                UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb
                input_ptr += IW4;
            }
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
                    auto src = GiLoadFloat32(input_ptr + ih * IW4 + iw * pack_size);
                    GiStoreFloat32(
                            patchT + iho * pack_size * alpha + iwo * pack_size, src);
                }
            }
        }
    }

    static void transform(
            const float* patchT, float* input_transform_buf, size_t unit_idx,
            size_t nr_units_in_tile, size_t ic, size_t IC) {
        // BT * d * B

        size_t ICB = IC / pack_size;
        size_t icb = ic / pack_size;

        GI_FLOAT32_t d0, d1, d2, d3, d4, d5, d6, d7, d8;
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MADD(a, b, c, d) GiMultiplyAddScalarFloat32(a, b, *(c + d))
        const float* v0 = input_parameters + 0;
        const float* v1 = input_parameters + 4;
        const float* v2 = input_parameters + 8;
        const float* v3 = input_parameters + 12;
        const float* v4 = input_parameters + 16;
        const float* v5 = input_parameters + 20;
        const float* v6 = input_parameters + 24;
#define MSUB(a, b, c, d) GiMultiplySubScalarFloat32(a, b, *(c + d))
#else
#define MADD(a, b, c, d) GiSimdFmaLane(a, b, c, d)
#define MSUB(a, b, c, d) GiFmsqLaneQFloat32(a, b, c, d)
        GI_FLOAT32_t v0 = GiLoadFloat32(input_parameters + 0);
        GI_FLOAT32_t v1 = GiLoadFloat32(input_parameters + 4);
        GI_FLOAT32_t v2 = GiLoadFloat32(input_parameters + 8);
        GI_FLOAT32_t v3 = GiLoadFloat32(input_parameters + 12);
        GI_FLOAT32_t v4 = GiLoadFloat32(input_parameters + 16);
        GI_FLOAT32_t v5 = GiLoadFloat32(input_parameters + 20);
        GI_FLOAT32_t v6 = GiLoadFloat32(input_parameters + 24);
#endif

        //! B
        //!    1.5      0       0       0       0      0      0      0       0
        //!     -1   -1.5     1.5   -0.75    0.75     -3      3     -1     1.5
        //! -7.875   -0.5    -2.5   0.125  -0.875     -4     -8      0      -1
        //!   5.25  7.375  -5.375       4    -3.5   7.75   0.25   5.25  -7.875
        //!  7.875  2.125  10.625  -0.625   4.375      5     10      0    5.25
        //!  -5.25  -5.75   -2.75   -4.25    1.75  -5.75  -4.25  -5.25   7.875
        //!   -1.5   -0.5    -2.5     0.5    -3.5     -1     -2      0   -5.25
        //！     1      1       1       1       1      1      1      1    -1.5
        //!      0      0       0       0       0      0      0      0       1

        // 1.5f,     0.75f,   3.0f,    7.875f,         v0
        // 0.5f,     2.5f,    0.125f,  0.875f,         v1
        // 4.0f,     8.0f,    5.25f,   7.375f,         v2
        // 5.375f,   3.5f,    7.75f,   0.25f,          v3
        // 2.125f,   10.625f, 0.625f,  4.375f,         v4
        // 5.0f,     10.0f,   5.75f,   2.75f,          v5
        // 4.25f,    1.75f,   2.0f,    0.0f,           v6

#define cb(i)                                                                     \
    d0 = GiLoadFloat32(patchT + i * alpha * pack_size + 0 * pack_size);           \
    d1 = GiLoadFloat32(patchT + i * alpha * pack_size + 1 * pack_size);           \
    d2 = GiLoadFloat32(patchT + i * alpha * pack_size + 2 * pack_size);           \
    d3 = GiLoadFloat32(patchT + i * alpha * pack_size + 3 * pack_size);           \
    d4 = GiLoadFloat32(patchT + i * alpha * pack_size + 4 * pack_size);           \
    d5 = GiLoadFloat32(patchT + i * alpha * pack_size + 5 * pack_size);           \
    d6 = GiLoadFloat32(patchT + i * alpha * pack_size + 6 * pack_size);           \
    d7 = GiLoadFloat32(patchT + i * alpha * pack_size + 7 * pack_size);           \
    auto t##i##8 = GiLoadFloat32(patchT + i * alpha * pack_size + 8 * pack_size); \
    auto t##i##0 = d7;                                                            \
    auto t##i##1 = d7;                                                            \
    auto t##i##2 = d7;                                                            \
    auto t##i##3 = d7;                                                            \
    auto t##i##4 = d7;                                                            \
    auto t##i##5 = d7;                                                            \
    auto t##i##6 = d7;                                                            \
    auto t##i##7 = d7;                                                            \
    t##i##8 = MSUB(t##i##8, d7, v0, 0);                                           \
    t##i##0 = SUBF(t##i##0, d1);                                                  \
    t##i##1 = MSUB(t##i##1, d1, v0, 0);                                           \
    t##i##2 = MADD(t##i##2, d1, v0, 0);                                           \
    t##i##3 = MSUB(t##i##3, d1, v0, 1);                                           \
    t##i##4 = MADD(t##i##4, d1, v0, 1);                                           \
    t##i##5 = MSUB(t##i##5, d1, v0, 2);                                           \
    t##i##6 = MADD(t##i##6, d1, v0, 2);                                           \
    t##i##7 = SUBF(t##i##7, d1);                                                  \
    t##i##8 = MADD(t##i##8, d1, v0, 0);                                           \
    t##i##0 = MSUB(t##i##0, d2, v0, 3);                                           \
    t##i##1 = MSUB(t##i##1, d2, v1, 0);                                           \
    t##i##2 = MSUB(t##i##2, d2, v1, 1);                                           \
    t##i##3 = MADD(t##i##3, d2, v1, 2);                                           \
    t##i##4 = MSUB(t##i##4, d2, v1, 3);                                           \
    t##i##5 = MSUB(t##i##5, d2, v2, 0);                                           \
    t##i##6 = MSUB(t##i##6, d2, v2, 1);                                           \
    t##i##8 = SUBF(t##i##8, d2);                                                  \
    t##i##0 = MADD(t##i##0, d3, v2, 2);                                           \
    t##i##1 = MADD(t##i##1, d3, v2, 3);                                           \
    t##i##2 = MSUB(t##i##2, d3, v3, 0);                                           \
    t##i##3 = MADD(t##i##3, d3, v2, 0);                                           \
    t##i##4 = MSUB(t##i##4, d3, v3, 1);                                           \
    t##i##5 = MADD(t##i##5, d3, v3, 2);                                           \
    t##i##6 = MADD(t##i##6, d3, v3, 3);                                           \
    t##i##7 = MADD(t##i##7, d3, v2, 2);                                           \
    t##i##8 = MSUB(t##i##8, d3, v0, 3);                                           \
    t##i##0 = MADD(t##i##0, d4, v0, 3);                                           \
    t##i##1 = MADD(t##i##1, d4, v4, 0);                                           \
    t##i##2 = MADD(t##i##2, d4, v4, 1);                                           \
    t##i##3 = MSUB(t##i##3, d4, v4, 2);                                           \
    t##i##4 = MADD(t##i##4, d4, v4, 3);                                           \
    t##i##5 = MADD(t##i##5, d4, v5, 0);                                           \
    t##i##6 = MADD(t##i##6, d4, v5, 1);                                           \
    t##i##8 = MADD(t##i##8, d4, v2, 2);                                           \
    t##i##0 = MSUB(t##i##0, d5, v2, 2);                                           \
    t##i##1 = MSUB(t##i##1, d5, v5, 2);                                           \
    t##i##2 = MSUB(t##i##2, d5, v5, 3);                                           \
    t##i##3 = MSUB(t##i##3, d5, v6, 0);                                           \
    t##i##4 = MADD(t##i##4, d5, v6, 1);                                           \
    t##i##5 = MSUB(t##i##5, d5, v5, 2);                                           \
    t##i##6 = MSUB(t##i##6, d5, v6, 0);                                           \
    t##i##7 = MSUB(t##i##7, d5, v2, 2);                                           \
    t##i##8 = MADD(t##i##8, d5, v0, 3);                                           \
    t##i##0 = MSUB(t##i##0, d6, v0, 0);                                           \
    t##i##1 = MSUB(t##i##1, d6, v1, 0);                                           \
    t##i##2 = MSUB(t##i##2, d6, v1, 1);                                           \
    t##i##3 = MADD(t##i##3, d6, v1, 0);                                           \
    t##i##4 = MSUB(t##i##4, d6, v3, 1);                                           \
    t##i##5 = SUBF(t##i##5, d6);                                                  \
    t##i##6 = MSUB(t##i##6, d6, v6, 2);                                           \
    t##i##8 = MSUB(t##i##8, d6, v2, 2);                                           \
    t##i##0 = MADD(t##i##0, d0, v0, 0);

        UNROLL_CALL_RAW(9, cb);
#undef cb

#define cb(i)                                                                  \
    d8 = t8##i;                                                                \
    d0 = t7##i;                                                                \
    d1 = t7##i;                                                                \
    d2 = t7##i;                                                                \
    d3 = t7##i;                                                                \
    d4 = t7##i;                                                                \
    d5 = t7##i;                                                                \
    d6 = t7##i;                                                                \
    d7 = t7##i;                                                                \
    d8 = MSUB(d8, t7##i, v0, 0);                                               \
    d0 = SUBF(d0, t1##i);                                                      \
    d1 = MSUB(d1, t1##i, v0, 0);                                               \
    d2 = MADD(d2, t1##i, v0, 0);                                               \
    d3 = MSUB(d3, t1##i, v0, 1);                                               \
    d4 = MADD(d4, t1##i, v0, 1);                                               \
    d5 = MSUB(d5, t1##i, v0, 2);                                               \
    d6 = MADD(d6, t1##i, v0, 2);                                               \
    d7 = SUBF(d7, t1##i);                                                      \
    d8 = MADD(d8, t1##i, v0, 0);                                               \
    d0 = MSUB(d0, t2##i, v0, 3);                                               \
    d1 = MSUB(d1, t2##i, v1, 0);                                               \
    d2 = MSUB(d2, t2##i, v1, 1);                                               \
    d3 = MADD(d3, t2##i, v1, 2);                                               \
    d4 = MSUB(d4, t2##i, v1, 3);                                               \
    d5 = MSUB(d5, t2##i, v2, 0);                                               \
    d6 = MSUB(d6, t2##i, v2, 1);                                               \
    d8 = SUBF(d8, t2##i);                                                      \
    d0 = MADD(d0, t3##i, v2, 2);                                               \
    d1 = MADD(d1, t3##i, v2, 3);                                               \
    d2 = MSUB(d2, t3##i, v3, 0);                                               \
    d3 = MADD(d3, t3##i, v2, 0);                                               \
    d4 = MSUB(d4, t3##i, v3, 1);                                               \
    d5 = MADD(d5, t3##i, v3, 2);                                               \
    d6 = MADD(d6, t3##i, v3, 3);                                               \
    d7 = MADD(d7, t3##i, v2, 2);                                               \
    d8 = MSUB(d8, t3##i, v0, 3);                                               \
    d0 = MADD(d0, t4##i, v0, 3);                                               \
    d1 = MADD(d1, t4##i, v4, 0);                                               \
    d2 = MADD(d2, t4##i, v4, 1);                                               \
    d3 = MSUB(d3, t4##i, v4, 2);                                               \
    d4 = MADD(d4, t4##i, v4, 3);                                               \
    d5 = MADD(d5, t4##i, v5, 0);                                               \
    d6 = MADD(d6, t4##i, v5, 1);                                               \
    d8 = MADD(d8, t4##i, v2, 2);                                               \
    d0 = MSUB(d0, t5##i, v2, 2);                                               \
    d1 = MSUB(d1, t5##i, v5, 2);                                               \
    d2 = MSUB(d2, t5##i, v5, 3);                                               \
    d3 = MSUB(d3, t5##i, v6, 0);                                               \
    d4 = MADD(d4, t5##i, v6, 1);                                               \
    d5 = MSUB(d5, t5##i, v5, 2);                                               \
    d6 = MSUB(d6, t5##i, v6, 0);                                               \
    d7 = MSUB(d7, t5##i, v2, 2);                                               \
    d8 = MADD(d8, t5##i, v0, 3);                                               \
    d0 = MSUB(d0, t6##i, v0, 0);                                               \
    d1 = MSUB(d1, t6##i, v1, 0);                                               \
    d2 = MSUB(d2, t6##i, v1, 1);                                               \
    d3 = MADD(d3, t6##i, v1, 0);                                               \
    d4 = MSUB(d4, t6##i, v3, 1);                                               \
    d5 = SUBF(d5, t6##i);                                                      \
    d6 = MSUB(d6, t6##i, v6, 2);                                               \
    d8 = MSUB(d8, t6##i, v2, 2);                                               \
    d0 = MADD(d0, t0##i, v0, 0);                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (0 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d0);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (1 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d1);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (2 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d2);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (3 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d3);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (4 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d4);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (5 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d5);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (6 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d6);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (7 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d7);                                                               \
    GiStoreFloat32(                                                            \
            input_transform_buf +                                              \
                    (8 * alpha + i) * ICB * nr_units_in_tile * pack_size +     \
                    icb * nr_units_in_tile * pack_size + unit_idx * pack_size, \
            d8);

        UNROLL_CALL_RAW(9, cb);
#undef cb
#undef MADD
#undef MSUB
    }
};

template <BiasMode bmode, typename Op>
struct OutputTransformF73_NCHW44 {
    static void transform(
            const float* output_transform_buf, const float* bias, float* output,
            float* transform_mid_buf, size_t oh_start, size_t ow_start, size_t OH,
            size_t OW, size_t oc_start, size_t oc_end, size_t oc_index, size_t unit_idx,
            size_t nr_units_in_tile, const DType& src_dtype, const DType& dst_dtype) {
        MEGDNN_MARK_USED_VAR(transform_mid_buf);
        Op op(src_dtype, dst_dtype);
        //! AT * m * A

        size_t oc = oc_start + oc_index;
        size_t OCB = (oc_end - oc_start) / pack_size;
        size_t ocb = oc_index / pack_size;

#define cb(m, n)                                                   \
    auto v##m##n = GiLoadFloat32(                                  \
            output_transform_buf +                                 \
            (m * alpha + n) * OCB * nr_units_in_tile * pack_size + \
            ocb * nr_units_in_tile * pack_size + unit_idx * pack_size);

        UNROLL_CALL_NOWRAPPER_D2(9, 9, cb);
#undef cb

        /**
         * A
         *
         * 1    0      0       0        0         0          0
         * 1    1      1       1        1         1          1
         * 1   -1      1      -1        1        -1          1
         * 1    2      4       8       16        32         64
         * 1   -2      4      -8       16       -32         64
         * 1  0.5   0.25   0.125   0.0625   0.03125   0.015625
         * 1 -0.5   0.25  -0.125   0.0625  -0.03125   0.015625
         * 1  1.5   2.25   3.375   5.0625   7.59375  11.390625
         * 0    0      0       0        0         0          1
         */
        /*
         * v1addv2 = v1##m + v2##m;
         * v1subv2 = v1##m - v2##m;
         * v3addv4 = v3##m + v4##m;
         * v3subv4 = v3##m - v4##m;
         * v5addv6 = v5##m + v6##m;
         * v5subv6 = v5##m - v6##m;
         * auto t0##m = v0##m + v1addv2 + v3addv4 + v5addv6 + v7##m;
         * auto t1##m = v1subv2 + v3subv4 * 2.f + v5subv6 * 0.5f + v7##m * 1.5f;
         * auto t2##m = v1addv2 + v3addv4 * 4.f + v5addv6 * 0.25f + v7##m * 2.25f;
         * auto t3##m = v1subv2 + v3subv4 * 8.f + v5subv6 * 0.125f + v7##m * 3.375f;
         * auto t4##m = v1addv2 + v3addv4 * 16.f + v5addv6 * 0.0625f + v7##m * 5.0625f;
         * auto t5##m = v1subv2 + v3subv4 * 32.f + v5subv6 * 0.03125f + v7##m
         * * 7.59375f; auto t6##m = v1addv2 + v3addv4 * 64.f + v5addv6 * 0.015625f +
         * v7##m * 11.390625f + v8##m;
         */

        GI_FLOAT32_t v1addv2, v1subv2, v3addv4, v3subv4, v5addv6, v5subv6;
#define cb(m)                                                                          \
    v1addv2 = ADDF(v1##m, v2##m);                                                      \
    v1subv2 = SUBF(v1##m, v2##m);                                                      \
    v3addv4 = ADDF(v3##m, v4##m);                                                      \
    v3subv4 = SUBF(v3##m, v4##m);                                                      \
    v5addv6 = ADDF(v5##m, v6##m);                                                      \
    v5subv6 = SUBF(v5##m, v6##m);                                                      \
    auto t0##m = ADDF(ADDF(ADDF(ADDF(v0##m, v1addv2), v3addv4), v5addv6), v7##m);      \
    auto t1##m =                                                                       \
            ADDF(ADDF(ADDF(v1subv2, MULSF(v3subv4, 2.f)), MULSF(v5subv6, 0.5f)),       \
                 MULSF(v7##m, 1.5f));                                                  \
    auto t2##m =                                                                       \
            ADDF(ADDF(ADDF(v1addv2, MULSF(v3addv4, 4.f)), MULSF(v5addv6, 0.25f)),      \
                 MULSF(v7##m, 2.25f));                                                 \
    auto t3##m =                                                                       \
            ADDF(ADDF(ADDF(v1subv2, MULSF(v3subv4, 8.f)), MULSF(v5subv6, 0.125f)),     \
                 MULSF(v7##m, 3.375f));                                                \
    auto t4##m =                                                                       \
            ADDF(ADDF(ADDF(v1addv2, MULSF(v3addv4, 16.f)), MULSF(v5addv6, 0.0625f)),   \
                 MULSF(v7##m, 5.0625f));                                               \
    auto t5##m =                                                                       \
            ADDF(ADDF(ADDF(v1subv2, MULSF(v3subv4, 32.f)), MULSF(v5subv6, 0.03125f)),  \
                 MULSF(v7##m, 7.59375f));                                              \
    auto t6##m = ADDF(                                                                 \
            ADDF(ADDF(ADDF(v1addv2, MULSF(v3addv4, 64.f)), MULSF(v5addv6, 0.015625f)), \
                 MULSF(v7##m, 11.390625f)),                                            \
            v8##m);

        UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb

        /*
         * v1addv2 = t##m##1 + t##m##2;
         * v1subv2 = t##m##1 - t##m##2;
         * v3addv4 = t##m##3 + t##m##4;
         * v3subv4 = t##m##3 - t##m##4;
         * v5addv6 = t##m##5 + t##m##6;
         * v5subv6 = t##m##5 - t##m##6;
         * v##m##0 = t##m##0 + v1addv2 + v3addv4 + v5addv6 + t##m##7;
         * v##m##1 = v1subv2 + v3subv4 * 2.f + v5subv6 * 0.5f + t##m##7 * 1.5f;
         * v##m##2 = v1addv2 + v3addv4 * 4.f + v5addv6 * 0.25f + t##m##7 * 2.25f;
         * v##m##3 = v1subv2 + v3subv4 * 8.f + v5subv6 * 0.125f + t##m##7 * 3.375;
         * v##m##4 = v1addv2 + v3addv4 * 16.f + v5addv6 * 0.0625f + t##m##7 * 5.0625f;
         * v##m##5 = v1subv2 + v3subv4 * 32.f + v5subv6 * 0.03125f + t##m##7 * 7.59375f;
         * v##m##6 = v1addv2 + v3addv4 * 64.f + v5addv6 * 0.015625f + t##m##7
         * * 11.390625f + t##m##8;
         */
#define cb(m)                                                                          \
    v1addv2 = ADDF(t##m##1, t##m##2);                                                  \
    v1subv2 = SUBF(t##m##1, t##m##2);                                                  \
    v3addv4 = ADDF(t##m##3, t##m##4);                                                  \
    v3subv4 = SUBF(t##m##3, t##m##4);                                                  \
    v5addv6 = ADDF(t##m##5, t##m##6);                                                  \
    v5subv6 = SUBF(t##m##5, t##m##6);                                                  \
    v##m##0 = ADDF(ADDF(ADDF(ADDF(t##m##0, v1addv2), v3addv4), v5addv6), t##m##7);     \
    v##m##1 =                                                                          \
            ADDF(ADDF(ADDF(v1subv2, MULSF(v3subv4, 2.f)), MULSF(v5subv6, 0.5f)),       \
                 MULSF(t##m##7, 1.5f));                                                \
    v##m##2 =                                                                          \
            ADDF(ADDF(ADDF(v1addv2, MULSF(v3addv4, 4.f)), MULSF(v5addv6, 0.25f)),      \
                 MULSF(t##m##7, 2.25f));                                               \
    v##m##3 =                                                                          \
            ADDF(ADDF(ADDF(v1subv2, MULSF(v3subv4, 8.f)), MULSF(v5subv6, 0.125f)),     \
                 MULSF(t##m##7, 3.375));                                               \
    v##m##4 =                                                                          \
            ADDF(ADDF(ADDF(v1addv2, MULSF(v3addv4, 16.f)), MULSF(v5addv6, 0.0625f)),   \
                 MULSF(t##m##7, 5.0625f));                                             \
    v##m##5 =                                                                          \
            ADDF(ADDF(ADDF(v1subv2, MULSF(v3subv4, 32.f)), MULSF(v5subv6, 0.03125f)),  \
                 MULSF(t##m##7, 7.59375f));                                            \
    v##m##6 = ADDF(                                                                    \
            ADDF(ADDF(ADDF(v1addv2, MULSF(v3addv4, 64.f)), MULSF(v5addv6, 0.015625f)), \
                 MULSF(t##m##7, 11.390625f)),                                          \
            t##m##8);

        UNROLL_CALL_NOWRAPPER(7, cb);
#undef cb

        GI_FLOAT32_t vbias;
        if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias = GiLoadFloat32(bias + oc);

#define cb(m, n) v##m##n = ADDF(v##m##n, vbias);
            UNROLL_CALL_RAW_D2(7, 7, cb);
#undef cb
        }
        if (bmode != BiasMode::BIAS) {
#define cb(m, n) v##m##n = op(CONCAT(v##m, n));
            UNROLL_CALL_RAW_D2(7, 7, cb);
#undef cb
        }
#define out_save(oho, owo)                                                           \
    do {                                                                             \
        size_t oh = oh_start + oho;                                                  \
        size_t ow = ow_start + owo;                                                  \
        if (oh < OH && ow < OW) {                                                    \
            if (bmode == BiasMode::BIAS) {                                           \
                v##oho##owo = ADDF(                                                  \
                        v##oho##owo, GiLoadFloat32(                                  \
                                             bias + oc * OH * OW +                   \
                                             oh * OW * pack_size + ow * pack_size)); \
                v##oho##owo = op(v##oho##owo);                                       \
            }                                                                        \
            GiStoreFloat32(                                                          \
                    output + oc * OH * OW + oh * OW * pack_size + ow * pack_size,    \
                    v##oho##owo);                                                    \
        }                                                                            \
    } while (0);
        UNROLL_CALL_RAW_D2(7, 7, out_save);
    }
#undef out_save
};
}  // namespace

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_F73_mk4_f_nchw44)

void winograd_F73_mk4_f_nchw44::filter(
        const float* filter, float* filter_transform_buf, float* transform_mid_buf,
        size_t OC, size_t IC, size_t oc_start, size_t oc_end) {
    constexpr size_t pack_size = 4;
    // Gg * GT
    // G
    // 0.6666667       0.0000000       0.0000000
    // 0.4444444       0.4444444       0.4444444
    // 0.0888889      -0.0888889       0.0888889
    // 0.0222222       0.0444444       0.0888889
    //-0.0031746       0.0063492      -0.0126984
    //-0.7111111      -0.3555556      -0.1777778
    //-0.3555556       0.1777778      -0.0888889
    //-0.1523810      -0.2285714      -0.3428572
    // 0.0000000       0.0000000       1.0000000
    MEGDNN_MARK_USED_VAR(transform_mid_buf);
    megdnn_assert(
            (oc_end - oc_start) % pack_size == 0 && oc_start % pack_size == 0 &&
                    oc_end % pack_size == 0 && IC % pack_size == 0 &&
                    OC % pack_size == 0,
            "NCHW44 Winograd filter transform requires both OC and IC "
            "are times of 4");

    size_t ICB = IC / pack_size;

    for (size_t ocb = oc_start / pack_size; ocb < oc_end / pack_size; ocb++) {
        for (size_t icb = 0; icb < ICB; icb++) {
            for (size_t ic_inner = 0; ic_inner < pack_size; ic_inner++) {
                const float* fptr = filter +
                                    (ocb * ICB + icb) * KERNEL_SIZE * KERNEL_SIZE *
                                            pack_size * pack_size +
                                    ic_inner * pack_size;

#define cb(m, n)           \
    GI_FLOAT32_t g##m##n = \
            GiLoadFloat32(fptr + (m * KERNEL_SIZE + n) * pack_size * pack_size);
                UNROLL_CALL_NOWRAPPER_D2(3, 3, cb)
#undef cb

/*
 * auto wd##n##0 = g##0##n * 0.6666667f;
 * auto wd##n##1 = (g##0##n + g##1##n + g##2##n) * 0.4444444f;
 * auto wd##n##2 = (g##0##n - g##1##n + g##2##n) * 0.0888889f;
 * auto wd##n##3 =
 *         g##0##n * 0.0222222f + g##1##n * 0.0444444f + g##2##n *
 * 0.0888889f; auto wd##n##4 = g##0##n * -0.0031746f + g##1##n *
 * 0.0063492f + g##2##n * -0.0126984f; auto wd##n##5 = g##0##n *
 * -0.7111111f + g##1##n * -0.3555556f + g##2##n * -0.1777778f; auto
 * wd##n##6 = g##0##n * -0.3555556f + g##1##n * 0.1777778f + g##2##n *
 * -0.0888889f; auto wd##n##7 = g##0##n * -0.1523810f + g##1##n *
 * -0.2285714f + g##2##n * -0.3428572f;
 */
#define FILTER_TRANSFORM(n, wd, g)                                               \
    auto wd##n##0 = MULSF(g##0##n, 0.6666667f);                                  \
    auto wd##n##1 = MULSF(ADDF(ADDF(g##0##n, g##1##n), g##2##n), 0.4444444f);    \
    auto wd##n##2 = MULSF(ADDF(SUBF(g##0##n, g##1##n), g##2##n), 0.0888889f);    \
    auto wd##n##3 =                                                              \
            ADDF(ADDF(MULSF(g##0##n, 0.0222222f), MULSF(g##1##n, 0.0444444f)),   \
                 MULSF(g##2##n, 0.0888889f));                                    \
    auto wd##n##4 =                                                              \
            ADDF(ADDF(MULSF(g##0##n, -0.0031746f), MULSF(g##1##n, 0.0063492f)),  \
                 MULSF(g##2##n, -0.0126984f));                                   \
    auto wd##n##5 =                                                              \
            ADDF(ADDF(MULSF(g##0##n, -0.7111111f), MULSF(g##1##n, -0.3555556f)), \
                 MULSF(g##2##n, -0.1777778f));                                   \
    auto wd##n##6 =                                                              \
            ADDF(ADDF(MULSF(g##0##n, -0.3555556f), MULSF(g##1##n, 0.1777778f)),  \
                 MULSF(g##2##n, -0.0888889f));                                   \
    auto wd##n##7 =                                                              \
            ADDF(ADDF(MULSF(g##0##n, -0.1523810f), MULSF(g##1##n, -0.2285714f)), \
                 MULSF(g##2##n, -0.3428572f));                                   \
    auto wd##n##8 = g##2##n;
                UNROLL_CALL_RAW(3, FILTER_TRANSFORM, wd, g);
                UNROLL_CALL_RAW(9, FILTER_TRANSFORM, ret, wd);
#undef FILTER_TRANSFORM
#define cb_save(m, n)                                                                 \
    GiStoreFloat32(                                                                   \
            filter_transform_buf + (m * alpha + n) * OC * IC + ocb * IC * pack_size + \
                    icb * pack_size * pack_size + ic_inner * pack_size,               \
            ret##m##n);
                UNROLL_CALL_NOWRAPPER_D2(9, 9, cb_save)
#undef cb_save
            }
        }
    }
}

void winograd_F73_mk4_f_nchw44::input(
        const float* input, float* input_transform_buf, float* transform_mid_buf,
        size_t IH, size_t IW, size_t IC, size_t PH, size_t PW, size_t unit_start_idx,
        size_t nr_units_in_tile) {
    constexpr size_t pack_size = 4;
    megdnn_assert(IC % pack_size == 0);
    constexpr int alpha = 3 + 7 - 1;

    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    float* patch = transform_mid_buf;
    float* patchT = transform_mid_buf + pack_size * alpha * alpha;

    for (size_t ic = 0; ic < IC; ic += pack_size) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransformF73_NCHW44::prepare<true>(
                        input, patch, patchT, ih_start, iw_start, IH, IW, ic, IC);
                InputTransformF73_NCHW44::transform(
                        patchT, input_transform_buf, unit_idx, nr_units_in_tile, ic,
                        IC);

            } else {
                InputTransformF73_NCHW44::prepare<false>(
                        input, patch, patchT, ih_start, iw_start, IH, IW, ic, IC);
                InputTransformF73_NCHW44::transform(
                        patchT, input_transform_buf, unit_idx, nr_units_in_tile, ic,
                        IC);
            }
        }
    }
}

void winograd_F73_mk4_f_nchw44::output(
        const float* output_transform_buf, const float* bias, float* output,
        float* transform_mid_buf, BiasMode bmode, NonlineMode nonline_mode, size_t OH,
        size_t OW, size_t oc_start, size_t oc_end, size_t unit_start_idx,
        size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...)                                                 \
    for (size_t oc = oc_start; oc < oc_end; oc += pack_size) {                       \
        size_t oc_index = oc - oc_start;                                             \
        rep(unit_idx, nr_units_in_tile) {                                            \
            size_t index = unit_start_idx + unit_idx;                                \
            auto nh = index / units_w;                                               \
            auto nw = index % units_w;                                               \
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;                                \
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;                                \
            OutputTransformF73_NCHW44<_bmode MEGDNN_COMMA _nonline_op>::transform(   \
                    output_transform_buf, bias, output, transform_mid_buf, oh_start, \
                    ow_start, OH, OW, oc_start, oc_end, oc_index, unit_idx,          \
                    nr_units_in_tile, src_dtype, dst_dtype);                         \
        }                                                                            \
    }

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    constexpr size_t pack_size = 4;

    size_t OC = oc_end - oc_start;
    megdnn_assert(
            OC % pack_size == 0 && oc_start % pack_size == 0 && oc_end % pack_size == 0,
            "NCHW44 Winograd filter transform requires OC is times of 4");

    GI_DISPATCH_CONV_WINOGRAD_BIAS(
            megdnn_fallback_winograd_fp32_F73_mk4, cb, float, float, bmode,
            nonline_mode);
#undef cb
}

}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
