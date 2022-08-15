#pragma once
#include "megdnn/opr_param_defs.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/gi/fp32/helper.h"

namespace megdnn {
namespace fallback {

/*
 * wd##0 = d##0;
 * tmp0 = (d##0 + d##2) * -0.2222222f
 * tmp1 = d##1 * -0.2222222
 * wd##1 = tmp0 + tmp1
 * wd##2 = tmp0 - tmp1
 * tmp0 = d##0 * 0.0111111f + d##2 * 0.0444444f
 * tmp1 = d##1 * 0.0222222f
 * wd##3 = tmp0 + tmp1
 * wd##4 = tmp0 - tmp1
 * tmp0 = d##0 * 0.7111111f + d##2 * 0.1777778f
 * tmp1 = d##1 * 0.3555556f
 * wd##5 = tmp0 + tmp1
 * wd##6 = tmp0 - tmp1
 * wd##7 = d##2
 */
template <param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT>
struct FilterTransform6X3 {
#define FILTER_TRANSFORM(d, wd, ADDC, SUBC, MULC)                    \
    do {                                                             \
        wd##0 = d##0;                                                \
        auto tmp0 = MULC(ADDC(d##0, d##2), -0.2222222f);             \
        auto tmp1 = MULC(d##1, -0.2222222f);                         \
        wd##1 = ADDC(tmp0, tmp1);                                    \
        wd##2 = SUBC(tmp0, tmp1);                                    \
        tmp0 = ADDC(MULC(d##0, 0.0111111f), MULC(d##2, 0.0444444f)); \
        tmp1 = MULC(d##1, 0.0222222f);                               \
        wd##3 = ADDC(tmp0, tmp1);                                    \
        wd##4 = SUBC(tmp0, tmp1);                                    \
        tmp0 = ADDC(MULC(d##0, 0.7111111f), MULC(d##2, 0.1777778f)); \
        tmp1 = MULC(d##1, 0.3555556f);                               \
        wd##5 = ADDC(tmp0, tmp1);                                    \
        wd##6 = SUBC(tmp0, tmp1);                                    \
        wd##7 = d##2;                                                \
    } while (0);

    static void transform(
            const float* filter, float* filter_transform_buf, float* transform_mid_buf,
            size_t OC, size_t IC, size_t oc_start, size_t oc_end) {
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

        constexpr size_t alpha = 6 + 3 - 1;
        size_t OCB = OC / 4;
        size_t ICB = IC / 4;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                const float* fptr = filter + (oc * IC + ic) * 3 * 3;

                GI_FLOAT32_t g0 = GiLoadFloat32(fptr);
                GI_FLOAT32_t g1 = GiLoadFloat32(fptr + 3);

                GI_FLOAT32_t g2 = GiLoadFloat32(fptr + 6 - 1);
                GI_FLOAT32_t zeros = GiZeroFloat32();
                g2 = GiExtqFloat32(g2, zeros, 1);

#define cb(i) GI_FLOAT32_t wd##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

                FILTER_TRANSFORM(g, wd, ADDF, SUBF, MULSF);

                size_t ocb = oc / 4;
                size_t oc4 = oc % 4;
                size_t icb = ic / 4;
                size_t ic4 = ic % 4;
#if MEGDNN_AARCH64
#define cb(i) GI_FLOAT32_V2_t wdt##i;
                UNROLL_CALL_NOWRAPPER(3, cb);
#undef cb

#define cb(i) GI_FLOAT32_V2_t ret##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                TRANSPOSE_8x3(wd, wdt);
                FILTER_TRANSFORM(wdt, ret, ADDFV2, SUBFV2, MULSFV2);

#define cb(i) GiStoreFloat32V2(transform_mid_buf + i * alpha, ret##i);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                rep(i, alpha) rep(j, alpha) {
                    if (format == param::MatrixMul::Format::DEFAULT) {
                        filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC + oc] =
                                transform_mid_buf[j * alpha + i];
                    } else {
                        filter_transform_buf
                                [(i * alpha + j) * OCB * ICB * 4 * 4 +
                                 ocb * ICB * 4 * 4 + icb * 4 * 4 + ic4 * 4 + oc4] =
                                        transform_mid_buf[j * alpha + i];
                    }
                }

#else
#define cb(i)                                                                          \
    do {                                                                               \
        mid_buf1[0] = GET_VECTOR_ELEM(wd, i, 0);                                       \
        auto tmp0 =                                                                    \
                (GET_VECTOR_ELEM(wd, i, 0) + GET_VECTOR_ELEM(wd, i, 2)) * -0.2222222f; \
        auto tmp1 = GET_VECTOR_ELEM(wd, i, 1) * -0.2222222f;                           \
        mid_buf1[1] = tmp0 + tmp1;                                                     \
        mid_buf1[2] = tmp0 - tmp1;                                                     \
        tmp0 = GET_VECTOR_ELEM(wd, i, 0) * 0.0111111f +                                \
               GET_VECTOR_ELEM(wd, i, 2) * 0.0444444f;                                 \
        tmp1 = GET_VECTOR_ELEM(wd, i, 1) * 0.0222222f;                                 \
        mid_buf1[3] = tmp0 + tmp1;                                                     \
        mid_buf1[4] = tmp0 - tmp1;                                                     \
        tmp0 = GET_VECTOR_ELEM(wd, i, 0) * 0.7111111f +                                \
               GET_VECTOR_ELEM(wd, i, 2) * 0.1777778f;                                 \
        tmp1 = GET_VECTOR_ELEM(wd, i, 1) * 0.3555556f;                                 \
        mid_buf1[5] = tmp0 + tmp1;                                                     \
        mid_buf1[6] = tmp0 - tmp1;                                                     \
        mid_buf1[7] = GET_VECTOR_ELEM(wd, i, 2);                                       \
        mid_buf1 += 8;                                                                 \
    } while (0);
#define GET_VECTOR_ELEM(s, i, idx) GiExtractLane##idx##Float32(CONCAT(s, i))

                float* mid_buf1 = transform_mid_buf;
                UNROLL_CALL_NOWRAPPER(8, cb);
                mid_buf1 = transform_mid_buf;
#undef cb

                rep(i, alpha) rep(j, alpha) {
                    if (format == param::MatrixMul::Format::DEFAULT) {
                        filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC + oc] =
                                transform_mid_buf[i * alpha + j];
                    } else {
                        filter_transform_buf
                                [(i * alpha + j) * OCB * ICB * 4 * 4 +
                                 ocb * ICB * 4 * 4 + icb * 4 * 4 + ic4 * 4 + oc4] =
                                        transform_mid_buf[i * alpha + j];
                    }
                }
#endif
            }
        }
    }
};
#undef FILTER_TRANSFORM
#undef GET_VECTOR_ELEM

template <param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT>
struct FilterTransform4X3 {
#define FILTER_TRANSFORM(d, wd, ADDC, SUBC, MULC)                    \
    do {                                                             \
        wd##0 = MULC(d##0, 0.25f);                                   \
        auto tmp0 = MULC(ADDC(d##0, d##2), -0.1666667f);             \
        auto tmp1 = MULC(d##1, -0.1666667f);                         \
        wd##1 = ADDC(tmp0, tmp1);                                    \
        wd##2 = SUBC(tmp0, tmp1);                                    \
        tmp0 = ADDC(MULC(d##0, 0.0416667f), MULC(d##2, 0.1666667f)); \
        tmp1 = MULC(d##1, 0.0833333f);                               \
        wd##3 = ADDC(tmp0, tmp1);                                    \
        wd##4 = SUBC(tmp0, tmp1);                                    \
        wd##5 = d##2;                                                \
    } while (0);

    static void transform(
            const float* filter, float* filter_transform_buf, float* transform_mid_buf,
            size_t OC, size_t IC, size_t oc_start, size_t oc_end) {
        constexpr size_t alpha = 4 + 3 - 1;
        size_t OCB = OC / 4;
        size_t ICB = IC / 4;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                const float* fptr = filter + (oc * IC + ic) * 3 * 3;

                GI_FLOAT32_t g0 = GiLoadFloat32(fptr);
                GI_FLOAT32_t g1 = GiLoadFloat32(fptr + 3);

                GI_FLOAT32_t g2 = GiLoadFloat32(fptr + 6 - 1);
                GI_FLOAT32_t zeros = GiZeroFloat32();
                g2 = GiExtqFloat32(g2, zeros, 1);

#define cb(i) GI_FLOAT32_t wd##i = GiZeroFloat32();
#if MEGDNN_AARCH64
                UNROLL_CALL_NOWRAPPER(8, cb);
#else
                UNROLL_CALL_NOWRAPPER(6, cb);
#endif
#undef cb

                FILTER_TRANSFORM(g, wd, ADDF, SUBF, MULSF);

                size_t ocb = oc / 4;
                size_t oc4 = oc % 4;
                size_t icb = ic / 4;
                size_t ic4 = ic % 4;
#if MEGDNN_AARCH64

#define cb(i) GI_FLOAT32_V2_t wdt##i;
                UNROLL_CALL_NOWRAPPER(3, cb);
#undef cb

#define cb(i) GI_FLOAT32_V2_t ret##i;
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

                TRANSPOSE_8x3(wd, wdt);
                FILTER_TRANSFORM(wdt, ret, ADDFV2, SUBFV2, MULSFV2);

#define cb(i) GiStoreFloat32V2(transform_mid_buf + i * alpha, ret##i);
                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
                rep(i, alpha) rep(j, alpha) {
                    if (format == param::MatrixMul::Format::DEFAULT) {
                        filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC + oc] =
                                transform_mid_buf[j * alpha + i];
                    } else {
                        filter_transform_buf
                                [(i * alpha + j) * OCB * ICB * 4 * 4 +
                                 ocb * ICB * 4 * 4 + icb * 4 * 4 + ic4 * 4 + oc4] =
                                        transform_mid_buf[j * alpha + i];
                    }
                }

#else
#define cb(i)                                                                          \
    do {                                                                               \
        mid_buf1[0] = GET_VECTOR_ELEM(wd, i, 0) * 0.25f;                               \
        auto tmp0 =                                                                    \
                (GET_VECTOR_ELEM(wd, i, 0) + GET_VECTOR_ELEM(wd, i, 2)) * -0.1666667f; \
        auto tmp1 = GET_VECTOR_ELEM(wd, i, 1) * -0.1666667f;                           \
        mid_buf1[1] = tmp0 + tmp1;                                                     \
        mid_buf1[2] = tmp0 - tmp1;                                                     \
        tmp0 = GET_VECTOR_ELEM(wd, i, 0) * 0.0416667f +                                \
               GET_VECTOR_ELEM(wd, i, 2) * 0.1666667f;                                 \
        tmp1 = GET_VECTOR_ELEM(wd, i, 1) * 0.0833333f;                                 \
        mid_buf1[3] = tmp0 + tmp1;                                                     \
        mid_buf1[4] = tmp0 - tmp1;                                                     \
        mid_buf1[5] = GET_VECTOR_ELEM(wd, i, 2);                                       \
        mid_buf1 += 6;                                                                 \
    } while (0);
#define GET_VECTOR_ELEM(s, i, idx) GiExtractLane##idx##Float32(CONCAT(s, i))

                float* mid_buf1 = transform_mid_buf;
                UNROLL_CALL_NOWRAPPER(6, cb);
                mid_buf1 = transform_mid_buf;
#undef cb

                rep(i, alpha) rep(j, alpha) {
                    if (format == param::MatrixMul::Format::DEFAULT) {
                        filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC + oc] =
                                transform_mid_buf[i * alpha + j];
                    } else {
                        filter_transform_buf
                                [(i * alpha + j) * OCB * ICB * 4 * 4 +
                                 ocb * ICB * 4 * 4 + icb * 4 * 4 + ic4 * 4 + oc4] =
                                        transform_mid_buf[i * alpha + j];
                    }
                }
#endif
            }
        }
    }
};
#undef FILTER_TRANSFORM
#undef GET_VECTOR_ELEM

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
