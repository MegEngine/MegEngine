/**
 * \file dnn/src/arm_common/conv_bias/fp32/filter_transform.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/arm_common/conv_bias/fp32/helper.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace arm_common {

template <param::MatrixMul::Format format=param::MatrixMul::Format::DEFAULT>
struct FilterTransform6X3 {
#define FILTER_TRANSFORM(d, wd)                       \
    do {                                              \
        wd##0 = d##0;                                 \
        auto tmp0 = (d##0 + d##2) * -0.2222222f;      \
        auto tmp1 = d##1 * -0.2222222f;               \
        wd##1 = tmp0 + tmp1;                          \
        wd##2 = tmp0 - tmp1;                          \
        tmp0 = d##0 * 0.0111111f + d##2 * 0.0444444f; \
        tmp1 = d##1 * 0.0222222f;                     \
        wd##3 = tmp0 + tmp1;                          \
        wd##4 = tmp0 - tmp1;                          \
        tmp0 = d##0 * 0.7111111f + d##2 * 0.1777778f; \
        tmp1 = d##1 * 0.3555556f;                     \
        wd##5 = tmp0 + tmp1;                          \
        wd##6 = tmp0 - tmp1;                          \
        wd##7 = d##2;                                 \
    } while (0);

    static void transform(const float* filter, float* filter_transform_buf,
                          float* transform_mid_buf, size_t OC, size_t IC,
                          size_t oc_start, size_t oc_end) {
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

                Vector<float, 4> g0 = Vector<float, 4>::load(fptr);
                Vector<float, 4> g1 = Vector<float, 4>::load(fptr + 3);

                Vector<float, 4> g2 = Vector<float, 4>::load(fptr + 6 - 1);
                float32x4_t zeros = vdupq_n_f32(0.0f);
                g2.value = vextq_f32(g2.value, zeros, 1);

#define cb(i) Vector<float, 4> wd##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) Vector<float, 8> wdt##i;
                UNROLL_CALL_NOWRAPPER(3, cb);
#undef cb

#define cb(i) Vector<float, 8> ret##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

                FILTER_TRANSFORM(g, wd);

                size_t ocb = oc / 4;
                size_t oc4 = oc % 4;
                size_t icb = ic / 4;
                size_t ic4 = ic % 4;
#if MEGDNN_AARCH64
                TRANSPOSE_8x3(wd, wdt);
                FILTER_TRANSFORM(wdt, ret);

#define cb(i) ret##i.save(transform_mid_buf + i * alpha);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                rep(i, alpha) rep(j, alpha) {
                    if (format == param::MatrixMul::Format::DEFAULT) {
                        filter_transform_buf[(i * alpha + j) * OC * IC +
                                             ic * OC + oc] =
                                transform_mid_buf[j * alpha + i];
                    } else {
                        filter_transform_buf[(i * alpha + j) * OCB * ICB * 4 *
                                                     4 +
                                             ocb * ICB * 4 * 4 + icb * 4 * 4 +
                                             ic4 * 4 + oc4] =
                                transform_mid_buf[j * alpha + i];
                    }
                }

#else

#define cb(i)                                                                 \
    do {                                                                      \
        mid_buf1[0] = GET_VECTOR_ELEM(wd, i, 0);                              \
        auto tmp0 = (GET_VECTOR_ELEM(wd, i, 0) + GET_VECTOR_ELEM(wd, i, 2)) * \
                    -0.2222222f;                                              \
        auto tmp1 = GET_VECTOR_ELEM(wd, i, 1) * -0.2222222f;                  \
        mid_buf1[1] = tmp0 + tmp1;                                            \
        mid_buf1[2] = tmp0 - tmp1;                                            \
        tmp0 = GET_VECTOR_ELEM(wd, i, 0) * 0.0111111f +                       \
               GET_VECTOR_ELEM(wd, i, 2) * 0.0444444f;                        \
        tmp1 = GET_VECTOR_ELEM(wd, i, 1) * 0.0222222f;                        \
        mid_buf1[3] = tmp0 + tmp1;                                            \
        mid_buf1[4] = tmp0 - tmp1;                                            \
        tmp0 = GET_VECTOR_ELEM(wd, i, 0) * 0.7111111f +                       \
               GET_VECTOR_ELEM(wd, i, 2) * 0.1777778f;                        \
        tmp1 = GET_VECTOR_ELEM(wd, i, 1) * 0.3555556f;                        \
        mid_buf1[5] = tmp0 + tmp1;                                            \
        mid_buf1[6] = tmp0 - tmp1;                                            \
        mid_buf1[7] = GET_VECTOR_ELEM(wd, i, 2);                              \
        mid_buf1 += 8;                                                        \
    } while (0);
#define GET_VECTOR_ELEM(s, i, idx) vgetq_lane_f32(CONCAT(s, i).value, idx)

                float* mid_buf1 = transform_mid_buf;
                UNROLL_CALL_NOWRAPPER(8, cb);
                mid_buf1 = transform_mid_buf;
#undef cb

                rep(i, alpha) rep(j, alpha) {
                    if (format == param::MatrixMul::Format::DEFAULT) {
                        filter_transform_buf[(i * alpha + j) * OC * IC +
                                             ic * OC + oc] =
                                transform_mid_buf[i * alpha + j];
                    } else {
                        filter_transform_buf[(i * alpha + j) * OCB * ICB * 4 *
                                                     4 +
                                             ocb * ICB * 4 * 4 + icb * 4 * 4 +
                                             ic4 * 4 + oc4] =
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

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
