/**
 * \file dnn/src/x86/pooling/mean_pooling_w2x2_s2x2_avx.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./pooling_special_cases.h"

#include <immintrin.h>
#include <string.h>

#include "../avx_helper.h"

namespace megdnn {
namespace x86 {

void mean_pooling_w2x2_s2x2_avx(const float *src, const int src_h, const int src_w,
        float *dst, const int dst_h, const int dst_w,
        const int pad_h, const int pad_w,
        bool is_include)
{
    (void)dst_h;
    // calculate boundaries
    const int dst_h_beg = (pad_h + 1) / 2; // x >= pad / stride
    const int dst_h_end = (src_h + pad_h) / 2; // x < (n + pad) / stride
    const int dst_w_beg = (pad_w + 1) / 2;
    const int dst_w_end = (src_w + pad_w) / 2;
    const float coef = 0.25;
    {
        // brute-force with padding
        int idst_h, idst_w;
        size_t count;
#define CALCULATE1 \
        const int isrc_h = -pad_h + 2*idst_h; \
        const float *src_d = src + isrc_h * src_w; \
        float *dst_d = dst + idst_h * dst_w;
#define CALCULATE2 \
        const int isrc_w = -pad_w + 2*idst_w; \
        const float *src_dd = src_d + isrc_w; \
        float *dst_dd = dst_d + idst_w; \
        *dst_dd = 0; \
        count = 0; \
        if (isrc_h >= 0 && isrc_h < src_h && isrc_w >= 0 && isrc_w < src_w) { \
            *dst_dd += *src_dd; \
            ++count; \
        } \
        if (isrc_h >= 0 && isrc_h < src_h && isrc_w+1 >= 0 && isrc_w+1 < src_w) { \
            *dst_dd += *(src_dd+1); \
            ++count; \
        } \
        if (isrc_h+1 >= 0 && isrc_h+1 < src_h && isrc_w >= 0 && isrc_w < src_w) { \
            *dst_dd += *(src_dd+src_w); \
            ++count; \
        } \
        if (isrc_h+1 >= 0 && isrc_h+1 < src_h && isrc_w+1 >= 0 && isrc_w+1 < src_w) { \
            *dst_dd += *(src_dd+src_w+1); \
            ++count; \
        } \
        if (is_include) { \
            *dst_dd *= coef; \
        } else { \
            *dst_dd /= static_cast<float>(count); \
        }

        for (idst_h = 0; idst_h < dst_h_beg; ++idst_h) {
            CALCULATE1
            for (idst_w = 0; idst_w < dst_w; ++idst_w) {
                CALCULATE2
            }
        }

        for (idst_h = dst_h_end; idst_h < dst_h; ++idst_h) {
            CALCULATE1
            for (idst_w = 0; idst_w < dst_w; ++idst_w) {
                CALCULATE2
            }
        }

        for (idst_h = dst_h_beg; idst_h < dst_h_end; ++idst_h) {
            CALCULATE1
            for (idst_w = 0; idst_w < dst_w_beg; ++idst_w) {
                CALCULATE2
            }
        }

        for (idst_h = dst_h_beg; idst_h < dst_h_end; ++idst_h) {
            CALCULATE1
            for (idst_w = dst_w_end; idst_w < dst_w; ++idst_w) {
                CALCULATE2
            }
        }
#undef CALCULATE1
#undef CALCULATE2
    }
    int idst_h;
    for (idst_h = dst_h_beg; idst_h + 4 <= dst_h_end; idst_h += 4) {
        const int isrc_h = -pad_h + 2 * idst_h;
        const float *src_d = src + isrc_h * src_w;
        float *dst_d = dst + idst_h * dst_w;
        int idst_w;
        for (idst_w = dst_w_beg; idst_w + 8 <= dst_w_end; idst_w += 8) {
            const int isrc_w = -pad_w + 2 * idst_w;
            const float *src_dd = src_d + isrc_w;
            float *dst_dd = dst_d + idst_w;

            __m256 va0, vb0, vc0, vd0,
                   va1, vb1, vc1, vd1,
                   va2, vb2, vc2, vd2,
                   va3, vb3, vc3, vd3;

            va0 = _mm256_loadu2_m128_emulate(src_dd + 0*src_w + 8, src_dd + 0*src_w);
            vb0 = _mm256_loadu2_m128_emulate(src_dd + 0*src_w + 12, src_dd + 0*src_w + 4);
            vc0 = _mm256_loadu2_m128_emulate(src_dd + 1*src_w + 8, src_dd + 1*src_w);
            vd0 = _mm256_loadu2_m128_emulate(src_dd + 1*src_w + 12, src_dd + 1*src_w + 4);
            va1 = _mm256_loadu2_m128_emulate(src_dd + 2*src_w + 8, src_dd + 2*src_w);
            vb1 = _mm256_loadu2_m128_emulate(src_dd + 2*src_w + 12, src_dd + 2*src_w + 4);
            vc1 = _mm256_loadu2_m128_emulate(src_dd + 3*src_w + 8, src_dd + 3*src_w);
            vd1 = _mm256_loadu2_m128_emulate(src_dd + 3*src_w + 12, src_dd + 3*src_w + 4);
            va2 = _mm256_loadu2_m128_emulate(src_dd + 4*src_w + 8, src_dd + 4*src_w);
            vb2 = _mm256_loadu2_m128_emulate(src_dd + 4*src_w + 12, src_dd + 4*src_w + 4);
            vc2 = _mm256_loadu2_m128_emulate(src_dd + 5*src_w + 8, src_dd + 5*src_w);
            vd2 = _mm256_loadu2_m128_emulate(src_dd + 5*src_w + 12, src_dd + 5*src_w + 4);
            va3 = _mm256_loadu2_m128_emulate(src_dd + 6*src_w + 8, src_dd + 6*src_w);
            vb3 = _mm256_loadu2_m128_emulate(src_dd + 6*src_w + 12, src_dd + 6*src_w + 4);
            vc3 = _mm256_loadu2_m128_emulate(src_dd + 7*src_w + 8, src_dd + 7*src_w);
            vd3 = _mm256_loadu2_m128_emulate(src_dd + 7*src_w + 12, src_dd + 7*src_w + 4);

            va0 = _mm256_add_ps(va0, vc0);
            vb0 = _mm256_add_ps(vb0, vd0);
            va1 = _mm256_add_ps(va1, vc1);
            vb1 = _mm256_add_ps(vb1, vd1);
            va2 = _mm256_add_ps(va2, vc2);
            vb2 = _mm256_add_ps(vb2, vd2);
            va3 = _mm256_add_ps(va3, vc3);
            vb3 = _mm256_add_ps(vb3, vd3);

            // use vc0 as temp storage
            vc0 = _mm256_broadcast_ss(&coef);

            va0 = _mm256_hadd_ps(va0, vb0);
            va1 = _mm256_hadd_ps(va1, vb1);
            va2 = _mm256_hadd_ps(va2, vb2);
            va3 = _mm256_hadd_ps(va3, vb3);

            va0 = _mm256_mul_ps(va0, vc0);
            va1 = _mm256_mul_ps(va1, vc0);
            va2 = _mm256_mul_ps(va2, vc0);
            va3 = _mm256_mul_ps(va3, vc0);

            _mm256_storeu_ps(dst_dd + 0*dst_w, va0);
            _mm256_storeu_ps(dst_dd + 1*dst_w, va1);
            _mm256_storeu_ps(dst_dd + 2*dst_w, va2);
            _mm256_storeu_ps(dst_dd + 3*dst_w, va3);
        }
        const int rem = dst_w_end - idst_w;
        int h;
        for (h = 0; h < 4; ++h) {
            float ans[8] = {0};
            int i;
            for (i = 0; i < rem; ++i) {
                ans[i] += src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 0];
                ans[i] += src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 1];
            }
            for (i = 0; i < rem; ++i) {
                ans[i] += src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 0];
                ans[i] += src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 1];
            }
            for (i = 0; i < rem; ++i) {
                ans[i] *= coef;
            }
            memcpy(dst_d + h*dst_w + idst_w, ans, sizeof(float) * rem);
        }
    }
    if (idst_h + 2 <= dst_h_end) {
        const int isrc_h = -pad_h + 2 * idst_h;
        const float *src_d = src + isrc_h * src_w;
        float *dst_d = dst + idst_h * dst_w;
        int idst_w;
        for (idst_w = dst_w_beg; idst_w + 8 <= dst_w_end; idst_w += 8) {
            const int isrc_w = -pad_w + 2 * idst_w;
            const float *src_dd = src_d + isrc_w;
            float *dst_dd = dst_d + idst_w;

            __m256 va0, vb0, vc0, vd0,
                   va1, vb1, vc1, vd1;

            va0 = _mm256_loadu2_m128_emulate(src_dd + 0*src_w + 8, src_dd + 0*src_w);
            vb0 = _mm256_loadu2_m128_emulate(src_dd + 0*src_w + 12, src_dd + 0*src_w + 4);
            vc0 = _mm256_loadu2_m128_emulate(src_dd + 1*src_w + 8, src_dd + 1*src_w);
            vd0 = _mm256_loadu2_m128_emulate(src_dd + 1*src_w + 12, src_dd + 1*src_w + 4);
            va1 = _mm256_loadu2_m128_emulate(src_dd + 2*src_w + 8, src_dd + 2*src_w);
            vb1 = _mm256_loadu2_m128_emulate(src_dd + 2*src_w + 12, src_dd + 2*src_w + 4);
            vc1 = _mm256_loadu2_m128_emulate(src_dd + 3*src_w + 8, src_dd + 3*src_w);
            vd1 = _mm256_loadu2_m128_emulate(src_dd + 3*src_w + 12, src_dd + 3*src_w + 4);

            va0 = _mm256_add_ps(va0, vc0);
            vb0 = _mm256_add_ps(vb0, vd0);
            va1 = _mm256_add_ps(va1, vc1);
            vb1 = _mm256_add_ps(vb1, vd1);

            // use vc0 as temp storage
            vc0 = _mm256_broadcast_ss(&coef);

            va0 = _mm256_hadd_ps(va0, vb0);
            va1 = _mm256_hadd_ps(va1, vb1);

            va0 = _mm256_mul_ps(va0, vc0);
            va1 = _mm256_mul_ps(va1, vc0);

            _mm256_storeu_ps(dst_dd + 0*dst_w, va0);
            _mm256_storeu_ps(dst_dd + 1*dst_w, va1);
        }
        const int rem = dst_w_end - idst_w;
        int h;
        for (h = 0; h < 2; ++h) {
            float ans[8] = {0};
            int i;
            for (i = 0; i < rem; ++i) {
                ans[i] += src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 0];
                ans[i] += src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 1];
            }
            for (i = 0; i < rem; ++i) {
                ans[i] += src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 0];
                ans[i] += src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 1];
            }
            for (i = 0; i < rem; ++i) {
                ans[i] *= coef;
            }
            memcpy(dst_d + h*dst_w + idst_w, ans, sizeof(float) * rem);
        }
        idst_h += 2;
    }
    if (idst_h + 1 <= dst_h_end) {
        const int isrc_h = -pad_h + 2 * idst_h;
        const float *src_d = src + isrc_h * src_w;
        float *dst_d = dst + idst_h * dst_w;
        int idst_w;
        for (idst_w = dst_w_beg; idst_w + 8 <= dst_w_end; idst_w += 8) {
            const int isrc_w = -pad_w + 2 * idst_w;
            const float *src_dd = src_d + isrc_w;
            float *dst_dd = dst_d + idst_w;

            __m256 va0, vb0, vc0, vd0;

            va0 = _mm256_loadu2_m128_emulate(src_dd + 0*src_w + 8, src_dd + 0*src_w);
            vb0 = _mm256_loadu2_m128_emulate(src_dd + 0*src_w + 12, src_dd + 0*src_w + 4);
            vc0 = _mm256_loadu2_m128_emulate(src_dd + 1*src_w + 8, src_dd + 1*src_w);
            vd0 = _mm256_loadu2_m128_emulate(src_dd + 1*src_w + 12, src_dd + 1*src_w + 4);

            va0 = _mm256_add_ps(va0, vc0);
            vb0 = _mm256_add_ps(vb0, vd0);

            // use vc0 as temp storage
            vc0 = _mm256_broadcast_ss(&coef);

            va0 = _mm256_hadd_ps(va0, vb0);

            va0 = _mm256_mul_ps(va0, vc0);

            _mm256_storeu_ps(dst_dd + 0*dst_w, va0);
        }
        const int rem = dst_w_end - idst_w;
        int h;
        for (h = 0; h < 1; ++h) {
            float ans[8] = {0};
            int i;
            for (i = 0; i < rem; ++i) {
                ans[i] += src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 0];
                ans[i] += src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 1];
            }
            for (i = 0; i < rem; ++i) {
                ans[i] += src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 0];
                ans[i] += src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 1];
            }
            for (i = 0; i < rem; ++i) {
                ans[i] *= coef;
            }
            memcpy(dst_d + h*dst_w + idst_w, ans, sizeof(float) * rem);
        }
        idst_h += 1;
    }
}

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
