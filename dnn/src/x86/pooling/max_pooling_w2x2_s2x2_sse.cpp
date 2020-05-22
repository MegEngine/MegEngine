/**
 * \file dnn/src/x86/pooling/max_pooling_w2x2_s2x2_sse.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./pooling_special_cases.h"

#include <xmmintrin.h>
#include <string.h>
#include <float.h>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

namespace megdnn {
namespace x86 {

void max_pooling_w2x2_s2x2_sse(const float *src, const int src_h, const int src_w,
        float *dst, const int dst_h, const int dst_w,
        const int pad_h, const int pad_w)
{
    (void)dst_h;
    // calculate boundaries
    const int dst_h_beg = (pad_h + 1) / 2; // x >= pad / stride
    const int dst_h_end = (src_h + pad_h) / 2; // x < (n + pad) / stride
    const int dst_w_beg = (pad_w + 1) / 2;
    const int dst_w_end = (src_w + pad_w) / 2;
    // 0202
#define POOLING_IMM0 0x88u
    // 1313
#define POOLING_IMM1 0xddu
    {
        // brute-force with padding
        int idst_h, idst_w;
#define CALCULATE1 \
        const int isrc_h = -pad_h + 2*idst_h; \
        const float *src_d = src + isrc_h * src_w; \
        float *dst_d = dst + idst_h * dst_w;
#define CALCULATE2 \
        const int isrc_w = -pad_w + 2*idst_w; \
        const float *src_dd = src_d + isrc_w; \
        float *dst_dd = dst_d + idst_w; \
        *dst_dd = -FLT_MAX; \
        if (isrc_h >= 0 && isrc_h < src_h && isrc_w >= 0 && isrc_w < src_w) { \
            *dst_dd = MAX(*dst_dd, *src_dd); \
        } \
        if (isrc_h >= 0 && isrc_h < src_h && isrc_w+1 >= 0 && isrc_w+1 < src_w) { \
            *dst_dd = MAX(*dst_dd, *(src_dd+1)); \
        } \
        if (isrc_h+1 >= 0 && isrc_h+1 < src_h && isrc_w >= 0 && isrc_w < src_w) { \
            *dst_dd = MAX(*dst_dd, *(src_dd+src_w)); \
        } \
        if (isrc_h+1 >= 0 && isrc_h+1 < src_h && isrc_w+1 >= 0 && isrc_w+1 < src_w) { \
            *dst_dd = MAX(*dst_dd, *(src_dd+src_w+1)); \
        } \

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
        for (idst_w = dst_w_beg; idst_w + 4 <= dst_w_end; idst_w += 4) {
            const int isrc_w = -pad_w + 2 * idst_w;
            const float *src_dd = src_d + isrc_w;
            float *dst_dd = dst_d + idst_w;

            __m128 va0, vb0, vc0, vd0,
                   va1, vb1, vc1, vd1,
                   va2, vb2, vc2, vd2,
                   va3, vb3, vc3, vd3;

            va0 = _mm_loadu_ps(src_dd + 0*src_w + 0);
            vb0 = _mm_loadu_ps(src_dd + 0*src_w + 4);
            vc0 = _mm_loadu_ps(src_dd + 1*src_w + 0);
            vd0 = _mm_loadu_ps(src_dd + 1*src_w + 4);
            va1 = _mm_loadu_ps(src_dd + 2*src_w + 0);
            vb1 = _mm_loadu_ps(src_dd + 2*src_w + 4);
            vc1 = _mm_loadu_ps(src_dd + 3*src_w + 0);
            vd1 = _mm_loadu_ps(src_dd + 3*src_w + 4);
            va2 = _mm_loadu_ps(src_dd + 4*src_w + 0);
            vb2 = _mm_loadu_ps(src_dd + 4*src_w + 4);
            vc2 = _mm_loadu_ps(src_dd + 5*src_w + 0);
            vd2 = _mm_loadu_ps(src_dd + 5*src_w + 4);
            va3 = _mm_loadu_ps(src_dd + 6*src_w + 0);
            vb3 = _mm_loadu_ps(src_dd + 6*src_w + 4);
            vc3 = _mm_loadu_ps(src_dd + 7*src_w + 0);
            vd3 = _mm_loadu_ps(src_dd + 7*src_w + 4);

            va0 = _mm_max_ps(va0, vc0);
            vb0 = _mm_max_ps(vb0, vd0);
            va1 = _mm_max_ps(va1, vc1);
            vb1 = _mm_max_ps(vb1, vd1);
            va2 = _mm_max_ps(va2, vc2);
            vb2 = _mm_max_ps(vb2, vd2);
            va3 = _mm_max_ps(va3, vc3);
            vb3 = _mm_max_ps(vb3, vd3);

            vc0 = _mm_shuffle_ps(va0, vb0, POOLING_IMM0);
            vd0 = _mm_shuffle_ps(va0, vb0, POOLING_IMM1);
            vc1 = _mm_shuffle_ps(va1, vb1, POOLING_IMM0);
            vd1 = _mm_shuffle_ps(va1, vb1, POOLING_IMM1);
            vc2 = _mm_shuffle_ps(va2, vb2, POOLING_IMM0);
            vd2 = _mm_shuffle_ps(va2, vb2, POOLING_IMM1);
            vc3 = _mm_shuffle_ps(va3, vb3, POOLING_IMM0);
            vd3 = _mm_shuffle_ps(va3, vb3, POOLING_IMM1);

            va0 = _mm_max_ps(vc0, vd0);
            va1 = _mm_max_ps(vc1, vd1);
            va2 = _mm_max_ps(vc2, vd2);
            va3 = _mm_max_ps(vc3, vd3);

            _mm_storeu_ps(dst_dd + 0*dst_w, va0);
            _mm_storeu_ps(dst_dd + 1*dst_w, va1);
            _mm_storeu_ps(dst_dd + 2*dst_w, va2);
            _mm_storeu_ps(dst_dd + 3*dst_w, va3);
        }
        const int rem = dst_w_end - idst_w;
        int h;
        for (h = 0; h < 4; ++h) {
            float ans[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
            int i;
            for (i = 0; i < rem; ++i) {
                ans[i] = MAX(ans[i], src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 0]);
                ans[i] = MAX(ans[i], src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 1]);
            }
            for (i = 0; i < rem; ++i) {
                ans[i] = MAX(ans[i], src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 0]);
                ans[i] = MAX(ans[i], src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 1]);
            }
            memcpy(dst_d + h*dst_w + idst_w, ans, sizeof(float) * rem);
        }
    }
    if (idst_h + 2 <= dst_h_end) {
        const int isrc_h = -pad_h + 2 * idst_h;
        const float *src_d = src + isrc_h * src_w;
        float *dst_d = dst + idst_h * dst_w;
        int idst_w;
        for (idst_w = dst_w_beg; idst_w + 4 <= dst_w_end; idst_w += 4) {
            const int isrc_w = -pad_w + 2 * idst_w;
            const float *src_dd = src_d + isrc_w;
            float *dst_dd = dst_d + idst_w;

            __m128 va0, vb0, vc0, vd0,
                   va1, vb1, vc1, vd1;

            va0 = _mm_loadu_ps(src_dd + 0*src_w + 0);
            vb0 = _mm_loadu_ps(src_dd + 0*src_w + 4);
            vc0 = _mm_loadu_ps(src_dd + 1*src_w + 0);
            vd0 = _mm_loadu_ps(src_dd + 1*src_w + 4);
            va1 = _mm_loadu_ps(src_dd + 2*src_w + 0);
            vb1 = _mm_loadu_ps(src_dd + 2*src_w + 4);
            vc1 = _mm_loadu_ps(src_dd + 3*src_w + 0);
            vd1 = _mm_loadu_ps(src_dd + 3*src_w + 4);

            va0 = _mm_max_ps(va0, vc0);
            vb0 = _mm_max_ps(vb0, vd0);
            va1 = _mm_max_ps(va1, vc1);
            vb1 = _mm_max_ps(vb1, vd1);

            vc0 = _mm_shuffle_ps(va0, vb0, POOLING_IMM0);
            vd0 = _mm_shuffle_ps(va0, vb0, POOLING_IMM1);
            vc1 = _mm_shuffle_ps(va1, vb1, POOLING_IMM0);
            vd1 = _mm_shuffle_ps(va1, vb1, POOLING_IMM1);

            va0 = _mm_max_ps(vc0, vd0);
            va1 = _mm_max_ps(vc1, vd1);

            _mm_storeu_ps(dst_dd + 0*dst_w, va0);
            _mm_storeu_ps(dst_dd + 1*dst_w, va1);
        }
        const int rem = dst_w_end - idst_w;
        int h;
        for (h = 0; h < 2; ++h) {
            float ans[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
            int i;
            for (i = 0; i < rem; ++i) {
                ans[i] = MAX(ans[i], src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 0]);
                ans[i] = MAX(ans[i], src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 1]);
            }
            for (i = 0; i < rem; ++i) {
                ans[i] = MAX(ans[i], src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 0]);
                ans[i] = MAX(ans[i], src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 1]);
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
        for (idst_w = dst_w_beg; idst_w + 4 <= dst_w_end; idst_w += 4) {
            const int isrc_w = -pad_w + 2 * idst_w;
            const float *src_dd = src_d + isrc_w;
            float *dst_dd = dst_d + idst_w;

            __m128 va0, vb0, vc0, vd0;

            va0 = _mm_loadu_ps(src_dd + 0*src_w + 0);
            vb0 = _mm_loadu_ps(src_dd + 0*src_w + 4);
            vc0 = _mm_loadu_ps(src_dd + 1*src_w + 0);
            vd0 = _mm_loadu_ps(src_dd + 1*src_w + 4);

            va0 = _mm_max_ps(va0, vc0);
            vb0 = _mm_max_ps(vb0, vd0);

            vc0 = _mm_shuffle_ps(va0, vb0, POOLING_IMM0);
            vd0 = _mm_shuffle_ps(va0, vb0, POOLING_IMM1);

            va0 = _mm_max_ps(vc0, vd0);

            _mm_storeu_ps(dst_dd + 0*dst_w, va0);
        }
        const int rem = dst_w_end - idst_w;
        int h;
        for (h = 0; h < 1; ++h) {
            float ans[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
            int i;
            for (i = 0; i < rem; ++i) {
                ans[i] = MAX(ans[i], src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 0]);
                ans[i] = MAX(ans[i], src_d[(2*h+0)*src_w + -pad_w + (idst_w+i)*2 + 1]);
            }
            for (i = 0; i < rem; ++i) {
                ans[i] = MAX(ans[i], src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 0]);
                ans[i] = MAX(ans[i], src_d[(2*h+1)*src_w + -pad_w + (idst_w+i)*2 + 1]);
            }
            memcpy(dst_d + h*dst_w + idst_w, ans, sizeof(float) * rem);
        }
        idst_h += 1;
    }
#undef POOLING_IMM0
#undef POOLING_IMM1
}

} // namespace x86
} // namespace megdnn
// vim: syntax=cpp.doxygen
