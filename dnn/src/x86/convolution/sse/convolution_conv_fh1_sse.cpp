/**
 * \file dnn/src/x86/convolution/sse/convolution_conv_fh1_sse.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#define SIMD_H1                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
        }                                                                     \
    } while (0)

#define SIMD_H2                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
        }                                                                     \
    } while (0)

#define SIMD_H3                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
        }                                                                     \
    } while (0)

#define SIMD_H4                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
        }                                                                     \
    } while (0)

#define SIMD_H5                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
        }                                                                     \
    } while (0)

#define SIMD_H6                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
        }                                                                     \
    } while (0)

#define SIMD_H7                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            __m128 res6;                                                      \
            res6 = _mm_loadu_ps(dst_dd + 6 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 6 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res6 = _mm_add_ps(res6, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
            _mm_storeu_ps(dst_dd + 6 * dst_w, res6);                          \
        }                                                                     \
    } while (0)

#define SIMD_H8                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            __m128 res6;                                                      \
            res6 = _mm_loadu_ps(dst_dd + 6 * dst_w);                          \
            __m128 res7;                                                      \
            res7 = _mm_loadu_ps(dst_dd + 7 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 6 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res6 = _mm_add_ps(res6, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 7 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res7 = _mm_add_ps(res7, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
            _mm_storeu_ps(dst_dd + 6 * dst_w, res6);                          \
            _mm_storeu_ps(dst_dd + 7 * dst_w, res7);                          \
        }                                                                     \
    } while (0)

#define SIMD_H9                                                               \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            __m128 res6;                                                      \
            res6 = _mm_loadu_ps(dst_dd + 6 * dst_w);                          \
            __m128 res7;                                                      \
            res7 = _mm_loadu_ps(dst_dd + 7 * dst_w);                          \
            __m128 res8;                                                      \
            res8 = _mm_loadu_ps(dst_dd + 8 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 6 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res6 = _mm_add_ps(res6, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 7 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res7 = _mm_add_ps(res7, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 8 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res8 = _mm_add_ps(res8, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
            _mm_storeu_ps(dst_dd + 6 * dst_w, res6);                          \
            _mm_storeu_ps(dst_dd + 7 * dst_w, res7);                          \
            _mm_storeu_ps(dst_dd + 8 * dst_w, res8);                          \
        }                                                                     \
    } while (0)

#define SIMD_H10                                                              \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            __m128 res6;                                                      \
            res6 = _mm_loadu_ps(dst_dd + 6 * dst_w);                          \
            __m128 res7;                                                      \
            res7 = _mm_loadu_ps(dst_dd + 7 * dst_w);                          \
            __m128 res8;                                                      \
            res8 = _mm_loadu_ps(dst_dd + 8 * dst_w);                          \
            __m128 res9;                                                      \
            res9 = _mm_loadu_ps(dst_dd + 9 * dst_w);                          \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 6 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res6 = _mm_add_ps(res6, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 7 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res7 = _mm_add_ps(res7, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 8 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res8 = _mm_add_ps(res8, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 9 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res9 = _mm_add_ps(res9, tmp1);                                \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
            _mm_storeu_ps(dst_dd + 6 * dst_w, res6);                          \
            _mm_storeu_ps(dst_dd + 7 * dst_w, res7);                          \
            _mm_storeu_ps(dst_dd + 8 * dst_w, res8);                          \
            _mm_storeu_ps(dst_dd + 9 * dst_w, res9);                          \
        }                                                                     \
    } while (0)

#define SIMD_H11                                                              \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            __m128 res6;                                                      \
            res6 = _mm_loadu_ps(dst_dd + 6 * dst_w);                          \
            __m128 res7;                                                      \
            res7 = _mm_loadu_ps(dst_dd + 7 * dst_w);                          \
            __m128 res8;                                                      \
            res8 = _mm_loadu_ps(dst_dd + 8 * dst_w);                          \
            __m128 res9;                                                      \
            res9 = _mm_loadu_ps(dst_dd + 9 * dst_w);                          \
            __m128 res10;                                                     \
            res10 = _mm_loadu_ps(dst_dd + 10 * dst_w);                        \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 6 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res6 = _mm_add_ps(res6, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 7 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res7 = _mm_add_ps(res7, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 8 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res8 = _mm_add_ps(res8, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 9 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res9 = _mm_add_ps(res9, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 10 * src_w);                     \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res10 = _mm_add_ps(res10, tmp1);                              \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
            _mm_storeu_ps(dst_dd + 6 * dst_w, res6);                          \
            _mm_storeu_ps(dst_dd + 7 * dst_w, res7);                          \
            _mm_storeu_ps(dst_dd + 8 * dst_w, res8);                          \
            _mm_storeu_ps(dst_dd + 9 * dst_w, res9);                          \
            _mm_storeu_ps(dst_dd + 10 * dst_w, res10);                        \
        }                                                                     \
    } while (0)

#define SIMD_H12                                                              \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            __m128 res6;                                                      \
            res6 = _mm_loadu_ps(dst_dd + 6 * dst_w);                          \
            __m128 res7;                                                      \
            res7 = _mm_loadu_ps(dst_dd + 7 * dst_w);                          \
            __m128 res8;                                                      \
            res8 = _mm_loadu_ps(dst_dd + 8 * dst_w);                          \
            __m128 res9;                                                      \
            res9 = _mm_loadu_ps(dst_dd + 9 * dst_w);                          \
            __m128 res10;                                                     \
            res10 = _mm_loadu_ps(dst_dd + 10 * dst_w);                        \
            __m128 res11;                                                     \
            res11 = _mm_loadu_ps(dst_dd + 11 * dst_w);                        \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 6 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res6 = _mm_add_ps(res6, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 7 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res7 = _mm_add_ps(res7, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 8 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res8 = _mm_add_ps(res8, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 9 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res9 = _mm_add_ps(res9, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 10 * src_w);                     \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res10 = _mm_add_ps(res10, tmp1);                              \
                tmp0 = _mm_loadu_ps(src_dd + 11 * src_w);                     \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res11 = _mm_add_ps(res11, tmp1);                              \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
            _mm_storeu_ps(dst_dd + 6 * dst_w, res6);                          \
            _mm_storeu_ps(dst_dd + 7 * dst_w, res7);                          \
            _mm_storeu_ps(dst_dd + 8 * dst_w, res8);                          \
            _mm_storeu_ps(dst_dd + 9 * dst_w, res9);                          \
            _mm_storeu_ps(dst_dd + 10 * dst_w, res10);                        \
            _mm_storeu_ps(dst_dd + 11 * dst_w, res11);                        \
        }                                                                     \
    } while (0)

#define SIMD_H13                                                              \
    do {                                                                      \
        const size_t sh = dh;                                                 \
        const float* src_d = src + sh * src_w;                                \
        float* dst_d = dst + dh * dst_w;                                      \
        size_t dw = dst_w_beg;                                                \
        for (; dw < dst_w_end; dw += 4) {                                     \
            const size_t sw = dw;                                             \
            float* dst_dd = dst_d + dw;                                       \
            __m128 tmp0;                                                      \
            __m128 tmp1;                                                      \
            __m128 res0;                                                      \
            res0 = _mm_loadu_ps(dst_dd + 0 * dst_w);                          \
            __m128 res1;                                                      \
            res1 = _mm_loadu_ps(dst_dd + 1 * dst_w);                          \
            __m128 res2;                                                      \
            res2 = _mm_loadu_ps(dst_dd + 2 * dst_w);                          \
            __m128 res3;                                                      \
            res3 = _mm_loadu_ps(dst_dd + 3 * dst_w);                          \
            __m128 res4;                                                      \
            res4 = _mm_loadu_ps(dst_dd + 4 * dst_w);                          \
            __m128 res5;                                                      \
            res5 = _mm_loadu_ps(dst_dd + 5 * dst_w);                          \
            __m128 res6;                                                      \
            res6 = _mm_loadu_ps(dst_dd + 6 * dst_w);                          \
            __m128 res7;                                                      \
            res7 = _mm_loadu_ps(dst_dd + 7 * dst_w);                          \
            __m128 res8;                                                      \
            res8 = _mm_loadu_ps(dst_dd + 8 * dst_w);                          \
            __m128 res9;                                                      \
            res9 = _mm_loadu_ps(dst_dd + 9 * dst_w);                          \
            __m128 res10;                                                     \
            res10 = _mm_loadu_ps(dst_dd + 10 * dst_w);                        \
            __m128 res11;                                                     \
            res11 = _mm_loadu_ps(dst_dd + 11 * dst_w);                        \
            __m128 res12;                                                     \
            res12 = _mm_loadu_ps(dst_dd + 12 * dst_w);                        \
            for (size_t fw = 0; fw < flt_w; ++fw) {                           \
                const float* src_dd = src_d + sw + fw;                        \
                __m128 vf0 = _mm_set1_ps(filter[0 * flt_w + flt_w - fw - 1]); \
                tmp0 = _mm_loadu_ps(src_dd + 0 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res0 = _mm_add_ps(res0, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 1 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res1 = _mm_add_ps(res1, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 2 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res2 = _mm_add_ps(res2, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 3 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res3 = _mm_add_ps(res3, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 4 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res4 = _mm_add_ps(res4, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 5 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res5 = _mm_add_ps(res5, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 6 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res6 = _mm_add_ps(res6, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 7 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res7 = _mm_add_ps(res7, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 8 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res8 = _mm_add_ps(res8, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 9 * src_w);                      \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res9 = _mm_add_ps(res9, tmp1);                                \
                tmp0 = _mm_loadu_ps(src_dd + 10 * src_w);                     \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res10 = _mm_add_ps(res10, tmp1);                              \
                tmp0 = _mm_loadu_ps(src_dd + 11 * src_w);                     \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res11 = _mm_add_ps(res11, tmp1);                              \
                tmp0 = _mm_loadu_ps(src_dd + 12 * src_w);                     \
                tmp1 = _mm_mul_ps(tmp0, vf0);                                 \
                res12 = _mm_add_ps(res12, tmp1);                              \
            }                                                                 \
            _mm_storeu_ps(dst_dd + 0 * dst_w, res0);                          \
            _mm_storeu_ps(dst_dd + 1 * dst_w, res1);                          \
            _mm_storeu_ps(dst_dd + 2 * dst_w, res2);                          \
            _mm_storeu_ps(dst_dd + 3 * dst_w, res3);                          \
            _mm_storeu_ps(dst_dd + 4 * dst_w, res4);                          \
            _mm_storeu_ps(dst_dd + 5 * dst_w, res5);                          \
            _mm_storeu_ps(dst_dd + 6 * dst_w, res6);                          \
            _mm_storeu_ps(dst_dd + 7 * dst_w, res7);                          \
            _mm_storeu_ps(dst_dd + 8 * dst_w, res8);                          \
            _mm_storeu_ps(dst_dd + 9 * dst_w, res9);                          \
            _mm_storeu_ps(dst_dd + 10 * dst_w, res10);                        \
            _mm_storeu_ps(dst_dd + 11 * dst_w, res11);                        \
            _mm_storeu_ps(dst_dd + 12 * dst_w, res12);                        \
        }                                                                     \
    } while (0)

#include <xmmintrin.h>
#include <algorithm>

#include "../convolution_direct_special_cases.h"

namespace megdnn {
namespace x86 {
namespace detail {

void convolution_conv_fh1_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w)
{
    (void)src_h;
    const size_t dst_h_beg = 0;
    const size_t dst_h_end = dst_h;
    const size_t dst_w_beg = 0;
    const size_t dst_w_end = dst_w;

    size_t dh = dst_h_beg;
    for (; dh + 13 <= dst_h_end; dh += 13) {
        SIMD_H13;
    }
    switch (dst_h_end - dh) {
        case 1:
            SIMD_H1;
            break;
        case 2:
            SIMD_H2;
            break;
        case 3:
            SIMD_H3;
            break;
        case 4:
            SIMD_H4;
            break;
        case 5:
            SIMD_H5;
            break;
        case 6:
            SIMD_H6;
            break;
        case 7:
            SIMD_H7;
            break;
        case 8:
            SIMD_H8;
            break;
        case 9:
            SIMD_H9;
            break;
        case 10:
            SIMD_H10;
            break;
        case 11:
            SIMD_H11;
            break;
        case 12:
            SIMD_H12;
            break;
    }
}

} // namespace detail
} // namespace x86
} // namespace megdnn
#undef SIMD_H1
#undef SIMD_H2
#undef SIMD_H3
#undef SIMD_H4
#undef SIMD_H5
#undef SIMD_H6
#undef SIMD_H7
#undef SIMD_H8
#undef SIMD_H9
#undef SIMD_H10
#undef SIMD_H11
#undef SIMD_H12
#undef SIMD_H13
