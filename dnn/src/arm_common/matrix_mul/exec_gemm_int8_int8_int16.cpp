/**
 * \file dnn/src/arm_common/matrix_mul/exec_gemm_int8_int8_int16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/matrix_mul/exec_gemm_int8_int8_int16.h"

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

namespace {

inline int8x8_t vreinterpret_s8_s8(int8x8_t x) { return x; }

void packA(const int8_t *src, int8_t *dst, size_t M, size_t K)
{
#if 0
    // naive impl
    megdnn_assert(M % 8 == 0);
    for (size_t m = 0; m+8 <= M; m += 8) {
    for (size_t k = 0; k < K; ++k) {
        for (size_t m2 = m; m2 < m+8; ++m2) *(dst++) = src[m2*K + k];
    }
    }
#else
    // 8x8 block at a time
    size_t m = 0;
    int8_t * __restrict dptr = dst;
    for (; m+8 <= M; m += 8) {
        size_t k = 0;
        for (; k+8 <= K; k += 8) {
            const int8_t * __restrict sptr = src + (m*K + k);
            int8x8_t  l0 = vld1_s8(sptr + 0*K),
                      l1 = vld1_s8(sptr + 1*K),
                      l2 = vld1_s8(sptr + 2*K),
                      l3 = vld1_s8(sptr + 3*K),
                      l4 = vld1_s8(sptr + 4*K),
                      l5 = vld1_s8(sptr + 5*K),
                      l6 = vld1_s8(sptr + 6*K),
                      l7 = vld1_s8(sptr + 7*K);
            // do transpose
#define TRANS(lhs, rhs, bit) { \
    auto tmp = vtrn_s ## bit(vreinterpret_s ## bit ## _s8(lhs), \
            vreinterpret_s ## bit ## _s8(rhs)); \
    lhs = vreinterpret_s8_s ## bit(tmp.val[0]); \
    rhs = vreinterpret_s8_s ## bit(tmp.val[1]); \
}
            TRANS(l0, l4, 32);
            TRANS(l1, l5, 32);
            TRANS(l2, l6, 32);
            TRANS(l3, l7, 32);
            TRANS(l0, l2, 16);
            TRANS(l1, l3, 16);
            TRANS(l4, l6, 16);
            TRANS(l5, l7, 16);
            TRANS(l0, l1, 8);
            TRANS(l2, l3, 8);
            TRANS(l4, l5, 8);
            TRANS(l6, l7, 8);
#undef TRANS
            vst1_s8(dptr, l0); dptr += 8;
            vst1_s8(dptr, l1); dptr += 8;
            vst1_s8(dptr, l2); dptr += 8;
            vst1_s8(dptr, l3); dptr += 8;
            vst1_s8(dptr, l4); dptr += 8;
            vst1_s8(dptr, l5); dptr += 8;
            vst1_s8(dptr, l6); dptr += 8;
            vst1_s8(dptr, l7); dptr += 8;
        }
        for (; k < K; ++k) {
            const int8_t * __restrict sptr = src + (m*K + k);
            for (size_t i = 0; i < 8; ++i) *(dptr++) = *(sptr + i*K);
        }
    }
    if (m < M) {
        for (size_t k = 0; k < K; ++k) {
            const int8_t * __restrict sptr = src + (m*K + k);
            for (size_t i = 0; i < 8; ++i) {
                *(dptr++) = (m+i < M ? *(sptr + i*K) : 0);
            }
        }
    }
#endif
}

#define LOAD(i) \
    int8x8_t l ## i = vld1_s8(sptr); \
    int8x8_t s ## i = vld1_s8(sptr + 8); \
    sptr += LDB;

#define STORE(i) \
    vst1_s8(dptr, l ## i); \
    dptr += 8; \
    vst1_s8(dptr, s ## i); \
    dptr += 8;

#define TRANS(i) \
    int8x8_t l ## i = vld1_s8(sptr); \
    int8x8_t s ## i = vld1_s8(sptr + 8); \
    sptr += N; \
    vst1_s8(dptr, l ## i); \
    dptr += 8; \
    vst1_s8(dptr, s ## i); \
    dptr += 8;

void packB(const int8_t *src, int8_t *dst, size_t K, size_t N, size_t LDB)
{
#if 0
    megdnn_assert(N % 8 == 0);
    for (size_t n = 0; n+8 <= N; n += 8)
    for (size_t k = 0; k < K; ++k)
    {
        for (size_t n2 = n; n2 < n+8; ++n2) *(dst++) = src[k*N + n2];
    }
#else
    int8_t * __restrict dptr = dst;
    size_t n = 0;
    for(; n+16 <=N; n += 16) {
        size_t k = 0;
        for (; k+8 <= K; k += 8) {
            const int8_t * __restrict sptr = src + k * LDB + n;

            LOAD(0);
            LOAD(1);
            LOAD(2);
            LOAD(3);
            LOAD(4);
            LOAD(5);
            LOAD(6);
            LOAD(7);
#undef LOAD
            STORE(0);
            STORE(1);
            STORE(2);
            STORE(3);
            STORE(4);
            STORE(5);
            STORE(6);
            STORE(7);
#undef STORE
#undef TRANS

        }
        for (; k < K; ++k) {
            const int8_t * __restrict sptr = src + k * LDB + n;
            int8x8_t l = vld1_s8(sptr);
            int8x8_t s = vld1_s8(sptr + 8);
            vst1_s8(dptr, l); dptr += 8;
            vst1_s8(dptr, s); dptr += 8;
        }
    }
    for (; n+8 <= N; n += 8) {
        size_t k = 0;
        for (; k+8 <= K; k += 8) {
            const int8_t * __restrict sptr = src + k * LDB + n;
            int8x8_t l0 = vld1_s8(sptr + 0*N),
                     l1 = vld1_s8(sptr + 1*N),
                     l2 = vld1_s8(sptr + 2*N),
                     l3 = vld1_s8(sptr + 3*N),
                     l4 = vld1_s8(sptr + 4*N),
                     l5 = vld1_s8(sptr + 5*N),
                     l6 = vld1_s8(sptr + 6*N),
                     l7 = vld1_s8(sptr + 7*N);
            vst1_s8(dptr, l0); dptr += 8;
            vst1_s8(dptr, l1); dptr += 8;
            vst1_s8(dptr, l2); dptr += 8;
            vst1_s8(dptr, l3); dptr += 8;
            vst1_s8(dptr, l4); dptr += 8;
            vst1_s8(dptr, l5); dptr += 8;
            vst1_s8(dptr, l6); dptr += 8;
            vst1_s8(dptr, l7); dptr += 8;
        }
        for (; k < K; ++k) {
            const int8_t * __restrict sptr = src + k * LDB + n;
            int8x8_t l = vld1_s8(sptr);
            vst1_s8(dptr, l); dptr += 8;
        }
    }
    if (n < N) {
        for (size_t k = 0; k < K; ++k) {
            const int8_t * __restrict sptr = src + k * LDB + n;
            int8_t l[8] = {0};
            for (size_t i = 0; n+i < N; ++i) l[i] = sptr[i];
            for (size_t i = 0; i < 8; ++i) *(dptr++) = l[i];
        }
    }
#endif
}

} // anonymous namespace

//#include <iostream>

namespace megdnn {
namespace arm_common {

#define GAO(i) { \
    tmp = vdup_lane_s8(a, i); \
    l ## i = vmlal_s8(l ## i, tmp, b); \
}

#define STORE_REMAIN_N(i, p) \
    if(plen > p) \
        Cptr[p] = vgetq_lane_s16(l##i, p); \
    else \
        break;

#define STORE_PARTRIAL_N(i) { \
    while(1) { \
        STORE_REMAIN_N(i, 0) \
        STORE_REMAIN_N(i, 1) \
        STORE_REMAIN_N(i, 2) \
        STORE_REMAIN_N(i, 3) \
        STORE_REMAIN_N(i, 4) \
        STORE_REMAIN_N(i, 5) \
        STORE_REMAIN_N(i, 6) \
        break; \
    } \
    Cptr += N; \
}

#define STORE_PARTRIAL_M(i) { \
    if(plen > i) { \
        vst1q_s16(Cptr, l##i); \
        Cptr += N; \
    } \
    else \
        break; \
}

#define GAO_16(i) { \
    tmp = vdup_lane_s8(a, i); \
    l ## i = vmlal_s8(l ## i, tmp, b0); \
    s ## i = vmlal_s8(s ## i, tmp, b1); \
}

#define STORE_16(i) { \
    vst1q_s16(Cptr, l##i); \
    vst1q_s16(Cptr + 8, s##i); \
    Cptr += N; \
}

#define STORE_REMAIN_N_16(i, p) \
    if(plen > p) \
        Cptr[8+p] = vgetq_lane_s16(s##i, p); \
    else \
        break;

#define STORE_PARTRIAL_N_16(i) { \
    while(1) { \
        vst1q_s16(Cptr, l##i); \
        STORE_REMAIN_N_16(i, 0) \
        STORE_REMAIN_N_16(i, 1) \
        STORE_REMAIN_N_16(i, 2) \
        STORE_REMAIN_N_16(i, 3) \
        STORE_REMAIN_N_16(i, 4) \
        STORE_REMAIN_N_16(i, 5) \
        STORE_REMAIN_N_16(i, 6) \
        break; \
    } \
    Cptr += N; \
}

#define STORE_PARTRIAL_M_16(i) { \
    if(plen > i) \
        STORE_16(i) \
    else \
        break; \
}

void exec_gemm_int8_int8_int16(const int8_t *A_, const int8_t *B_, int16_t *C,
        size_t M, size_t K, size_t N,size_t LDB,
        int8_t *w0, int8_t *w1)
{
    // for test
    //printf("matrix_mul M %ld, K %ld, N %ld \n", M, K, N);
    packA(A_, w0, M, K);
    packB(B_, w1, K, N, LDB);

    const int8_t * A = w0;
    const int8_t * B = w1;
    for (size_t m = 0; m < M; m += 8) {
        size_t n = 0;
        for (; n + 16 <= N; n += 16) {
        //for (; n + 7 < N; n += 16) {
            int16x8_t l0 = vdupq_n_s16(0),
                      l1 = vdupq_n_s16(0),
                      l2 = vdupq_n_s16(0),
                      l3 = vdupq_n_s16(0),
                      l4 = vdupq_n_s16(0),
                      l5 = vdupq_n_s16(0),
                      l6 = vdupq_n_s16(0),
                      l7 = vdupq_n_s16(0),
                      s0 = vdupq_n_s16(0),
                      s1 = vdupq_n_s16(0),
                      s2 = vdupq_n_s16(0),
                      s3 = vdupq_n_s16(0),
                      s4 = vdupq_n_s16(0),
                      s5 = vdupq_n_s16(0),
                      s6 = vdupq_n_s16(0),
                      s7 = vdupq_n_s16(0);

            const int8_t * __restrict Aptr = A + m*K;
            const int8_t * __restrict Bptr = B + n*K;

            for (size_t k = 0; k < K; ++k) {
                int8x8_t tmp;
                int8x8_t a = vld1_s8(Aptr),
                         b0 = vld1_s8(Bptr),
                         b1 = vld1_s8(Bptr + 8);
                Aptr += 8;
                Bptr += 16;
                //__builtin_prefetch(Aptr, 0, 0);
                __builtin_prefetch(Bptr, 0, 0);

                GAO_16(0);
                GAO_16(1);
                GAO_16(2);
                GAO_16(3);
                GAO_16(4);
                GAO_16(5);
                GAO_16(6);
                GAO_16(7);


            }

            int16_t * __restrict Cptr = C + m*N + n;

            if (m+8 <= M) { // sub-case 1: m+8 <= M && n+16 <= N
                STORE_16(0)
                STORE_16(1)
                STORE_16(2)
                STORE_16(3)
                STORE_16(4)
                STORE_16(5)
                STORE_16(6)
                STORE_16(7)
            } else {
                size_t plen = M - m;
                while(1) {
                    STORE_PARTRIAL_M_16(0)
                    STORE_PARTRIAL_M_16(1)
                    STORE_PARTRIAL_M_16(2)
                    STORE_PARTRIAL_M_16(3)
                    STORE_PARTRIAL_M_16(4)
                    STORE_PARTRIAL_M_16(5)
                    STORE_PARTRIAL_M_16(6)
                    break;
                }
            }
        }

        for (; n < N; n += 8) {
            int16x8_t l0 = vdupq_n_s16(0),
                      l1 = vdupq_n_s16(0),
                      l2 = vdupq_n_s16(0),
                      l3 = vdupq_n_s16(0),
                      l4 = vdupq_n_s16(0),
                      l5 = vdupq_n_s16(0),
                      l6 = vdupq_n_s16(0),
                      l7 = vdupq_n_s16(0);
            const int8_t * __restrict Aptr = A + m*K;
            const int8_t * __restrict Bptr = B + n*K;
            for (size_t k = 0; k < K; ++k) {
                int8x8_t a = vld1_s8(Aptr),
                         b = vld1_s8(Bptr);
                int8x8_t tmp;
                GAO(0);
                GAO(1);
                GAO(2);
                GAO(3);
                GAO(4);
                GAO(5);
                GAO(6);
                GAO(7);
                Aptr += 8;
                Bptr += 8;
            }
            int16_t * __restrict Cptr = C + m*N + n;

            if (m+8 <= M && n+8 <= N) {
                vst1q_s16(Cptr + 0*N, l0);
                vst1q_s16(Cptr + 1*N, l1);
                vst1q_s16(Cptr + 2*N, l2);
                vst1q_s16(Cptr + 3*N, l3);
                vst1q_s16(Cptr + 4*N, l4);
                vst1q_s16(Cptr + 5*N, l5);
                vst1q_s16(Cptr + 6*N, l6);
                vst1q_s16(Cptr + 7*N, l7);
            }  else if (m+8 <=M && n+8 > N) { // m+8<=M && n+8<=N && n+8>N
                size_t plen = N - n;
                STORE_PARTRIAL_N(0)
                STORE_PARTRIAL_N(1)
                STORE_PARTRIAL_N(2)
                STORE_PARTRIAL_N(3)
                STORE_PARTRIAL_N(4)
                STORE_PARTRIAL_N(5)
                STORE_PARTRIAL_N(6)
                STORE_PARTRIAL_N(7)
            } else if(n+8 <= N) { // m+8>M && n+8<=N
                size_t plen = M - m;
                while(1) {
                    STORE_PARTRIAL_M(0)
                    STORE_PARTRIAL_M(1)
                    STORE_PARTRIAL_M(2)
                    STORE_PARTRIAL_M(3)
                    STORE_PARTRIAL_M(4)
                    STORE_PARTRIAL_M(5)
                    STORE_PARTRIAL_M(6)
                    break;
                }
            } else {
                int16_t cache[8*8];
                vst1q_s16(cache + 0*8, l0);
                vst1q_s16(cache + 1*8, l1);
                vst1q_s16(cache + 2*8, l2);
                vst1q_s16(cache + 3*8, l3);
                vst1q_s16(cache + 4*8, l4);
                vst1q_s16(cache + 5*8, l5);
                vst1q_s16(cache + 6*8, l6);
                vst1q_s16(cache + 7*8, l7);

                for (size_t i = 0; m+i < M && i < 8; ++i)
                for (size_t j = 0; n+j < N && j < 8; ++j)
                {
                    Cptr[i*N + j] = cache[i*8 + j];
                }

            }
        }
    }
}
#undef GAO
#undef STORE_REMAIN_N
#undef STORE_PARTRIAL_N
#undef STORE_PARTRIAL_M

#undef GAO_16
#undef STORE_16
#undef STORE_REMAIN_N_16
#undef STORE_PARTRIAL_N_16
#undef STORE_PARTRIAL_M_16

} // namespace arm_common
} // namespace megdnn

// vim: syntax=cpp.doxygen

