#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#include "src/arm_common/matrix_mul/fp16/hgemv.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/fallback/matrix_mul/gemm_common.h"

namespace {

#define UNROLL_OUT(cb, step) UNROLL_CALL_RAW(step, cb)

void hgemv_naive_n(
        const __fp16* __restrict A, const __fp16* __restrict B, __fp16* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 1);
#define vaddvq_f16(v) \
    ((v)[0] + (v)[1] + (v)[2] + (v)[3] + (v)[4] + (v)[5] + (v)[6] + (v)[7])
    size_t m = 0;
    for (; m + 4 <= M; m += 4) {
        float16x8_t a0, a1, a2, a3, b0;
        float16x8_t sum0, sum1, sum2, sum3;
        sum0 = vdupq_n_f16(0.f);
        sum1 = vdupq_n_f16(0.f);
        sum2 = vdupq_n_f16(0.f);
        sum3 = vdupq_n_f16(0.f);
        size_t k = 0;
        for (; k + 8 <= K; k += 8) {
            a0 = vld1q_f16(A + (m + 0) * Astride + k);
            a1 = vld1q_f16(A + (m + 1) * Astride + k);
            a2 = vld1q_f16(A + (m + 2) * Astride + k);
            a3 = vld1q_f16(A + (m + 3) * Astride + k);
            b0 = vld1q_f16(B + k);
            sum0 = vmlaq_f16(sum0, a0, b0);
            sum1 = vmlaq_f16(sum1, a1, b0);
            sum2 = vmlaq_f16(sum2, a2, b0);
            sum3 = vmlaq_f16(sum3, a3, b0);
        }
        for (; k < K; ++k) {
            sum0[0] += A[(m + 0) * Astride + k] * B[k];
            sum1[0] += A[(m + 1) * Astride + k] * B[k];
            sum2[0] += A[(m + 2) * Astride + k] * B[k];
            sum3[0] += A[(m + 3) * Astride + k] * B[k];
        }
        C[(m + 0) * Cstride] = vaddvq_f16(sum0);
        C[(m + 1) * Cstride] = vaddvq_f16(sum1);
        C[(m + 2) * Cstride] = vaddvq_f16(sum2);
        C[(m + 3) * Cstride] = vaddvq_f16(sum3);
    }
    for (; m + 2 <= M; m += 2) {
        float16x8_t a0, a1, b0;
        float16x8_t sum0, sum1;
        sum0 = vdupq_n_f16(0.f);
        sum1 = vdupq_n_f16(0.f);
        size_t k = 0;
        for (; k + 8 <= K; k += 8) {
            a0 = vld1q_f16(A + (m + 0) * Astride + k);
            a1 = vld1q_f16(A + (m + 1) * Astride + k);
            b0 = vld1q_f16(B + k);
            sum0 = vmlaq_f16(sum0, a0, b0);
            sum1 = vmlaq_f16(sum1, a1, b0);
        }
        for (; k < K; ++k) {
            sum0[0] += A[(m + 0) * Astride + k] * B[k];
            sum1[0] += A[(m + 1) * Astride + k] * B[k];
        }
        C[(m + 0) * Cstride] = vaddvq_f16(sum0);
        C[(m + 1) * Cstride] = vaddvq_f16(sum1);
    }
    for (; m < M; m += 1) {
        float16x8_t a0, b0;
        float16x8_t sum0;
        sum0 = vdupq_n_f16(0.f);
        size_t k = 0;
        for (; k + 8 <= K; k += 8) {
            a0 = vld1q_f16(A + (m + 0) * Astride + k);
            b0 = vld1q_f16(B + k);
            sum0 = vfmaq_f16(sum0, a0, b0);
        }
        for (; k < K; k += 1) {
            sum0[0] += A[(m + 0) * Astride + k] * B[k];
        }
        C[(m + 0) * Cstride] = vaddvq_f16(sum0);
    }
#undef vaddvq_f16
}
}  // namespace

#if defined(__aarch64__)
#define VFMAQ_N_F16(a, b, n) vfmaq_n_f16(a, b, n)
#else
#define VFMAQ_N_F16(a, b, n) vaddq_f16(a, vmulq_n_f16(b, n))
#endif

#if defined(__aarch64__)
#define VFMA_N_F16(a, b, n) vfma_n_f16(a, b, n)
#else
#define VFMA_N_F16(a, b, n) vadd_f16(a, vmul_n_f16(b, n))
#endif

void megdnn::arm_common::gemv_like(
        const __fp16* __restrict A, const __fp16* __restrict B, __fp16* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert((M <= 4) || (M == 8 && K <= 2) || (N == 1 && Bstride == 1));
    if (N == 1) {
        return hgemv_naive_n(A, B, C, M, N, K, Astride, Bstride, Cstride);
    }
    size_t m = 0;
    for (; m + 4 <= M; m += 4) {
        size_t k = 0;
        memset(C + m * Cstride, 0, 4 * sizeof(__fp16) * N);
        for (; k + 4 <= K; k += 4) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k], a01 = A[m * Astride + k + 1],
                   a02 = A[m * Astride + k + 2], a03 = A[m * Astride + k + 3];
            __fp16 a10 = A[(m + 1) * Astride + k], a11 = A[(m + 1) * Astride + k + 1],
                   a12 = A[(m + 1) * Astride + k + 2],
                   a13 = A[(m + 1) * Astride + k + 3];
            __fp16 a20 = A[(m + 2) * Astride + k], a21 = A[(m + 2) * Astride + k + 1],
                   a22 = A[(m + 2) * Astride + k + 2],
                   a23 = A[(m + 2) * Astride + k + 3];
            __fp16 a30 = A[(m + 3) * Astride + k], a31 = A[(m + 3) * Astride + k + 1],
                   a32 = A[(m + 3) * Astride + k + 2],
                   a33 = A[(m + 3) * Astride + k + 3];
            for (; n + 8 <= N; n += 8) {
                float16x8_t b0, b1, b2, b3;
                float16x8_t c0, c1, c2, c3;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMAQ_N_F16(c1, b##i, a1##i);
#define calculate_row2(i) c2 = VFMAQ_N_F16(c2, b##i, a2##i);
#define calculate_row3(i) c3 = VFMAQ_N_F16(c3, b##i, a3##i);
                UNROLL_OUT(calculate_row0, 4)
                UNROLL_OUT(calculate_row1, 4)
                UNROLL_OUT(calculate_row2, 4)
                UNROLL_OUT(calculate_row3, 4)
#undef calculate_row0
#undef calculate_row1
#undef calculate_row2
#undef calculate_row3
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n + 4 <= N; n += 4) {
                float16x4_t b0, b1, b2, b3;
                float16x4_t c0, c1, c2, c3;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMA_N_F16(c1, b##i, a1##i);
#define calculate_row2(i) c2 = VFMA_N_F16(c2, b##i, a2##i);
#define calculate_row3(i) c3 = VFMA_N_F16(c3, b##i, a3##i);
                UNROLL_OUT(calculate_row0, 4)
                UNROLL_OUT(calculate_row1, 4)
                UNROLL_OUT(calculate_row2, 4)
                UNROLL_OUT(calculate_row3, 4)
#undef calculate_row0
#undef calculate_row1
#undef calculate_row2
#undef calculate_row3
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n < N; n += 1) {
                __fp16 b0, b1, b2, b3;
                __fp16 c0, c1, c2, c3;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
                c0 += a00 * b0 + a01 * b1 + a02 * b2 + a03 * b3;
                c1 += a10 * b0 + a11 * b1 + a12 * b2 + a13 * b3;
                c2 += a20 * b0 + a21 * b1 + a22 * b2 + a23 * b3;
                c3 += a30 * b0 + a31 * b1 + a32 * b2 + a33 * b3;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
        }
        for (; k + 2 <= K; k += 2) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k], a01 = A[m * Astride + k + 1];
            __fp16 a10 = A[(m + 1) * Astride + k], a11 = A[(m + 1) * Astride + k + 1];
            __fp16 a20 = A[(m + 2) * Astride + k], a21 = A[(m + 2) * Astride + k + 1];
            __fp16 a30 = A[(m + 3) * Astride + k], a31 = A[(m + 3) * Astride + k + 1];
            for (; n + 8 <= N; n += 8) {
                float16x8_t b0, b1;
                float16x8_t c0, c1, c2, c3;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMAQ_N_F16(c1, b##i, a1##i);
#define calculate_row2(i) c2 = VFMAQ_N_F16(c2, b##i, a2##i);
#define calculate_row3(i) c3 = VFMAQ_N_F16(c3, b##i, a3##i);
                UNROLL_OUT(calculate_row0, 2)
                UNROLL_OUT(calculate_row1, 2)
                UNROLL_OUT(calculate_row2, 2)
                UNROLL_OUT(calculate_row3, 2)
#undef calculate_row0
#undef calculate_row1
#undef calculate_row2
#undef calculate_row3
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n + 4 <= N; n += 4) {
                float16x4_t b0, b1;
                float16x4_t c0, c1, c2, c3;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMA_N_F16(c1, b##i, a1##i);
#define calculate_row2(i) c2 = VFMA_N_F16(c2, b##i, a2##i);
#define calculate_row3(i) c3 = VFMA_N_F16(c3, b##i, a3##i);
                UNROLL_OUT(calculate_row0, 2)
                UNROLL_OUT(calculate_row1, 2)
                UNROLL_OUT(calculate_row2, 2)
                UNROLL_OUT(calculate_row3, 2)
#undef calculate_row0
#undef calculate_row1
#undef calculate_row2
#undef calculate_row3
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n < N; n += 1) {
                __fp16 b0, b1;
                __fp16 c0, c1, c2, c3;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
                c0 += a00 * b0 + a01 * b1;
                c1 += a10 * b0 + a11 * b1;
                c2 += a20 * b0 + a21 * b1;
                c3 += a30 * b0 + a31 * b1;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
        }
        for (; k < K; k += 1) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k];
            __fp16 a10 = A[(m + 1) * Astride + k];
            __fp16 a20 = A[(m + 2) * Astride + k];
            __fp16 a30 = A[(m + 3) * Astride + k];
            for (; n + 8 <= N; n += 8) {
                float16x8_t b0;
                float16x8_t c0, c1, c2, c3;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMAQ_N_F16(c1, b##i, a1##i);
#define calculate_row2(i) c2 = VFMAQ_N_F16(c2, b##i, a2##i);
#define calculate_row3(i) c3 = VFMAQ_N_F16(c3, b##i, a3##i);
                UNROLL_OUT(calculate_row0, 1)
                UNROLL_OUT(calculate_row1, 1)
                UNROLL_OUT(calculate_row2, 1)
                UNROLL_OUT(calculate_row3, 1)
#undef calculate_row0
#undef calculate_row1
#undef calculate_row2
#undef calculate_row3
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n + 4 <= N; n += 4) {
                float16x4_t b0;
                float16x4_t c0, c1, c2, c3;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMA_N_F16(c1, b##i, a1##i);
#define calculate_row2(i) c2 = VFMA_N_F16(c2, b##i, a2##i);
#define calculate_row3(i) c3 = VFMA_N_F16(c3, b##i, a3##i);
                UNROLL_OUT(calculate_row0, 1)
                UNROLL_OUT(calculate_row1, 1)
                UNROLL_OUT(calculate_row2, 1)
                UNROLL_OUT(calculate_row3, 1)
#undef calculate_row0
#undef calculate_row1
#undef calculate_row2
#undef calculate_row3
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n < N; n += 1) {
                __fp16 b0;
                __fp16 c0, c1, c2, c3;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
                c0 = c0 + a00 * b0;
                c1 = c1 + a10 * b0;
                c2 = c2 + a20 * b0;
                c3 = c3 + a30 * b0;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
        }
    }
    for (; m + 2 <= M; m += 2) {
        size_t k = 0;
        memset(C + m * Cstride, 0, 2 * sizeof(__fp16) * N);
        for (; k + 4 <= K; k += 4) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k], a01 = A[m * Astride + k + 1],
                   a02 = A[m * Astride + k + 2], a03 = A[m * Astride + k + 3];
            __fp16 a10 = A[(m + 1) * Astride + k], a11 = A[(m + 1) * Astride + k + 1],
                   a12 = A[(m + 1) * Astride + k + 2],
                   a13 = A[(m + 1) * Astride + k + 3];
            for (; n + 8 <= N; n += 8) {
                float16x8_t b0, b1, b2, b3;
                float16x8_t c0, c1;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMAQ_N_F16(c1, b##i, a1##i);
                UNROLL_OUT(calculate_row0, 4)
                UNROLL_OUT(calculate_row1, 4)
#undef calculate_row0
#undef calculate_row1
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n + 4 <= N; n += 4) {
                float16x4_t b0, b1, b2, b3;
                float16x4_t c0, c1;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMA_N_F16(c1, b##i, a1##i);
                UNROLL_OUT(calculate_row0, 4)
                UNROLL_OUT(calculate_row1, 4)
#undef calculate_row0
#undef calculate_row1
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n < N; n += 1) {
                __fp16 b0, b1, b2, b3;
                __fp16 c0, c1;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
                c0 += a00 * b0 + a01 * b1 + a02 * b2 + a03 * b3;
                c1 += a10 * b0 + a11 * b1 + a12 * b2 + a13 * b3;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
        }
        for (; k + 2 <= K; k += 2) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k], a01 = A[m * Astride + k + 1];
            __fp16 a10 = A[(m + 1) * Astride + k], a11 = A[(m + 1) * Astride + k + 1];
            for (; n + 8 <= N; n += 8) {
                float16x8_t b0, b1;
                float16x8_t c0, c1;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMAQ_N_F16(c1, b##i, a1##i);
                UNROLL_OUT(calculate_row0, 2)
                UNROLL_OUT(calculate_row1, 2)
#undef calculate_row0
#undef calculate_row1
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n + 4 <= N; n += 4) {
                float16x4_t b0, b1;
                float16x4_t c0, c1;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMA_N_F16(c1, b##i, a1##i);
                UNROLL_OUT(calculate_row0, 2)
                UNROLL_OUT(calculate_row1, 2)
#undef calculate_row0
#undef calculate_row1
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n < N; n += 1) {
                __fp16 b0, b1;
                __fp16 c0, c1;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
                c0 += a00 * b0 + a01 * b1;
                c1 += a10 * b0 + a11 * b1;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
        }
        for (; k < K; k += 1) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k];
            __fp16 a10 = A[(m + 1) * Astride + k];
            for (; n + 8 <= N; n += 8) {
                float16x8_t b0;
                float16x8_t c0, c1;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMAQ_N_F16(c1, b##i, a1##i);
                UNROLL_OUT(calculate_row0, 1)
                UNROLL_OUT(calculate_row1, 1)
#undef calculate_row0
#undef calculate_row1
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n + 4 <= N; n += 4) {
                float16x4_t b0;
                float16x4_t c0, c1;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#define calculate_row1(i) c1 = VFMA_N_F16(c1, b##i, a1##i);
                UNROLL_OUT(calculate_row0, 1)
                UNROLL_OUT(calculate_row1, 1)
#undef calculate_row0
#undef calculate_row1
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n < N; n += 1) {
                __fp16 b0;
                __fp16 c0, c1;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
                c0 = c0 + a00 * b0;
                c1 = c1 + a10 * b0;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
        }
    }
    for (; m < M; m += 1) {
        size_t k = 0;
        memset(C + m * Cstride, 0, sizeof(__fp16) * N);
        for (; k + 4 <= K; k += 4) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k], a01 = A[m * Astride + k + 1],
                   a02 = A[m * Astride + k + 2], a03 = A[m * Astride + k + 3];
            {
#if !defined(__aarch64__)
                float16x8_t va00 = vdupq_n_f16(a00), va01 = vdupq_n_f16(a01),
                            va02 = vdupq_n_f16(a02), va03 = vdupq_n_f16(a03);
#endif
                for (; n + 8 <= N; n += 8) {
                    float16x8_t b0, b1, b2, b3;
                    float16x8_t c0;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                    UNROLL_OUT(loadC, 1)
                    UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#if defined(__aarch64__)
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#else
#define calculate_row0(i) c0 = vfmaq_f16(c0, b##i, va0##i);
#endif
                    UNROLL_OUT(calculate_row0, 4)
#undef calculate_row0
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                    UNROLL_OUT(vstore, 1)
#undef vstore
                }
            }
            {
#if !defined(__aarch64__)
                float16x4_t va00 = vdup_n_f16(a00), va01 = vdup_n_f16(a01),
                            va02 = vdup_n_f16(a02), va03 = vdup_n_f16(a03);
#endif
                for (; n + 4 <= N; n += 4) {
                    float16x4_t b0, b1, b2, b3;
                    float16x4_t c0;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                    UNROLL_OUT(loadC, 1)
                    UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#if defined(__aarch64__)
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#else
#define calculate_row0(i) c0 = vfma_f16(c0, b##i, va0##i);
#endif
                    UNROLL_OUT(calculate_row0, 4)
#undef calculate_row0
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                    UNROLL_OUT(vstore, 1)
#undef vstore
                }
            }
            for (; n < N; n += 1) {
                __fp16 b0, b1, b2, b3;
                __fp16 c0;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
                c0 += a00 * b0 + a01 * b1 + a02 * b2 + a03 * b3;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
        }
        for (; k + 2 <= K; k += 2) {
            __fp16 a00 = A[m * Astride + k], a01 = A[m * Astride + k + 1];
            size_t n = 0;
            {
#if !defined(__aarch64__)
                float16x8_t va00 = vdupq_n_f16(a00), va01 = vdupq_n_f16(a01);
#endif
                for (; n + 8 <= N; n += 8) {
                    float16x8_t b0, b1;
                    float16x8_t c0;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                    UNROLL_OUT(loadC, 1)
                    UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#if defined(__aarch64__)
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#else
#define calculate_row0(i) c0 = vfmaq_f16(c0, b##i, va0##i);
#endif
                    UNROLL_OUT(calculate_row0, 2)
#undef calculate_row0
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                    UNROLL_OUT(vstore, 1)
#undef vstore
                }
            }
            {
#if !defined(__aarch64__)
                float16x4_t va00 = vdup_n_f16(a00), va01 = vdup_n_f16(a01);
#endif
                for (; n + 4 <= N; n += 4) {
                    float16x4_t b0, b1;
                    float16x4_t c0;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                    UNROLL_OUT(loadC, 1)
                    UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#if defined(__aarch64__)
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#else
#define calculate_row0(i) c0 = vfma_f16(c0, b##i, va0##i);
#endif
                    UNROLL_OUT(calculate_row0, 2)
#undef calculate_row0
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                    UNROLL_OUT(vstore, 1)
#undef vstore
                }
            }
            for (; n < N; n += 1) {
                __fp16 b0, b1;
                __fp16 c0;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
                c0 += a00 * b0 + a01 * b1;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
        }
        for (; k < K; k += 1) {
            size_t n = 0;
            __fp16 a00 = A[m * Astride + k];
            {
#if !defined(__aarch64__)
                float16x8_t va00 = vdupq_n_f16(a00);
#endif
                for (; n + 8 <= N; n += 8) {
                    float16x8_t b0;
                    float16x8_t c0;
#define loadB(i) b##i = vld1q_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f16(C + (m + i) * Cstride + n);
                    UNROLL_OUT(loadC, 1)
                    UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#if defined(__aarch64__)
#define calculate_row0(i) c0 = VFMAQ_N_F16(c0, b##i, a0##i);
#else
#define calculate_row0(i) c0 = vfmaq_f16(c0, b##i, va0##i);
#endif
                    UNROLL_OUT(calculate_row0, 1)
#undef calculate_row0
#define vstore(i) vst1q_f16(C + (m + i) * Cstride + n, c##i);
                    UNROLL_OUT(vstore, 1)
#undef vstore
                }
            }
            {
#if !defined(__aarch64__)
                float16x4_t va00 = vdup_n_f16(a00);
#endif
                for (; n + 4 <= N; n += 4) {
                    float16x4_t b0;
                    float16x4_t c0;
#define loadB(i) b##i = vld1_f16(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f16(C + (m + i) * Cstride + n);
                    UNROLL_OUT(loadC, 1)
                    UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#if defined(__aarch64__)
#define calculate_row0(i) c0 = VFMA_N_F16(c0, b##i, a0##i);
#else
#define calculate_row0(i) c0 = vfma_f16(c0, b##i, va0##i);
#endif
                    UNROLL_OUT(calculate_row0, 1)
#undef calculate_row0
#define vstore(i) vst1_f16(C + (m + i) * Cstride + n, c##i);
                    UNROLL_OUT(vstore, 1)
#undef vstore
                }
            }
            for (; n < N; n += 1) {
                __fp16 b0;
                __fp16 c0;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
                c0 = c0 + a00 * b0;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
        }
    }
}

#undef VFMA_N_F16
#undef VFMAQ_N_F16

bool megdnn::arm_common::is_hgemv_preferred(
        bool transposeA, bool transposeB, size_t M, size_t N, size_t K, size_t /*LDA*/,
        size_t LDB, size_t /*LDC*/) {
    if (transposeA)
        return false;
    if (transposeB)
        return false;

    return M <= 4 || (M <= 8 && K <= 2) || (N == 1 && LDB == 1);
}

#endif
// vim: syntax=cpp.doxygen
