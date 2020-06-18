/**
 * \file dnn/src/arm_common/matrix_mul/fp32/exec_sgemv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/matrix_mul/fp32/exec_sgemv.h"
#include <cstddef>
#include "include/megdnn/oprs.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fp32_sgemv)

using namespace megdnn;
using namespace arm_common;

namespace {

#define UNROLL_OUT(cb, step) UNROLL_CALL_RAW(step, cb)

#if !defined(__aarch64__)
#define vaddvq_f32(v) (v)[0] + (v)[1] + (v)[2] + (v)[3]
#endif
void sgemv_naive_n(const float* __restrict A, const float* __restrict B,
                   float* __restrict C, size_t M, size_t N, size_t K,
                   size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 1);
#define reset_acc(i) acc##i = 0;
#define acc_calu(i) acc##i += A[(m + i) * Astride + k] * B[k];
#define vdupq_sum(i) sum##i = vdupq_n_f32(0.f);
#define loadA(i) a##i = vld1q_f32(A + (m + i) * Astride + k);
#define loadB(i) b##i = vld1q_f32(B + k);
#define calculate(i) sum##i = vmlaq_f32(sum##i, a##i, b0);
#define vstore(i) C[(m + i) * Cstride] = vaddvq_f32(sum##i) + acc##i;
    size_t m = 0;
    for (; m < M; m += 1) {
        float acc0;
        float32x4_t a0, b0;
        float32x4_t sum0;
        UNROLL_OUT(vdupq_sum, 1)
        size_t k = 0;
        for (; k + 4 <= K; k += 4) {
            UNROLL_OUT(loadA, 1)
            UNROLL_OUT(loadB, 1)
            UNROLL_OUT(calculate, 1)
        }
        UNROLL_OUT(reset_acc, 1)
        for (; k < K; ++k) {
            UNROLL_OUT(acc_calu, 1)
        }
        UNROLL_OUT(vstore, 1)
    }
#undef vdupq_sum
#undef loadA
#undef loadB
#undef calculate
#undef vstore
}
#if !defined(__aarch64__)
#undef vaddvq_f32
#endif

void sgemv_naive_m(const float* __restrict A, const float* __restrict B,
                   float* __restrict C, size_t M, size_t N, size_t K,
                   size_t Astride, size_t Bstride, size_t Cstride) {
    size_t m = 0;
    for (; m + 4 <= M; m += 4) {
        size_t k = 0;
        memset(C + m * Cstride, 0, 4 * sizeof(float) * N);
        for (; k + 4 <= K; k += 4) {
            size_t n = 0;
            for (; n + 4 <= N; n += 4) {
                float32x4_t a00, a01, a02, a03, a10, a11, a12, a13, a20, a21,
                        a22, a23, a30, a31, a32, a33;
                float32x4_t b0, b1, b2, b3;
                float32x4_t c0, c1, c2, c3;
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
#define loadA0(i) a0##i = vdupq_n_f32(A[(m + 0) * Astride + k + i]);
#define loadA1(i) a1##i = vdupq_n_f32(A[(m + 1) * Astride + k + i]);
#define loadA2(i) a2##i = vdupq_n_f32(A[(m + 2) * Astride + k + i]);
#define loadA3(i) a3##i = vdupq_n_f32(A[(m + 3) * Astride + k + i]);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 4)
                UNROLL_OUT(loadA0, 4)
                UNROLL_OUT(loadA1, 4)
                UNROLL_OUT(loadA2, 4)
                UNROLL_OUT(loadA3, 4)
#undef loadB
#undef loadC
#undef loadA0
#undef loadA1
#undef loadA2
#undef loadA3
#define calculate_row0(i) c0 = vmlaq_f32(c0, b##i, a0##i);
#define calculate_row1(i) c1 = vmlaq_f32(c1, b##i, a1##i);
#define calculate_row2(i) c2 = vmlaq_f32(c2, b##i, a2##i);
#define calculate_row3(i) c3 = vmlaq_f32(c3, b##i, a3##i);
                UNROLL_OUT(calculate_row0, 4)
                UNROLL_OUT(calculate_row1, 4)
                UNROLL_OUT(calculate_row2, 4)
                UNROLL_OUT(calculate_row3, 4)
#undef calculate_row0
#undef calculate_row1
#undef calculate_row2
#undef calculate_row3
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x4_t a0, a1, a2, a3;
                float32x2_t b0, b1, b2, b3;
                float32x2_t c0, c1, c2, c3;
#define loadA(i) a##i = vld1q_f32(A + (m + i) * Astride + k);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadA, 4)
                UNROLL_OUT(loadB, 4)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_lane_f32(c##i, b0, vget_low_f32(a##i), 0);
#define calculateB1(i) c##i = vmla_lane_f32(c##i, b1, vget_low_f32(a##i), 1);
#define calculateB2(i) c##i = vmla_lane_f32(c##i, b2, vget_high_f32(a##i), 0);
#define calculateB3(i) c##i = vmla_lane_f32(c##i, b3, vget_high_f32(a##i), 1);
                UNROLL_OUT(calculateB0, 4)
                UNROLL_OUT(calculateB1, 4)
                UNROLL_OUT(calculateB2, 4)
                UNROLL_OUT(calculateB3, 4)
#undef calculateB0
#undef calculateB1
#undef calculateB2
#undef calculateB3
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22,
                        a23, a30, a31, a32, a33;
                float b0, b1, b2, b3;
                float c0, c1, c2, c3;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[m * Astride + k + i];
#define loadA1(i) a1##i = A[(m + 1) * Astride + k + i];
#define loadA2(i) a2##i = A[(m + 2) * Astride + k + i];
#define loadA3(i) a3##i = A[(m + 3) * Astride + k + i];
                UNROLL_OUT(loadA0, 4)
                UNROLL_OUT(loadA1, 4)
                UNROLL_OUT(loadA2, 4)
                UNROLL_OUT(loadA3, 4)
#undef loadA0
#undef loadA1
#undef loadA2
#undef loadA3
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
            for (; n + 4 <= N; n += 4) {
                float32x2_t a0, a1, a2, a3;
                float32x4_t b0, b1;
                float32x4_t c0, c1, c2, c3;
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
#define loadA(i) a##i = vld1_f32(A + (m + i) * Astride + k);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadA, 4)
                UNROLL_OUT(loadB, 2)
#undef loadA
#undef loadC
#undef loadB
#define calculateB0(i) c##i = vmlaq_lane_f32(c##i, b0, a##i, 0);
#define calculateB1(i) c##i = vmlaq_lane_f32(c##i, b1, a##i, 1);
                UNROLL_OUT(calculateB0, 4)
                UNROLL_OUT(calculateB1, 4)
#undef calculateB0
#undef calculateB1
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x2_t a0, a1, a2, a3;
                float32x2_t b0, b1;
                float32x2_t c0, c1, c2, c3;
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vld1_f32(A + (m + i) * Astride + k);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadA, 4)
                UNROLL_OUT(loadB, 2)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_lane_f32(c##i, b0, a##i, 0);
#define calculateB1(i) c##i = vmla_lane_f32(c##i, b1, a##i, 1);
                UNROLL_OUT(calculateB0, 4)
                UNROLL_OUT(calculateB1, 4)
#undef calculateB0
#undef calculateB1
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a01, a10, a11, a20, a21, a30, a31;
                float b0, b1;
                float c0, c1, c2, c3;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[(m + 0) * Astride + k + i];
#define loadA1(i) a1##i = A[(m + 1) * Astride + k + i];
#define loadA2(i) a2##i = A[(m + 2) * Astride + k + i];
#define loadA3(i) a3##i = A[(m + 3) * Astride + k + i];
                UNROLL_OUT(loadA0, 2)
                UNROLL_OUT(loadA1, 2)
                UNROLL_OUT(loadA2, 2)
                UNROLL_OUT(loadA3, 2)
#undef loadA0
#undef loadA1
#undef loadA2
#undef loadA3
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
            for (; n + 4 <= N; n += 4) {
                float32x4_t a0, a1, a2, a3;
                float32x4_t b0;
                float32x4_t c0, c1, c2, c3;
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vdupq_n_f32(A[(m + i) * Astride + k]);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadA, 4)
                UNROLL_OUT(loadB, 1)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmlaq_f32(c##i, a##i, b0);
                UNROLL_OUT(calculateB0, 4)
#undef calculateB0
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x2_t a0, a1, a2, a3;
                float32x2_t b0;
                float32x2_t c0, c1, c2, c3;
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vdup_n_f32(A[(m + i) * Astride + k]);
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadA, 4)
                UNROLL_OUT(loadB, 1)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_f32(c##i, a##i, b0);
                UNROLL_OUT(calculateB0, 4)
#undef calculateB0
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 4)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a10, a20, a30;
                float b0;
                float c0, c1, c2, c3;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 4)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[(m + 0) * Astride + k + i];
#define loadA1(i) a1##i = A[(m + 1) * Astride + k + i];
#define loadA2(i) a2##i = A[(m + 2) * Astride + k + i];
#define loadA3(i) a3##i = A[(m + 3) * Astride + k + i];
                UNROLL_OUT(loadA0, 1)
                UNROLL_OUT(loadA1, 1)
                UNROLL_OUT(loadA2, 1)
                UNROLL_OUT(loadA3, 1)
#undef loadA0
#undef loadA1
#undef loadA2
#undef loadA3
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
        memset(C + m * Cstride, 0, 2 * sizeof(float) * N);
        for (; k + 4 <= K; k += 4) {
            size_t n = 0;
            for (; n + 4 <= N; n += 4) {
                float32x4_t a00, a01, a02, a03, a10, a11, a12, a13;
                float32x4_t b0, b1, b2, b3;
                float32x4_t c0, c1;
#define loadA0(i) a0##i = vdupq_n_f32(A[(m + 0) * Astride + k + i]);
#define loadA1(i) a1##i = vdupq_n_f32(A[(m + 1) * Astride + k + i]);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 4)
                UNROLL_OUT(loadA0, 4)
                UNROLL_OUT(loadA1, 4)
#undef loadB
#undef loadC
#undef loadA0
#undef loadA1
#define calculate_row0(i) c0 = vmlaq_f32(c0, b##i, a0##i);
#define calculate_row1(i) c1 = vmlaq_f32(c1, b##i, a1##i);
                UNROLL_OUT(calculate_row0, 4)
                UNROLL_OUT(calculate_row1, 4)
#undef calculate_row0
#undef calculate_row1
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x4_t a0, a1;
                float32x2_t b0, b1, b2, b3;
                float32x2_t c0, c1;
#define loadA(i) a##i = vld1q_f32(A + (m + i) * Astride + k);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadA, 2)
                UNROLL_OUT(loadB, 4)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_lane_f32(c##i, b0, vget_low_f32(a##i), 0);
#define calculateB1(i) c##i = vmla_lane_f32(c##i, b1, vget_low_f32(a##i), 1);
#define calculateB2(i) c##i = vmla_lane_f32(c##i, b2, vget_high_f32(a##i), 0);
#define calculateB3(i) c##i = vmla_lane_f32(c##i, b3, vget_high_f32(a##i), 1);
                UNROLL_OUT(calculateB0, 2)
                UNROLL_OUT(calculateB1, 2)
                UNROLL_OUT(calculateB2, 2)
                UNROLL_OUT(calculateB3, 2)
#undef calculateB0
#undef calculateB1
#undef calculateB2
#undef calculateB3
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a01, a02, a03, a10, a11, a12, a13;
                float b0, b1, b2, b3;
                float c0, c1;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[m * Astride + k + i];
#define loadA1(i) a1##i = A[(m + 1) * Astride + k + i];
                UNROLL_OUT(loadA0, 4)
                UNROLL_OUT(loadA1, 4)
#undef loadA0
#undef loadA1
                c0 += a00 * b0 + a01 * b1 + a02 * b2 + a03 * b3;
                c1 += a10 * b0 + a11 * b1 + a12 * b2 + a13 * b3;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
        }
        for (; k + 2 <= K; k += 2) {
            size_t n = 0;
            for (; n + 4 <= N; n += 4) {
                float32x2_t a0, a1;
                float32x4_t b0, b1;
                float32x4_t c0, c1;
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
#define loadA(i) a##i = vld1_f32(A + (m + i) * Astride + k);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadA, 2)
                UNROLL_OUT(loadB, 2)
#undef loadA
#undef loadC
#undef loadB
#define calculateB0(i) c##i = vmlaq_lane_f32(c##i, b0, a##i, 0);
#define calculateB1(i) c##i = vmlaq_lane_f32(c##i, b1, a##i, 1);
                UNROLL_OUT(calculateB0, 2)
                UNROLL_OUT(calculateB1, 2)
#undef calculateB0
#undef calculateB1
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x2_t a0, a1;
                float32x2_t b0, b1;
                float32x2_t c0, c1;
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vld1_f32(A + (m + i) * Astride + k);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadA, 2)
                UNROLL_OUT(loadB, 2)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_lane_f32(c##i, b0, a##i, 0);
#define calculateB1(i) c##i = vmla_lane_f32(c##i, b1, a##i, 1);
                UNROLL_OUT(calculateB0, 2)
                UNROLL_OUT(calculateB1, 2)
#undef calculateB0
#undef calculateB1
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a01, a10, a11;
                float b0, b1;
                float c0, c1;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[(m + 0) * Astride + k + i];
#define loadA1(i) a1##i = A[(m + 1) * Astride + k + i];
                UNROLL_OUT(loadA0, 2)
                UNROLL_OUT(loadA1, 2)
#undef loadA0
#undef loadA1
                c0 += a00 * b0 + a01 * b1;
                c1 += a10 * b0 + a11 * b1;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
        }
        for (; k < K; k += 1) {
            size_t n = 0;
            for (; n + 4 <= N; n += 4) {
                float32x4_t a0, a1;
                float32x4_t b0;
                float32x4_t c0, c1;
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vdupq_n_f32(A[(m + i) * Astride + k]);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadA, 2)
                UNROLL_OUT(loadB, 1)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmlaq_f32(c##i, a##i, b0);
                UNROLL_OUT(calculateB0, 2)
#undef calculateB0
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x2_t a0, a1;
                float32x2_t b0;
                float32x2_t c0, c1;
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vdup_n_f32(A[(m + i) * Astride + k]);
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadA, 2)
                UNROLL_OUT(loadB, 1)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_f32(c##i, a##i, b0);
                UNROLL_OUT(calculateB0, 2)
#undef calculateB0
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 2)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a10;
                float b0;
                float c0, c1;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 2)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[(m + 0) * Astride + k + i];
#define loadA1(i) a1##i = A[(m + 1) * Astride + k + i];
                UNROLL_OUT(loadA0, 1)
                UNROLL_OUT(loadA1, 1)
#undef loadA0
#undef loadA1
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
        memset(C + m * Cstride, 0, sizeof(float) * N);
        for (; k + 4 <= K; k += 4) {
            size_t n = 0;
            for (; n + 4 <= N; n += 4) {
                float32x4_t a00, a01, a02, a03;
                float32x4_t b0, b1, b2, b3;
                float32x4_t c0;
#define loadA0(i) a0##i = vdupq_n_f32(A[m * Astride + k + i]);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadB, 4)
                UNROLL_OUT(loadA0, 4)
#undef loadB
#undef loadC
#undef loadA0
#define calculate_row0(i) c0 = vmlaq_f32(c0, b##i, a0##i);
                UNROLL_OUT(calculate_row0, 4)
#undef calculate_row0
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x4_t a0;
                float32x2_t b0, b1, b2, b3;
                float32x2_t c0;
#define loadA(i) a##i = vld1q_f32(A + (m + i) * Astride + k);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadA, 1)
                UNROLL_OUT(loadB, 4)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_lane_f32(c##i, b0, vget_low_f32(a##i), 0);
#define calculateB1(i) c##i = vmla_lane_f32(c##i, b1, vget_low_f32(a##i), 1);
#define calculateB2(i) c##i = vmla_lane_f32(c##i, b2, vget_high_f32(a##i), 0);
#define calculateB3(i) c##i = vmla_lane_f32(c##i, b3, vget_high_f32(a##i), 1);
                UNROLL_OUT(calculateB0, 1)
                UNROLL_OUT(calculateB1, 1)
                UNROLL_OUT(calculateB2, 1)
                UNROLL_OUT(calculateB3, 1)
#undef calculateB0
#undef calculateB1
#undef calculateB2
#undef calculateB3
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a01, a02, a03;
                float b0, b1, b2, b3;
                float c0;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadB, 4)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[m * Astride + k + i];
                UNROLL_OUT(loadA0, 4)
#undef loadA0
                c0 += a00 * b0 + a01 * b1 + a02 * b2 + a03 * b3;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
        }
        for (; k + 2 <= K; k += 2) {
            size_t n = 0;
            for (; n + 4 <= N; n += 4) {
                float32x2_t a0;
                float32x4_t b0, b1;
                float32x4_t c0;
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
#define loadA(i) a##i = vld1_f32(A + (m + i) * Astride + k);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadA, 1)
                UNROLL_OUT(loadB, 2)
#undef loadA
#undef loadC
#undef loadB
#define calculateB0(i) c##i = vmlaq_lane_f32(c##i, b0, a##i, 0);
#define calculateB1(i) c##i = vmlaq_lane_f32(c##i, b1, a##i, 1);
                UNROLL_OUT(calculateB0, 1)
                UNROLL_OUT(calculateB1, 1)
#undef calculateB0
#undef calculateB1
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x2_t a0;
                float32x2_t b0, b1;
                float32x2_t c0;
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vld1_f32(A + (m + i) * Astride + k);
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadA, 1)
                UNROLL_OUT(loadB, 2)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_lane_f32(c##i, b0, a##i, 0);
#define calculateB1(i) c##i = vmla_lane_f32(c##i, b1, a##i, 1);
                UNROLL_OUT(calculateB0, 1)
                UNROLL_OUT(calculateB1, 1)
#undef calculateB0
#undef calculateB1
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00, a01;
                float b0, b1;
                float c0;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadB, 2)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[(m + 0) * Astride + k + i];
                UNROLL_OUT(loadA0, 2)
#undef loadA0
                c0 += a00 * b0 + a01 * b1;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
        }
        for (; k < K; k += 1) {
            size_t n = 0;
            for (; n + 4 <= N; n += 4) {
                float32x4_t a0;
                float32x4_t b0;
                float32x4_t c0;
#define loadC(i) c##i = vld1q_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1q_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vdupq_n_f32(A[(m + i) * Astride + k]);
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadA, 1)
                UNROLL_OUT(loadB, 1)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmlaq_f32(c##i, a##i, b0);
                UNROLL_OUT(calculateB0, 1)
#undef calculateB0
#define vstore(i) vst1q_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
            for (; n + 2 <= N; n += 2) {
                float32x2_t a0;
                float32x2_t b0;
                float32x2_t c0;
#define loadC(i) c##i = vld1_f32(C + (m + i) * Cstride + n);
#define loadB(i) b##i = vld1_f32(B + (k + i) * Bstride + n);
#define loadA(i) a##i = vdup_n_f32(A[(m + i) * Astride + k]);
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadA, 1)
                UNROLL_OUT(loadB, 1)
#undef loadA
#undef loadB
#undef loadC
#define calculateB0(i) c##i = vmla_f32(c##i, a##i, b0);
                UNROLL_OUT(calculateB0, 1)
#undef calculateB0
#define vstore(i) vst1_f32(C + (m + i) * Cstride + n, c##i);
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
            for (; n < N; n += 1) {
                float a00;
                float b0;
                float c0;
#define loadC(i) c##i = C[(m + i) * Cstride + n];
#define loadB(i) b##i = B[(k + i) * Bstride + n];
                UNROLL_OUT(loadC, 1)
                UNROLL_OUT(loadB, 1)
#undef loadB
#undef loadC
#define loadA0(i) a0##i = A[(m + 0) * Astride + k + i];
                UNROLL_OUT(loadA0, 1)
#undef loadA0
                c0 = c0 + a00 * b0;
#define vstore(i) C[(m + i) * Cstride + n] = c##i;
                UNROLL_OUT(vstore, 1)
#undef vstore
            }
        }
    }
}

void sgemv_naive_n_mk4(const float* __restrict A, const float* __restrict B,
                       float* __restrict C, size_t M, size_t N, size_t K,
                       size_t Astride, size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(N == 1 && Bstride == PACK_SIZE && M % PACK_SIZE == 0 &&
                  K % PACK_SIZE == 0);
    auto Aptr = A;
    auto Cptr = C;
    size_t m = 0;
    while (m < M) {
        auto Aptr0 = Aptr;
        auto Cptr0 = Cptr;
        float32x4_t c[4];
#define INIT(step) c[step] = vdupq_n_f32(0.0f);
        UNROLL_CALL_RAW(4, INIT)
#undef INIT
        auto Bptr = B;
        size_t k = 0;
        while (k < K) {
            float32x4_t b = vld1q_f32(Bptr);
            float32x4x2_t a[2];
#define LOAD_A(step) a[step] = vld1q_f32_x2(Aptr0 + step * 8);
            UNROLL_CALL_RAW(2, LOAD_A)
#undef LOAD_A

#define COMPT(step) \
    c[step] = vfmaq_laneq_f32(c[step], a[step / 2].val[step % 2], b, step % 4);
            UNROLL_CALL_RAW(4, COMPT)
#undef COMPT
            Bptr += Bstride;
            Aptr0 += PACK_SIZE * PACK_SIZE;
            k += PACK_SIZE;
        }

#define ADD_C(step, stride) c[step] = vaddq_f32(c[step], c[step + stride]);
        UNROLL_CALL_RAW(2, ADD_C, 2)
        UNROLL_CALL_RAW(1, ADD_C, 1)
#undef ADD_C
        vst1q_f32(Cptr0, c[0]);

        Aptr += Astride;
        Cptr += Cstride;
        m += PACK_SIZE;
    }
}

}  // namespace

namespace megdnn {
namespace arm_common {

void gemv_like(const float* __restrict A, const float* __restrict B,
               float* __restrict C, size_t M, size_t N, size_t K,
               size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(M < 8 || (M == 8 && K <= 2) || (N == 1 && Bstride == 1));
    if (N == 1) {
        MIDOUT_BEGIN(megdnn_fp32_sgemv, midout_iv("F32_GEMV_NCHW_N"_hash)) {
            return sgemv_naive_n(A, B, C, M, N, K, Astride, Bstride, Cstride);
        }
        MIDOUT_END();
    } else {
        MIDOUT_BEGIN(megdnn_fp32_sgemv, midout_iv("F32_GEMV_NCHW_M"_hash)) {
            return sgemv_naive_m(A, B, C, M, N, K, Astride, Bstride, Cstride);
        }
        MIDOUT_END();
    }
}

void gemv_like_mk4(const float* __restrict A, const float* __restrict B,
                   float* __restrict C, size_t M, size_t N, size_t K,
                   size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 4);
    MIDOUT_BEGIN(megdnn_fp32_sgemv, midout_iv("F32_GEMV_NCHW44_N"_hash)) {
        return sgemv_naive_n_mk4(A, B, C, M, N, K, Astride, Bstride, Cstride);
    }
    MIDOUT_END();
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
