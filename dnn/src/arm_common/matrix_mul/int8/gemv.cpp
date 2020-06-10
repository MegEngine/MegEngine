/**
 * \file dnn/src/arm_common/matrix_mul/int8/gemv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cstddef>
#include "src/arm_common/matrix_mul/int8/gemv.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "megdnn/oprs.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_int8_gemv)

using namespace megdnn;
using namespace arm_common;

#if !__ARM_FEATURE_DOTPROD

namespace {

void gemv_naive_n(const int8_t* __restrict A, const int8_t* __restrict B,
                  int32_t* __restrict C, size_t M, size_t N, size_t K,
                  size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 1);
    size_t m = 0;
    for (; m + 2 <= M; m += 2) {
        int32_t acc0 = 0, acc1 = 0;
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(A + m * Astride + k);
            int8x16_t a1 = vld1q_s8(A + (m + 1) * Astride + k);

            int8x16_t b0 = vld1q_s8(B + k);

            int16x8_t c0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
            c0 = vmlal_high_s8(c0, a0, b0);

            int16x8_t c1 = vmull_s8(vget_low_s8(a1), vget_low_s8(b0));
            c1 = vmlal_high_s8(c1, a1, b0);
            acc0 += vaddlvq_s16(c0);
            acc1 += vaddlvq_s16(c1);
        }

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t a1 = vld1_s8(A + (m + 1) * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);

            int16x8_t c0 = vmull_s8(a0, b0);

            int16x8_t c1 = vmull_s8(a1, b0);
            acc0 += vaddlvq_s16(c0);
            acc1 += vaddlvq_s16(c1);
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(A[m * Astride + k]) * B[k];
            acc1 += static_cast<int32_t>(A[(m + 1) * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc0;
        C[(m + 1) * Cstride] = acc1;
    }

    for (; m < M; ++m) {
        int32_t acc0 = 0;
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(A + m * Astride + k);
            int8x16_t b0 = vld1q_s8(B + k);

            int16x8_t c0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
            c0 = vmlal_high_s8(c0, a0, b0);

            acc0 += vaddlvq_s16(c0);
        }

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);

            int16x8_t c0 = vmull_s8(a0, b0);
            acc0 += vaddlvq_s16(c0);
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(A[m * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc0;
    }
}
}  // namespace
#endif

#if __ARM_FEATURE_DOTPROD
namespace {

void gemv_naive_n(const int8_t* __restrict A, const int8_t* __restrict B,
                  int32_t* __restrict C, size_t M, size_t N, size_t K,
                  size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 1);
    size_t m = 0;
    for (; m + 2 <= M; m += 2) {
        int32_t acc[4];
        int32x4_t acc_neon = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int64x2_t a0 = vreinterpretq_s64_s8(vld1q_s8(A + m * Astride + k));
            int64x2_t a1 =
                    vreinterpretq_s64_s8(vld1q_s8(A + (m + 1) * Astride + k));
            //! the first 8 elements is m, the last 8 elements is m + 1
            int8x16_t a2 = vreinterpretq_s8_s64(vzip1q_s64(a0, a1));
            int8x16_t a3 = vreinterpretq_s8_s64(vzip2q_s64(a0, a1));

            int64x2_t b0 = vreinterpretq_s64_s8(vld1q_s8(B + k));
            int8x16_t b2 = vreinterpretq_s8_s64(vzip1q_s64(b0, b0));
            int8x16_t b3 = vreinterpretq_s8_s64(vzip2q_s64(b0, b0));

            acc_neon = vdotq_s32(acc_neon, a2, b2);
            acc_neon = vdotq_s32(acc_neon, a3, b3);
        }
        vst1q_s32(acc, acc_neon);

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t a1 = vld1_s8(A + (m + 1) * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);
            uint32x2_t zero = vdup_n_s32(0);
            acc[0] += vaddv_s32(vdot_s32(zero, a0, b0));
            zero = vdup_n_s32(0);
            acc[3] += vaddv_s32(vdot_s32(zero, a1, b0));
        }

        for (; k < K; ++k) {
            acc[0] += static_cast<int32_t>(A[m * Astride + k]) * B[k];
            acc[3] += static_cast<int32_t>(A[(m + 1) * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc[0] + acc[1];
        C[(m + 1) * Cstride] = acc[2] + acc[3];
    }

    for (; m < M; ++m) {
        int32_t acc[4];
        int32x4_t acc_neon = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(A + m * Astride + k);
            int8x16_t b0 = vld1q_s8(B + k);
            acc_neon = vdotq_s32(acc_neon, a0, b0);
        }
        vst1q_s32(acc, acc_neon);

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);
            uint32x2_t zero = vdup_n_s32(0);
            acc[0] += vaddv_s32(vdot_s32(zero, a0, b0));
        }

        for (; k < K; ++k) {
            acc[0] += static_cast<int32_t>(A[m * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc[0] + acc[1] + acc[2] + acc[3];
    }
}
}  // namespace
#endif

bool matmul::is_gemv_like_preferred_int8(bool transposeA, bool transposeB,
                                         size_t M, size_t N, size_t K,
                                         size_t LDA, size_t LDB, size_t LDC) {
    MEGDNN_MARK_USED_VAR(LDA);
    MEGDNN_MARK_USED_VAR(LDB);
    MEGDNN_MARK_USED_VAR(LDC);
    MEGDNN_MARK_USED_VAR(M);
    MEGDNN_MARK_USED_VAR(K);
    if (transposeA)
        return false;
    if (transposeB)
        return false;

    return N == 1 && LDB == 1;
}

void matmul::gemv_like_int8(const int8_t* __restrict A,
                            const int8_t* __restrict B, int32_t* __restrict C,
                            size_t M, size_t N, size_t K, size_t Astride,
                            size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv) {
        return gemv_naive_n(A, B, C, M, N, K, Astride, Bstride, Cstride);
    } MIDOUT_END();
}


// vim: syntax=cpp.doxygen
