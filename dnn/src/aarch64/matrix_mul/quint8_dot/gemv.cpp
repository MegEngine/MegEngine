/**
 * \file dnn/src/aarch64/matrix_mul/quint8_dot/gemv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/matrix_mul/quint8_dot/gemv.h"
#include <cstddef>
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/common/unroll_macro.h"

#if __ARM_FEATURE_DOTPROD

namespace {

void gemv_naive_n(const uint8_t* __restrict A, const uint8_t* __restrict B,
                  int32_t* __restrict C, size_t M, size_t N, size_t K,
                  size_t Astride, size_t Bstride, size_t Cstride,
                  uint8_t zero_point_A, uint8_t zero_point_B) {
    int32_t zAB = static_cast<int32_t>(zero_point_A) *
                  static_cast<int32_t>(zero_point_B) * K;
    uint8x16_t zAq = vdupq_n_u8(zero_point_A);
    uint8x16_t zBq = vdupq_n_u8(zero_point_B);
    uint8x8_t zA = vdup_n_u8(zero_point_A);
    uint8x8_t zB = vdup_n_u8(zero_point_B);
    megdnn_assert(N == 1 && Bstride == 1);
    size_t m = 0;
    for (; m + 2 <= M; m += 2) {
        int32_t acc_zA, acc_zB, acc_zB2;
        int32_t acc[4];
        size_t k = 0;
        uint32x4_t acc_neon = vdupq_n_u32(0);
        {
            uint32x4_t acc_zA_neon = vdupq_n_u32(0);
            uint32x4_t acc_zB_neon = vdupq_n_u32(0);
            uint32x4_t acc_zB2_neon = vdupq_n_u32(0);
            for (; k + 16 <= K; k += 16) {
                uint8x16_t elem = vld1q_u8(A + m * Astride + k);
                acc_zB_neon = vdotq_u32(acc_zB_neon, zBq, elem);
                uint64x2_t a0 = vreinterpretq_u64_u8(elem);
                elem = vld1q_u8(A + (m + 1) * Astride + k);
                acc_zB2_neon = vdotq_u32(acc_zB2_neon, zBq, elem);
                uint64x2_t a1 = vreinterpretq_u64_u8(elem);
                //! the first 8 elements is m, the last 8 elements is m + 1
                uint8x16_t a2 = vreinterpretq_u8_u64(vzip1q_u64(a0, a1));
                uint8x16_t a3 = vreinterpretq_u8_u64(vzip2q_u64(a0, a1));

                elem = vld1q_u8(B + k);
                acc_zA_neon = vdotq_u32(acc_zA_neon, zAq, elem);
                uint64x2_t b0 = vreinterpretq_u64_u8(elem);
                uint8x16_t b2 = vreinterpretq_u8_u64(vzip1q_u64(b0, b0));
                uint8x16_t b3 = vreinterpretq_u8_u64(vzip2q_u64(b0, b0));

                acc_neon = vdotq_u32(acc_neon, a2, b2);
                acc_neon = vdotq_u32(acc_neon, a3, b3);
            }
            vst1q_s32(acc, vreinterpretq_s32_u32(acc_neon));
            acc_zA = vaddvq_u32(acc_zA_neon);
            acc_zB = vaddvq_u32(acc_zB_neon);
            acc_zB2 = vaddvq_u32(acc_zB2_neon);
        }

        {
            uint32x2_t acc_zA_neon = vdup_n_u32(0);
            uint32x2_t acc_zB_neon = vdup_n_u32(0);
            uint32x2_t acc_zB2_neon = vdup_n_u32(0);
            for (; k + 8 <= K; k += 8) {
                uint8x8_t a0 = vld1_u8(A + m * Astride + k);
                uint8x8_t a1 = vld1_u8(A + (m + 1) * Astride + k);
                uint8x8_t b0 = vld1_u8(B + k);
                uint32x2_t zero = vdup_n_u32(0);
                acc[0] += vaddv_u32(vdot_u32(zero, a0, b0));
                zero = vdup_n_u32(0);
                acc[3] += vaddv_u32(vdot_u32(zero, a1, b0));

                acc_zB_neon = vdot_u32(acc_zB_neon, a0, zB);
                acc_zB2_neon = vdot_u32(acc_zB2_neon, a1, zB);
                acc_zA_neon = vdot_u32(acc_zA_neon, b0, zA);
            }

            acc_zA += vaddv_u32(acc_zA_neon);
            acc_zB += vaddv_u32(acc_zB_neon);
            acc_zB2 += vaddv_u32(acc_zB2_neon);
        }

        for (; k < K; ++k) {
            acc[0] += static_cast<int32_t>(A[m * Astride + k]) * B[k];
            acc[3] += static_cast<int32_t>(A[(m + 1) * Astride + k]) * B[k];
            acc_zA += static_cast<int32_t>(B[k]) * zero_point_A;
            acc_zB += static_cast<int32_t>(A[m * Astride + k]) * zero_point_B;
            acc_zB2 += static_cast<int32_t>(A[(m + 1) * Astride + k]) *
                       zero_point_B;
        }
        C[m * Cstride] = acc[0] + acc[1] + zAB - acc_zA - acc_zB;
        C[(m + 1) * Cstride] = acc[2] + acc[3] + zAB - acc_zA - acc_zB2;
    }

    for (; m < M; ++m) {
        int32_t acc[4];
        int32_t acc_zA, acc_zB;
        uint32x4_t acc_neon = vdupq_n_u32(0);
        size_t k = 0;
        {
            uint32x4_t acc_zA_neon = vdupq_n_u32(0);
            uint32x4_t acc_zB_neon = vdupq_n_u32(0);
            for (; k + 16 <= K; k += 16) {
                uint8x16_t a0 = vld1q_u8(A + m * Astride + k);
                uint8x16_t b0 = vld1q_u8(B + k);
                acc_neon = vdotq_u32(acc_neon, a0, b0);
                acc_zB_neon = vdotq_u32(acc_zB_neon, zBq, a0);
                acc_zA_neon = vdotq_u32(acc_zA_neon, zAq, b0);
            }
            vst1q_s32(acc, vreinterpretq_s32_u32(acc_neon));
            acc_zA = vaddvq_u32(acc_zA_neon);
            acc_zB = vaddvq_u32(acc_zB_neon);
        }

        {
            uint32x2_t acc_zA_neon = vdup_n_u32(0);
            uint32x2_t acc_zB_neon = vdup_n_u32(0);
            for (; k + 8 <= K; k += 8) {
                uint8x8_t a0 = vld1_u8(A + m * Astride + k);
                uint8x8_t b0 = vld1_u8(B + k);
                uint32x2_t zero = vdup_n_u32(0);
                acc[0] += vaddv_u32(vdot_u32(zero, a0, b0));

                acc_zB_neon = vdot_u32(acc_zB_neon, a0, zB);
                acc_zA_neon = vdot_u32(acc_zA_neon, b0, zA);
            }
            acc_zA += vaddv_u32(acc_zA_neon);
            acc_zB += vaddv_u32(acc_zB_neon);
        }

        for (; k < K; ++k) {
            acc[0] += static_cast<int32_t>(A[m * Astride + k]) * B[k];
            acc_zA += static_cast<int32_t>(B[k]) * zero_point_A;
            acc_zB += static_cast<int32_t>(A[m * Astride + k]) * zero_point_B;
        }
        C[m * Cstride] =
                acc[0] + acc[1] + acc[2] + acc[3] + zAB - acc_zA - acc_zB;
    }
}

}  // namespace

bool megdnn::aarch64::matmul::is_gemv_like_preferred_quint8(
        bool transposeA, bool transposeB, size_t M, size_t N, size_t K,
        size_t /* LDA */, size_t LDB, size_t /* LDC */) {
    if (transposeA)
        return false;
    if (transposeB)
        return false;
    MEGDNN_MARK_USED_VAR(K);
    MEGDNN_MARK_USED_VAR(M);
    //! rebenchmark gemv in sdm855
    return (N == 1 && LDB == 1);
}

void megdnn::aarch64::matmul::gemv_like_quint8(
        const uint8_t* __restrict A, const uint8_t* __restrict B,
        int32_t* __restrict C, size_t M, size_t N, size_t K, size_t Astride,
        size_t Bstride, size_t Cstride, uint8_t zero_point_A,
        uint8_t zero_point_B) {
    megdnn_assert(N == 1);
    return gemv_naive_n(A, B, C, M, N, K, Astride, Bstride, Cstride,
                        zero_point_A, zero_point_B);
}

#endif

// vim: syntax=cpp.doxygen
