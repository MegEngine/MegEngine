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
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/matrix_mul/int8/gemv.h"
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

void gemv_naive_n_mk4(const int8_t* __restrict A, const int8_t* __restrict B,
                      int32_t* __restrict C, size_t M, size_t N, size_t K,
                      size_t Astride, size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(N == 1 && Bstride == 4);
    auto Aptr = A;
    size_t m = 0;
    for (; m < M; m += PACK_SIZE) {
        auto Bptr = B;
        auto Aptr0 = Aptr;
        int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16x4_t a = vld4q_s8(Aptr0);
            int8x16_t b = vld1q_s8(Bptr);
            int16x8_t c[4];

            c[0] = vmull_s8(vget_low_s8(a.val[0]), vget_low_s8(b));
            c[1] = vmull_s8(vget_low_s8(a.val[1]), vget_low_s8(b));
            c[2] = vmull_s8(vget_low_s8(a.val[2]), vget_low_s8(b));
            c[3] = vmull_s8(vget_low_s8(a.val[3]), vget_low_s8(b));

            c[0] = vmlal_high_s8(c[0], a.val[0], b);
            c[1] = vmlal_high_s8(c[1], a.val[1], b);
            c[2] = vmlal_high_s8(c[2], a.val[2], b);
            c[3] = vmlal_high_s8(c[3], a.val[3], b);

            acc0 += vaddlvq_s16(c[0]);
            acc1 += vaddlvq_s16(c[1]);
            acc2 += vaddlvq_s16(c[2]);
            acc3 += vaddlvq_s16(c[3]);

            Bptr += 16;
            Aptr0 += PACK_SIZE * 16;
        }

        for (; k + 8 <= K; k += 8) {
            int8x8x4_t a = vld4_s8(Aptr0);
            int8x8_t b = vld1_s8(Bptr);
            int16x8_t c[4];

            c[0] = vmull_s8(a.val[0], b);
            c[1] = vmull_s8(a.val[1], b);
            c[2] = vmull_s8(a.val[2], b);
            c[3] = vmull_s8(a.val[3], b);

            acc0 += vaddlvq_s16(c[0]);
            acc1 += vaddlvq_s16(c[1]);
            acc2 += vaddlvq_s16(c[2]);
            acc3 += vaddlvq_s16(c[3]);

            Bptr += 8;
            Aptr0 += PACK_SIZE * 8;
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(*(Aptr0 + 0)) * B[k];
            acc1 += static_cast<int32_t>(*(Aptr0 + 1)) * B[k];
            acc2 += static_cast<int32_t>(*(Aptr0 + 2)) * B[k];
            acc3 += static_cast<int32_t>(*(Aptr0 + 3)) * B[k];
            Aptr0 += 4;
        }

        C[0] = acc0;
        C[1] = acc1;
        C[2] = acc2;
        C[3] = acc3;

        Aptr += Astride;
        C += Cstride;
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

void gemv_naive_n_mk4(const int8_t* __restrict A, const int8_t* __restrict B,
                      int32_t* __restrict C, size_t M, size_t N, size_t K,
                      size_t Astride, size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(N == 1 && Bstride == 4);

    auto Aptr = A;
    size_t m = 0;
    for (; m < M; m += PACK_SIZE) {
        auto Bptr = B;
        auto Aptr0 = Aptr;
        int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        size_t k = 0;
        if (k + 16 <= K) {
            int32x4_t acc_neon[4];
            acc_neon[0] = vdupq_n_s32(0);
            acc_neon[1] = vdupq_n_s32(0);
            acc_neon[2] = vdupq_n_s32(0);
            acc_neon[3] = vdupq_n_s32(0);
            for (; k + 16 <= K; k += 16) {
                int8x16x4_t a = vld4q_s8(Aptr0);
                int8x16_t b = vld1q_s8(Bptr);

                acc_neon[0] = vdotq_s32(acc_neon[0], a.val[0], b);
                acc_neon[1] = vdotq_s32(acc_neon[1], a.val[1], b);
                acc_neon[2] = vdotq_s32(acc_neon[2], a.val[2], b);
                acc_neon[3] = vdotq_s32(acc_neon[3], a.val[3], b);

                Bptr += 16;
                Aptr0 += PACK_SIZE * 16;
            }
            acc0 = vaddvq_s32(acc_neon[0]);
            acc1 = vaddvq_s32(acc_neon[1]);
            acc2 = vaddvq_s32(acc_neon[2]);
            acc3 = vaddvq_s32(acc_neon[3]);
        }

        if (k + 8 <= K) {
            int32x2_t acc_neon[4];
            acc_neon[0] = vdup_n_s32(0);
            acc_neon[1] = vdup_n_s32(0);
            acc_neon[2] = vdup_n_s32(0);
            acc_neon[3] = vdup_n_s32(0);

            int8x8x4_t a = vld4_s8(Aptr0);
            int8x8_t b = vld1_s8(Bptr);
            acc_neon[0] = vdot_s32(acc_neon[0], a.val[0], b);
            acc_neon[1] = vdot_s32(acc_neon[1], a.val[1], b);
            acc_neon[2] = vdot_s32(acc_neon[2], a.val[2], b);
            acc_neon[3] = vdot_s32(acc_neon[3], a.val[3], b);

            Bptr += 8;
            Aptr0 += PACK_SIZE * 8;
            k += 8;

            acc0 += vaddv_s32(acc_neon[0]);
            acc1 += vaddv_s32(acc_neon[1]);
            acc2 += vaddv_s32(acc_neon[2]);
            acc3 += vaddv_s32(acc_neon[3]);
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(*(Aptr0 + 0)) * B[k];
            acc1 += static_cast<int32_t>(*(Aptr0 + 1)) * B[k];
            acc2 += static_cast<int32_t>(*(Aptr0 + 2)) * B[k];
            acc3 += static_cast<int32_t>(*(Aptr0 + 3)) * B[k];
            Aptr0 += 4;
        }

        C[0] = acc0;
        C[1] = acc1;
        C[2] = acc2;
        C[3] = acc3;

        Aptr += Astride;
        C += Cstride;
    }
}

void gemv_naive_n_mk4_dot(const int8_t* __restrict A,
                          const int8_t* __restrict B, int32_t* __restrict C,
                          size_t M, size_t N, size_t K, size_t Astride,
                          size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(N == 1 && Bstride == 4);

    auto Aptr = A;
    size_t m = 0;
    for (; m < M; m += PACK_SIZE) {
        auto Bptr = B;
        auto Aptr0 = Aptr;
        size_t k = 0;
        int32x4_t acc_neon;
        acc_neon = vdupq_n_s32(0);
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(Aptr0);
            int8x16_t a1 = vld1q_s8(Aptr0 + 16);
            int8x16_t a2 = vld1q_s8(Aptr0 + 32);
            int8x16_t a3 = vld1q_s8(Aptr0 + 48);
            int8x16_t b = vld1q_s8(Bptr);
            acc_neon = vdotq_laneq_s32(acc_neon, a0, b, 0);
            acc_neon = vdotq_laneq_s32(acc_neon, a1, b, 1);
            acc_neon = vdotq_laneq_s32(acc_neon, a2, b, 2);
            acc_neon = vdotq_laneq_s32(acc_neon, a3, b, 3);
            Bptr += 16;
            Aptr0 += PACK_SIZE * 16;
        }

        if (k + 8 <= K) {
            int8x16_t a0 = vld1q_s8(Aptr0);
            int8x16_t a1 = vld1q_s8(Aptr0 + 16);
            int8x8_t b = vld1_s8(Bptr);
            acc_neon = vdotq_lane_s32(acc_neon, a0, b, 0);
            acc_neon = vdotq_lane_s32(acc_neon, a1, b, 1);
            Bptr += 8;
            Aptr0 += PACK_SIZE * 8;
            k += 8;
        }

        if (k + 4 <= K) {
            int8x16_t a = vld1q_s8(Aptr0);
            int32_t tmp = *(reinterpret_cast<const int32_t*>(Bptr));
            int8x8_t b = vdup_n_s32(tmp);
            acc_neon = vdotq_lane_s32(acc_neon, a, b, 0);
        }

        vst1q_s32(C, acc_neon);
        Aptr += Astride;
        C += Cstride;
    }
}

}  // namespace
#endif

bool arm_common::is_gemv_like_preferred_int8(bool transposeA, bool transposeB,
                                             size_t M, size_t N, size_t K,
                                             size_t LDA, size_t LDB,
                                             size_t LDC) {
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

void arm_common::gemv_like(const int8_t* __restrict A,
                           const int8_t* __restrict B, int32_t* __restrict C,
                           size_t M, size_t N, size_t K, size_t Astride,
                           size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv,
                 midout_iv("INT8_gemv_like"_hash)) {
        return gemv_naive_n(A, B, C, M, N, K, Astride, Bstride, Cstride);
    }
    MIDOUT_END();
}

void arm_common::gemv_like_mk4(const int8_t* __restrict A,
                               const int8_t* __restrict B,
                               int32_t* __restrict C, size_t M, size_t N,
                               size_t K, size_t Astride, size_t Bstride,
                               size_t Cstride) {
    megdnn_assert(N == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv,
                 midout_iv("INT8_gemv_like_mk4"_hash)) {
        return gemv_naive_n_mk4(A, B, C, M, N, K, Astride, Bstride, Cstride);
    }
    MIDOUT_END();
}

#if __ARM_FEATURE_DOTPROD
void arm_common::gemv_like_mk4_dot(const int8_t* __restrict A,
                                   const int8_t* __restrict B,
                                   int32_t* __restrict C, size_t M, size_t N,
                                   size_t K, size_t Astride, size_t Bstride,
                                   size_t Cstride) {
    megdnn_assert(N == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv,
                 midout_iv("INT8_gemv_like_mk4_dot"_hash)) {
        return gemv_naive_n_mk4_dot(A, B, C, M, N, K, Astride, Bstride,
                                    Cstride);
    }
    MIDOUT_END();
}
#endif

// vim: syntax=cpp.doxygen
