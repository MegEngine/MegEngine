/**
 * \file dnn/src/armv7/matrix_mul/fp32/strategy_mk_4x12.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/matrix_mul/fp32/strategy.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

namespace {

// Overview of register layout:
//
// A 1x12 cell of Rhs is stored in 32bit in q1-q3
// A 4x1 cell of Lhs is stored in 132bit in q0
// A 4x12 block of accumulators is stored in 32bit in q4-q15.
//
//                +--------+--------+--------+
//                | q1[0-3]| q2[0-3]| q3[0-3]|
//           Rhs  +--------+--------+--------+
//
//                |        |        |        |
//
//    Lhs         |        |        |        |
//
//  +--+ - - - -  +--------+--------+--------+
//  |q0|          | q4[0-3]| q5[0-3]| q6[0-3]|
//  |q0|          | q7[0-3]| q8[0-3]| q9[0-3]|
//  |q0|          |q10[0-3]|q11[0-3]|q12[0-3]|
//  |q0|          |q13[0-3]|q14[0-3]|q15[0-3]|
//  +--+ - - - -  +--------+--------+--------+
//
//                        Accumulator
void kern_4x12(const float* packA, const float* packB, int K, float* output,
               int LDC, bool is_first_k) {
    MEGDNN_MARK_USED_VAR(LDC);
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    float* output0 = output;

    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "mov r1, %[output0]\n"
            "vld1.32 {d8-d11}, [r1]!\n"
            "vld1.32 {d12-d15}, [r1]!\n"
            "vld1.32 {d16-d19}, [r1]!\n"
            "vld1.32 {d20-d23}, [r1]!\n"
            "vld1.32 {d24-d27}, [r1]!\n"
            "vld1.32 {d28-d31}, [r1]!\n"

            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "b 2f\n"

            "1:\n"
            "veor.32 q4, q4, q4\n"
            "pld [%[output0]]\n"
            "veor.32 q5, q4, q4\n"
            "veor.32 q6, q4, q4\n"
            "veor.32 q7, q4, q4\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "veor.32 q8, q4, q4\n"
            "veor.32 q9, q4, q4\n"
            "veor.32 q10, q4, q4\n"
            "veor.32 q11, q4, q4\n"
            "vld1.32 {d4-d7}, [%[b_ptr]]!\n"
            "veor.32 q12, q4, q4\n"
            "veor.32 q13, q4, q4\n"
            "veor.32 q14, q4, q4\n"
            "veor.32 q15, q4, q4\n"

            "2: \n"
            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vmla.f32 q4, q0, d4[0]\n"
            "vmla.f32 q5, q0, d4[1]\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q0, d6[0]\n"
            "vmla.f32 q9, q0, d6[1]\n"
            "vmla.f32 q10, q0, d7[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q11, q0, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q12, q0, d4[0]\n"
            "vmla.f32 q13, q0, d4[1]\n"
            "vmla.f32 q14, q0, d5[0]\n"
            "vmla.f32 q15, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"

            "vmla.f32 q4,  q1, d6[0]\n"
            "subs %[K], %[K], #1\n"
            "vmla.f32 q5,  q1, d6[1]\n"
            "vmla.f32 q6, q1, d7[0]\n"
            "vmla.f32 q7, q1, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q1, d4[0]\n"
            "vmla.f32 q9, q1, d4[1]\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vmla.f32 q10, q1, d5[0]\n"
            "vmla.f32 q11, q1, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q12, q1, d6[0]\n"
            "vmla.f32 q13, q1, d6[1]\n"
            "vmla.f32 q14, q1, d7[0]\n"
            "vmla.f32 q15, q1, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "vmla.f32 q4,  q0, d4[0]\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q0, d6[0]\n"
            "vmla.f32 q9, q0, d6[1]\n"
            "vmla.f32 q10, q0, d7[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q11, q0, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q12, q0, d4[0]\n"
            "vmla.f32 q13, q0, d4[1]\n"
            "vmla.f32 q14, q0, d5[0]\n"
            "vmla.f32 q15, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"

            "vmla.f32 q4,  q1, d6[0]\n"
            "subs %[K], %[K], #1\n"
            "vmla.f32 q5,  q1, d6[1]\n"
            "vmla.f32 q6, q1, d7[0]\n"
            "vmla.f32 q7, q1, d7[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q1, d4[0]\n"
            "vmla.f32 q9, q1, d4[1]\n"
            "vst1.32 {d8-d11}, [%[output0]]!\n"
            "vmla.f32 q10, q1, d5[0]\n"
            "vmla.f32 q11, q1, d5[1]\n"
            "vst1.32 {d12-d15}, [%[output0]]!\n"
            "vmla.f32 q12, q1, d6[0]\n"
            "vmla.f32 q13, q1, d6[1]\n"
            "vst1.32 {d16-d19}, [%[output0]]!\n"
            "vmla.f32 q14, q1, d7[0]\n"
            "vmla.f32 q15, q1, d7[1]\n"
            "vst1.32 {d20-d23}, [%[output0]]!\n"
            "vst1.32 {d24-d27}, [%[output0]]!\n"
            "vst1.32 {d28-d31}, [%[output0]]!\n"

            "b 6f\n"

            // odd tail
            "5:\n"
            "vmla.f32 q4,  q0, d4[0]\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q8, q0, d6[0]\n"
            "vst1.32 {d8-d11}, [%[output0]]!\n"
            "vmla.f32 q9, q0, d6[1]\n"
            "vmla.f32 q10, q0, d7[0]\n"
            "vst1.32 {d12-d15}, [%[output0]]!\n"
            "vmla.f32 q11, q0, d7[1]\n"
            "vmla.f32 q12, q0, d4[0]\n"
            "vst1.32 {d16-d19}, [%[output0]]!\n"
            "vmla.f32 q13, q0, d4[1]\n"
            "vst1.32 {d20-d23}, [%[output0]]!\n"
            "vmla.f32 q14, q0, d5[0]\n"
            "vst1.32 {d24-d27}, [%[output0]]!\n"
            "vmla.f32 q15, q0, d5[1]\n"
            "vst1.32 {d28-d31}, [%[output0]]!\n"

            "6:\n"
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ is_first_k ] "+r"(is_first_k), [ oddk ] "+r"(oddk),
              [ output0 ] "+r"(output0)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
              "q11", "q12", "q13", "q14", "q15", "r1", "cc", "memory");
}




// Overview of register layout:
//
// A 2x4 cell of Rhs is stored in 32bit in v2 - v3
// A 4x2 cell of Lhs is stored in 32bit in v0 - v1
// A 4x4 block of accumulators is stored in 32bit in v4-v6
//
//                 +--------+
//                 | q2[0-3]|
//                 | q5[0-3]|
//           Rhs   +--------+
//
//                 |        |
//
//    Lhs          |        |
//
//  +--+   ---  -  +--------+
//  |q0|           | q8[0-3]|
//  |q0|           |q11[0-3]|
//  |q0|           |q14[0-3]|
//  |q0|           |q17[0-3]|
//  +--+   ---  -  +--------+
//
//                        Accumulator
void kern_4x4(const float* packA, const float* packB, int K, float* output,
              int LDC, bool is_first_k, int n_remain) {
    MEGDNN_MARK_USED_VAR(LDC);
    const float* a_ptr = packA;
    const float* b_ptr = packB;

    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

//clang-format off
#define LOAD_C                   \
    "cmp %[n_remain], #4\n"      \
    "blt 11f\n"                  \
    "vld1.32 {d8-d11}, [r1]!\n"  \
    "vld1.32 {d12-d15}, [r1]!\n" \
    "b 14f\n"                    \
    "11:\n"                      \
    "cmp %[n_remain], #3\n"      \
    "blt 12f\n"                  \
    "vld1.32 {d8-d11}, [r1]!\n"  \
    "vld1.32 {d12-d13}, [r1]!\n" \
    "b 14f\n"                    \
    "12:\n"                      \
    "cmp %[n_remain], #2\n"      \
    "blt 13f\n"                  \
    "vld1.32 {d8-d11}, [r1]\n"   \
    "b 14f\n"                    \
    "13:\n"                      \
    "vld1.32 {d8-d9}, [r1]\n"    \
    "14:\n"

#define STORE_C                         \
    "cmp %[n_remain], #4\n"             \
    "blt 21f\n"                         \
    "vst1.32 {d8-d11}, [%[output]]!\n"  \
    "vst1.32 {d12-d15}, [%[output]]!\n" \
    "b 24f\n"                           \
    "21:\n"                             \
    "cmp %[n_remain], #3\n"             \
    "blt 22f\n"                         \
    "vst1.32 {d8-d11}, [%[output]]!\n"  \
    "vst1.32 {d12-d13}, [%[output]]!\n" \
    "b 24f\n"                           \
    "22:\n"                             \
    "cmp %[n_remain], #2\n"             \
    "blt 23f\n"                         \
    "vst1.32 {d8-d11}, [%[output]]!\n"  \
    "b 24f\n"                           \
    "23:\n"                             \
    "vst1.32 {d8-d9}, [%[output]]!\n"   \
    "24:\n"
//clang-format on

    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "mov r1, %[output]\n" LOAD_C
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "b 2f\n"

            "1:\n"
            "veor.32 q4, q4, q4\n"
            "pld [%[output]]\n"
            "veor.32 q5, q4, q4\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "veor.32 q6, q4, q4\n"
            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "veor.32 q7, q4, q4\n"

            "2: \n"
            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vmla.f32 q4,  q0, d4[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"

            "vld1.32 {d4-d5}, [%[b_ptr]]!\n"
            "vmla.f32 q4,  q1, d6[0]\n"
            "subs %[K], %[K], #1\n"
            "vmla.f32 q5,  q1, d6[1]\n"
            "vld1.32 {d0-d1}, [%[a_ptr]]!\n"
            "vmla.f32 q6, q1, d7[0]\n"
            "vmla.f32 q7, q1, d7[1]\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "vmla.f32 q4,  q0, d4[0]\n"
            "vld1.32 {d2-d3}, [%[a_ptr]]!\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vld1.32 {d6-d7}, [%[b_ptr]]!\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"

            "vmla.f32 q4,  q1, d6[0]\n"
            "vmla.f32 q5,  q1, d6[1]\n"
            "vmla.f32 q6, q1, d7[0]\n"
            "vmla.f32 q7, q1, d7[1]\n"
            "b 6f\n"

            // odd tail
            "5:\n"
            "vmla.f32 q4,  q0, d4[0]\n"
            "vmla.f32 q5,  q0, d4[1]\n"
            "vmla.f32 q6, q0, d5[0]\n"
            "vmla.f32 q7, q0, d5[1]\n"

            "6:\n" STORE_C
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ is_first_k ] "+r"(is_first_k), [ oddk ] "+r"(oddk),
              [ output ] "+r"(output), [ n_remain ] "+r"(n_remain)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "r1", "cc",
              "memory");
#undef LOAD_C
#undef STORE_C
}

}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL(sgemm_mk4_pack_4x12);
//! Now no matmul mode of only packB support in conv1x1 and im2col, so just copy
//! the weight
void sgemm_mk4_pack_4x12::pack_A(float* out, const float* in, int ldin, int y0,
                                 int ymax, int k0, int kmax, bool) const {
    megdnn_assert(y0 % 4 == 0 && ymax % 4 == 0, "M must be time of 4");
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    constexpr int PACK_C_SIZE = 4;
    size_t cp_length = (kmax - k0) * PACK_C_SIZE;
    for (int m = y0; m < ymax; m += 4) {
        const float* src = in + (m / PACK_C_SIZE) * ldin + k0 * PACK_C_SIZE;
        memcpy(out, src, cp_length * sizeof(float));
        out += cp_length;
    }
}

void sgemm_mk4_pack_4x12::pack_B(float* out, const float* in, int ldin, int x0,
                                 int xmax, int k0, int kmax,
                                 bool transpose_B) const {
    megdnn_assert(!transpose_B);
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    float tmpbuff[16] = {0.0f};

    constexpr int PACK_C_SIZE = 4;
    int ksize = kmax - k0;
    int ksize12 = ksize * 12;
    int ksize4 = (ksize << 2);
    float* outptr_base = out;
    float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr = in + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;
        prefetch_3x(inptr);

        int x = x0;
        auto outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            auto outptr_interleave = outptr;
            transpose_1x12_4_s(inptr, outptr_interleave);
            outptr += ksize12;
        }
        outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            transpose_1x4_4_s(inptr, outptr_interleave);
            outptr += ksize4;
        }
        if (x < xmax) {
            memcpy(tmpbuff, inptr, sizeof(float) * (xmax - x) * PACK_C_SIZE);
            auto outptr_interleave = outptr;
            const float* tmp_ptr = &tmpbuff[0];
            transpose_1x4_4_s<float>(tmp_ptr, outptr_interleave);
            outptr += ksize4;
        }
        outptr_base += 12 * PACK_C_SIZE;
        outptr_base4 += 4 * PACK_C_SIZE;
    }
}

void sgemm_mk4_pack_4x12::kern(const float* packA, const float* packB, size_t M,
                               size_t N, size_t K, float* C, size_t LDC,
                               bool is_first_k, const float*, float*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == C_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Float32);
    constexpr int PACK_C_SIZE = 4;
    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 12;
    const int K12 = K * 12;
    const int K4 = K * 4;
    size_t m = 0;
    for (; m < M; m += A_INTERLEAVE) {
        float* output = C + (m / 4 * LDC);

        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x12(packA, cur_packB, K, output, LDC, is_first_k);
            output += PACK_C_SIZE * B_INTERLEAVE;
            cur_packB += K12;
        }
        for (; n < N; n += 4) {
            kern_4x4(packA, cur_packB, K, output, LDC, is_first_k,
                     std::min<size_t>(N - n, 4));
            output += PACK_C_SIZE * 4;
            cur_packB += K4;
        }
        packA += K4;
    }
}

// vim: syntax=cpp.doxygen
