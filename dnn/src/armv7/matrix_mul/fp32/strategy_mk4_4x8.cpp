/**
 * \file dnn/src/armv7/matrix_mul/fp32/strategy_mk4_4x8.cpp
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

void kern_4x1(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB = (LDB - 4) * sizeof(float);
    asm volatile(
            "subs %[K], %[K], #4\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"
            "vld1.32 {d12-d15}, [%[A]]!\n"
            "veor    q8,     q8 \n"
            "veor    q9,     q9 \n"
            "veor    q10,    q10 \n"
            "veor    q11,    q11 \n"

            "vld1.32 {d0-d1}, [%[B]]!\n"

            "vmla.f32 q8, q4, d0[0]\n"
            "vmla.f32 q9, q5, d0[1]\n"

            "beq 2f\n"

            "1:\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"
            "vmla.f32 q10, q6, d1[0]\n"
            "vmla.f32 q11, q7, d1[1]\n"

            "add %[B], %[B], %[LDB]\n"
            "vld1.32 {d0-d1}, [%[B]]!\n"
            "vld1.32 {d12-d15}, [%[A]]!\n"

            "vmla.f32 q8, q4, d0[0]\n"
            "vmla.f32 q9, q5, d0[1]\n"

            "subs %[K], %[K], #4\n"
            "bne 1b\n"

            "2:\n"

            "vmla.f32 q10, q6, d1[0]\n"
            "vmla.f32 q11, q7, d1[1]\n"
            "vadd.f32 q8,  q8, q10\n"
            "vadd.f32 q9,  q9, q11\n"
            "vadd.f32 q8,  q8, q9\n"

            "vst1.32 {d16, d17}, [%[C]]!\n"

            : [ A ] "+r"(A), [ B ] "+r"(B), [ K ] "+r"(K), [ C ] "+r"(C)
            : [ LDB ] "r"(LDB)
            : "d0", "d1", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
              "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "cc",
              "memory");
}

// Overview of register layout:
//
// A 8x4 cell of Rhs is stored in 32bit in q0-q3, load 4 register each time
// A 4x4 cell of Lhs is stored in 32bit in q4-q7
// A 4x8 block of accumulators is stored in 32bit in q8-q11.
//
//                 +--------+
//                 | q0-q3  |
//           Rhs   +--------+
//
//                 |        |
//
//    Lhs          |        |
//
//  +---+ - - - -  +--------+
//  | q4|          | q8-11  |
//  | q5|          |        |
//  | q6|          |        |
//  | q7|          |        |
//  +---+ - - - -  +--------+
//
//                        Accumulator
void kern_4x4(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    //! as each load 16 number from B, and pos add 16 * 4, we should minus it
    //! before we add stride
    LDB = (LDB - 16) * sizeof(float);
    asm volatile(
            "subs %[K], %[K], #4\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"
            "vld1.32 {d12-d15}, [%[A]]!\n"

            "vld1.32 {d0-d3}, [%[B]]!\n"
            "vld1.32 {d4-d7}, [%[B]]!\n"

            "vmul.f32 q8, q4, d0[0]\n"
            "vmul.f32 q9, q4, d2[0]\n"
            "vmul.f32 q10, q4, d4[0]\n"
            "vmul.f32 q11, q4, d6[0]\n"

            "vmla.f32 q8, q5, d0[1]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmla.f32 q11, q5, d6[1]\n"

            "beq 2f\n"

            "1:\n"

            "vld1.32 {d8-d11}, [%[A]]!\n"

            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q6, d7[0]\n"

            "add %[B], %[B], %[LDB]\n"

            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vld1.32 {d0-d1}, [%[B]]!\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vld1.32 {d2-d3}, [%[B]]!\n"
            "vmla.f32 q11, q7, d7[1]\n"
            "vld1.32 {d4-d5}, [%[B]]!\n"

            "vmla.f32 q8, q4, d0[0]\n"
            "vld1.32 {d6-d7}, [%[B]]!\n"
            "vmla.f32 q9, q4, d2[0]\n"
            "vmla.f32 q10, q4, d4[0]\n"
            "vmla.f32 q11, q4, d6[0]\n"

            "vld1.32 {d12-d15}, [%[A]]!\n"

            "vmla.f32 q8, q5, d0[1]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmla.f32 q11, q5, d6[1]\n"

            "subs %[K], %[K], #4\n"
            "bne 1b\n"

            "2:\n"

            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q6, d7[0]\n"

            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vmla.f32 q11, q7, d7[1]\n"

            "vst1.32 {d16, d17, d18, d19}, [%[C]]!\n"
            "vst1.32 {d20, d21, d22, d23}, [%[C]]!\n"

            : [A] "+r"(A), [B] "+r"(B), [K] "+r"(K), [C] "+r"(C)
            : [LDB] "r"(LDB)
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "cc", "memory");
}

// Overview of register layout:
//
// A 8x4 cell of Rhs is stored in 32bit in v0-v3, load 4 register each time
// A 4x4 cell of Lhs is stored in 32bit in v4-v7
// A 4x8 block of accumulators is stored in 32bit in q8-q15.
//
//                 +--------+--------+
//                 | v0-v3  | v0-v3  |
//           Rhs   +--------+--------+
//
//                 |        |        |
//
//    Lhs          |        |        |
//
//  +---+ - - - -  +--------+--------+
//  | v4|          | v8-11  | v12-15 |
//  | v5|          |        |        |
//  | v6|          |        |        |
//  | v7|          |        |        |
//  +---+ - - - -  +--------+--------+
//
//                        Accumulator
void kern_4x8(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB *= sizeof(float);
    //! as each load 32 number from B, the pos add 32 * 4, we should minus it
    //! before we add stride
    LDB -= 32 * sizeof(float);
    asm volatile(
            "vld1.32 {d8, d9, d10, d11}, [%[A]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[A]]!\n"

            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmul.f32 q8, q4, d0[0]\n"
            "vmla.f32 q8, q5, d0[1]\n"
            "vmul.f32 q9, q4, d2[0]\n"
            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vmul.f32 q10, q4, d4[0]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmul.f32 q11, q4, d6[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q5, d6[1]\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vmla.f32 q11, q6, d7[0]\n"
            "vmla.f32 q11, q7, d7[1]\n"

            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmul.f32 q12, q4, d0[0]\n"
            "vmla.f32 q12, q5, d0[1]\n"
            "vmul.f32 q13, q4, d2[0]\n"
            "vmla.f32 q12, q6, d1[0]\n"
            "vmla.f32 q13, q5, d2[1]\n"
            "vmla.f32 q12, q7, d1[1]\n"
            "vmla.f32 q13, q6, d3[0]\n"
            "vmla.f32 q13, q7, d3[1]\n"
            "vmul.f32 q14, q4, d4[0]\n"
            "vmla.f32 q14, q5, d4[1]\n"
            "vmul.f32 q15, q4, d6[0]\n"
            "vmla.f32 q14, q6, d5[0]\n"
            "vmla.f32 q15, q5, d6[1]\n"
            "vmla.f32 q14, q7, d5[1]\n"
            "vmla.f32 q15, q6, d7[0]\n"
            "vmla.f32 q15, q7, d7[1]\n"

            "add %[B], %[B], %[LDB]\n"
            "subs %[K], %[K], #4\n"
            "cmp %[K], #0\n"
            "beq 2f\n"

            "1:\n"
            "vld1.32 {d8, d9, d10, d11}, [%[A]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[A]]!\n"

            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmla.f32 q8, q4, d0[0]\n"
            "vmla.f32 q8, q5, d0[1]\n"
            "vmla.f32 q9, q4, d2[0]\n"
            "vmla.f32 q8, q6, d1[0]\n"
            "vmla.f32 q9, q5, d2[1]\n"
            "vmla.f32 q8, q7, d1[1]\n"
            "vmla.f32 q9, q6, d3[0]\n"
            "vmla.f32 q9, q7, d3[1]\n"
            "vld1.32 {d0, d1, d2, d3}, [%[B]]!\n"
            "vmla.f32 q10, q4, d4[0]\n"
            "vmla.f32 q10, q5, d4[1]\n"
            "vmla.f32 q11, q4, d6[0]\n"
            "vmla.f32 q10, q6, d5[0]\n"
            "vmla.f32 q11, q5, d6[1]\n"
            "vmla.f32 q10, q7, d5[1]\n"
            "vmla.f32 q11, q6, d7[0]\n"
            "vmla.f32 q11, q7, d7[1]\n"

            "vld1.32 {d4, d5, d6, d7}, [%[B]]!\n"
            "vmla.f32 q12, q4, d0[0]\n"
            "vmla.f32 q12, q5, d0[1]\n"
            "vmla.f32 q13, q4, d2[0]\n"
            "vmla.f32 q12, q6, d1[0]\n"
            "vmla.f32 q13, q5, d2[1]\n"
            "vmla.f32 q12, q7, d1[1]\n"
            "vmla.f32 q13, q6, d3[0]\n"
            "vmla.f32 q13, q7, d3[1]\n"
            "vmla.f32 q14, q4, d4[0]\n"
            "vmla.f32 q14, q5, d4[1]\n"
            "vmla.f32 q15, q4, d6[0]\n"
            "vmla.f32 q14, q6, d5[0]\n"
            "vmla.f32 q15, q5, d6[1]\n"
            "vmla.f32 q14, q7, d5[1]\n"
            "vmla.f32 q15, q6, d7[0]\n"
            "vmla.f32 q15, q7, d7[1]\n"

            "add %[B], %[B], %[LDB]\n"
            "subs %[K], %[K], #4\n"
            "cmp %[K], #0\n"
            "bne 1b\n"
            "2:\n"
            "vst1.32 {d16, d17, d18, d19}, [%[C]]!\n"
            "vst1.32 {d20, d21, d22, d23}, [%[C]]!\n"
            "vst1.32 {d24, d25, d26, d27}, [%[C]]!\n"
            "vst1.32 {d28, d29, d30, d31}, [%[C]]!\n"
            : [A] "+r"(A), [B] "+r"(B), [K] "+r"(K), [C] "+r"(C)
            : [LDB] "r"(LDB)
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31", "cc", "memory");
}

}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(sgemm_nopack_4x8);

void sgemm_nopack_4x8::kern(const float* A, size_t LDA, const float* B,
                            size_t LDB, float* C, size_t LDC, size_t M,
                            size_t K, size_t N, const float*, void*, bool trA,
                            bool trB) const {
    constexpr size_t MB = 4;
    constexpr size_t KB = 4;
    constexpr size_t NB = 8;
    constexpr size_t NB_HALF = 4;

    megdnn_assert(!trA && !trB && M % MB == 0 && K % KB == 0);

    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)
    for (size_t m = 0; m < M; m += MB) {
        float* output = C + (m / MB) * LDC;
        const float* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_4x8(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_4x4(A, cur_B, LDB, K, output);
            cur_B += KB * NB_HALF;
            output += MB * NB_HALF;
            n += 4;
        }
        while (n < N) {
            kern_4x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    }
}

// vim: syntax=cpp.doxygen
