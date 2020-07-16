/**
 * \file dnn/src/armv7/matrix_mul/fp16/strategy_mk8_4x8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/matrix_mul/fp16/strategy.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

namespace {

void kern_8x1(const dt_float16* a_ptr, const dt_float16* b_ptr, int LDB, int K,
              dt_float16* output) {
    LDB = (LDB - 4) * sizeof(dt_float16);
    asm volatile(
            "subs %[K], #8\n"

            "vld1.32 {d0}, [%[b_ptr]]!\n"
            "vld1.32 {d1}, [%[b_ptr]], %[LDB]\n"
            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"

            "vmul.f16 q12, q4, d0[0]\n"
            "vmul.f16 q13, q5, d0[1]\n"
            "vmul.f16 q14, q6, d0[2]\n"
            "vmul.f16 q15, q7, d0[3]\n"

            "beq 2f\n"

            "1:\n"
            "vmla.f16 q12, q8, d1[0]\n"
            "vld1.32 {d0}, [%[b_ptr]]!\n"
            "vmla.f16 q13, q9, d1[1]\n"
            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"
            "vmla.f16 q14, q10, d1[2]\n"
            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vmla.f16 q15, q11, d1[3]\n"

            "vmla.f16 q12, q4, d0[0]\n"
            "vld1.32 {d1}, [%[b_ptr]], %[LDB]\n"
            "vmla.f16 q13, q5, d0[1]\n"
            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vmla.f16 q14, q6, d0[2]\n"
            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"
            "vmla.f16 q15, q7, d0[3]\n"

            "subs %[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "vmla.f16 q12, q8, d1[0]\n"
            "vmla.f16 q13, q9, d1[1]\n"
            "vmla.f16 q14, q10, d1[2]\n"
            "vmla.f16 q15, q11, d1[3]\n"

            "vadd.f16 q12, q12, q14\n"
            "vadd.f16 q13, q13, q15\n"
            "vadd.f16 q12, q12, q13\n"

            "vst1.32 {d24, d25}, [%[output]]!\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "d0", "d1", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
              "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
              "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc", "memory");
}

// Overview of register layout:
//
// A 8x1 cell of Rhs is stored in 16bit in v4-v11
// A 4x1 cell of Lhs is stored in 16bit in v0-v3
// A 4x1 block of accumulators is stored in 16bit in v12-v15.
//
//                  Rhs +--------+
//                      | v4[0-7]|
//                      | v5[0-7]|
//                      | v6[0-7]|
//                      | v7[0-7]|
//                      | v8[0-7]|
//                      | v9[0-7]|
//                      |v10[0-7]|
//                      |v11[0-7]|
//                      +--------+
//      Lhs
//  +--------+ - - - - -+--------+
//  | v0[0-7]|          |v12[0-7]|
//  | v1[0-7]|          |v13[0-7]|
//  | v2[0-7]|          |v14[0-7]|
//  | v3[0-7]|          |v15[0-7]|
//  +--------+          +--------+--------+
//                      Accumulator
void kern_8x4(const dt_float16* a_ptr, const dt_float16* b_ptr, int LDB, int K,
              dt_float16* output) {
    //! As each load 64 number from B, but the pos add 48 * 2, so we minus 48
    //! here.
    LDB = (LDB - 16) * sizeof(dt_float16);

    asm volatile(
            "subs %[K], #8\n"

            "vld1.32 {d0, d1, d2, d3}, [%[b_ptr]]!\n"
            "vld1.32 {d4, d5, d6, d7}, [%[b_ptr]], %[LDB]\n"
            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"

            "vmul.f16 q12, q4, d0[0]\n"
            "vmul.f16 q13, q4, d2[0]\n"
            "vmul.f16 q14, q4, d4[0]\n"
            "vmul.f16 q15, q4, d6[0]\n"

            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vmla.f16 q12, q5, d0[1]\n"
            "vmla.f16 q13, q5, d2[1]\n"
            "vmla.f16 q14, q5, d4[1]\n"
            "vmla.f16 q15, q5, d6[1]\n"

            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vmla.f16 q12, q6, d0[2]\n"
            "vmla.f16 q13, q6, d2[2]\n"
            "vmla.f16 q14, q6, d4[2]\n"
            "vmla.f16 q15, q6, d6[2]\n"

            "vmla.f16 q12, q7, d0[3]\n"
            "vmla.f16 q13, q7, d2[3]\n"
            "vmla.f16 q14, q7, d4[3]\n"
            "vmla.f16 q15, q7, d6[3]\n"

            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"
            "vmla.f16 q12, q8, d1[0]\n"
            "vmla.f16 q13, q8, d3[0]\n"
            "vmla.f16 q14, q8, d5[0]\n"
            "vmla.f16 q15, q8, d7[0]\n"

            "vmla.f16 q12, q9, d1[1]\n"
            "vmla.f16 q13, q9, d3[1]\n"
            "vmla.f16 q14, q9, d5[1]\n"
            "vmla.f16 q15, q9, d7[1]\n"

            "vmla.f16 q12, q10, d1[2]\n"
            "vmla.f16 q13, q10, d3[2]\n"
            "vmla.f16 q14, q10, d5[2]\n"
            "vmla.f16 q15, q10, d7[2]\n"

            "vmla.f16 q12, q11, d1[3]\n"
            "vmla.f16 q13, q11, d3[3]\n"
            "vmla.f16 q14, q11, d5[3]\n"
            "vmla.f16 q15, q11, d7[3]\n"

            "beq 2f\n"

            "1:\n"
            "vld1.32 {d0, d1, d2, d3}, [%[b_ptr]]!\n"
            "vld1.32 {d4, d5, d6, d7}, [%[b_ptr]], %[LDB]\n"
            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"

            "vmla.f16 q12, q4, d0[0]\n"
            "vmla.f16 q13, q4, d2[0]\n"
            "vmla.f16 q14, q4, d4[0]\n"
            "vmla.f16 q15, q4, d6[0]\n"

            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vmla.f16 q12, q5, d0[1]\n"
            "vmla.f16 q13, q5, d2[1]\n"
            "vmla.f16 q14, q5, d4[1]\n"
            "vmla.f16 q15, q5, d6[1]\n"

            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vmla.f16 q12, q6, d0[2]\n"
            "vmla.f16 q13, q6, d2[2]\n"
            "vmla.f16 q14, q6, d4[2]\n"
            "vmla.f16 q15, q6, d6[2]\n"

            "vmla.f16 q12, q7, d0[3]\n"
            "vmla.f16 q13, q7, d2[3]\n"
            "vmla.f16 q14, q7, d4[3]\n"
            "vmla.f16 q15, q7, d6[3]\n"

            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"
            "vmla.f16 q12, q8, d1[0]\n"
            "vmla.f16 q13, q8, d3[0]\n"
            "vmla.f16 q14, q8, d5[0]\n"
            "vmla.f16 q15, q8, d7[0]\n"

            "vmla.f16 q12, q9, d1[1]\n"
            "vmla.f16 q13, q9, d3[1]\n"
            "vmla.f16 q14, q9, d5[1]\n"
            "vmla.f16 q15, q9, d7[1]\n"

            "vmla.f16 q12, q10, d1[2]\n"
            "vmla.f16 q13, q10, d3[2]\n"
            "vmla.f16 q14, q10, d5[2]\n"
            "vmla.f16 q15, q10, d7[2]\n"

            "vmla.f16 q12, q11, d1[3]\n"
            "vmla.f16 q13, q11, d3[3]\n"
            "vmla.f16 q14, q11, d5[3]\n"
            "vmla.f16 q15, q11, d7[3]\n"

            "subs %[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "vst1.32 {d24, d25, d26, d27}, [%[output]]!\n"
            "vst1.32 {d28, d29, d30, d31}, [%[output]]!\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31", "cc", "memory");
}

}  // anonymous namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(gemm_nopack_f16_4x8);

void gemm_nopack_f16_4x8::kern(const dt_float16* A, size_t LDA,
                               const dt_float16* B, size_t LDB, dt_float16* C,
                               size_t LDC, size_t M, size_t K, size_t N,
                               const dt_float16*, void*, bool trA,
                               bool trB) const {
    constexpr static size_t MB = 8;
    constexpr static size_t KB = 8;
    constexpr static size_t NB = 4;

    megdnn_assert(!trA && !trB && M % MB == 0 && K % KB == 0);

    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)
    for (size_t m = 0; m < M; m += MB) {
        dt_float16* output = C + (m / MB) * LDC;
        const dt_float16* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_8x4(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        while (n < N) {
            kern_8x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    }
}

#endif
// vim: syntax=cpp.doxygen
