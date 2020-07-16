/**
 * \file dnn/src/armv7/matrix_mul/int16x16x32/strategy_mk8_4x8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/matrix_mul/int16x16x32/strategy.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

namespace {

void kern_8x1(const dt_int16* a_ptr, const dt_int16* b_ptr, int LDB, int K,
              dt_int32* output) {
    //! As each load 16 number from B, but the pos add 16 * 2, so we minus 16
    //! here.
    LDB = (LDB - 4) * sizeof(dt_int16);

    asm volatile(
            "subs %[K], #8\n"
            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"

            "vld1.32 {d0}, [%[b_ptr]]!\n"
            "vld1.32 {d1}, [%[b_ptr]], %[LDB]\n"

            "vmull.s16 q12, d8, d0[0]\n"
            "vmull.s16 q13, d9, d0[0]\n"
            "vmull.s16 q14, d10, d0[1]\n"
            "vmull.s16 q15, d11, d0[1]\n"

            "vmlal.s16 q12, d12, d0[2]\n"
            "vmlal.s16 q13, d13, d0[2]\n"
            "vmlal.s16 q14, d14, d0[3]\n"
            "vmlal.s16 q15, d15, d0[3]\n"

            "beq 2f\n"

            "1:\n"

            "vld1.32 {d8, d9, d10, d11}, [%[a_ptr]]!\n"
            "vld1.32 {d12, d13, d14, d15}, [%[a_ptr]]!\n"
            "vld1.32 {d0}, [%[b_ptr]]!\n"

            "vmlal.s16 q12, d16, d1[0]\n"
            "vmlal.s16 q13, d17, d1[0]\n"
            "vmlal.s16 q14, d18, d1[1]\n"
            "vmlal.s16 q15, d19, d1[1]\n"

            "vmlal.s16 q12, d20, d1[2]\n"
            "vmlal.s16 q13, d21, d1[2]\n"
            "vmlal.s16 q14, d22, d1[3]\n"
            "vmlal.s16 q15, d23, d1[3]\n"

            "vld1.32 {d1}, [%[b_ptr]], %[LDB]\n"
            "vld1.32 {d16, d17, d18, d19}, [%[a_ptr]]!\n"
            "vld1.32 {d20, d21, d22, d23}, [%[a_ptr]]!\n"

            "vmlal.s16 q12, d8, d0[0]\n"
            "vmlal.s16 q13, d9, d0[0]\n"
            "vmlal.s16 q14, d10, d0[1]\n"
            "vmlal.s16 q15, d11, d0[1]\n"

            "vmlal.s16 q12, d12, d0[2]\n"
            "vmlal.s16 q13, d13, d0[2]\n"
            "vmlal.s16 q14, d14, d0[3]\n"
            "vmlal.s16 q15, d15, d0[3]\n"

            "subs %[K], %[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "vmlal.s16 q12, d16, d1[0]\n"
            "vmlal.s16 q13, d17, d1[0]\n"
            "vmlal.s16 q14, d18, d1[1]\n"
            "vmlal.s16 q15, d19, d1[1]\n"

            "vmlal.s16 q12, d20, d1[2]\n"
            "vmlal.s16 q13, d21, d1[2]\n"
            "vmlal.s16 q14, d22, d1[3]\n"
            "vmlal.s16 q15, d23, d1[3]\n"

            "vadd.s32 q12, q12, q14\n"
            "vadd.s32 q13, q13, q15\n"

            "vst1.32 {d24, d25, d26, d27}, [%[output]]!\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "d0", "d1", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
              "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
              "d25", "d26", "d27", "d28", "d29", "d30", "d31", "cc", "memory");
}

// Overview of register layout:
//
// A 4x8 cell of Rhs is stored in 16bit in q0-q3
// A 8x8 cell of Lhs is stored in 16bit in q4-q7
// A 2x8 block of accumulators is stored in 32bit in q8-v15.
//
//                  Rhs +--------+
//                      | q4[0-7]|
//                      | q5[0-7]|
//                      | q6[0-7]|
//                      | q7[0-7]|
//                      +--------+
//      Lhs
//  +--------+ - - - - -+--------+--------+
//  | q0[0-7]|          | q8[0-3]| v9[0-3]|
//  | q1[0-7]|          |q10[0-3]|v11[0-3]|
//  | q2[0-7]|          |q12[0-3]|v13[0-3]|
//  | q3[0-7]|          |q14[0-3]|v15[0-3]|
//  +--------+          +--------+--------+
//                      Accumulator
void kern_8x4(const dt_int16* a_ptr, const dt_int16* b_ptr, int LDB, int K,
              dt_int32* output) {
    //! As each load 16 number from B, but the pos add 16 * 2, so we minus 16
    //! here.
    LDB = (LDB - 16) * sizeof(dt_int16);

    asm volatile(
            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vld1.32 {d0, d1, d2, d3}, [%[b_ptr]]!\n"
            "subs %[K], #8\n"

            "vld1.32 {d4, d5, d6, d7}, [%[b_ptr]], %[LDB]\n"
            "vmull.s16 q8, d8, d0[0]\n"
            "vmull.s16 q10, d8, d2[0]\n"
            "vmull.s16 q12, d8, d4[0]\n"
            "vmull.s16 q14, d8, d6[0]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmull.s16 q9, d9, d0[0]\n"
            "vmull.s16 q11, d9, d2[0]\n"
            "vmull.s16 q13, d9, d4[0]\n"
            "vmull.s16 q15, d9, d6[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d0[1]\n"
            "vmlal.s16 q10, d10, d2[1]\n"
            "vmlal.s16 q12, d10, d4[1]\n"
            "vmlal.s16 q14, d10, d6[1]\n"
            "vmlal.s16 q9, d11, d0[1]\n"
            "vmlal.s16 q11, d11, d2[1]\n"
            "vmlal.s16 q13, d11, d4[1]\n"
            "vmlal.s16 q15, d11, d6[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d0[2]\n"
            "vmlal.s16 q10, d12, d2[2]\n"
            "vmlal.s16 q12, d12, d4[2]\n"
            "vmlal.s16 q14, d12, d6[2]\n"
            "vmlal.s16 q9, d13, d0[2]\n"
            "vmlal.s16 q11, d13, d2[2]\n"
            "vmlal.s16 q13, d13, d4[2]\n"
            "vmlal.s16 q15, d13, d6[2]\n"

            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d14, d0[3]\n"
            "vmlal.s16 q10, d14, d2[3]\n"
            "vmlal.s16 q12, d14, d4[3]\n"
            "vmlal.s16 q14, d14, d6[3]\n"
            "vmlal.s16 q9, d15, d0[3]\n"
            "vmlal.s16 q11, d15, d2[3]\n"
            "vmlal.s16 q13, d15, d4[3]\n"
            "vmlal.s16 q15, d15, d6[3]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d8, d1[0]\n"
            "vmlal.s16 q10, d8, d3[0]\n"
            "vmlal.s16 q12, d8, d5[0]\n"
            "vmlal.s16 q14, d8, d7[0]\n"
            "vmlal.s16 q9, d9, d1[0]\n"
            "vmlal.s16 q11, d9, d3[0]\n"
            "vmlal.s16 q13, d9, d5[0]\n"
            "vmlal.s16 q15, d9, d7[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d1[1]\n"
            "vmlal.s16 q10, d10, d3[1]\n"
            "vmlal.s16 q12, d10, d5[1]\n"
            "vmlal.s16 q14, d10, d7[1]\n"
            "vmlal.s16 q9, d11, d1[1]\n"
            "vmlal.s16 q11, d11, d3[1]\n"
            "vmlal.s16 q13, d11, d5[1]\n"
            "vmlal.s16 q15, d11, d7[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d1[2]\n"
            "vmlal.s16 q10, d12, d3[2]\n"
            "vmlal.s16 q12, d12, d5[2]\n"
            "vmlal.s16 q14, d12, d7[2]\n"
            "vmlal.s16 q9, d13, d1[2]\n"
            "vmlal.s16 q11, d13, d3[2]\n"
            "vmlal.s16 q13, d13, d5[2]\n"
            "vmlal.s16 q15, d13, d7[2]\n"

            "beq 2f\n"

            "1:\n"
            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d14, d1[3]\n"
            "vmlal.s16 q10, d14, d3[3]\n"
            "vmlal.s16 q9, d15, d1[3]\n"
            "vmlal.s16 q11, d15, d3[3]\n"

            "vld1.32 {d0, d1, d2, d3}, [%[b_ptr]]!\n"
            "vmlal.s16 q12, d14, d5[3]\n"
            "vmlal.s16 q14, d14, d7[3]\n"
            "vmlal.s16 q13, d15, d5[3]\n"
            "vmlal.s16 q15, d15, d7[3]\n"

            "vld1.32 {d4, d5, d6, d7}, [%[b_ptr]], %[LDB]\n"
            "vmlal.s16 q8, d8, d0[0]\n"
            "vmlal.s16 q10, d8, d2[0]\n"
            "vmlal.s16 q12, d8, d4[0]\n"
            "vmlal.s16 q14, d8, d6[0]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmlal.s16 q9, d9, d0[0]\n"
            "vmlal.s16 q11, d9, d2[0]\n"
            "vmlal.s16 q13, d9, d4[0]\n"
            "vmlal.s16 q15, d9, d6[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d0[1]\n"
            "vmlal.s16 q10, d10, d2[1]\n"
            "vmlal.s16 q12, d10, d4[1]\n"
            "vmlal.s16 q14, d10, d6[1]\n"
            "vmlal.s16 q9, d11, d0[1]\n"
            "vmlal.s16 q11, d11, d2[1]\n"
            "vmlal.s16 q13, d11, d4[1]\n"
            "vmlal.s16 q15, d11, d6[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d0[2]\n"
            "vmlal.s16 q10, d12, d2[2]\n"
            "vmlal.s16 q12, d12, d4[2]\n"
            "vmlal.s16 q14, d12, d6[2]\n"
            "vmlal.s16 q9, d13, d0[2]\n"
            "vmlal.s16 q11, d13, d2[2]\n"
            "vmlal.s16 q13, d13, d4[2]\n"
            "vmlal.s16 q15, d13, d6[2]\n"

            "vld1.32 {d8, d9}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d14, d0[3]\n"
            "vmlal.s16 q10, d14, d2[3]\n"
            "vmlal.s16 q12, d14, d4[3]\n"
            "vmlal.s16 q14, d14, d6[3]\n"
            "vmlal.s16 q9, d15, d0[3]\n"
            "vmlal.s16 q11, d15, d2[3]\n"
            "vmlal.s16 q13, d15, d4[3]\n"
            "vmlal.s16 q15, d15, d6[3]\n"

            "vld1.32 {d10, d11}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d8, d1[0]\n"
            "vmlal.s16 q10, d8, d3[0]\n"
            "vmlal.s16 q12, d8, d5[0]\n"
            "vmlal.s16 q14, d8, d7[0]\n"
            "vmlal.s16 q9, d9, d1[0]\n"
            "vmlal.s16 q11, d9, d3[0]\n"
            "vmlal.s16 q13, d9, d5[0]\n"
            "vmlal.s16 q15, d9, d7[0]\n"

            "vld1.32 {d12, d13}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d10, d1[1]\n"
            "vmlal.s16 q10, d10, d3[1]\n"
            "vmlal.s16 q12, d10, d5[1]\n"
            "vmlal.s16 q14, d10, d7[1]\n"
            "vmlal.s16 q9, d11, d1[1]\n"
            "vmlal.s16 q11, d11, d3[1]\n"
            "vmlal.s16 q13, d11, d5[1]\n"
            "vmlal.s16 q15, d11, d7[1]\n"

            "vld1.32 {d14, d15}, [%[a_ptr]]!\n"
            "vmlal.s16 q8, d12, d1[2]\n"
            "vmlal.s16 q10, d12, d3[2]\n"
            "vmlal.s16 q12, d12, d5[2]\n"
            "vmlal.s16 q14, d12, d7[2]\n"
            "vmlal.s16 q9, d13, d1[2]\n"
            "vmlal.s16 q11, d13, d3[2]\n"
            "vmlal.s16 q13, d13, d5[2]\n"
            "vmlal.s16 q15, d13, d7[2]\n"

            "subs %[K], %[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "vmlal.s16 q8, d14, d1[3]\n"
            "vmlal.s16 q10, d14, d3[3]\n"
            "vmlal.s16 q9, d15, d1[3]\n"
            "vmlal.s16 q11, d15, d3[3]\n"
            "vst1.32 {d16, d17, d18, d19}, [%[output]]!\n"
            "vmlal.s16 q12, d14, d5[3]\n"
            "vmlal.s16 q14, d14, d7[3]\n"
            "vmlal.s16 q13, d15, d5[3]\n"
            "vmlal.s16 q15, d15, d7[3]\n"
            "vst1.32 {d20, d21, d22, d23}, [%[output]]!\n"
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

MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(gemm_nopack_s16_4x8);

void gemm_nopack_s16_4x8::kern(const dt_int16* A, size_t LDA, const dt_int16* B,
                               size_t LDB, dt_int32* C, size_t LDC, size_t M,
                               size_t K, size_t N, const dt_int32*, void*,
                               bool trA, bool trB) const {
    constexpr static size_t MB = 8;
    constexpr static size_t KB = 8;
    constexpr static size_t NB = 4;

    megdnn_assert(!trA && !trB && M % MB == 0 && K % KB == 0);

    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)
    for (size_t m = 0; m < M; m += MB) {
        dt_int32* output = C + (m / MB) * LDC;
        const dt_int16* cur_B = B;
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

// vim: syntax=cpp.doxygen
