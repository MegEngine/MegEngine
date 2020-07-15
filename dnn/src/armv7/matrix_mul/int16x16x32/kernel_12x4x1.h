/**
 * \file dnn/src/armv7/matrix_mul/int16x16x32/kernel_12x4x1.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"

namespace megdnn {
namespace armv7 {
namespace matmul_12x4x1 {
/**
 * Overview of register layout:
 *
 * A 12x4 cell of Rhs is stored in 16bit in d3
 * A 1x4 cell of Lhs is stored in 16bit in d0 d1 d2
 * A 4x4 block of accumulators is stored in 32bit in q4-q15
 *
 *                     +--------+
 *                     | d3[0-3]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +--------+ - - - - +---------
 *  |d0[0]|            | q4[0-3]|
 *  |d0[1]|            | q5[0-3]|
 *  |d0[2]|            | q6[0-3]|
 *  |d0[3]|            | q7[0-3]|
 *  |d1[0]|            | q8[0-3]|
 *  |d1[1]|            | q9[0-3]|
 *  |d1[2]|            | q10[0-3]|
 *  |d1[3]|            | q11[0-3]|
 *  |d2[0]|            | q12[0-3]|
 *  |d2[1]|            | q13[0-3]|
 *  |d2[2]|            | q14[0-3]|
 *  |d2[3]|            | q15[0-3]|
 *  +--------+ - - - - +---------
 *
 *                     Accumulator
 */
static void kern_12x4(const int16_t* packA, const int16_t* packB, int K,
                      int32_t* output, int LDC, bool is_first_k) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

    int asmLDC = LDC * sizeof(int32_t);
    int oddLDC = LDC * 2;
    int32_t* outptr_row0 = output;
    int32_t* outptr_row2 = outptr_row0 + oddLDC;
    int32_t* outptr_row4 = outptr_row2 + oddLDC;
    int32_t* outptr_row6 = outptr_row4 + oddLDC;
    int32_t* outptr_row8 = outptr_row6 + oddLDC;
    int32_t* outptr_row10 = outptr_row8 + oddLDC;
    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "vld1.32 {d8,d9}  ,[%[outptr_row0]]\n"
            "vld1.32 {d12,d13},[%[outptr_row2]]\n"
            "vld1.32 {d16,d17},[%[outptr_row4]]\n"
            "vld1.32 {d20,d21},[%[outptr_row6]]\n"
            "vld1.32 {d24,d25},[%[outptr_row8]]\n"
            "vld1.32 {d28,d29},[%[outptr_row10]]\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vld1.16 {d10,d11},[%[outptr_row0]]\n"
            "vld1.16 {d14,d15},[%[outptr_row2]]\n"
            "vld1.16 {d18,d19},[%[outptr_row4]]\n"
            "vld1.16 {d22,d23},[%[outptr_row6]]\n"
            "vld1.16 {d26,d27},[%[outptr_row8]]\n"
            "vld1.16 {d30,d31},[%[outptr_row10]]\n"
            "2:\n"
            "pld [%[b_ptr],#64]\n"
            "vld1.16 {d3},[%[b_ptr]]!\n"
            "pld [%[a_ptr],#196]\n"
            "vld1.16 {d0},[%[a_ptr]]!\n"
            "vld1.16 {d1},[%[a_ptr]]!\n"
            "vld1.16 {d2},[%[a_ptr]]!\n"
            "vmlal.s16 q4,d3,d0[0]\n"
            "vmlal.s16 q5,d3,d0[1]\n"
            "vmlal.s16 q6,d3,d0[2]\n"
            "vmlal.s16 q7,d3,d0[3]\n"

            "vmlal.s16 q8 ,d3,d1[0]\n"
            "vmlal.s16 q9 ,d3,d1[1]\n"
            "vmlal.s16 q10,d3,d1[2]\n"
            "vmlal.s16 q11,d3,d1[3]\n"

            "vmlal.s16 q12,d3,d2[0]\n"
            "vmlal.s16 q13,d3,d2[1]\n"
            "vmlal.s16 q14,d3,d2[2]\n"
            "vmlal.s16 q15,d3,d2[3]\n"
            "subs %[K], %[K], #1\n"
            "bne 2b\n"
            "vst1.32 {d10,d11},[%[outptr_row0]]\n"
            "vst1.32 {d14,d15},[%[outptr_row2]]\n"
            "vst1.32 {d18,d19},[%[outptr_row4]]\n"
            "vst1.32 {d22,d23},[%[outptr_row6]]\n"
            "vst1.32 {d26,d27},[%[outptr_row8]]\n"
            "vst1.32 {d30,d31},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "sub %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "sub %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "sub %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "sub %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "sub %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d8,d9}  ,[%[outptr_row0]]\n"
            "vst1.32 {d12,d13},[%[outptr_row2]]\n"
            "vst1.32 {d16,d17},[%[outptr_row4]]\n"
            "vst1.32 {d20,d21},[%[outptr_row6]]\n"
            "vst1.32 {d24,d25},[%[outptr_row8]]\n"
            "vst1.32 {d28,d29},[%[outptr_row10]]\n"
            "b 4f \n"
            "1:\n"  // handle fisrt reduce 1 cmp
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"

            "veor.s32 q8, q8, q8\n"
            "veor.s32 q9, q9, q9\n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q11, q11, q11\n"

            "veor.s32 q12, q12, q12\n"
            "veor.s32 q13, q13, q13\n"
            "veor.s32 q14, q14, q14\n"
            "veor.s32 q15, q15, q15\n"
            "3:\n"
            "pld [%[b_ptr],#64]\n"
            "vld1.16 {d3},[%[b_ptr]]!\n"
            "pld [%[a_ptr],#196]\n"
            "vld1.16 {d0},[%[a_ptr]]!\n"
            "vld1.16 {d1},[%[a_ptr]]!\n"
            "vld1.16 {d2},[%[a_ptr]]!\n"
            "vmlal.s16 q4,d3,d0[0]\n"
            "vmlal.s16 q5,d3,d0[1]\n"
            "vmlal.s16 q6,d3,d0[2]\n"
            "vmlal.s16 q7,d3,d0[3]\n"

            "vmlal.s16 q8 ,d3,d1[0]\n"
            "vmlal.s16 q9 ,d3,d1[1]\n"
            "vmlal.s16 q10,d3,d1[2]\n"
            "vmlal.s16 q11,d3,d1[3]\n"

            "vmlal.s16 q12,d3,d2[0]\n"
            "vmlal.s16 q13,d3,d2[1]\n"
            "vmlal.s16 q14,d3,d2[2]\n"
            "vmlal.s16 q15,d3,d2[3]\n"
            "subs %[K], %[K], #1\n"
            "bne 3b\n"
            "vst1.32 {d8,d9}  ,[%[outptr_row0]]\n"
            "vst1.32 {d12,d13},[%[outptr_row2]]\n"
            "vst1.32 {d16,d17},[%[outptr_row4]]\n"
            "vst1.32 {d20,d21},[%[outptr_row6]]\n"
            "vst1.32 {d24,d25},[%[outptr_row8]]\n"
            "vst1.32 {d28,d29},[%[outptr_row10]]\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d10,d11},[%[outptr_row0]]\n"
            "vst1.32 {d14,d15},[%[outptr_row2]]\n"
            "vst1.32 {d18,d19},[%[outptr_row4]]\n"
            "vst1.32 {d22,d23},[%[outptr_row6]]\n"
            "vst1.32 {d26,d27},[%[outptr_row8]]\n"
            "vst1.32 {d30,d31},[%[outptr_row10]]\n"
            "4: \n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [asmLDC] "+r"(asmLDC), [is_first_k] "+r"(is_first_k),
              [outptr_row0] "+r"(outptr_row0), [outptr_row2] "+r"(outptr_row2),
              [outptr_row4] "+r"(outptr_row4), [outptr_row6] "+r"(outptr_row6),
              [outptr_row8] "+r"(outptr_row8), [outptr_row10] "+r"(outptr_row10)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d18", "d20", "d21", "d22",
              "d24", "d26", "d28", "d30", "cc", "memory");
}

static void kern_12x123(const int16_t* packA, const int16_t* packB, int K,
                        int32_t* output, int LDC, bool is_first_k,
                        size_t n_remain) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;
    int asmLDC = LDC * sizeof(int32_t);
    int oddLDC = LDC * 2;
    int32_t* outptr_row0 = output;
    int32_t* outptr_row2 = outptr_row0 + oddLDC;
    int32_t* outptr_row4 = outptr_row2 + oddLDC;
    int32_t* outptr_row6 = outptr_row4 + oddLDC;
    int32_t* outptr_row8 = outptr_row6 + oddLDC;
    int32_t* outptr_row10 = outptr_row8 + oddLDC;

    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "cmp %[n_remain] ,#3 \n"
            "beq 5f\n"
            "cmp %[n_remain],#2\n"
            "beq 6f\n"
            "cmp %[n_remain],#1\n"
            "beq 7f\n"
            "5: \n"
            "vld1.32 {d8}    ,[%[outptr_row0]]!\n"
            "vld1.32 {d9[0]} ,[%[outptr_row0]]\n"
            "vld1.32 {d12}   ,[%[outptr_row2]]!\n"
            "vld1.32 {d13[0]},[%[outptr_row2]]\n"
            "vld1.32 {d16}   ,[%[outptr_row4]]!\n"
            "vld1.32 {d17[0]},[%[outptr_row4]]\n"
            "vld1.32 {d20}   ,[%[outptr_row6]]!\n"
            "vld1.32 {d20[0]},[%[outptr_row6]]\n"
            "vld1.32 {d24}   ,[%[outptr_row8]]!\n"
            "vld1.32 {d25[0]},[%[outptr_row8]]\n"
            "vld1.32 {d28}   ,[%[outptr_row10]]!\n"
            "vld1.32 {d29[0]},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8\n"
            "sub %[outptr_row2],%[outptr_row2],#8\n"
            "sub %[outptr_row4],%[outptr_row4],#8\n"
            "sub %[outptr_row6],%[outptr_row6],#8\n"
            "sub %[outptr_row8],%[outptr_row8],#8\n"
            "sub %[outptr_row10],%[outptr_row10],#8\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vld1.32 {d10}   ,[%[outptr_row0]]!\n"
            "vld1.32 {d11[0]},[%[outptr_row0]]\n"
            "vld1.32 {d14}   ,[%[outptr_row2]]!\n"
            "vld1.32 {d15[0]},[%[outptr_row2]]\n"
            "vld1.32 {d18}   ,[%[outptr_row4]]!\n"
            "vld1.32 {d19[0]},[%[outptr_row4]]\n"
            "vld1.32 {d22}   ,[%[outptr_row6]]!\n"
            "vld1.32 {d23[0]},[%[outptr_row6]]\n"
            "vld1.32 {d26}   ,[%[outptr_row8]]!\n"
            "vld1.32 {d27[0]},[%[outptr_row8]]\n"
            "vld1.32 {d30}   ,[%[outptr_row10]]!\n"
            "vld1.32 {d31[0]},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8\n"
            "sub %[outptr_row2],%[outptr_row2],#8\n"
            "sub %[outptr_row4],%[outptr_row4],#8\n"
            "sub %[outptr_row6],%[outptr_row6],#8\n"
            "sub %[outptr_row8],%[outptr_row8],#8\n"
            "sub %[outptr_row10],%[outptr_row10],#8\n"
            "b 2f\n"
            "6: \n"
            "vld1.32 {d8} ,[%[outptr_row0]]\n"
            "vld1.32 {d12},[%[outptr_row2]]\n"
            "vld1.32 {d16},[%[outptr_row4]]\n"
            "vld1.32 {d20},[%[outptr_row6]]\n"
            "vld1.32 {d24},[%[outptr_row8]]\n"
            "vld1.32 {d28},[%[outptr_row10]]\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vld1.32 {d10},[%[outptr_row0]]\n"
            "vld1.32 {d14},[%[outptr_row2]]\n"
            "vld1.32 {d18},[%[outptr_row4]]\n"
            "vld1.32 {d22},[%[outptr_row6]]\n"
            "vld1.32 {d26},[%[outptr_row8]]\n"
            "vld1.32 {d30},[%[outptr_row10]]\n"
            "b 2f\n"
            "7: \n"
            "vld1.32 {d8[0]} ,[%[outptr_row0]]\n"
            "vld1.32 {d12[0]},[%[outptr_row2]]\n"
            "vld1.32 {d16[0]},[%[outptr_row4]]\n"
            "vld1.32 {d20[0]},[%[outptr_row6]]\n"
            "vld1.32 {d24[0]},[%[outptr_row8]]\n"
            "vld1.32 {d28[0]},[%[outptr_row10]]\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vld1.32 {d10[0]},[%[outptr_row0]]\n"
            "vld1.32 {d14[0]},[%[outptr_row2]]\n"
            "vld1.32 {d18[0]},[%[outptr_row4]]\n"
            "vld1.32 {d22[0]},[%[outptr_row6]]\n"
            "vld1.32 {d26[0]},[%[outptr_row8]]\n"
            "vld1.32 {d30[0]},[%[outptr_row10]]\n"
            "b 2f \n"
            "1:\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"

            "veor.s32 q8, q8, q8\n"
            "veor.s32 q9, q9, q9\n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q11, q11, q11\n"

            "veor.s32 q12, q12, q12\n"
            "veor.s32 q13, q13, q13\n"
            "veor.s32 q14, q14, q14\n"
            "veor.s32 q15, q15, q15\n"
            "2:\n"
            "pld [%[b_ptr],#16]\n"
            "vld1.16 {d3},[%[b_ptr]]!\n"
            "pld [%[a_ptr],#196]\n"
            "vld1.16 {d0},[%[a_ptr]]!\n"
            "vld1.16 {d1},[%[a_ptr]]!\n"
            "vld1.16 {d2},[%[a_ptr]]!\n"
            "vmlal.s16 q4,d3,d0[0]\n"
            "vmlal.s16 q5,d3,d0[1]\n"
            "vmlal.s16 q6,d3,d0[2]\n"
            "vmlal.s16 q7,d3,d0[3]\n"

            "vmlal.s16 q8 ,d3,d1[0]\n"
            "vmlal.s16 q9 ,d3,d1[1]\n"
            "vmlal.s16 q10,d3,d1[2]\n"
            "vmlal.s16 q11,d3,d1[3]\n"

            "vmlal.s16 q12,d3,d2[0]\n"
            "vmlal.s16 q13,d3,d2[1]\n"
            "vmlal.s16 q14,d3,d2[2]\n"
            "vmlal.s16 q15,d3,d2[3]\n"
            "subs %[K], %[K], #1\n"
            "bne 2b\n"
            "cmp %[is_first_k], #1\n"
            "beq 3f\n"
            "cmp %[n_remain] ,#3 \n"
            "beq 5f\n"
            "cmp %[n_remain] ,#2 \n"
            "beq 6f\n"
            "cmp %[n_remain] ,#1 \n"
            "beq 7f\n"
            "5: \n"
            "vst1.32 {d10}   ,[%[outptr_row0]]!\n"
            "vst1.32 {d11[0]},[%[outptr_row0]]\n"
            "vst1.32 {d14}   ,[%[outptr_row2]]!\n"
            "vst1.32 {d15[0]},[%[outptr_row2]]\n"
            "vst1.32 {d18}   ,[%[outptr_row4]]!\n"
            "vst1.32 {d19[0]},[%[outptr_row4]]\n"
            "vst1.32 {d22}   ,[%[outptr_row6]]!\n"
            "vst1.32 {d23[0]},[%[outptr_row6]]\n"
            "vst1.32 {d26}   ,[%[outptr_row8]]!\n"
            "vst1.32 {d27[0]},[%[outptr_row8]]\n"
            "vst1.32 {d30}   ,[%[outptr_row10]]!\n"
            "vst1.32 {d31[0]},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8\n"
            "sub %[outptr_row2],%[outptr_row2],#8\n"
            "sub %[outptr_row4],%[outptr_row4],#8\n"
            "sub %[outptr_row6],%[outptr_row6],#8\n"
            "sub %[outptr_row8],%[outptr_row8],#8\n"
            "sub %[outptr_row10],%[outptr_row10],#8\n"
            "sub %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "sub %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "sub %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "sub %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "sub %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "sub %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d8}    ,[%[outptr_row0]]!\n"
            "vst1.32 {d9[0]} ,[%[outptr_row0]]\n"
            "vst1.32 {d12}   ,[%[outptr_row2]]!\n"
            "vst1.32 {d13[0]},[%[outptr_row2]]\n"
            "vst1.32 {d16}   ,[%[outptr_row4]]!\n"
            "vst1.32 {d17[0]},[%[outptr_row4]]\n"
            "vst1.32 {d20}   ,[%[outptr_row6]]!\n"
            "vst1.32 {d21[0]},[%[outptr_row6]]\n"
            "vst1.32 {d24}   ,[%[outptr_row8]]!\n"
            "vst1.32 {d25[0]},[%[outptr_row8]]\n"
            "vst1.32 {d28}   ,[%[outptr_row10]]!\n"
            "vst1.32 {d29[0]},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8\n"
            "sub %[outptr_row2],%[outptr_row2],#8\n"
            "sub %[outptr_row4],%[outptr_row4],#8\n"
            "sub %[outptr_row6],%[outptr_row6],#8\n"
            "sub %[outptr_row8],%[outptr_row8],#8\n"
            "sub %[outptr_row10],%[outptr_row10],#8\n"

            "b 4f\n"
            "6: \n"
            "vst1.32 {d10}   ,[%[outptr_row0]]\n"
            "vst1.32 {d14}   ,[%[outptr_row2]]\n"
            "vst1.32 {d18}   ,[%[outptr_row4]]\n"
            "vst1.32 {d22}   ,[%[outptr_row6]]\n"
            "vst1.32 {d26}   ,[%[outptr_row8]]\n"
            "vst1.32 {d30}   ,[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "sub %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "sub %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "sub %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "sub %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "sub %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d8}    ,[%[outptr_row0]]\n"
            "vst1.32 {d12}   ,[%[outptr_row2]]\n"
            "vst1.32 {d16}   ,[%[outptr_row4]]\n"
            "vst1.32 {d20}   ,[%[outptr_row6]]\n"
            "vst1.32 {d24}   ,[%[outptr_row8]]\n"
            "vst1.32 {d28}   ,[%[outptr_row10]]\n"
            "b 4f\n"
            "7: \n"
            "vst1.32 {d10[0]},[%[outptr_row0]]\n"
            "vst1.32 {d14[0]},[%[outptr_row2]]\n"
            "vst1.32 {d18[0]},[%[outptr_row4]]\n"
            "vst1.32 {d22[0]},[%[outptr_row6]]\n"
            "vst1.32 {d26[0]},[%[outptr_row8]]\n"
            "vst1.32 {d30[0]},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "sub %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "sub %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "sub %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "sub %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "sub %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d8[0]} ,[%[outptr_row0]]\n"
            "vst1.32 {d12[0]},[%[outptr_row2]]\n"
            "vst1.32 {d16[0]},[%[outptr_row4]]\n"
            "vst1.32 {d20[0]},[%[outptr_row6]]\n"
            "vst1.32 {d24[0]},[%[outptr_row8]]\n"
            "vst1.32 {d28[0]},[%[outptr_row10]]\n"
            "b 4f\n"
            "3: \n"  // first k
            "cmp %[n_remain] ,#3 \n"
            "beq 5f\n"
            "cmp %[n_remain] ,#2 \n"
            "beq 6f\n"
            "cmp %[n_remain] ,#1 \n"
            "beq 7f\n"
            "5:\n"
            "vst1.32 {d8}    ,[%[outptr_row0]]!\n"
            "vst1.32 {d9[0]} ,[%[outptr_row0]]\n"
            "vst1.32 {d12}   ,[%[outptr_row2]]!\n"
            "vst1.32 {d13[0]},[%[outptr_row2]]\n"
            "vst1.32 {d16}   ,[%[outptr_row4]]!\n"
            "vst1.32 {d17[0]},[%[outptr_row4]]\n"
            "vst1.32 {d20}   ,[%[outptr_row6]]!\n"
            "vst1.32 {d21[0]},[%[outptr_row6]]\n"
            "vst1.32 {d24}   ,[%[outptr_row8]]!\n"
            "vst1.32 {d25[0]},[%[outptr_row8]]\n"
            "vst1.32 {d28}   ,[%[outptr_row10]]!\n"
            "vst1.32 {d29[0]},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8\n"
            "sub %[outptr_row2],%[outptr_row2],#8\n"
            "sub %[outptr_row4],%[outptr_row4],#8\n"
            "sub %[outptr_row6],%[outptr_row6],#8\n"
            "sub %[outptr_row8],%[outptr_row8],#8\n"
            "sub %[outptr_row10],%[outptr_row10],#8\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d10}   ,[%[outptr_row0]]!\n"
            "vst1.32 {d11[0]},[%[outptr_row0]]\n"
            "vst1.32 {d14}   ,[%[outptr_row2]]!\n"
            "vst1.32 {d15[0]},[%[outptr_row2]]\n"
            "vst1.32 {d18}   ,[%[outptr_row4]]!\n"
            "vst1.32 {d19[0]},[%[outptr_row4]]\n"
            "vst1.32 {d22}   ,[%[outptr_row6]]!\n"
            "vst1.32 {d23[0]},[%[outptr_row6]]\n"
            "vst1.32 {d26}   ,[%[outptr_row8]]!\n"
            "vst1.32 {d27[0]},[%[outptr_row8]]\n"
            "vst1.32 {d30}   ,[%[outptr_row10]]!\n"
            "vst1.32 {d31[0]},[%[outptr_row10]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8\n"
            "sub %[outptr_row2],%[outptr_row2],#8\n"
            "sub %[outptr_row4],%[outptr_row4],#8\n"
            "sub %[outptr_row6],%[outptr_row6],#8\n"
            "sub %[outptr_row8],%[outptr_row8],#8\n"
            "sub %[outptr_row10],%[outptr_row10],#8\n"
            "b 4f\n"
            "6:\n"
            "vst1.32 {d8}    ,[%[outptr_row0]]\n"
            "vst1.32 {d12}   ,[%[outptr_row2]]\n"
            "vst1.32 {d16}   ,[%[outptr_row4]]\n"
            "vst1.32 {d20}   ,[%[outptr_row6]]\n"
            "vst1.32 {d24}   ,[%[outptr_row8]]\n"
            "vst1.32 {d28}   ,[%[outptr_row10]]\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d10}   ,[%[outptr_row0]]\n"
            "vst1.32 {d14}   ,[%[outptr_row2]]\n"
            "vst1.32 {d18}   ,[%[outptr_row4]]\n"
            "vst1.32 {d22}   ,[%[outptr_row6]]\n"
            "vst1.32 {d26}   ,[%[outptr_row8]]\n"
            "vst1.32 {d30}   ,[%[outptr_row10]]\n"
            "b 4f\n"
            "7: \n"
            "vst1.32 {d8[0]} ,[%[outptr_row0]]\n"
            "vst1.32 {d12[0]},[%[outptr_row2]]\n"
            "vst1.32 {d16[0]},[%[outptr_row4]]\n"
            "vst1.32 {d20[0]},[%[outptr_row6]]\n"
            "vst1.32 {d24[0]},[%[outptr_row8]]\n"
            "vst1.32 {d28[0]},[%[outptr_row10]]\n"
            "add %[outptr_row0],%[outptr_row0],%[asmLDC]\n"
            "add %[outptr_row2],%[outptr_row2],%[asmLDC]\n"
            "add %[outptr_row4],%[outptr_row4],%[asmLDC]\n"
            "add %[outptr_row6],%[outptr_row6],%[asmLDC]\n"
            "add %[outptr_row8],%[outptr_row8],%[asmLDC]\n"
            "add %[outptr_row10],%[outptr_row10],%[asmLDC]\n"
            "vst1.32 {d10[0]},[%[outptr_row0]]\n"
            "vst1.32 {d14[0]},[%[outptr_row2]]\n"
            "vst1.32 {d18[0]},[%[outptr_row4]]\n"
            "vst1.32 {d22[0]},[%[outptr_row6]]\n"
            "vst1.32 {d26[0]},[%[outptr_row8]]\n"
            "vst1.32 {d30[0]},[%[outptr_row10]]\n"
            "4:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [n_remain] "+r"(n_remain), [asmLDC] "+r"(asmLDC),
              [is_first_k] "+r"(is_first_k), [outptr_row0] "+r"(outptr_row0),
              [outptr_row2] "+r"(outptr_row2), [outptr_row4] "+r"(outptr_row4),
              [outptr_row6] "+r"(outptr_row6), [outptr_row8] "+r"(outptr_row8),
              [outptr_row10] "+r"(outptr_row10)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
              "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29",
              "d30", "d31", "cc", "memory");
}

static void kern_4x4(const int16_t* packA, const int16_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

    int32_t* outptr_row0 = output;
    int32_t* outptr_row1 = outptr_row0 + LDC;
    int32_t* outptr_row2 = outptr_row1 + LDC;
    int32_t* outptr_row3 = outptr_row2 + LDC;

    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "vld1.32 {d8,d9}  ,[%[outptr_row0]]\n"
            "vld1.32 {d10,d11},[%[outptr_row1]]\n"
            "vld1.32 {d12,d13},[%[outptr_row2]]\n"
            "vld1.32 {d14,d15},[%[outptr_row3]]\n"
            "b 2f \n"
            "1:\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"
            "2:\n"
            "pld [%[b_ptr],#64]\n"
            "vld1.16 {d3},[%[b_ptr]]!\n"
            "pld [%[a_ptr],#64]\n"
            "vld1.16 {d0},[%[a_ptr]]!\n"
            "vmlal.s16 q4,d3,d0[0]\n"
            "vmlal.s16 q5,d3,d0[1]\n"
            "vmlal.s16 q6,d3,d0[2]\n"
            "vmlal.s16 q7,d3,d0[3]\n"
            "subs %[K], %[K], #1\n"
            "bne 2b\n"
            "vst1.32 {d8,d9}  ,[%[outptr_row0]]\n"
            "vst1.32 {d10,d11},[%[outptr_row1]]\n"
            "vst1.32 {d12,d13},[%[outptr_row2]]\n"
            "vst1.32 {d14,d15},[%[outptr_row3]]\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [is_first_k] "+r"(is_first_k), [outptr_row0] "+r"(outptr_row0),
              [outptr_row1] "+r"(outptr_row1), [outptr_row2] "+r"(outptr_row2),
              [outptr_row3] "+r"(outptr_row3)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d18", "d20", "d21", "d22",
              "d24", "d26", "d28", "d30", "cc", "memory");
}

static void kern_4x123(const int16_t* packA, const int16_t* packB, int K,
                       int32_t* output, int LDC, bool is_first_k,
                       int n_remain) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

    int32_t* outptr_row0 = output;
    int32_t* outptr_row1 = outptr_row0 + LDC;
    int32_t* outptr_row2 = outptr_row1 + LDC;
    int32_t* outptr_row3 = outptr_row2 + LDC;

    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "cmp %[n_remain],#3\n"
            "beq 3f\n"
            "cmp %[n_remain],#2\n"
            "beq 4f\n"
            "cmp %[n_remain],#1\n"
            "beq 5f\n"
            "3: \n"
            "vld1.32 {d8} ,[%[outptr_row0]]!\n"
            "vld1.32 {d9[0]} ,[%[outptr_row0]]\n"
            "vld1.32 {d10},[%[outptr_row1]]!\n"
            "vld1.32 {d11[0]},[%[outptr_row1]]\n"
            "vld1.32 {d12},[%[outptr_row2]]!\n"
            "vld1.32 {d13[0]},[%[outptr_row2]]\n"
            "vld1.32 {d14},[%[outptr_row3]]!\n"
            "vld1.32 {d15[0]},[%[outptr_row3]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8\n"
            "sub %[outptr_row1],%[outptr_row1],#8\n"
            "sub %[outptr_row2],%[outptr_row2],#8\n"
            "sub %[outptr_row3],%[outptr_row3],#8\n"
            "b 2f\n"
            "4:\n"
            "vld1.32 {d8} ,[%[outptr_row0]]\n"
            "vld1.32 {d10},[%[outptr_row1]]\n"
            "vld1.32 {d12},[%[outptr_row2]]\n"
            "vld1.32 {d14},[%[outptr_row3]]\n"
            "b 2f\n"
            "5:\n"
            "vld1.32 {d8[0]} ,[%[outptr_row0]]\n"
            "vld1.32 {d10[0]},[%[outptr_row1]]\n"
            "vld1.32 {d12[0]},[%[outptr_row2]]\n"
            "vld1.32 {d14[0]},[%[outptr_row3]]\n"
            "b 2f \n"
            "1:\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"
            "2:\n"
            "pld [%[b_ptr],#16]\n"
            "vld1.16 {d3},[%[b_ptr]]!\n"
            "pld [%[a_ptr],#64]\n"
            "vld1.16 {d0},[%[a_ptr]]!\n"
            "vmlal.s16 q4,d3,d0[0]\n"
            "vmlal.s16 q5,d3,d0[1]\n"
            "vmlal.s16 q6,d3,d0[2]\n"
            "vmlal.s16 q7,d3,d0[3]\n"
            "subs %[K], %[K], #1\n"
            "bne 2b\n"
            "cmp %[n_remain],#3\n"
            "beq 3f\n"
            "cmp %[n_remain],#2\n"
            "beq 4f\n"
            "cmp %[n_remain],#1\n"
            "beq 5f\n"
            "3:\n"
            "vst1.32 {d8} ,[%[outptr_row0]]!\n"
            "vst1.32 {d9[0]} ,[%[outptr_row0]]\n"
            "vst1.32 {d10},[%[outptr_row1]]!\n"
            "vst1.32 {d11[0]},[%[outptr_row1]]\n"
            "vst1.32 {d12},[%[outptr_row2]]!\n"
            "vst1.32 {d13[0]},[%[outptr_row2]]\n"
            "vst1.32 {d14},[%[outptr_row3]]!\n"
            "vst1.32 {d15[0]},[%[outptr_row3]]\n"
            "b 6f\n"
            "4:\n"
            "vst1.32 {d8} ,[%[outptr_row0]]\n"
            "vst1.32 {d10},[%[outptr_row1]]\n"
            "vst1.32 {d12},[%[outptr_row2]]\n"
            "vst1.32 {d14},[%[outptr_row3]]\n"
            "b 6f\n"
            "5:\n"
            "vst1.32 {d8[0]} ,[%[outptr_row0]]\n"
            "vst1.32 {d10[0]},[%[outptr_row1]]\n"
            "vst1.32 {d12[0]},[%[outptr_row2]]\n"
            "vst1.32 {d14[0]},[%[outptr_row3]]\n"
            "6:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [n_remain] "+r"(n_remain), [is_first_k] "+r"(is_first_k),
              [outptr_row0] "+r"(outptr_row0), [outptr_row1] "+r"(outptr_row1),
              [outptr_row2] "+r"(outptr_row2), [outptr_row3] "+r"(outptr_row3)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d18", "d20", "d21", "d22",
              "d24", "d26", "d28", "d30", "cc", "memory");
}

static void kern_1x4(const int16_t* packA, const int16_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k) {
    MEGDNN_MARK_USED_VAR(LDC);
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

    int32_t* outptr_row0 = output;
    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "pld [%[outptr_row0],#64]\n"
            "vld1.32 {d8,d9} ,[%[outptr_row0]]\n"
            "b 2f \n"
            "1:\n"
            "veor.s32 q4, q4, q4\n"
            "2:\n"
            "pld [%[b_ptr],#64]\n"
            "pld [%[a_ptr],#16]\n"
            "vld1.16 {d3},[%[b_ptr]]!\n"
            "vld1.16 {d0[0]},[%[a_ptr]]!\n"
            "vmlal.s16 q4,d3,d0[0]\n"
            "subs %[K], %[K], #1\n"
            "bne 2b\n"
            "vst1.32 {d8,d9} ,[%[outptr_row0]]\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [is_first_k] "+r"(is_first_k), [outptr_row0] "+r"(outptr_row0)
            :
            : "d0", "d3", "d4", "d8", "d9", "cc", "memory");
}

/************************************************
 *this kern can hanle 1xk mul kx1 kx2 kx3  get 1x1 1x2 1x3
 *123 stands for n remain 1 2 3
 ************************************************/
static void kern_1x123(const int16_t* packA, const int16_t* packB, int K,
                       int32_t* output, int LDC, bool is_first_k,
                       int n_remain) {
    MEGDNN_MARK_USED_VAR(LDC);
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;
    int32_t* outptr_row0 = output;
    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"
            "cmp %[n_remain],#3\n"
            "beq 3f\n"
            "cmp %[n_remain],#2\n"
            "beq 4f\n"
            "cmp %[n_remain],#1\n"
            "beq 5f\n"
            "3:\n"
            "vld1.32 {d8} ,[%[outptr_row0]]!\n"
            "vld1.32 {d9[0]} ,[%[outptr_row0]]\n"
            "sub %[outptr_row0],%[outptr_row0],#8 \n"
            "b 2f\n"
            "4:\n"
            "vld1.32 {d8} ,[%[outptr_row0]]\n"
            "b 2f\n"
            "5:\n"
            "vld1.32 {d8[0]} ,[%[outptr_row0]]\n"
            "b 2f \n"
            "1:\n"
            "veor.s32 q4, q4, q4\n"
            "2:\n"
            "vld1.16 {d3},[%[b_ptr]]!\n"
            "vld1.16 {d0[0]},[%[a_ptr]]!\n"
            "vmlal.s16 q4,d3,d0[0]\n"
            "subs %[K], %[K], #1\n"
            "bne 2b\n"
            "cmp %[n_remain],#3\n"
            "beq 3f\n"
            "cmp %[n_remain],#2\n"
            "beq 4f\n"
            "cmp %[n_remain],#1\n"
            "beq 5f\n"
            "3:\n"
            "vst1.32 {d8}    ,[%[outptr_row0]]!\n"
            "vst1.32 {d9[0]} ,[%[outptr_row0]]\n"
            "b 7f\n"
            "4:\n"
            "vst1.32 {d8} ,[%[outptr_row0]]\n"
            "b 7f\n"
            "5:\n"
            "vst1.32 {d8[0]} ,[%[outptr_row0]]\n"
            "7:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [n_remain] "+r"(n_remain), [is_first_k] "+r"(is_first_k),
              [outptr_row0] "+r"(outptr_row0)
            :
            : "d0", "d3", "d8", "d9", "cc", "memory");
}

static void gemm_s16x16x32_12x4_pack_A_n(dt_int16* outptr,
                                         const dt_int16* inptr, int ldin,
                                         int y0, int ymax, int k0, int kmax) {
    int y = y0;
    int K = kmax - k0;
    for (; y + 11 < ymax; y += 12) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;
        const int16_t* inptr4 = inptr3 + ldin;
        const int16_t* inptr5 = inptr4 + ldin;
        const int16_t* inptr6 = inptr5 + ldin;
        const int16_t* inptr7 = inptr6 + ldin;
        const int16_t* inptr8 = inptr7 + ldin;
        const int16_t* inptr9 = inptr8 + ldin;
        const int16_t* inptr10 = inptr9 + ldin;
        const int16_t* inptr11 = inptr10 + ldin;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            transpose_12x4_1_h(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, inptr8, inptr9, inptr10, inptr11,
                               ldin, outptr);
        }

        for (; k < kmax; k++) {
            transpose_12x1(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                           inptr6, inptr7, inptr8, inptr9, inptr10, inptr11,
                           outptr);
        }
    }

    for (; y + 3 < ymax; y += 4) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;

        int k = 0;
        for (; k + 3 < K; k += 4) {
            transpose_4x4_1_h(inptr0, inptr1, inptr2, inptr3, outptr);
        }
        for (; k < K; k++) {
            transpose_4x1(inptr0, inptr1, inptr2, inptr3, outptr);
        }
    }
    for (; y < ymax; y++) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        std::memcpy(outptr, inptr0, sizeof(int16_t) * K);
        outptr += K;
    }
}

static void gemm_s16x16x32_12x4_transpose_pack_A_n(dt_int16* out,
                                                   const dt_int16* in, int ldin,
                                                   int x0, int xmax, int k0,
                                                   int kmax) {
    const int ksize = kmax - k0;
    const int ksize12 = ksize * 12;
    const int ksize4 = ksize * 4;
    int16_t* outptr = out;
    int16_t* outptr_interleave = out;
    int16_t* outptr_base = out;
    int16_t* outptr_times4_base = out + (xmax - x0) / 12 * ksize12;
    int16_t* outptr_times1_base =
            outptr_times4_base + ((xmax - x0) % 12) / 4 * ksize4;
    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const int16_t* inptr0 = in + k * ldin + x0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;

        int x = x0;
        outptr = outptr_base;

        for (; x + 11 < xmax; x += 12) {
            outptr_interleave = outptr;

            interleave_4x12_1_h(inptr0, inptr1, inptr2, inptr3,
                                outptr_interleave);
            outptr += ksize12;
        }
        outptr = outptr_times4_base;
        for (; x + 3 < xmax; x += 4) {
            outptr_interleave = outptr;
            interleave_4x4_1_h(inptr0, inptr1, inptr2, inptr3,
                               outptr_interleave);
            outptr += ksize4;
        }

        outptr = outptr_times1_base;
        for (; x < xmax; x++) {
            outptr_interleave = outptr;
            transpose_4x1(inptr0, inptr1, inptr2, inptr3, outptr_interleave);
            outptr += ksize;
        }
        outptr_base += 48;
        outptr_times4_base += 16;
        outptr_times1_base += 4;
    }
    for (; k < kmax; k++) {
        const int16_t* inptr0 = in + k * ldin + x0;
        prefetch_2x(inptr0);

        int x = x0;
        outptr = outptr_base;

        for (; x + 11 < xmax; x += 12) {
            outptr_interleave = outptr;
            interleave_1x12_1_h(inptr0, outptr_interleave);
            outptr += ksize12;
        }
        outptr = outptr_times4_base;
        for (; x + 3 < xmax; x += 4) {
            outptr_interleave = outptr;
            interleave_1x4_1_h(inptr0, outptr_interleave);
            outptr += ksize4;
        }

        outptr = outptr_times1_base;
        for (; x < xmax; x++) {
            outptr_interleave = outptr;
            *outptr_interleave++ = *inptr0++;
            outptr += ksize;
        }
        outptr_base += 12;
        outptr_times4_base += 4;
        outptr_times1_base += 1;
    }
}

static void gemm_s16x16x32_12x4_pack_B_n(dt_int16* out, const dt_int16* in,
                                         int ldin, int x0, int xmax, int k0,
                                         int kmax) {
    const int ksize = kmax - k0;
    const int ksize4 = ksize * 4;
    int16_t* outptr = out;
    int16_t* outptr_base = out;
    int16_t* outptr_interleave = NULL;
    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const int16_t* inptr0 = in + k * ldin + x0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;

        int x = x0;
        outptr = outptr_base;
        for (; x + 3 < xmax; x += 4) {
            outptr_interleave = outptr;
            interleave_4x4_1_h(inptr0, inptr1, inptr2, inptr3,
                               outptr_interleave);
            outptr += ksize4;
        }
        if (x < xmax) {
            outptr_interleave = outptr;
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr_interleave, 4,
                         xmax - x);
            outptr += ksize4;
        }
        outptr_base += 4 * 4;
    }
    for (; k < kmax; k++) {
        const int16_t* inptr0 = in + k * ldin + x0;
        prefetch_2x(inptr0);

        int x = x0;
        outptr = outptr_base;
        for (; x + 3 < xmax; x += 4) {
            outptr_interleave = outptr;
            int16x4_t vdata = vld1_s16(inptr0);
            vst1_s16(outptr_interleave, vdata);
            inptr0 += 4;
            outptr += ksize4;
        }
        if (x < xmax) {
            int remain = xmax - x;
            outptr_interleave = outptr;
            interleave_helper(inptr0, outptr_interleave, 4, remain);
            outptr += ksize4;
        }
        outptr_base += 4;
    }
}

static void gemm_s16x16x32_12x4_transpose_pack_B_n(dt_int16* outptr,
                                                   const dt_int16* inptr,
                                                   int ldin, int y0, int ymax,
                                                   int k0, int kmax) {
    int K = kmax - k0;
    int y = y0;
    int16_t* out = outptr;
    int16_t zerobuff[4];
    std::memset(zerobuff, 0, sizeof(int16_t) * 4);
    for (; y + 3 < ymax; y += 4) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;
        int k = 0;
        for (; k + 3 < K; k += 4) {
            transpose_4x4_1_h(inptr0, inptr1, inptr2, inptr3, out);
        }
        for (; k < K; k++) {
            transpose_4x1(inptr0, inptr1, inptr2, inptr3, out);
        }
    }
    if (y < ymax) {
        const int16_t *inptr0, *inptr1, *inptr2, *inptr3;
        inptr0 = inptr + y * ldin + k0;
        inptr1 = inptr0 + ldin;
        inptr2 = inptr1 + ldin;

        switch (y + 3 - ymax) {
            case 2:
                inptr1 = zerobuff;
                MEGDNN_FALLTHRU
            case 1:
                inptr2 = zerobuff;
                MEGDNN_FALLTHRU
            case 0:
                inptr3 = zerobuff;
                break;
            default:
                megdnn_assert(0);
        }
        int k = 0;
        for (; k + 3 < K; k += 4) {
            transpose_4x4_1_h(inptr0, inptr1, inptr2, inptr3, out);
        }
        for (; k < K; k++) {
            transpose_4x1(inptr0, inptr1, inptr2, inptr3, out);
        }
    }
}
}  // namespace matmul_12x4x1
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
