/**
 * \file dnn/src/aarch64/matrix_mul/int8_dot/kernel_8x12x4.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if __ARM_FEATURE_DOTPROD

#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_8x12x4 {

// Overview of register layout:
//
// A 12x4 cell of Rhs is stored in 8bit in q2-q4.
// A 8x4x2 cell of Lhs is stored in 8bit in q0-q1,q5-q6
// A 8x12 block of accumulators is stored in 8bit in q8--q31.
//
//                            +--------+--------+--------+
//                            |v2[0-16]|v3[0-16]|v4[0-16]|
//                       Rhs  +--------+--------+--------+
//
//                            |        |        |        |
//
//    Lhs                     |        |        |        |
//
//  +-------+-------+ - - - - +--------+--------+--------+
//  |v0[0-4]|v5[0-4]|         | v8[0-4]|v16[0-4]|v24[0-4]|
//  |v0[0-4]|v5[0-4]|         | v9[0-4]|v17[0-4]|v25[0-4]|
//  |v0[0-4]|v5[0-4]|         |v10[0-4]|v18[0-4]|v26[0-4]|
//  |v0[0-4]|v5[0-4]|         |v11[0-4]|v19[0-4]|v27[0-4]|
//  |v1[0-4]|v6[0-4]|         |v12[0-4]|v20[0-4]|v28[0-4]|
//  |v1[0-4]|v6[0-4]|         |v13[0-4]|v21[0-4]|v29[0-4]|
//  |v1[0-4]|v6[0-4]|         |v14[0-4]|v22[0-4]|v30[0-4]|
//  |v1[0-4]|v6[0-4]|         |v15[0-4]|v23[0-4]|v31[0-4]|
//  +-------+-------+ - - - - +--------+--------+--------+
//
//                            Accumulator

/**
 * \note The performance of reorder instruction and use prefetch is almost the
 * same, I test in kirin980 with small and big core, here i just keep both the
 * implementation.
 */
#if 1
static void kern_8x12(const int8_t* packA, const int8_t* packB, int K,
                      int32_t* output, int LDC, bool is_first_k) {
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;
    int32x4_t a0;
    int32x4_t a1;
    int32x4_t b0;
    int32x4_t b1;
    int32x4_t b2;
    int32x4_t a0a;
    int32x4_t a1a;
    LDC = LDC * sizeof(int32_t);

    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    int32_t* outptr4;
    int32_t* outptr5;
    int32_t* outptr6;
    int32_t* outptr7;

    asm volatile (
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "add %[outptr4], %[outptr3], %x[LDC]\n"
            "add %[outptr5], %[outptr4], %x[LDC]\n"
            "add %[outptr6], %[outptr5], %x[LDC]\n"
            "add %[outptr7], %[outptr6], %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 5f\n"
            // we can not use ld1, as it can not encode {v8, v16, v24}
            "ldp q8, q16, [%[outptr0]]\n"
            "ldr q24, [%[outptr0], #32]\n"
            "ldp q9, q17, [%[outptr1]]\n"
            "ldr q25, [%[outptr1], #32]\n"
            "ldp q10, q18, [%[outptr2]]\n"
            "ldr q26, [%[outptr2], #32]\n"
            "ldp q11, q19, [%[outptr3]]\n"
            "ldr q27, [%[outptr3], #32]\n"
            "ldp q12, q20, [%[outptr4]]\n"
            "ldr q28, [%[outptr4], #32]\n"
            "ldp q13, q21, [%[outptr5]]\n"
            "ldr q29, [%[outptr5], #32]\n"
            "ldp q14, q22, [%[outptr6]]\n"
            "ldr q30, [%[outptr6], #32]\n"
            "ldp q15, q23, [%[outptr7]]\n"
            "ldr q31, [%[outptr7], #32]\n"
            "b 6f\n"

            "5:\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v15.16b, v15.16b, v15.16b\n"

            "eor v16.16b, v16.16b, v16.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"
            "eor v18.16b, v18.16b, v18.16b\n"
            "eor v19.16b, v19.16b, v19.16b\n"
            "eor v20.16b, v20.16b, v20.16b\n"
            "eor v21.16b, v21.16b, v21.16b\n"
            "eor v22.16b, v22.16b, v22.16b\n"
            "eor v23.16b, v23.16b, v23.16b\n"

            "eor v24.16b, v24.16b, v24.16b\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"
            "eor v31.16b, v31.16b, v31.16b\n"

            "6: \n"
            // Initialize result registers, load initial operands, prime prefetches.
            "ldr  %q[a0], [%[a_ptr]]\n"
            "ldr  %q[b0], [%[b_ptr]]\n"
            "ldr  %q[a1], [%[a_ptr], #16]\n"
            "ldr  %q[b1], [%[b_ptr], #16]\n"
            ASM_PREFETCH("[%[b_ptr], #64]")
            ASM_PREFETCH("[%[a_ptr], #64]")
            ASM_PREFETCH("[%[b_ptr], #128]")
            ASM_PREFETCH("[%[a_ptr], #128]")
            ASM_PREFETCH("[%[b_ptr], #192]")
            ASM_PREFETCH("[%[b_ptr], #256]")
            ASM_PREFETCH("[%[a_ptr], #192]")
            ASM_PREFETCH("[%[b_ptr], #320]")
            ASM_PREFETCH("[%[a_ptr], #256]")
            ASM_PREFETCH("[%[b_ptr], #384]")

            // Skip loop if we are doing zero iterations of it.
            "cbz  %w[k], 4f\n"

            // Loop proper
            "1:\n"
            "sdot  v8.4s , %[b0].16b, %[a0].4b[0]\n"
            "sdot    v9.4s , %[b0].16b, %[a0].4b[1]\n"

            "ldr  %q[b2], [%[b_ptr], #32]\n"
            "sdot  v10.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0].4b[3]\n"
            "ldr  %q[a0a], [%[a_ptr], #32]\n"
            "sdot   v12.4s, %[b0].16b, %[a1].4b[0]\n"
            "sdot  v13.4s, %[b0].16b, %[a1].4b[1]\n"
            "ldr  %q[a1a], [%[a_ptr], #48]\n"
            "sdot  v14.4s, %[b0].16b, %[a1].4b[2]\n"
            "sdot  v15.4s, %[b0].16b, %[a1].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr], #48]\n"

            "sdot  v16.4s, %[b1].16b, %[a0].4b[0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0].4b[1]\n"
            ASM_PREFETCH("[%[a_ptr], #320]")
            "sdot  v18.4s, %[b1].16b, %[a0].4b[2]\n"
            "sdot  v19.4s, %[b1].16b, %[a0].4b[3]\n"
            "sdot  v20.4s, %[b1].16b, %[a1].4b[0]\n"
            "sdot  v21.4s, %[b1].16b, %[a1].4b[1]\n"
            "sdot  v22.4s, %[b1].16b, %[a1].4b[2]\n"
            "sdot  v23.4s, %[b1].16b, %[a1].4b[3]\n"
            "ldr  %q[b1], [%[b_ptr], #64]\n"

            "sdot  v24.4s, %[b2].16b, %[a0].4b[0]\n"
            "sdot  v25.4s, %[b2].16b, %[a0].4b[1]\n"
            ASM_PREFETCH("[%[b_ptr], #448]")
            "sdot  v26.4s, %[b2].16b, %[a0].4b[2]\n"
            "sdot  v27.4s, %[b2].16b, %[a0].4b[3]\n"
            "sdot  v28.4s, %[b2].16b, %[a1].4b[0]\n"
            "sdot  v29.4s, %[b2].16b, %[a1].4b[1]\n"
            "sdot  v30.4s, %[b2].16b, %[a1].4b[2]\n"
            "sdot  v31.4s, %[b2].16b, %[a1].4b[3]\n"
            "ldr  %q[b2], [%[b_ptr], #80]\n"

            "sdot  v8.4s , %[b0].16b, %[a0a].4b[0]\n"
            "sdot  v9.4s , %[b0].16b, %[a0a].4b[1]\n"
            "ldr  %q[a0], [%[a_ptr], #64]\n"
            "sdot  v10.4s, %[b0].16b, %[a0a].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0a].4b[3]\n"
            "sdot   v12.4s, %[b0].16b, %[a1a].4b[0]\n"
            "ldr  %q[a1], [%[a_ptr], #80]\n"
            "sdot   v13.4s, %[b0].16b, %[a1a].4b[1]\n"
            "sdot  v14.4s, %[b0].16b, %[a1a].4b[2]\n"
            "sdot  v15.4s, %[b0].16b, %[a1a].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr], #96]\n"

            "sdot  v16.4s, %[b1].16b, %[a0a].4b[0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0a].4b[1]\n"
            ASM_PREFETCH("[%[b_ptr], #512]")
            "sdot  v18.4s, %[b1].16b, %[a0a].4b[2]\n"
            "sdot  v19.4s, %[b1].16b, %[a0a].4b[3]\n"
            "sdot  v20.4s, %[b1].16b, %[a1a].4b[0]\n"
            "sdot  v21.4s, %[b1].16b, %[a1a].4b[1]\n"
            "sdot  v22.4s, %[b1].16b, %[a1a].4b[2]\n"
            "sdot  v23.4s, %[b1].16b, %[a1a].4b[3]\n"
            "ldr  %q[b1], [%[b_ptr], #112]\n"

            "sdot  v24.4s, %[b2].16b, %[a0a].4b[0]\n"
            "sdot  v25.4s, %[b2].16b, %[a0a].4b[1]\n"
            "add  %[a_ptr], %[a_ptr], #64\n"
            "sdot  v26.4s, %[b2].16b, %[a0a].4b[2]\n"
            "sdot  v27.4s, %[b2].16b, %[a0a].4b[3]\n"
            "add  %[b_ptr], %[b_ptr], #96\n"
            "sdot  v28.4s, %[b2].16b, %[a1a].4b[0]\n"
            "sdot  v29.4s, %[b2].16b, %[a1a].4b[1]\n"
            "subs  %w[k], %w[k], #1\n"
            "sdot  v30.4s, %[b2].16b, %[a1a].4b[2]\n"
            "sdot  v31.4s, %[b2].16b, %[a1a].4b[3]\n"
            "bne  1b\n"

            // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
            "4:\n"

            // Branch to alternative tail for odd K
            "cbnz  %w[oddk], 2f\n"

            // Detached final iteration (even K)
            "sdot  v8.4s , %[b0].16b, %[a0].4b[0]\n"
            "sdot   v9.4s , %[b0].16b, %[a0].4b[1]\n"
            "ldr  %q[b2], [%[b_ptr], #32]\n"
            "sdot  v10.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0].4b[3]\n"
            "ldr  %q[a0a], [%[a_ptr], #32]\n"
            "sdot   v12.4s, %[b0].16b, %[a1].4b[0]\n"
            "sdot   v13.4s, %[b0].16b, %[a1].4b[1]\n"
            "ldr  %q[a1a], [%[a_ptr], #48]\n"
            "sdot  v14.4s, %[b0].16b, %[a1].4b[2]\n"
            "sdot  v15.4s, %[b0].16b, %[a1].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr], #48]\n"

            "sdot  v16.4s, %[b1].16b, %[a0].4b[0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0].4b[1]\n"
            "sdot  v18.4s, %[b1].16b, %[a0].4b[2]\n"
            "sdot  v19.4s, %[b1].16b, %[a0].4b[3]\n"
            "sdot  v20.4s, %[b1].16b, %[a1].4b[0]\n"
            "sdot  v21.4s, %[b1].16b, %[a1].4b[1]\n"
            "sdot  v22.4s, %[b1].16b, %[a1].4b[2]\n"
            "sdot  v23.4s, %[b1].16b, %[a1].4b[3]\n"
            "ldr  %q[b1], [%[b_ptr], #64]\n"

            "sdot  v24.4s, %[b2].16b, %[a0].4b[0]\n"
            "sdot  v25.4s, %[b2].16b, %[a0].4b[1]\n"
            "add  %[a_ptr], %[a_ptr], #64\n"
            "sdot  v26.4s, %[b2].16b, %[a0].4b[2]\n"
            "sdot  v27.4s, %[b2].16b, %[a0].4b[3]\n"
            "sdot  v28.4s, %[b2].16b, %[a1].4b[0]\n"
            "sdot  v29.4s, %[b2].16b, %[a1].4b[1]\n"
            "sdot  v30.4s, %[b2].16b, %[a1].4b[2]\n"
            "sdot  v31.4s, %[b2].16b, %[a1].4b[3]\n"
            "ldr  %q[b2], [%[b_ptr], #80]\n"

            "sdot  v8.4s , %[b0].16b, %[a0a].4b[0]\n"

            "sdot  v16.4s, %[b1].16b, %[a0a].4b[0]\n"
            "add  %[b_ptr], %[b_ptr], #96\n"
            "sdot   v9.4s , %[b0].16b, %[a0a].4b[1]\n"
            "str  q8, [%[outptr0], #0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0a].4b[1]\n"
            "str  q16, [%[outptr0], #16]\n"
            "sdot  v24.4s, %[b2].16b, %[a0a].4b[0]\n"
            "str  q24, [%[outptr0], #32]\n"

            "sdot  v25.4s, %[b2].16b, %[a0a].4b[1]\n"
            "str  q9, [%[outptr1], #0]\n"
            "sdot  v10.4s, %[b0].16b, %[a0a].4b[2]\n"
            "str  q17, [%[outptr1], #16]\n"
            "sdot  v18.4s, %[b1].16b, %[a0a].4b[2]\n"
            "str  q25, [%[outptr1], #32]\n"
            "sdot  v26.4s, %[b2].16b, %[a0a].4b[2]\n"
            "str  q10, [%[outptr2], #0]\n"

            "sdot  v11.4s, %[b0].16b, %[a0a].4b[3]\n"
            "str  q18, [%[outptr2], #16]\n"
            "sdot  v19.4s, %[b1].16b, %[a0a].4b[3]\n"
            "str  q26, [%[outptr2], #32]\n"
            "sdot  v27.4s, %[b2].16b, %[a0a].4b[3]\n"
            "str  q11, [%[outptr3], #0]\n"

            "sdot   v12.4s, %[b0].16b, %[a1a].4b[0]\n"
            "str  q19, [%[outptr3], #16]\n"
            "sdot  v20.4s, %[b1].16b, %[a1a].4b[0]\n"
            "str  q27, [%[outptr3], #32]\n"
            "sdot  v28.4s, %[b2].16b, %[a1a].4b[0]\n"
            "str  q12, [%[outptr4], #0]\n"

            "sdot   v13.4s, %[b0].16b, %[a1a].4b[1]\n"
            "str  q20, [%[outptr4], #16]\n"
            "sdot  v21.4s, %[b1].16b, %[a1a].4b[1]\n"
            "str  q28, [%[outptr4], #32]\n"
            "sdot  v29.4s, %[b2].16b, %[a1a].4b[1]\n"
            "str  q13, [%[outptr5], #0]\n"

            "sdot  v14.4s, %[b0].16b, %[a1a].4b[2]\n"
            "str  q21, [%[outptr5], #16]\n"
            "sdot  v22.4s, %[b1].16b, %[a1a].4b[2]\n"
            "str  q29, [%[outptr5], #32]\n"
            "sdot  v30.4s, %[b2].16b, %[a1a].4b[2]\n"
            "str  q14, [%[outptr6], #0]\n"

            "sdot  v15.4s, %[b0].16b, %[a1a].4b[3]\n"
            "str  q22, [%[outptr6], #16]\n"
            "sdot  v23.4s, %[b1].16b, %[a1a].4b[3]\n"
            "str  q30, [%[outptr6], #32]\n"
            "sdot  v31.4s, %[b2].16b, %[a1a].4b[3]\n"
            "str  q15, [%[outptr7], #0]\n"

            "b  3f\n"

            // Detached final iteration (odd K)
            "2:\n"
            "sdot  v8.4s , %[b0].16b, %[a0].4b[0]\n"
            "ldr  %q[b2], [%[b_ptr], #32]\n"
            "sdot  v16.4s, %[b1].16b, %[a0].4b[0]\n"
            "sdot   v9.4s , %[b0].16b, %[a0].4b[1]\n"
            "str  q8, [%[outptr0], #0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0].4b[1]\n"
            "str  q16, [%[outptr0], #16]\n"
            "sdot  v24.4s, %[b2].16b, %[a0].4b[0]\n"
            "add  %[b_ptr], %[b_ptr], #48\n"
            "add  %[a_ptr], %[a_ptr], #32\n"
            "str  q24, [%[outptr0], #32]\n"
            "sdot  v25.4s, %[b2].16b, %[a0].4b[1]\n"
            "str  q9, [%[outptr1], #0]\n"

            "sdot  v10.4s, %[b0].16b, %[a0].4b[2]\n"
            "str  q17, [%[outptr1], #16]\n"
            "sdot  v18.4s, %[b1].16b, %[a0].4b[2]\n"
            "str  q25, [%[outptr1], #32]\n"
            "sdot  v26.4s, %[b2].16b, %[a0].4b[2]\n"
            "str  q10, [%[outptr2], #0]\n"

            "sdot  v11.4s, %[b0].16b, %[a0].4b[3]\n"
            "str  q18, [%[outptr2], #16]\n"
            "sdot  v19.4s, %[b1].16b, %[a0].4b[3]\n"
            "str  q26, [%[outptr2], #32]\n"
            "sdot  v27.4s, %[b2].16b, %[a0].4b[3]\n"
            "str  q11, [%[outptr3], #0]\n"

            "sdot   v12.4s, %[b0].16b, %[a1].4b[0]\n"
            "str  q19, [%[outptr3], #16]\n"
            "sdot  v20.4s, %[b1].16b, %[a1].4b[0]\n"
            "str  q27, [%[outptr3], #32]\n"
            "sdot  v28.4s, %[b2].16b, %[a1].4b[0]\n"
            "str  q12, [%[outptr4], #0]\n"

            "sdot   v13.4s, %[b0].16b, %[a1].4b[1]\n"
            "str  q20, [%[outptr4], #16]\n"
            "sdot  v21.4s, %[b1].16b, %[a1].4b[1]\n"
            "str  q28, [%[outptr4], #32]\n"
            "sdot  v29.4s, %[b2].16b, %[a1].4b[1]\n"
            "str  q13, [%[outptr5], #0]\n"

            "sdot  v14.4s, %[b0].16b, %[a1].4b[2]\n"
            "str  q21, [%[outptr5], #16]\n"
            "sdot  v22.4s, %[b1].16b, %[a1].4b[2]\n"
            "str  q29, [%[outptr5], #32]\n"
            "sdot  v30.4s, %[b2].16b, %[a1].4b[2]\n"
            "str  q14, [%[outptr6], #0]\n"

            "sdot  v15.4s, %[b0].16b, %[a1].4b[3]\n"
            "str  q22, [%[outptr6], #16]\n"
            "sdot  v23.4s, %[b1].16b, %[a1].4b[3]\n"
            "str  q30, [%[outptr6], #32]\n"
            "sdot  v31.4s, %[b2].16b, %[a1].4b[3]\n"
            "str  q15, [%[outptr7], #0]\n"


            // Common tail
            "3:\n"
            "str  q23, [%[outptr7], #16]\n"
            "str  q31, [%[outptr7], #32]\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr),[oddk] "+r" (oddk),
              [is_first_k] "+r" (is_first_k), [k] "+r" (k), [LDC] "+r" (LDC),
              [a0] "=w" (a0), [a1] "=w" (a1), [a0a] "=w" (a0a), [a1a] "=w" (a1a),
              [b0] "=w" (b0), [b1] "=w" (b1), [b2] "=w" (b2),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3),
              [outptr4] "=r"(outptr4), [outptr5] "=r"(outptr5),
              [outptr6] "=r"(outptr6), [outptr7] "=r"(outptr7)
            :
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14",
              "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc",
              "memory"
            );
}
#else
static void kern_8x12(const int8_t* packA, const int8_t* packB, int K,
                      int32_t* output, int LDC, bool is_first_k) {
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;

    int32x4_t a0;
    int32x4_t a1;
    int32x4_t b0;
    int32x4_t b1;
    int32x4_t b2;
    int32x4_t a0a;
    int32x4_t a1a;
    LDC = LDC * sizeof(int32_t);

    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    int32_t* outptr4;
    int32_t* outptr5;
    int32_t* outptr6;
    int32_t* outptr7;

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "add %[outptr4], %[outptr3], %x[LDC]\n"
            "add %[outptr5], %[outptr4], %x[LDC]\n"
            "add %[outptr6], %[outptr5], %x[LDC]\n"
            "add %[outptr7], %[outptr6], %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"
            // we can not use ld1, as it can not encode {v8, v16, v24}
            "ldp q8, q16, [%[outptr0]]\n"
            "ldr q24, [%[outptr0], #32]\n"
            "ldp q9, q17, [%[outptr1]]\n"
            "ldr q25, [%[outptr1], #32]\n"
            "ldp q10, q18, [%[outptr2]]\n"
            "ldr q26, [%[outptr2], #32]\n"
            "ldp q11, q19, [%[outptr3]]\n"
            "ldr q27, [%[outptr3], #32]\n"
            "ldp q12, q20, [%[outptr4]]\n"
            "ldr q28, [%[outptr4], #32]\n"
            "ldp q13, q21, [%[outptr5]]\n"
            "ldr q29, [%[outptr5], #32]\n"
            "ldp q14, q22, [%[outptr6]]\n"
            "ldr q30, [%[outptr6], #32]\n"
            "ldp q15, q23, [%[outptr7]]\n"
            "ldr q31, [%[outptr7], #32]\n"
            "b 2f\n"

            "1:\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v15.16b, v15.16b, v15.16b\n"

            "eor v16.16b, v16.16b, v16.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"
            "eor v18.16b, v18.16b, v18.16b\n"
            "eor v19.16b, v19.16b, v19.16b\n"
            "eor v20.16b, v20.16b, v20.16b\n"
            "eor v21.16b, v21.16b, v21.16b\n"
            "eor v22.16b, v22.16b, v22.16b\n"
            "eor v23.16b, v23.16b, v23.16b\n"

            "eor v24.16b, v24.16b, v24.16b\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"
            "eor v31.16b, v31.16b, v31.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"
            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "ldr  %q[b2], [%[b_ptr]], #16\n"
            "sdot  v8.4s, %[b0].16b, %[a0].4b[0]\n"
            "sdot  v9.4s, %[b0].16b, %[a0].4b[1]\n"
            "sdot  v10.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0].4b[3]\n"
            "sdot  v12.4s, %[b0].16b, %[a1].4b[0]\n"
            "sdot  v13.4s, %[b0].16b, %[a1].4b[1]\n"
            "sdot  v14.4s, %[b0].16b, %[a1].4b[2]\n"
            "sdot  v15.4s, %[b0].16b, %[a1].4b[3]\n"
            "sdot  v16.4s, %[b1].16b, %[a0].4b[0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0].4b[1]\n"
            "sdot  v18.4s, %[b1].16b, %[a0].4b[2]\n"
            "sdot  v19.4s, %[b1].16b, %[a0].4b[3]\n"
            "sdot  v20.4s, %[b1].16b, %[a1].4b[0]\n"
            "sdot  v21.4s, %[b1].16b, %[a1].4b[1]\n"
            "sdot  v22.4s, %[b1].16b, %[a1].4b[2]\n"
            "sdot  v23.4s, %[b1].16b, %[a1].4b[3]\n"
            "sdot  v24.4s, %[b2].16b, %[a0].4b[0]\n"
            "sdot  v25.4s, %[b2].16b, %[a0].4b[1]\n"
            "sdot  v26.4s, %[b2].16b, %[a0].4b[2]\n"
            "sdot  v27.4s, %[b2].16b, %[a0].4b[3]\n"
            "sdot  v28.4s, %[b2].16b, %[a1].4b[0]\n"
            "sdot  v29.4s, %[b2].16b, %[a1].4b[1]\n"
            "sdot  v30.4s, %[b2].16b, %[a1].4b[2]\n"
            "sdot  v31.4s, %[b2].16b, %[a1].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[a1a], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "ldr  %q[b2], [%[b_ptr]], #16\n"
            "sdot  v8.4s, %[b0].16b, %[a0].4b[0]\n"
            "sdot  v9.4s, %[b0].16b, %[a0].4b[1]\n"
            "sdot  v10.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0].4b[3]\n"
            "sdot  v12.4s, %[b0].16b, %[a1].4b[0]\n"
            "sdot  v13.4s, %[b0].16b, %[a1].4b[1]\n"
            "sdot  v14.4s, %[b0].16b, %[a1].4b[2]\n"
            "sdot  v15.4s, %[b0].16b, %[a1].4b[3]\n"
            "sdot  v16.4s, %[b1].16b, %[a0].4b[0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0].4b[1]\n"
            "sdot  v18.4s, %[b1].16b, %[a0].4b[2]\n"
            "sdot  v19.4s, %[b1].16b, %[a0].4b[3]\n"
            "sdot  v20.4s, %[b1].16b, %[a1].4b[0]\n"
            "sdot  v21.4s, %[b1].16b, %[a1].4b[1]\n"
            "sdot  v22.4s, %[b1].16b, %[a1].4b[2]\n"
            "sdot  v23.4s, %[b1].16b, %[a1].4b[3]\n"
            "sdot  v24.4s, %[b2].16b, %[a0].4b[0]\n"
            "sdot  v25.4s, %[b2].16b, %[a0].4b[1]\n"
            "sdot  v26.4s, %[b2].16b, %[a0].4b[2]\n"
            "sdot  v27.4s, %[b2].16b, %[a0].4b[3]\n"
            "sdot  v28.4s, %[b2].16b, %[a1].4b[0]\n"
            "sdot  v29.4s, %[b2].16b, %[a1].4b[1]\n"
            "sdot  v30.4s, %[b2].16b, %[a1].4b[2]\n"
            "sdot  v31.4s, %[b2].16b, %[a1].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "ldr  %q[b2], [%[b_ptr]], #16\n"
            "sdot  v8.4s, %[b0].16b, %[a0a].4b[0]\n"
            "sdot  v9.4s, %[b0].16b, %[a0a].4b[1]\n"
            "sdot  v10.4s, %[b0].16b, %[a0a].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0a].4b[3]\n"
            "sdot  v12.4s, %[b0].16b, %[a1a].4b[0]\n"
            "sdot  v13.4s, %[b0].16b, %[a1a].4b[1]\n"
            "sdot  v14.4s, %[b0].16b, %[a1a].4b[2]\n"
            "sdot  v15.4s, %[b0].16b, %[a1a].4b[3]\n"
            "sdot  v16.4s, %[b1].16b, %[a0a].4b[0]\n"
            "sdot  v17.4s, %[b1].16b, %[a0a].4b[1]\n"
            "sdot  v18.4s, %[b1].16b, %[a0a].4b[2]\n"
            "sdot  v19.4s, %[b1].16b, %[a0a].4b[3]\n"
            "sdot  v20.4s, %[b1].16b, %[a1a].4b[0]\n"
            "sdot  v21.4s, %[b1].16b, %[a1a].4b[1]\n"
            "sdot  v22.4s, %[b1].16b, %[a1a].4b[2]\n"
            "sdot  v23.4s, %[b1].16b, %[a1a].4b[3]\n"
            "sdot  v24.4s, %[b2].16b, %[a0a].4b[0]\n"
            "sdot  v25.4s, %[b2].16b, %[a0a].4b[1]\n"
            "sdot  v26.4s, %[b2].16b, %[a0a].4b[2]\n"
            "sdot  v27.4s, %[b2].16b, %[a0a].4b[3]\n"
            "sdot  v28.4s, %[b2].16b, %[a1a].4b[0]\n"
            "sdot  v29.4s, %[b2].16b, %[a1a].4b[1]\n"
            "sdot  v30.4s, %[b2].16b, %[a1a].4b[2]\n"
            "sdot  v31.4s, %[b2].16b, %[a1a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n"
            "stp q8, q16, [%[outptr0]]\n"
            "str q24, [%[outptr0], #32]\n"
            "stp q9, q17, [%[outptr1]]\n"
            "str q25, [%[outptr1], #32]\n"
            "stp q10, q18, [%[outptr2]]\n"
            "str q26, [%[outptr2], #32]\n"
            "stp q11, q19, [%[outptr3]]\n"
            "str q27, [%[outptr3], #32]\n"
            "stp q12, q20, [%[outptr4]]\n"
            "str q28, [%[outptr4], #32]\n"
            "stp q13, q21, [%[outptr5]]\n"
            "str q29, [%[outptr5], #32]\n"
            "stp q14, q22, [%[outptr6]]\n"
            "str q30, [%[outptr6], #32]\n"
            "stp q15, q23, [%[outptr7]]\n"
            "str q31, [%[outptr7], #32]\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [a0] "+w"(a0),
              [a1] "+w"(a1), [a0a] "+w"(a0a), [a1a] "+w"(a1a), [b0] "+w"(b0),
              [b1] "+w"(b1), [b2] "+w"(b2), [k] "+r"(k), [LDC] "+r"(LDC),
              [oddk] "+r"(oddk), [is_first_k] "+r"(is_first_k),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3),
              [outptr4] "=r"(outptr4), [outptr5] "=r"(outptr5),
              [outptr6] "=r"(outptr6), [outptr7] "=r"(outptr7)
            :
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory");
}

#endif

// Overview of register layout:
//
// A 12x4 cell of Rhs is stored in 8bit in q2-q4.
// A 8x4x2 cell of Lhs is stored in 8bit in q0-q1,q5-q6
// A 8x12 block of accumulators is stored in 8bit in q8--q31.
//
//                            +--------+--------+--------+
//                            |v1[0-16]|v2[0-16]|v3[0-16]|
//                       Rhs  +--------+--------+--------+
//                            |v5[0-16]|v6[0-16]|v7[0-16]|
//                            +--------+--------+--------+
//
//                            |        |        |        |
//
//    Lhs                     |        |        |        |
//
//  +-------+-------+ - - - - +--------+--------+--------+
//  |v0[0-4]|v4[0-4]|         | v8[0-4]|v12[0-4]|v16[0-4]|
//  |v0[0-4]|v4[0-4]|         | v9[0-4]|v13[0-4]|v17[0-4]|
//  |v0[0-4]|v4[0-4]|         |v10[0-4]|v14[0-4]|v18[0-4]|
//  |v0[0-4]|v4[0-4]|         |v11[0-4]|v15[0-4]|v19[0-4]|
//  +-------+-------+ - - - - +--------+--------+--------+
//
//                            Accumulator

static void kern_4x12(const int8_t* packA, const int8_t* packB, int K,
                      int32_t* output, int LDC, bool is_first_k, int m_remain) {
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;
    int32x4_t a0;
    int32x4_t b0;
    int32x4_t b1;
    int32x4_t b2;
    int32x4_t a0a;
    int32x4_t b0a;
    int32x4_t b1a;
    int32x4_t b2a;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    size_t x0;

// clang-format off
#define LOAD_LINE(v1, v2, v3, m)            \
    "cbz %[x0], 100f\n"                        \
    "ldp " v1 "," v2 ", [%[outptr" m "]]\n" \
    "ldr " v3 ",  [%[outptr" m "], #32]\n"  \
    "subs %[x0], %[x0], #1\n"

#define LOAD_C \
    "mov %[x0], %x[m_remain]\n"            \
    LOAD_LINE("q8", "q12", "q16", "0")  \
    LOAD_LINE("q9", "q13", "q17", "1")  \
    LOAD_LINE("q10", "q14", "q18", "2") \
    LOAD_LINE("q11", "q15", "q19", "3") \
    "100:\n"

#define STORE_LINE(v1, v2, v3, m)          \
    "cbz %[x0], 101f\n"                       \
    "stp " v1 "," v2", [%[outptr" m "]]\n" \
    "str " v3 ", [%[outptr" m "], #32]\n"  \
    "subs %[x0], %[x0], #1\n"

#define STORE_C \
    "mov %[x0], %x[m_remain]\n"             \
    STORE_LINE("q8", "q12", "q16", "0")  \
    STORE_LINE("q9", "q13", "q17", "1")  \
    STORE_LINE("q10", "q14", "q18", "2") \
    STORE_LINE("q11", "q15", "q19", "3") \
    "101:\n"

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v15.16b, v15.16b, v15.16b\n"
            "eor v16.16b, v16.16b, v16.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"
            "eor v18.16b, v18.16b, v18.16b\n"
            "eor v19.16b, v19.16b, v19.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"

            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "ldr  %q[b2], [%[b_ptr]], #16\n"
            "sdot  v8.4s, %[b0].16b, %[a0].4b[0]\n"
            "sdot    v9.4s, %[b0].16b, %[a0].4b[1]\n"
            "sdot  v10.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0].4b[3]\n"
            "sdot  v12.4s, %[b1].16b, %[a0].4b[0]\n"
            "sdot    v13.4s, %[b1].16b, %[a0].4b[1]\n"
            "sdot  v14.4s, %[b1].16b, %[a0].4b[2]\n"
            "sdot  v15.4s, %[b1].16b, %[a0].4b[3]\n"
            "sdot  v16.4s, %[b2].16b, %[a0].4b[0]\n"
            "sdot    v17.4s, %[b2].16b, %[a0].4b[1]\n"
            "sdot  v18.4s, %[b2].16b, %[a0].4b[2]\n"
            "sdot  v19.4s, %[b2].16b, %[a0].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "ldr  %q[b2], [%[b_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "ldr  %q[b1a], [%[b_ptr]], #16\n"
            "ldr  %q[b2a], [%[b_ptr]], #16\n"

            "sdot  v8.4s, %[b0].16b, %[a0].4b[0]\n"
            "sdot    v9.4s, %[b0].16b, %[a0].4b[1]\n"
            "sdot  v10.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v11.4s, %[b0].16b, %[a0].4b[3]\n"
            "sdot  v12.4s, %[b1].16b, %[a0].4b[0]\n"
            "sdot    v13.4s, %[b1].16b, %[a0].4b[1]\n"
            "sdot  v14.4s, %[b1].16b, %[a0].4b[2]\n"
            "sdot  v15.4s, %[b1].16b, %[a0].4b[3]\n"
            "sdot  v16.4s, %[b2].16b, %[a0].4b[0]\n"
            "sdot    v17.4s, %[b2].16b, %[a0].4b[1]\n"
            "sdot  v18.4s, %[b2].16b, %[a0].4b[2]\n"
            "sdot  v19.4s, %[b2].16b, %[a0].4b[3]\n"
            "sdot  v8.4s , %[b0a].16b, %[a0a].4b[0]\n"
            "sdot    v9.4s , %[b0a].16b, %[a0a].4b[1]\n"
            "sdot  v10.4s, %[b0a].16b, %[a0a].4b[2]\n"
            "sdot  v11.4s, %[b0a].16b, %[a0a].4b[3]\n"
            "sdot  v12.4s, %[b1a].16b, %[a0a].4b[0]\n"
            "sdot    v13.4s, %[b1a].16b, %[a0a].4b[1]\n"
            "sdot  v14.4s, %[b1a].16b, %[a0a].4b[2]\n"
            "sdot  v15.4s, %[b1a].16b, %[a0a].4b[3]\n"
            "sdot  v16.4s, %[b2a].16b, %[a0a].4b[0]\n"
            "sdot    v17.4s, %[b2a].16b, %[a0a].4b[1]\n"
            "sdot  v18.4s, %[b2a].16b, %[a0a].4b[2]\n"
            "sdot  v19.4s, %[b2a].16b, %[a0a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k),
              [outptr0] "+r"(outptr0), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [m_remain] "+r"(m_remain),
              [LDC] "+r"(LDC), [a0] "=w"(a0), [a0a] "=w"(a0a), [b0] "=w"(b0),
              [b1] "=w"(b1), [b2] "=w"(b2), [b0a] "=w"(b0a), [b1a] "=w"(b1a),
              [b2a] "=w"(b2a), [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [x0] "=r"(x0)
            :
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "memory", "cc");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A (4x4)x2 cell of Rhs is stored in 8bit in q2-q3.
// A 4x4x2 cell of Lhs is stored in 8bit in q0-q1, q4-a5
// A 8x4 block of accumulators is stored in 8bit in q4--q7.
//
//                            +--------+
//                            |v2[0-16]|
//                       Rhs  +--------+
//                            |v3[0-16]|
//                            +--------+
//                            |        |
//
//    Lhs                     |        |
//
//  +-------+-------+ - - - - +--------+
//  |v0[0-4]|v4[0-4]|         | v6[0-4]|
//  |v0[0-4]|v4[0-4]|         | v7[0-4]|
//  |v0[0-4]|v4[0-4]|         | v8[0-4]|
//  |v0[0-4]|v4[0-4]|         | v9[0-4]|
//  |v1[0-4]|v5[0-4]|         |v10[0-4]|
//  |v1[0-4]|v5[0-4]|         |v11[0-4]|
//  |v1[0-4]|v5[0-4]|         |v12[0-4]|
//  |v1[0-4]|v5[0-4]|         |v13[0-4]|
//  +-------+-------+ - - - - +---------+
//
//                            Accumulator

static void kern_8x4(const int8_t* packA, const int8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int n_remain) {
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;
    int32x4_t a0;
    int32x4_t a1;
    int32x4_t b0;
    int32x4_t b0a;
    int32x4_t a0a;
    int32x4_t a1a;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    int32_t* outptr4;
    int32_t* outptr5;
    int32_t* outptr6;
    int32_t* outptr7;

    size_t x0;

// clang-format off
#define LOAD_LINE(reg_index, n)             \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                \
    "blt 100" n "f\n"                       \
    "ldr q" reg_index ", [%[x0]] \n"           \
    "b 101" n "f\n"                         \
    "100" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 101" n "f\n"                       \
    "ld1 {v" reg_index ".s}[0], [%[x0]], #4\n" \
    "cmp %w[n_remain], #1\n"                \
    "beq 101" n "f\n"                       \
    "ld1 {v" reg_index ".s}[1], [%[x0]], #4\n" \
    "cmp %w[n_remain], #2\n"                \
    "beq 101" n "f\n"                       \
    "ld1 {v" reg_index ".s}[2], [%[x0]], #4\n" \
    "101" n ":\n"


#define LOAD_C           \
    LOAD_LINE("6", "0")  \
    LOAD_LINE("7", "1")  \
    LOAD_LINE("8", "2")  \
    LOAD_LINE("9", "3")  \
    LOAD_LINE("10", "4") \
    LOAD_LINE("11", "5") \
    LOAD_LINE("12", "6") \
    LOAD_LINE("13", "7")

#define STORE_LINE(reg_index, n)            \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                \
    "blt 102" n "f\n"                       \
    "str q" reg_index ", [%[x0]]\n"            \
    "b 103" n "f\n"                         \
    "102" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 103" n "f\n"                       \
    "st1 {v" reg_index ".s}[0], [%[x0]], #4\n" \
    "cmp %w[n_remain], #1\n"                \
    "beq 103" n "f\n"                       \
    "st1 {v" reg_index ".s}[1], [%[x0]], #4\n" \
    "cmp %w[n_remain], #2\n"                \
    "beq 103" n "f\n"                       \
    "st1 {v" reg_index ".s}[2], [%[x0]], #4\n" \
    "103" n ":\n"

#define STORE_C           \
    STORE_LINE("6", "0")  \
    STORE_LINE("7", "1")  \
    STORE_LINE("8", "2")  \
    STORE_LINE("9", "3")  \
    STORE_LINE("10", "4") \
    STORE_LINE("11", "5") \
    STORE_LINE("12", "6") \
    STORE_LINE("13", "7")

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "add %[outptr4], %[outptr3], %x[LDC]\n"
            "add %[outptr5], %[outptr4], %x[LDC]\n"
            "add %[outptr6], %[outptr5], %x[LDC]\n"
            "add %[outptr7], %[outptr6], %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"

            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "sdot  v6.4s , %[b0].16b, %[a0].4b[0]\n"
            "sdot  v7.4s , %[b0].16b, %[a0].4b[1]\n"
            "sdot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "sdot  v10.4s, %[b0].16b, %[a1].4b[0]\n"
            "sdot  v11.4s, %[b0].16b, %[a1].4b[1]\n"
            "sdot  v12.4s, %[b0].16b, %[a1].4b[2]\n"
            "sdot  v13.4s, %[b0].16b, %[a1].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[a1a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "sdot  v6.4s , %[b0].16b, %[a0].4b[0]\n"
            "sdot  v7.4s , %[b0].16b, %[a0].4b[1]\n"
            "sdot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "sdot  v10.4s, %[b0].16b, %[a1].4b[0]\n"
            "sdot  v11.4s, %[b0].16b, %[a1].4b[1]\n"
            "sdot  v12.4s, %[b0].16b, %[a1].4b[2]\n"
            "sdot  v13.4s, %[b0].16b, %[a1].4b[3]\n"

            "sdot  v6.4s , %[b0a].16b, %[a0a].4b[0]\n"
            "sdot  v7.4s , %[b0a].16b, %[a0a].4b[1]\n"
            "sdot  v8.4s, %[b0a].16b, %[a0a].4b[2]\n"
            "sdot  v9.4s, %[b0a].16b, %[a0a].4b[3]\n"
            "sdot  v10.4s, %[b0a].16b, %[a1a].4b[0]\n"
            "sdot  v11.4s, %[b0a].16b, %[a1a].4b[1]\n"
            "sdot  v12.4s, %[b0a].16b, %[a1a].4b[2]\n"
            "sdot  v13.4s, %[b0a].16b, %[a1a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [LDC] "+r"(LDC),
              [oddk] "+r"(oddk), [is_first_k] "+r"(is_first_k),
              [n_remain] "+r"(n_remain), [k] "+r"(k), [outptr0] "+r"(outptr0),
              [a0] "=w"(a0), [a1] "=w"(a1), [a0a] "=w"(a0a), [a1a] "=w"(a1a),
              [b0] "=w"(b0), [b0a] "=w"(b0a), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3),
              [outptr4] "=r"(outptr4), [outptr5] "=r"(outptr5),
              [outptr6] "=r"(outptr6), [outptr7] "=r"(outptr7), [x0] "=r"(x0)
            :
            : "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "memory",
              "cc");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A 4x4x2 cell of Rhs is stored in 8bit in q2-q3.
// A 4x4x2 cell of Lhs is stored in 8bit in q0-q1
// A 4x4x2 block of accumulators is stored in 8bit in q4--q7.
//
//                            +--------+
//                            | v2[0-7]|
//                       Rhs  +--------+
//                            | v3[0-7]|
//                            +--------+
//                            |        |
//
//    Lhs                     |        |
//
//  +-------+-------+ - - - - +--------+
//  |v0[0-4]|v1[0-4]|         | v4[0-7]|
//  |v0[0-4]|v1[0-4]|         | v5[0-7]|
//  |v0[0-4]|v1[0-4]|         | v6[0-7]|
//  |v0[0-4]|v1[0-4]|         | v7[0-7]|
//  +-------+-------+ - - - - +--------+
//
//                            Accumulator

static void kern_4x4(const int8_t* packA, const int8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 4;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(packA);
    const int32_t* b_ptr = reinterpret_cast<const int32_t*>(packB);
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;
    int32x4_t a0;
    int32x4_t a0a;
    int32x4_t b0;
    int32x4_t b0a;
    LDC = LDC * sizeof(int32_t);

    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    size_t x0, x1;

// clang-format off
#define LOAD_LINE(reg_index, n)                          \
    "cbz %[x1], 102f\n"                                     \
    "mov %[x0], %[outptr" n "]\n"                           \
    "cmp %w[n_remain], #4\n"                             \
    "blt 100" n "f\n"                                    \
    "ldr q" reg_index ", [%[x0]]\n"                         \
    "b 101" n "f\n"                                      \
    "100" n ":\n"                                        \
    "cmp %w[n_remain], #0\n"                             \
    "beq 101" n "f\n"                                    \
    "ld1 {v" reg_index ".s}[0], [%[x0]], #4\n"              \
    "cmp %w[n_remain], #1\n"                             \
    "beq 101" n "f\n"                                    \
    "ld1 {v" reg_index ".s}[1], [%[x0]], #4\n"              \
    "cmp %w[n_remain], #2\n"                             \
    "beq 101" n "f\n"                                    \
    "ld1 {v" reg_index ".s}[2], [%[x0]], #4\n"              \
    "101" n ":\n" \
    "subs %[x1], %[x1], #1\n"

#define LOAD_C               \
    "mov %[x1], %x[m_remain]\n" \
    LOAD_LINE("4", "0")      \
    LOAD_LINE("5", "1")      \
    LOAD_LINE("6", "2")      \
    LOAD_LINE("7", "3")      \
    "102:\n"

#define STORE_LINE(reg_index, n)            \
    "cbz %[x1], 105f\n"                        \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                \
    "blt 103" n "f\n"                       \
    "str q" reg_index ", [%[x0]]\n"            \
    "b 104" n "f\n"                         \
    "103" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 104" n "f\n"                       \
    "st1 {v" reg_index ".s}[0], [%[x0]], #4\n" \
    "cmp %w[n_remain], #1\n"                \
    "beq 104" n "f\n"                       \
    "st1 {v" reg_index ".s}[1], [%[x0]], #4\n" \
    "cmp %w[n_remain], #2\n"                \
    "beq 104" n "f\n"                       \
    "st1 {v" reg_index ".s}[2], [%[x0]], #4\n" \
    "104" n ":\n" \
    "subs %[x1], %[x1], #1\n"

#define STORE_C              \
    "mov %[x1], %x[m_remain]\n" \
    STORE_LINE("4", "0")     \
    STORE_LINE("5", "1")     \
    STORE_LINE("6", "2")     \
    STORE_LINE("7", "3")     \
    "105:\n"

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"  //
            LOAD_C      //

            "b 2f\n"

            "1:\n"
            "eor v4.16b,  v4.16b,  v4.16b\n"
            "eor v5.16b,  v5.16b,  v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"

            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "sdot  v4.4s , %[b0].16b, %[a0].4b[0]\n"
            "sdot  v5.4s , %[b0].16b, %[a0].4b[1]\n"
            "sdot  v6.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v7.4s, %[b0].16b, %[a0].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "sdot  v4.4s , %[b0].16b, %[a0].4b[0]\n"
            "sdot  v5.4s , %[b0].16b, %[a0].4b[1]\n"
            "sdot  v6.4s, %[b0].16b, %[a0].4b[2]\n"
            "sdot  v7.4s, %[b0].16b, %[a0].4b[3]\n"
            "sdot  v4.4s , %[b0a].16b, %[a0a].4b[0]\n"
            "sdot  v5.4s , %[b0a].16b, %[a0a].4b[1]\n"
            "sdot  v6.4s, %[b0a].16b, %[a0a].4b[2]\n"
            "sdot  v7.4s, %[b0a].16b, %[a0a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [n_remain] "+r"(n_remain),
              [m_remain] "+r"(m_remain), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [k] "+r"(k), [a0] "=w"(a0),
              [a0a] "=w"(a0a), [b0] "=w"(b0), [b0a] "=w"(b0a),
              [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [x0] "=r"(x0), [x1] "=r"(x1)
            :
            : "v4", "v5", "v6", "v7", "memory", "cc");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_s8_8x12_pack_A_n(dt_int8* outptr, const dt_int8* inptr,
                                  int ldin, int y0, int ymax, int k0,
                                  int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 7 < ymax; y += 8) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        const int8_t* inptr6 = inptr5 + ldin;
        const int8_t* inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int K = kmax - k0;
        //! read 8 * 4 in each row
        for (; K > 15; K -= 16) {
            interleave_8x4_4_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr);
        }

        if (K > 0) {
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr, 4, K);
        }
    }
    for (; y < ymax; y += 4) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        //! read 4 * 4 in each row
        for (; K > 15; K -= 16) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, K);
        }
    }
}

static void gemm_s8_8x12_pack_A_t(dt_int8* out, const dt_int8* in, int ldin,
                                  int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize8 = round_up<int>(ksize, 4) * 8;
    const int ksize4 = round_up(ksize, 4) * 4;
    int8_t* outptr = out;
    int8_t* outptr_base = out;
    //! 4x4 block output start pos
    int8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = in + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = x0;
        outptr = outptr_base;
        for (; x + 7 < xmax; x += 8) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_8x4_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += ksize8;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, 4);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 8 * 4;
        outptr_base4 += 4 * 4;
    }
}

static void gemm_s8_8x12_pack_B_n(dt_int8* out, const dt_int8* in, int ldin,
                                  int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize12 = round_up<int>(ksize, 4) * 12;
    const int ksize4 = round_up(ksize, 4) * 4;
    int8_t* outptr = out;
    int8_t* outptr_base = out;
    //! 4x4 block output start pos
    int8_t* outptr_base4 = out + ((xmax - x0) / 12) * ksize12;

    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = in + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = x0;
        outptr = outptr_base;
        for (; x + 11 < xmax; x += 12) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_12x4_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += ksize12;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, 4);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 12 * 4;
        outptr_base4 += 4 * 4;
    }
}

static void gemm_s8_8x12_pack_B_t(dt_int8* outptr, const dt_int8* inptr,
                                  int ldin, int y0, int ymax, int k0,
                                  int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 11 < ymax; y += 12) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        const int8_t* inptr6 = inptr5 + ldin;
        const int8_t* inptr7 = inptr6 + ldin;
        const int8_t* inptr8 = inptr7 + ldin;
        const int8_t* inptr9 = inptr8 + ldin;
        const int8_t* inptr10 = inptr9 + ldin;
        const int8_t* inptr11 = inptr10 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);
        prefetch_2x(inptr8);
        prefetch_2x(inptr9);
        prefetch_2x(inptr10);
        prefetch_2x(inptr11);

        int K = kmax - k0;
        //! read 12 * 4 in each row
        for (; K > 15; K -= 16) {
            interleave_12x4_4_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                                inptr6, inptr7, inptr8, inptr9, inptr10,
                                inptr11, outptr);
        }

        if (K > 0) {
            interleave_12(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                          inptr6, inptr7, inptr8, inptr9, inptr10, inptr11,
                          outptr, 4, K);
        }
    }
    for (; y < ymax; y += 4) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        //! read 4 * 4 in each row
        for (; K > 15; K -= 16) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, K);
        }
    }
}

}  // namespace matmul_8x12x4
}  // namespace aarch64
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
