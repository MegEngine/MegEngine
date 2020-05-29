/**
 * \file dnn/src/aarch64/matrix_mul/int8_dot/kernel_mk4_8x12x4.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#if __ARM_FEATURE_DOTPROD

#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_mk4_8x12x4 {

// Overview of register layout:
//
// A 12x4 cell of Rhs is stored in 8bit in q2-q4.
// A 8x4x2 cell of Lhs is stored in 8bit in q0-q1,q5-q6
// A 8x12 block of accumulators is stored in 8bit in q8--q31.
//
//                              +------------+------------+------------+
//                              |    v2[0-16]|    v3[0-16]|    v4[0-16]|
//                         Rhs  +------------+------------+------------+
//
//                              |            |            |            |
//
//    Lhs                       |            |            |            |
//
//  +--------+--------+ - - - - +------------+------------+------------+
//  |v0[0-16]|v5[0-16]|         | v8 v9v10v11|v16v17v18v19|v24v25v26v27|
//  |v1[0-16]|v6[0-16]|         |v12v13v14v15|v20v21v22v23|v28v29v30v31|
//  +--------+--------+ - - - - +------------+------------+------------+
//
//                            Accumulator

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

    asm volatile (
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 5f\n"
            // we can not use ld1, as it can not encode {v8, v16, v24}
            "ldp q8, q9, [%[outptr0]]\n"
            "ldp q10, q11, [%[outptr0], #32]\n"
            "ldp q16, q17, [%[outptr0], #64]\n"
            "ldp q18, q19, [%[outptr0], #96]\n"
            "ldp q24, q25, [%[outptr0], #128]\n"
            "ldp q26, q27, [%[outptr0], #160]\n"
            "ldp q12, q13, [%[outptr1]]\n"
            "ldp q14, q15, [%[outptr1], #32]\n"
            "ldp q20, q21, [%[outptr1], #64]\n"
            "ldp q22, q23, [%[outptr1], #96]\n"
            "ldp q28, q29, [%[outptr1], #128]\n"
            "ldp q30, q31, [%[outptr1], #160]\n"
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
            "sdot  v8.4s , %[a0].16b, %[b0].4b[0]\n"
            "sdot    v9.4s , %[a0].16b, %[b0].4b[1]\n"

            "ldr  %q[b2], [%[b_ptr], #32]\n"
            "sdot  v10.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v11.4s, %[a0].16b, %[b0].4b[3]\n"
            "ldr  %q[a0a], [%[a_ptr], #32]\n"
            "sdot   v12.4s, %[a1].16b, %[b0].4b[0]\n"
            "sdot  v13.4s, %[a1].16b, %[b0].4b[1]\n"
            "ldr  %q[a1a], [%[a_ptr], #48]\n"
            "sdot  v14.4s, %[a1].16b, %[b0].4b[2]\n"
            "sdot  v15.4s, %[a1].16b, %[b0].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr], #48]\n"

            "sdot  v16.4s, %[a0].16b, %[b1].4b[0]\n"
            "sdot  v17.4s, %[a0].16b, %[b1].4b[1]\n"
            ASM_PREFETCH("[%[a_ptr], #320]")
            "sdot  v18.4s, %[a0].16b, %[b1].4b[2]\n"
            "sdot  v19.4s, %[a0].16b, %[b1].4b[3]\n"
            "sdot  v20.4s, %[a1].16b, %[b1].4b[0]\n"
            "sdot  v21.4s, %[a1].16b, %[b1].4b[1]\n"
            "sdot  v22.4s, %[a1].16b, %[b1].4b[2]\n"
            "sdot  v23.4s, %[a1].16b, %[b1].4b[3]\n"
            "ldr  %q[b1], [%[b_ptr], #64]\n"

            "sdot  v24.4s, %[a0].16b, %[b2].4b[0]\n"
            "sdot  v25.4s, %[a0].16b, %[b2].4b[1]\n"
            ASM_PREFETCH("[%[b_ptr], #448]")
            "sdot  v26.4s, %[a0].16b, %[b2].4b[2]\n"
            "sdot  v27.4s, %[a0].16b, %[b2].4b[3]\n"
            "sdot  v28.4s, %[a1].16b, %[b2].4b[0]\n"
            "sdot  v29.4s, %[a1].16b, %[b2].4b[1]\n"
            "sdot  v30.4s, %[a1].16b, %[b2].4b[2]\n"
            "sdot  v31.4s, %[a1].16b, %[b2].4b[3]\n"
            "ldr  %q[b2], [%[b_ptr], #80]\n"

            "sdot  v8.4s , %[a0a].16b, %[b0].4b[0]\n"
            "sdot  v9.4s , %[a0a].16b, %[b0].4b[1]\n"
            "ldr  %q[a0], [%[a_ptr], #64]\n"
            "sdot  v10.4s, %[a0a].16b, %[b0].4b[2]\n"
            "sdot  v11.4s, %[a0a].16b, %[b0].4b[3]\n"
            "sdot   v12.4s, %[a1a].16b, %[b0].4b[0]\n"
            "ldr  %q[a1], [%[a_ptr], #80]\n"
            "sdot   v13.4s, %[a1a].16b, %[b0].4b[1]\n"
            "sdot  v14.4s, %[a1a].16b, %[b0].4b[2]\n"
            "sdot  v15.4s, %[a1a].16b, %[b0].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr], #96]\n"

            "sdot  v16.4s, %[a0a].16b, %[b1].4b[0]\n"
            "sdot  v17.4s, %[a0a].16b, %[b1].4b[1]\n"
            ASM_PREFETCH("[%[b_ptr], #512]")
            "sdot  v18.4s, %[a0a].16b, %[b1].4b[2]\n"
            "sdot  v19.4s, %[a0a].16b, %[b1].4b[3]\n"
            "sdot  v20.4s, %[a1a].16b, %[b1].4b[0]\n"
            "sdot  v21.4s, %[a1a].16b, %[b1].4b[1]\n"
            "sdot  v22.4s, %[a1a].16b, %[b1].4b[2]\n"
            "sdot  v23.4s, %[a1a].16b, %[b1].4b[3]\n"
            "ldr  %q[b1], [%[b_ptr], #112]\n"

            "sdot  v24.4s, %[a0a].16b, %[b2].4b[0]\n"
            "sdot  v25.4s, %[a0a].16b, %[b2].4b[1]\n"
            "add  %[a_ptr], %[a_ptr], #64\n"
            "sdot  v26.4s, %[a0a].16b, %[b2].4b[2]\n"
            "sdot  v27.4s, %[a0a].16b, %[b2].4b[3]\n"
            "add  %[b_ptr], %[b_ptr], #96\n"
            "sdot  v28.4s, %[a1a].16b, %[b2].4b[0]\n"
            "sdot  v29.4s, %[a1a].16b, %[b2].4b[1]\n"
            "subs  %w[k], %w[k], #1\n"
            "sdot  v30.4s, %[a1a].16b, %[b2].4b[2]\n"
            "sdot  v31.4s, %[a1a].16b, %[b2].4b[3]\n"
            "bne  1b\n"

            // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
            "4:\n"

            // Branch to alternative tail for odd K
            "cbnz  %w[oddk], 2f\n"

            // Detached final iteration (even K)
            "sdot  v8.4s , %[a0].16b, %[b0].4b[0]\n"
            "sdot   v9.4s , %[a0].16b, %[b0].4b[1]\n"
            "ldr  %q[b2], [%[b_ptr], #32]\n"
            "sdot  v10.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v11.4s, %[a0].16b, %[b0].4b[3]\n"
            "ldr  %q[a0a], [%[a_ptr], #32]\n"
            "sdot   v12.4s, %[a1].16b, %[b0].4b[0]\n"
            "sdot   v13.4s, %[a1].16b, %[b0].4b[1]\n"
            "ldr  %q[a1a], [%[a_ptr], #48]\n"
            "sdot  v14.4s, %[a1].16b, %[b0].4b[2]\n"
            "sdot  v15.4s, %[a1].16b, %[b0].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr], #48]\n"

            "sdot  v16.4s, %[a0].16b, %[b1].4b[0]\n"
            "sdot  v17.4s, %[a0].16b, %[b1].4b[1]\n"
            "sdot  v18.4s, %[a0].16b, %[b1].4b[2]\n"
            "sdot  v19.4s, %[a0].16b, %[b1].4b[3]\n"
            "sdot  v20.4s, %[a1].16b, %[b1].4b[0]\n"
            "sdot  v21.4s, %[a1].16b, %[b1].4b[1]\n"
            "sdot  v22.4s, %[a1].16b, %[b1].4b[2]\n"
            "sdot  v23.4s, %[a1].16b, %[b1].4b[3]\n"
            "ldr  %q[b1], [%[b_ptr], #64]\n"

            "sdot  v24.4s, %[a0].16b, %[b2].4b[0]\n"
            "sdot  v25.4s, %[a0].16b, %[b2].4b[1]\n"
            "add  %[a_ptr], %[a_ptr], #64\n"
            "sdot  v26.4s, %[a0].16b, %[b2].4b[2]\n"
            "sdot  v27.4s, %[a0].16b, %[b2].4b[3]\n"
            "sdot  v28.4s, %[a1].16b, %[b2].4b[0]\n"
            "sdot  v29.4s, %[a1].16b, %[b2].4b[1]\n"
            "sdot  v30.4s, %[a1].16b, %[b2].4b[2]\n"
            "sdot  v31.4s, %[a1].16b, %[b2].4b[3]\n"
            "ldr  %q[b2], [%[b_ptr], #80]\n"

            "sdot  v8.4s , %[a0a].16b, %[b0].4b[0]\n"

            "sdot  v16.4s, %[a0a].16b, %[b1].4b[0]\n"
            "add  %[b_ptr], %[b_ptr], #96\n"
            "sdot   v9.4s , %[a0a].16b, %[b0].4b[1]\n"
            "str  q8, [%[outptr0], #0]\n"
            "sdot  v17.4s, %[a0a].16b, %[b1].4b[1]\n"
            "str  q16, [%[outptr0], #64]\n"
            "sdot  v24.4s, %[a0a].16b, %[b2].4b[0]\n"
            "str  q24, [%[outptr0], #128]\n"

            "sdot  v25.4s, %[a0a].16b, %[b2].4b[1]\n"
            "str  q9, [%[outptr0], #16]\n"
            "sdot  v10.4s, %[a0a].16b, %[b0].4b[2]\n"
            "str  q17, [%[outptr0], #80]\n"
            "sdot  v18.4s, %[a0a].16b, %[b1].4b[2]\n"
            "str  q25, [%[outptr0], #144]\n"
            "sdot  v26.4s, %[a0a].16b, %[b2].4b[2]\n"
            "str  q10, [%[outptr0], #32]\n"

            "sdot  v11.4s, %[a0a].16b, %[b0].4b[3]\n"
            "str  q18, [%[outptr0], #96]\n"
            "sdot  v19.4s, %[a0a].16b, %[b1].4b[3]\n"
            "str  q26, [%[outptr0], #160]\n"
            "sdot  v27.4s, %[a0a].16b, %[b2].4b[3]\n"
            "str  q11, [%[outptr0], #48]\n"

            "sdot   v12.4s, %[a1a].16b, %[b0].4b[0]\n"
            "str  q19, [%[outptr0], #112]\n"
            "sdot  v20.4s, %[a1a].16b, %[b1].4b[0]\n"
            "str  q27, [%[outptr0], #176]\n"
            "sdot  v28.4s, %[a1a].16b, %[b2].4b[0]\n"
            "str  q12, [%[outptr1], #0]\n"

            "sdot   v13.4s, %[a1a].16b, %[b0].4b[1]\n"
            "str  q20, [%[outptr1], #64]\n"
            "sdot  v21.4s, %[a1a].16b, %[b1].4b[1]\n"
            "str  q28, [%[outptr1], #128]\n"
            "sdot  v29.4s, %[a1a].16b, %[b2].4b[1]\n"
            "str  q13, [%[outptr1], #16]\n"

            "sdot  v14.4s, %[a1a].16b, %[b0].4b[2]\n"
            "str  q21, [%[outptr1], #80]\n"
            "sdot  v22.4s, %[a1a].16b, %[b1].4b[2]\n"
            "str  q29, [%[outptr1], #144]\n"
            "sdot  v30.4s, %[a1a].16b, %[b2].4b[2]\n"
            "str  q14, [%[outptr1], #32]\n"

            "sdot  v15.4s, %[a1a].16b, %[b0].4b[3]\n"
            "str  q22, [%[outptr1], #96]\n"
            "sdot  v23.4s, %[a1a].16b, %[b1].4b[3]\n"
            "str  q30, [%[outptr1], #160]\n"
            "sdot  v31.4s, %[a1a].16b, %[b2].4b[3]\n"
            "str  q15, [%[outptr1], #48]\n"

            "b  3f\n"

            // Detached final iteration (odd K)
            "2:\n"
            "sdot  v8.4s , %[a0].16b, %[b0].4b[0]\n"
            "ldr  %q[b2], [%[b_ptr], #32]\n"
            "sdot  v16.4s, %[a0].16b, %[b1].4b[0]\n"
            "sdot   v9.4s , %[a0].16b, %[b0].4b[1]\n"
            "str  q8, [%[outptr0], #0]\n"
            "sdot  v17.4s, %[a0].16b, %[b1].4b[1]\n"
            "str  q16, [%[outptr0], #64]\n"
            "sdot  v24.4s, %[a0].16b, %[b2].4b[0]\n"
            "add  %[b_ptr], %[b_ptr], #48\n"
            "add  %[a_ptr], %[a_ptr], #32\n"
            "str  q24, [%[outptr0], #128]\n"
            "sdot  v25.4s, %[a0].16b, %[b2].4b[1]\n"
            "str  q9, [%[outptr0], #16]\n"

            "sdot  v10.4s, %[a0].16b, %[b0].4b[2]\n"
            "str  q17, [%[outptr0], #80]\n"
            "sdot  v18.4s, %[a0].16b, %[b1].4b[2]\n"
            "str  q25, [%[outptr0], #144]\n"
            "sdot  v26.4s, %[a0].16b, %[b2].4b[2]\n"
            "str  q10, [%[outptr0], #32]\n"

            "sdot  v11.4s, %[a0].16b, %[b0].4b[3]\n"
            "str  q18, [%[outptr0], #96]\n"
            "sdot  v19.4s, %[a0].16b, %[b1].4b[3]\n"
            "str  q26, [%[outptr0], #160]\n"
            "sdot  v27.4s, %[a0].16b, %[b2].4b[3]\n"
            "str  q11, [%[outptr0], #48]\n"

            "sdot   v12.4s, %[a1].16b, %[b0].4b[0]\n"
            "str  q19, [%[outptr0], #112]\n"
            "sdot  v20.4s, %[a1].16b, %[b1].4b[0]\n"
            "str  q27, [%[outptr0], #176]\n"
            "sdot  v28.4s, %[a1].16b, %[b2].4b[0]\n"
            "str  q12, [%[outptr1], #0]\n"

            "sdot   v13.4s, %[a1].16b, %[b0].4b[1]\n"
            "str  q20, [%[outptr1], #64]\n"
            "sdot  v21.4s, %[a1].16b, %[b1].4b[1]\n"
            "str  q28, [%[outptr1], #128]\n"
            "sdot  v29.4s, %[a1].16b, %[b2].4b[1]\n"
            "str  q13, [%[outptr1], #16]\n"

            "sdot  v14.4s, %[a1].16b, %[b0].4b[2]\n"
            "str  q21, [%[outptr1], #80]\n"
            "sdot  v22.4s, %[a1].16b, %[b1].4b[2]\n"
            "str  q29, [%[outptr1], #144]\n"
            "sdot  v30.4s, %[a1].16b, %[b2].4b[2]\n"
            "str  q14, [%[outptr1], #32]\n"

            "sdot  v15.4s, %[a1].16b, %[b0].4b[3]\n"
            "str  q22, [%[outptr1], #96]\n"
            "sdot  v23.4s, %[a1].16b, %[b1].4b[3]\n"
            "str  q30, [%[outptr1], #160]\n"
            "sdot  v31.4s, %[a1].16b, %[b2].4b[3]\n"
            "str  q15, [%[outptr1], #48]\n"


            // Common tail
            "3:\n"
            "str  q23, [%[outptr1], #112]\n"
            "str  q31, [%[outptr1], #176]\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr),[oddk] "+r" (oddk),
              [is_first_k] "+r" (is_first_k), [k] "+r" (k), [LDC] "+r" (LDC),
              [a0] "=w" (a0), [a1] "=w" (a1), [a0a] "=w" (a0a), [a1a] "=w" (a1a),
              [b0] "=w" (b0), [b1] "=w" (b1), [b2] "=w" (b2),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1)
            :
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14",
              "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc",
              "memory"
            );
}

// Overview of register layout:
//
// A (12x4)x2 cell of Rhs is stored in 8bit in q2-q7.
// A (4x4)x2 cell of Lhs is stored in 8bit in q0-q1
// A 4x12 block of accumulators is stored in 8bit in q8--q19.
//
//                              +------------+------------+------------+
//                              |    v2[0-16]|    v3[0-16]|    v4[0-16]|
//                         Rhs  +------------+------------+------------+
//                              |    v5[0-16]|    v6[0-16]|    v7[0-16]|
//                              +------------+------------+------------+
//    Lhs                       |            |            |            |
//
//  +--------+--------+ - - - - +------------+------------+------------+
//  |v0[0-16]|v1[0-16]|         | v8 v9v10v11|v12v13v14v15|v16v17v18v19|
//  +--------+--------+ - - - - +------------+------------+------------+
//
//                            Accumulator

static void kern_4x12(const int8_t* packA, const int8_t* packB, int K,
                      int32_t* output, int LDC, bool is_first_k) {
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

    asm volatile(
            // load accumulator C
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"
            "ldp q8, q9, [%[outptr0]]\n"
            "ldp q10, q11, [%[outptr0], #32]\n"
            "ldp q12, q13, [%[outptr0], #64]\n"
            "ldp q14, q15, [%[outptr0], #96]\n"
            "ldp q16, q17, [%[outptr0], #128]\n"
            "ldp q18, q19, [%[outptr0], #160]\n"

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
            "sdot  v8.4s, %[a0].16b, %[b0].4b[0]\n"
            "sdot    v9.4s, %[a0].16b, %[b0].4b[1]\n"
            "sdot  v10.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v11.4s, %[a0].16b, %[b0].4b[3]\n"
            "sdot  v12.4s, %[a0].16b, %[b1].4b[0]\n"
            "sdot    v13.4s, %[a0].16b, %[b1].4b[1]\n"
            "sdot  v14.4s, %[a0].16b, %[b1].4b[2]\n"
            "sdot  v15.4s, %[a0].16b, %[b1].4b[3]\n"
            "sdot  v16.4s, %[a0].16b, %[b2].4b[0]\n"
            "sdot    v17.4s, %[a0].16b, %[b2].4b[1]\n"
            "sdot  v18.4s, %[a0].16b, %[b2].4b[2]\n"
            "sdot  v19.4s, %[a0].16b, %[b2].4b[3]\n"

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

            "sdot  v8.4s, %[a0].16b, %[b0].4b[0]\n"
            "sdot    v9.4s, %[a0].16b, %[b0].4b[1]\n"
            "sdot  v10.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v11.4s, %[a0].16b, %[b0].4b[3]\n"
            "sdot  v12.4s, %[a0].16b, %[b1].4b[0]\n"
            "sdot    v13.4s, %[a0].16b, %[b1].4b[1]\n"
            "sdot  v14.4s, %[a0].16b, %[b1].4b[2]\n"
            "sdot  v15.4s, %[a0].16b, %[b1].4b[3]\n"
            "sdot  v16.4s, %[a0].16b, %[b2].4b[0]\n"
            "sdot    v17.4s, %[a0].16b, %[b2].4b[1]\n"
            "sdot  v18.4s, %[a0].16b, %[b2].4b[2]\n"
            "sdot  v19.4s, %[a0].16b, %[b2].4b[3]\n"
            "sdot  v8.4s , %[a0a].16b, %[b0a].4b[0]\n"
            "sdot    v9.4s , %[a0a].16b, %[b0a].4b[1]\n"
            "sdot  v10.4s, %[a0a].16b, %[b0a].4b[2]\n"
            "sdot  v11.4s, %[a0a].16b, %[b0a].4b[3]\n"
            "sdot  v12.4s, %[a0a].16b, %[b1a].4b[0]\n"
            "sdot    v13.4s, %[a0a].16b, %[b1a].4b[1]\n"
            "sdot  v14.4s, %[a0a].16b, %[b1a].4b[2]\n"
            "sdot  v15.4s, %[a0a].16b, %[b1a].4b[3]\n"
            "sdot  v16.4s, %[a0a].16b, %[b2a].4b[0]\n"
            "sdot    v17.4s, %[a0a].16b, %[b2a].4b[1]\n"
            "sdot  v18.4s, %[a0a].16b, %[b2a].4b[2]\n"
            "sdot  v19.4s, %[a0a].16b, %[b2a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n"
            "stp q8, q9, [%[outptr0]]\n"
            "stp q10, q11, [%[outptr0], #32]\n"
            "stp q12, q13, [%[outptr0], #64]\n"
            "stp q14, q15, [%[outptr0], #96]\n"
            "stp q16, q17, [%[outptr0], #128]\n"
            "stp q18, q19, [%[outptr0], #160]\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k),
              [outptr0] "+r"(outptr0), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [a0] "=w"(a0), [a0a] "=w"(a0a),
              [b0] "=w"(b0), [b1] "=w"(b1), [b2] "=w"(b2), [b0a] "=w"(b0a),
              [b1a] "=w"(b1a), [b2a] "=w"(b2a)
            :
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "memory", "cc");
}

// Overview of register layout:
//
// A (4x4)x2 cell of Rhs is stored in 8bit in q2-q7.
// A (8x4)x2 cell of Lhs is stored in 8bit in q0-q1, q4-q5
// A 8x4 block of accumulators is stored in 8bit in q6-q13.
//
//                              +------------+
//                              |    v2[0-16]|
//                         Rhs  +------------+
//                              |    v3[0-16]|
//                              +------------+
//    Lhs                       |            |
//
//  +--------+--------+ - - - - +------------+
//  |v0[0-16]|v4[0-16]|         | v6 v7 v8 v9|
//  +--------+--------+ - - - - +------------+
//  |v1[0-16]|v5[0-16]|         |v10v11v12v13|
//  +--------+--------+ - - - - +------------+
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

    size_t x0;

// clang-format off
#define LOAD_LINE(v0, v1, v2, v3, n)        \
    "mov %[x0], %[outptr" n "]\n"           \
    "cmp %w[n_remain], #4\n"                \
    "blt 100" n "f\n"                       \
    "ldr q" v0 ", [%[x0]] \n"               \
    "ldr q" v1 ", [%[x0], #16] \n"          \
    "ldr q" v2 ", [%[x0], #32] \n"          \
    "ldr q" v3 ", [%[x0], #48] \n"          \
    "b 101" n "f\n"                         \
    "100" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 101" n "f\n"                       \
    "ldr q" v0 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #1\n"                \
    "beq 101" n "f\n"                       \
    "ldr q" v1 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #2\n"                \
    "beq 101" n "f\n"                       \
    "ldr q" v2 ", [%[x0]], #16\n"           \
    "101" n ":\n"


#define LOAD_C                              \
    LOAD_LINE("6", "7", "8", "9", "0")      \
    LOAD_LINE("10", "11", "12", "13", "1")  \
    
#define STORE_LINE(v0, v1, v2, v3, n)       \
    "mov %[x0], %[outptr" n "]\n"           \
    "cmp %w[n_remain], #4\n"                \
    "blt 102" n "f\n"                       \
    "str q" v0 ", [%[x0]] \n"               \
    "str q" v1 ", [%[x0], #16] \n"          \
    "str q" v2 ", [%[x0], #32] \n"          \
    "str q" v3 ", [%[x0], #48] \n"          \
    "b 103" n "f\n"                         \
    "102" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 103" n "f\n"                       \
    "str q" v0 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #1\n"                \
    "beq 103" n "f\n"                       \
    "str q" v1 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #2\n"                \
    "beq 103" n "f\n"                       \
    "str q" v2 ", [%[x0]], #16\n"           \
    "103" n ":\n"

#define STORE_C                             \
    STORE_LINE("6", "7", "8", "9", "0")     \
    STORE_LINE("10", "11", "12", "13", "1")

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
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
            "sdot  v6.4s , %[a0].16b, %[b0].4b[0]\n"
            "sdot  v7.4s , %[a0].16b, %[b0].4b[1]\n"
            "sdot  v8.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v9.4s, %[a0].16b, %[b0].4b[3]\n"
            "sdot  v10.4s, %[a1].16b, %[b0].4b[0]\n"
            "sdot  v11.4s, %[a1].16b, %[b0].4b[1]\n"
            "sdot  v12.4s, %[a1].16b, %[b0].4b[2]\n"
            "sdot  v13.4s, %[a1].16b, %[b0].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[a1a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "sdot  v6.4s , %[a0].16b, %[b0].4b[0]\n"
            "sdot  v7.4s , %[a0].16b, %[b0].4b[1]\n"
            "sdot  v8.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v9.4s, %[a0].16b, %[b0].4b[3]\n"
            "sdot  v10.4s, %[a1].16b, %[b0].4b[0]\n"
            "sdot  v11.4s, %[a1].16b, %[b0].4b[1]\n"
            "sdot  v12.4s, %[a1].16b, %[b0].4b[2]\n"
            "sdot  v13.4s, %[a1].16b, %[b0].4b[3]\n"

            "sdot  v6.4s , %[a0a].16b, %[b0a].4b[0]\n"
            "sdot  v7.4s , %[a0a].16b, %[b0a].4b[1]\n"
            "sdot  v8.4s, %[a0a].16b, %[b0a].4b[2]\n"
            "sdot  v9.4s, %[a0a].16b, %[b0a].4b[3]\n"
            "sdot  v10.4s, %[a1a].16b, %[b0a].4b[0]\n"
            "sdot  v11.4s, %[a1a].16b, %[b0a].4b[1]\n"
            "sdot  v12.4s, %[a1a].16b, %[b0a].4b[2]\n"
            "sdot  v13.4s, %[a1a].16b, %[b0a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [LDC] "+r"(LDC),
              [oddk] "+r"(oddk), [is_first_k] "+r"(is_first_k),
              [n_remain] "+r"(n_remain), [k] "+r"(k), [outptr0] "+r"(outptr0),
              [a0] "=w"(a0), [a1] "=w"(a1), [a0a] "=w"(a0a), [a1a] "=w"(a1a),
              [b0] "=w"(b0), [b0a] "=w"(b0a), [outptr1] "=r"(outptr1),
              [x0] "=r"(x0)
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
// A (4x4)x2 cell of Rhs is stored in 8bit in q2-q3.
// A (4x4)x2 cell of Lhs is stored in 8bit in q0-q1
// A 4x4 block of accumulators is stored in 8bit in q4-q7.
//
//                              +------------+
//                              |    v2[0-16]|
//                         Rhs  +------------+
//                              |    v3[0-16]|
//                              +------------+
//    Lhs                       |            |
//
//  +--------+--------+ - - - - +------------+
//  |v0[0-16]|v4[0-16]|         | v4 v5 v6 v7|
//  +--------+--------+ - - - - +------------+
//                            Accumulator

static void kern_4x4(const int8_t* packA, const int8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int n_remain) {
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
    size_t x0;

// clang-format off
#define LOAD_LINE(v0, v1, v2, v3, n)        \
    "mov %[x0], %[outptr" n "]\n"           \
    "cmp %w[n_remain], #4\n"                \
    "blt 100" n "f\n"                       \
    "ldr q" v0 ", [%[x0]] \n"               \
    "ldr q" v1 ", [%[x0], #16] \n"          \
    "ldr q" v2 ", [%[x0], #32] \n"          \
    "ldr q" v3 ", [%[x0], #48] \n"          \
    "b 101" n "f\n"                         \
    "100" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 101" n "f\n"                       \
    "ldr q" v0 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #1\n"                \
    "beq 101" n "f\n"                       \
    "ldr q" v1 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #2\n"                \
    "beq 101" n "f\n"                       \
    "ldr q" v2 ", [%[x0]], #16\n"           \
    "101" n ":\n"

#define LOAD_C                              \
    LOAD_LINE("4", "5", "6", "7", "0")

#define STORE_LINE(v0, v1, v2, v3, n)       \
    "mov %[x0], %[outptr" n "]\n"           \
    "cmp %w[n_remain], #4\n"                \
    "blt 102" n "f\n"                       \
    "str q" v0 ", [%[x0]] \n"               \
    "str q" v1 ", [%[x0], #16] \n"          \
    "str q" v2 ", [%[x0], #32] \n"          \
    "str q" v3 ", [%[x0], #48] \n"          \
    "b 103" n "f\n"                         \
    "102" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 103" n "f\n"                       \
    "str q" v0 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #1\n"                \
    "beq 103" n "f\n"                       \
    "str q" v1 ", [%[x0]], #16\n"           \
    "cmp %w[n_remain], #2\n"                \
    "beq 103" n "f\n"                       \
    "str q" v2 ", [%[x0]], #16\n"           \
    "103" n ":\n"

#define STORE_C                             \
    STORE_LINE("4", "5", "6", "7", "0")
    // clang-format on

    asm volatile(
            // load accumulator C
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
            "sdot  v4.4s , %[a0].16b, %[b0].4b[0]\n"
            "sdot  v5.4s , %[a0].16b, %[b0].4b[1]\n"
            "sdot  v6.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v7.4s, %[a0].16b, %[b0].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "sdot  v4.4s , %[a0].16b, %[b0].4b[0]\n"
            "sdot  v5.4s , %[a0].16b, %[b0].4b[1]\n"
            "sdot  v6.4s, %[a0].16b, %[b0].4b[2]\n"
            "sdot  v7.4s, %[a0].16b, %[b0].4b[3]\n"
            "sdot  v4.4s , %[a0a].16b, %[b0a].4b[0]\n"
            "sdot  v5.4s , %[a0a].16b, %[b0a].4b[1]\n"
            "sdot  v6.4s, %[a0a].16b, %[b0a].4b[2]\n"
            "sdot  v7.4s, %[a0a].16b, %[b0a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [n_remain] "+r"(n_remain),
              [LDC] "+r"(LDC), [outptr0] "+r"(outptr0), [k] "+r"(k),
              [a0] "=w"(a0), [a0a] "=w"(a0a), [b0] "=w"(b0), [b0a] "=w"(b0a),
              [x0] "=r"(x0)
            :
            : "v4", "v5", "v6", "v7", "memory", "cc");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_mk4_s8_8x12_pack_A(dt_int8* outptr, const dt_int8* inptr,
                                    int ldin, int y0, int ymax, int k0,
                                    int kmax) {
    megdnn_assert(ymax % 4 == 0 && y0 % 4 == 0,
                  "mk4 matmul with m is not times of 4");
    megdnn_assert(kmax % 4 == 0 && k0 % 4 == 0,
                  "mk4 matmul with k is not times of 4");
    int y = y0;
    int start_y = y0 / 4;
    for (; y + 7 < ymax; y += 8, start_y += 2) {
        const int8_t* inptr0 = inptr + start_y * ldin + (k0 << 2);
        const int8_t* inptr1 = inptr0 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);

        int K = kmax - k0;
        //! read 2 * 4 in each row
        for (; K > 3; K -= 4) {
            interleave_2x4_4_b(inptr0, inptr1, outptr);
        }
    }
    for (; y + 3 < ymax; y += 4, start_y ++) {
        int K = kmax - k0;
        const int8_t* inptr0 = inptr + start_y * ldin + (k0 << 2);
        std::memcpy(outptr, inptr0, sizeof(dt_int8) * K * 4);
    }
}

static void gemm_mk4_s8_8x12_pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                    int x0, int xmax, int k0, int kmax) {
    const int ksize = kmax - k0;
    const int ksize12 = ksize * 12;
    const int ksize4 = ksize * 4;
    int8_t* outptr = out;
    int8_t* outptr_base = out;
    //! 4x4 block output start pos
    int8_t* outptr_base4 = out + ((xmax - x0) / 12) * ksize12;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const int8_t* inptr = in + (k >> 2) * ldin + (x0 << 2);
        prefetch_2x(inptr);

        int x = x0;
        outptr = outptr_base;
        for (; x + 11 < xmax; x += 12) {
            std::memcpy(outptr, inptr, 48);
            outptr += ksize12;
            inptr += 48;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            std::memcpy(outptr, inptr, 16);
            outptr += ksize4;
            inptr += 16;
        }

        if (x < xmax) {
            int i = 0;
            for (; i < xmax - x; i++) {
                *outptr++ = *inptr++;
                *outptr++ = *inptr++;
                *outptr++ = *inptr++;
                *outptr++ = *inptr++;
            }
            for (; i < 4; i++) {
                *outptr++ = 0;
                *outptr++ = 0;
                *outptr++ = 0;
                *outptr++ = 0;
            }
        }

        outptr_base += 48;
        outptr_base4 += 16;
    }
}

}  // namespace matmul_mk4_8x12x4
}  // namespace aarch64
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
