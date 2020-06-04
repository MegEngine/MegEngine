/**
 * \file dnn/src/aarch64/matrix_mul/int8/kernel_4x4x16.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if !(__ARM_FEATURE_DOTPROD)
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_4x4x16 {

/**
 * Overview of register layout:
 *
 * A 16x2 cell of Rhs is stored in 8bit in q2-q4.
 * A 8x2x2 cell of Lhs is stored in 8bit in q0-q1
 * A 8x16 block of accumulators is stored in 8bit in q8--q31.
 *
 * \warning Fast kernel operating on int8 operands.
 * It is assumed that one of the two int8 operands only takes values
 * in [-127, 127], while the other may freely range in [-128, 127].
 * The issue with both operands taking the value -128 is that:
 * -128*-128 + -128*-128 == -32768 overflows int16.
 * Every other expression a*b + c*d, for any int8 a,b,c,d, fits in int16
 * range. That is the basic idea of this kernel.
 *
 *
 *                     +--------+--------+---------+---------+
 *                     |v4[0-16]|v5[0-16]| v6[0-16]| v7[0-16]|
 *                Rhs  +--------+--------+---------+---------+
 *                     |v8[0-16]|v9[0-16]|v10[0-16]|v11[0-16]|
 *                     +--------+--------+---------+---------+
 *                     |        |        |         |         |
 *
 *    Lhs              |        |        |         |         |
 *
 *  +--------+ - - - - +-------------------------------------+
 *  |v0[0-16]|         |v16[0-4]|v17[0-4]| v18[0-4]| v19[0-4]|
 *  |v1[0-16]|         |v20[0-4]|v21[0-4]| v22[0-4]| v23[0-4]|
 *  |v2[0-16]|         |v24[0-4]|v25[0-4]| v26[0-4]| v27[0-4]|
 *  |v3[0-16]|         |v28[0-4]|v29[0-4]| v30[0-4]| v31[0-4]|
 *  +--------+ - - - - +-------------------------------------+
 *
 *                            Accumulator
 */

static void kern_4x4(const int8_t* packA, const int8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k) {
    K /= 16;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(int32_t);

    asm volatile (
            // load accumulator C
            "add x1, %[output], %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"

            "ldr q16, [%[output]]\n"
            "ldr q17, [x1]\n"
            "ldr q18, [x2]\n"
            "ldr q19, [x3]\n"
            "b 2f\n"

            "1:\n"
            "eor v16.16b,  v16.16b,  v16.16b\n"
            "eor v17.16b,  v17.16b,  v17.16b\n"
            "eor v18.16b,  v18.16b,  v18.16b\n"
            "eor v19.16b,  v19.16b,  v19.16b\n"

            "2: \n"
            "ldr q0, [%[a_ptr]]\n"
            "ldr q4, [%[b_ptr]]\n"
            "ldr q5, [%[b_ptr], #16]\n"
            "ldr q6, [%[b_ptr], #32]\n"
            "movi v20.4s, #0x0\n"
            "ldr q7, [%[b_ptr], #48]\n"
            "movi v21.4s, #0x0\n"
            "ldr q1, [%[a_ptr], #16]\n"
            "movi v22.4s, #0x0\n"
            "ldr q2, [%[a_ptr], #32]\n"
            "movi v23.4s, #0x0\n"
            "ldr q3, [%[a_ptr], #48]\n"
            "movi v24.4s, #0x0\n"
            ASM_PREFETCH("[%[b_ptr], #64]")
            "movi v25.4s, #0x0\n"
            ASM_PREFETCH("[%[a_ptr], #64]")
            "movi v26.4s, #0x0\n"
            ASM_PREFETCH("[%[b_ptr], #128]")
            "movi v27.4s, #0x0\n"
            ASM_PREFETCH("[%[a_ptr], #128]")
            "movi v28.4s, #0x0\n"
            ASM_PREFETCH("[%[b_ptr], #192]")
            "movi v29.4s, #0x0\n"
            ASM_PREFETCH("[%[a_ptr], #192]")
            "movi v30.4s, #0x0\n"
            ASM_PREFETCH("[%[b_ptr], #256]")
            "movi v31.4s, #0x0\n"
            ASM_PREFETCH("[%[a_ptr], #256]")

            // Start of unroll 0 (first iteration)
            "smull v12.8h, v0.8b, v4.8b\n"
            "smull v13.8h, v0.8b, v5.8b\n"

            // Skip loop if we are doing zero iterations of it.
            "cbz %w[k], 4f\n"

            // Unroll 0 continuation (branch target)
            "3:\n"
            "smull v14.8h, v0.8b, v6.8b\n"
            "subs %w[k], %w[k], #1\n"
            "smull v15.8h, v0.8b, v7.8b\n"
            "ldr q8, [%[b_ptr], #64]\n"
            "smlal2 v12.8h, v0.16b, v4.16b\n"
            "smlal2 v13.8h, v0.16b, v5.16b\n"
            "ldr q9, [%[b_ptr], #80]\n"
            "smlal2 v14.8h, v0.16b, v6.16b\n"
            "smlal2 v15.8h, v0.16b, v7.16b\n"
            "ldr  q0, [%[a_ptr], #64]\n"

            "sadalp v16.4s, v12.8h\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "sadalp v17.4s, v13.8h\n"
            "sadalp v18.4s, v14.8h\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "sadalp v19.4s, v15.8h\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ldr q10, [%[b_ptr], #96]\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "ldr q11, [%[b_ptr], #112]\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"
            "add %[b_ptr], %[b_ptr], #128\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"
            "ldr  q1, [%[a_ptr], #80]\n"

            "sadalp v20.4s, v12.8h\n"
            "smull v12.8h, v2.8b, v4.8b\n"
            "sadalp v21.4s, v13.8h\n"
            "sadalp v22.4s, v14.8h\n"
            "smull v13.8h, v2.8b, v5.8b\n"
            "sadalp v23.4s, v15.8h\n"
            "smull v14.8h, v2.8b, v6.8b\n"
            "smull v15.8h, v2.8b, v7.8b\n"
            "smlal2 v12.8h, v2.16b, v4.16b\n"
            ASM_PREFETCH("[%[b_ptr], #192]")
            "smlal2 v13.8h, v2.16b, v5.16b\n"
            "smlal2 v14.8h, v2.16b, v6.16b\n"
            ASM_PREFETCH("[%[a_ptr], #320]")
            "smlal2 v15.8h, v2.16b, v7.16b\n"
            "ldr  q2, [%[a_ptr], #96]\n"

            "sadalp v24.4s, v12.8h\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "sadalp v25.4s, v13.8h\n"
            "sadalp v26.4s, v14.8h\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v27.4s, v15.8h\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "ldr  q4, [%[b_ptr], #0]\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"
            "ldr  q3, [%[a_ptr], #112]\n"

            // Unroll 1
            "sadalp v28.4s, v12.8h\n"
            "smull v12.8h, v0.8b, v8.8b\n"
            "sadalp v29.4s, v13.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "smull v13.8h, v0.8b, v9.8b\n"
            "sadalp v31.4s, v15.8h\n"
            "smull v14.8h, v0.8b, v10.8b\n"
            "smull v15.8h, v0.8b, v11.8b\n"
            "ldr  q5, [%[b_ptr], #16]\n"
            "smlal2 v12.8h, v0.16b, v8.16b\n"
            "smlal2 v13.8h, v0.16b, v9.16b\n"
            "ldr  q6, [%[b_ptr], #32]\n"
            "smlal2 v14.8h, v0.16b, v10.16b\n"
            "smlal2 v15.8h, v0.16b, v11.16b\n"
            "ldr  q0, [%[a_ptr], #128]\n"

            "sadalp v16.4s, v12.8h\n"
            "smull v12.8h, v1.8b, v8.8b\n"
            "sadalp v17.4s, v13.8h\n"
            "sadalp v18.4s, v14.8h\n"
            "smull v13.8h, v1.8b, v9.8b\n"
            "sadalp v19.4s, v15.8h\n"
            "add %[a_ptr], %[a_ptr], #128\n"
            "smull v14.8h, v1.8b, v10.8b\n"
            "smull v15.8h, v1.8b, v11.8b\n"
            "ldr  q7, [%[b_ptr], #48]\n"
            "smlal2 v12.8h, v1.16b, v8.16b\n"
            "smlal2 v13.8h, v1.16b, v9.16b\n"
            "smlal2 v14.8h, v1.16b, v10.16b\n"
            "smlal2 v15.8h, v1.16b, v11.16b\n"
            "ldr  q1, [%[a_ptr], #16]\n"

            "sadalp v20.4s, v12.8h\n"
            "smull v12.8h, v2.8b, v8.8b\n"
            "sadalp v21.4s, v13.8h\n"
            "sadalp v22.4s, v14.8h\n"
            "smull v13.8h, v2.8b, v9.8b\n"
            "sadalp v23.4s, v15.8h\n"
            "smull v14.8h, v2.8b, v10.8b\n"
            "smull v15.8h, v2.8b, v11.8b\n"
            "smlal2 v12.8h, v2.16b, v8.16b\n"
            ASM_PREFETCH("[%[b_ptr], #256]")
            "smlal2 v13.8h, v2.16b, v9.16b\n"
            "smlal2 v14.8h, v2.16b, v10.16b\n"
            ASM_PREFETCH("[%[a_ptr], #256]")
            "smlal2 v15.8h, v2.16b, v11.16b\n"
            "ldr  q2, [%[a_ptr], #32]\n"

            "sadalp v24.4s, v12.8h\n"
            "smull v12.8h, v3.8b, v8.8b\n"
            "sadalp v25.4s, v13.8h\n"
            "sadalp v26.4s, v14.8h\n"
            "smull v13.8h, v3.8b, v9.8b\n"
            "sadalp v27.4s, v15.8h\n"
            "smull v14.8h, v3.8b, v10.8b\n"
            "smull v15.8h, v3.8b, v11.8b\n"
            "smlal2 v12.8h, v3.16b, v8.16b\n"
            "smlal2 v13.8h, v3.16b, v9.16b\n"
            "smlal2 v14.8h, v3.16b, v10.16b\n"
            "smlal2 v15.8h, v3.16b, v11.16b\n"
            "ldr  q3, [%[a_ptr], #48]\n"

            // Start of unroll 0 for next iteration.
            "sadalp v28.4s, v12.8h\n"
            "smull v12.8h, v0.8b, v4.8b\n"
            "sadalp v29.4s, v13.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "smull v13.8h, v0.8b, v5.8b\n"
            "sadalp v31.4s, v15.8h\n"
            "bne 3b\n"

            // Target to use when K=1 or 2 (i.e. zero iterations of main loop)
            "4:\n"

            // Branch to alternative tail for odd K
            "cbnz %w[oddk], 5f\n"

            // Detached final iteration (even K)
            "smull v14.8h, v0.8b, v6.8b\n"
            "smull v15.8h, v0.8b, v7.8b\n"
            "ldr q8, [%[b_ptr], #64]\n"
            "smlal2 v12.8h, v0.16b, v4.16b\n"
            "smlal2 v13.8h, v0.16b, v5.16b\n"
            "ldr q9, [%[b_ptr], #80]\n"
            "smlal2 v14.8h, v0.16b, v6.16b\n"
            "smlal2 v15.8h, v0.16b, v7.16b\n"
            "ldr  q0, [%[a_ptr], #64]\n"

            "sadalp v16.4s, v12.8h\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "sadalp v17.4s, v13.8h\n"
            "sadalp v18.4s, v14.8h\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "sadalp v19.4s, v15.8h\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ldr q10, [%[b_ptr], #96]\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "ldr q11, [%[b_ptr], #112]\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"
            "add %[b_ptr], %[b_ptr], #128\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"
            "ldr  q1, [%[a_ptr], #80]\n"

            "sadalp v20.4s, v12.8h\n"
            "smull v12.8h, v2.8b, v4.8b\n"
            "sadalp v21.4s, v13.8h\n"
            "sadalp v22.4s, v14.8h\n"
            "smull v13.8h, v2.8b, v5.8b\n"
            "sadalp v23.4s, v15.8h\n"
            "smull v14.8h, v2.8b, v6.8b\n"
            "smull v15.8h, v2.8b, v7.8b\n"
            "smlal2 v12.8h, v2.16b, v4.16b\n"
            "smlal2 v13.8h, v2.16b, v5.16b\n"
            "smlal2 v14.8h, v2.16b, v6.16b\n"
            "smlal2 v15.8h, v2.16b, v7.16b\n"
            "ldr  q2, [%[a_ptr], #96]\n"

            "sadalp v24.4s, v12.8h\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "sadalp v25.4s, v13.8h\n"
            "sadalp v26.4s, v14.8h\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v27.4s, v15.8h\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"
            "ldr  q3, [%[a_ptr], #112]\n"

            // Unroll 1
            "sadalp v28.4s, v12.8h\n"
            "smull v12.8h, v0.8b, v8.8b\n"
            "sadalp v29.4s, v13.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "smull v13.8h, v0.8b, v9.8b\n"
            "sadalp v31.4s, v15.8h\n"
            "smull v14.8h, v0.8b, v10.8b\n"
            "add %[a_ptr], %[a_ptr], #128\n"
            "smull v15.8h, v0.8b, v11.8b\n"
            "smlal2 v12.8h, v0.16b, v8.16b\n"
            "smlal2 v13.8h, v0.16b, v9.16b\n"
            "smlal2 v14.8h, v0.16b, v10.16b\n"
            "smlal2 v15.8h, v0.16b, v11.16b\n"

            "sadalp v16.4s, v12.8h\n"
            "smull v12.8h, v1.8b, v8.8b\n"
            "sadalp v17.4s, v13.8h\n"
            "sadalp v18.4s, v14.8h\n"
            "smull v13.8h, v1.8b, v9.8b\n"
            "sadalp v19.4s, v15.8h\n"
            "smull v14.8h, v1.8b, v10.8b\n"
            "smull v15.8h, v1.8b, v11.8b\n"
            "smlal2 v12.8h, v1.16b, v8.16b\n"
            "addp v16.4s, v16.4s, v17.4s\n"
            "smlal2 v13.8h, v1.16b, v9.16b\n"
            "addp v17.4s, v18.4s, v19.4s\n"
            "smlal2 v14.8h, v1.16b, v10.16b\n"
            "smlal2 v15.8h, v1.16b, v11.16b\n"

            "sadalp v20.4s, v12.8h\n"
            "smull v12.8h, v2.8b, v8.8b\n"
            "sadalp v21.4s, v13.8h\n"
            "sadalp v22.4s, v14.8h\n"
            "smull v13.8h, v2.8b, v9.8b\n"
            "sadalp v23.4s, v15.8h\n"
            "addp v16.4s, v16.4s, v17.4s\n"
            "smull v14.8h, v2.8b, v10.8b\n"
            "addp v18.4s, v20.4s, v21.4s\n"
            "addp v19.4s, v22.4s, v23.4s\n"
            "smull v15.8h, v2.8b, v11.8b\n"
            "smlal2 v12.8h, v2.16b, v8.16b\n"
            "str q16, [%[output]]\n"
            "smlal2 v13.8h, v2.16b, v9.16b\n"
            "smlal2 v14.8h, v2.16b, v10.16b\n"
            "smlal2 v15.8h, v2.16b, v11.16b\n"

            "sadalp v24.4s, v12.8h\n"
            "smull v12.8h, v3.8b, v8.8b\n"
            "sadalp v25.4s, v13.8h\n"
            "sadalp v26.4s, v14.8h\n"
            "smull v13.8h, v3.8b, v9.8b\n"
            "sadalp v27.4s, v15.8h\n"
            "addp v17.4s, v18.4s, v19.4s\n"
            "smull v14.8h, v3.8b, v10.8b\n"
            "addp v20.4s, v24.4s, v25.4s\n"
            "addp v21.4s, v26.4s, v27.4s\n"
            "smull v15.8h, v3.8b, v11.8b\n"
            "smlal2 v12.8h, v3.16b, v8.16b\n"
            "str q17, [x1]\n"
            "smlal2 v13.8h, v3.16b, v9.16b\n"
            "smlal2 v14.8h, v3.16b, v10.16b\n"
            "addp v18.4s, v20.4s, v21.4s\n"
            "smlal2 v15.8h, v3.16b, v11.16b\n"
            "b 6f\n"

            // Detached final iteration (odd K)
            "5:\n"
            "smull v14.8h, v0.8b, v6.8b\n"
            "add %[a_ptr], %[a_ptr], #64\n"
            "smull v15.8h, v0.8b, v7.8b\n"
            "add %[b_ptr], %[b_ptr], #64\n"
            "smlal2 v12.8h, v0.16b, v4.16b\n"
            "smlal2 v13.8h, v0.16b, v5.16b\n"
            "smlal2 v14.8h, v0.16b, v6.16b\n"
            "smlal2 v15.8h, v0.16b, v7.16b\n"

            "sadalp v16.4s, v12.8h\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "sadalp v17.4s, v13.8h\n"
            "sadalp v18.4s, v14.8h\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "sadalp v19.4s, v15.8h\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "addp v16.4s, v16.4s, v17.4s\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"
            "addp v17.4s, v18.4s, v19.4s\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "sadalp v20.4s, v12.8h\n"
            "smull v12.8h, v2.8b, v4.8b\n"
            "sadalp v21.4s, v13.8h\n"
            "sadalp v22.4s, v14.8h\n"
            "smull v13.8h, v2.8b, v5.8b\n"
            "sadalp v23.4s, v15.8h\n"
            "addp v16.4s, v16.4s, v17.4s\n"
            "smull v14.8h, v2.8b, v6.8b\n"
            "addp v18.4s, v20.4s, v21.4s\n"
            "addp v19.4s, v22.4s, v23.4s\n"
            "smull v15.8h, v2.8b, v7.8b\n"
            "smlal2 v12.8h, v2.16b, v4.16b\n"
            "str q16, [%[output]]\n"
            "smlal2 v13.8h, v2.16b, v5.16b\n"
            "smlal2 v14.8h, v2.16b, v6.16b\n"
            "smlal2 v15.8h, v2.16b, v7.16b\n"

            "sadalp v24.4s, v12.8h\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "sadalp v25.4s, v13.8h\n"
            "sadalp v26.4s, v14.8h\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v27.4s, v15.8h\n"
            "addp v17.4s, v18.4s, v19.4s\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "addp v20.4s, v24.4s, v25.4s\n"
            "addp v21.4s, v26.4s, v27.4s\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "str q17, [x1]\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "addp v18.4s, v20.4s, v21.4s\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"

            "6:\n"

            // Final additions
            "sadalp v28.4s, v12.8h\n"
            "str q18, [x2]\n"
            "sadalp v29.4s, v13.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"

            // Horizontal reduction, phase 1
            "addp v22.4s, v28.4s, v29.4s\n"
            "addp v23.4s, v30.4s, v31.4s\n"

            // Horizontal reduction, phase 2
            "addp v19.4s, v22.4s, v23.4s\n"
            "str q19, [x3]\n"

            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [oddk] "+r" (oddk),
              [is_first_k] "+r" (is_first_k), [k] "+r" (k), [LDC] "+r" (LDC),
              [output] "+r"(output)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20","v21","v22","v23","v24","v25","v26",
              "v27","v28","v29","v30","v31", "x1", "x2", "x3",
              "cc", "memory"
            );
}

static void kern_4x4_remain(const int8_t* packA, const int8_t* packB, int K,
                            int32_t* output, int LDC, bool is_first_k,
                            int m_remain, int n_remain) {
    megdnn_assert(K > 0);
    K /= 16;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);

// clang-format off
#define LOAD_LINE(reg_index, n)                          \
    "cbz x4, 102f\n"                                     \
    "mov x5, x" n "\n"                                   \
    "cmp %w[n_remain], #4\n"                             \
    "blt 100" n "f\n"                                    \
    "ldr q" reg_index ", [x5]\n"                         \
    "b 101" n "f\n"                                      \
    "100" n ":\n"                                        \
    "cmp %w[n_remain], #0\n"                             \
    "beq 101" n "f\n"                                    \
    "ld1 {v" reg_index ".s}[0], [x5], #4\n"              \
    "cmp %w[n_remain], #1\n"                             \
    "beq 101" n "f\n"                                    \
    "ld1 {v" reg_index ".s}[1], [x5], #4\n"              \
    "cmp %w[n_remain], #2\n"                             \
    "beq 101" n "f\n"                                    \
    "ld1 {v" reg_index ".s}[2], [x5], #4\n"              \
    "101" n ":\n" \
    "subs x4, x4, #1\n"

#define LOAD_C                \
    "mov x4, %x[m_remain]\n"  \
    LOAD_LINE("16", "0")      \
    LOAD_LINE("17", "1")      \
    LOAD_LINE("18", "2")      \
    LOAD_LINE("19", "3")      \
    "102:\n"

#define STORE_LINE(reg_index, n)            \
    "cbz x4, 105f\n"                        \
    "mov x5, x" n "\n"                      \
    "cmp %w[n_remain], #4\n"                \
    "blt 103" n "f\n"                       \
    "str q" reg_index ", [x5]\n"            \
    "b 104" n "f\n"                         \
    "103" n ":\n"                           \
    "cmp %w[n_remain], #0\n"                \
    "beq 104" n "f\n"                       \
    "st1 {v" reg_index ".s}[0], [x5], #4\n" \
    "cmp %w[n_remain], #1\n"                \
    "beq 104" n "f\n"                       \
    "st1 {v" reg_index ".s}[1], [x5], #4\n" \
    "cmp %w[n_remain], #2\n"                \
    "beq 104" n "f\n"                       \
    "st1 {v" reg_index ".s}[2], [x5], #4\n" \
    "104" n ":\n" \
    "subs x4, x4, #1\n"

#define STORE_C              \
    "mov x4, %x[m_remain]\n" \
    STORE_LINE("16", "0")     \
    STORE_LINE("17", "1")     \
    STORE_LINE("18", "2")     \
    STORE_LINE("19", "3")     \
    "105:\n"

    // clang-format on

    asm volatile(
            // load accumulator C
            "mov x0, %[output]\n"
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"

            LOAD_C  //
            "b 2f\n"

            "1:\n"
            "eor v16.16b,  v16.16b,  v16.16b\n"
            "eor v17.16b,  v17.16b,  v17.16b\n"
            "eor v18.16b,  v18.16b,  v18.16b\n"
            "eor v19.16b,  v19.16b,  v19.16b\n"
            "eor v20.16b,  v20.16b,  v20.16b\n"
            "eor v21.16b,  v21.16b,  v21.16b\n"
            "eor v22.16b,  v22.16b,  v22.16b\n"
            "eor v23.16b,  v23.16b,  v23.16b\n"
            "eor v24.16b,  v24.16b,  v24.16b\n"
            "eor v25.16b,  v25.16b,  v25.16b\n"
            "eor v26.16b,  v26.16b,  v26.16b\n"
            "eor v27.16b,  v27.16b,  v27.16b\n"
            "eor v28.16b,  v28.16b,  v28.16b\n"
            "eor v29.16b,  v29.16b,  v29.16b\n"
            "eor v30.16b,  v30.16b,  v30.16b\n"
            "eor v31.16b,  v31.16b,  v31.16b\n"

            "2: \n"
            "ldr q4, [%[b_ptr]]\n"
            "ldr q5, [%[b_ptr], #16]\n"
            "ldr q6, [%[b_ptr], #32]\n"
            "ldr q7, [%[b_ptr], #48]\n"
            "ldr q0, [%[a_ptr]]\n"
            "ldr q1, [%[a_ptr], #16]\n"
            "ldr q2, [%[a_ptr], #32]\n"
            "ldr q3, [%[a_ptr], #48]\n"

            "smull v12.8h, v0.8b, v4.8b\n"
            "smull v13.8h, v0.8b, v5.8b\n"
            "smull v14.8h, v0.8b, v6.8b\n"
            "smull v15.8h, v0.8b, v7.8b\n"
            "smlal2 v12.8h, v0.16b, v4.16b\n"
            "smlal2 v13.8h, v0.16b, v5.16b\n"
            "smlal2 v14.8h, v0.16b, v6.16b\n"
            "smlal2 v15.8h, v0.16b, v7.16b\n"
            "sadalp v16.4s, v12.8h\n"
            "sadalp v17.4s, v13.8h\n"
            "sadalp v18.4s, v14.8h\n"
            "sadalp v19.4s, v15.8h\n"

            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "sadalp v21.4s, v13.8h\n"
            "sadalp v22.4s, v14.8h\n"
            "sadalp v23.4s, v15.8h\n"

            "smull v12.8h, v2.8b, v4.8b\n"
            "smull v13.8h, v2.8b, v5.8b\n"
            "smull v14.8h, v2.8b, v6.8b\n"
            "smull v15.8h, v2.8b, v7.8b\n"
            "smlal2 v12.8h, v2.16b, v4.16b\n"
            "smlal2 v13.8h, v2.16b, v5.16b\n"
            "smlal2 v14.8h, v2.16b, v6.16b\n"
            "smlal2 v15.8h, v2.16b, v7.16b\n"
            "sadalp v24.4s, v12.8h\n"
            "sadalp v25.4s, v13.8h\n"
            "sadalp v26.4s, v14.8h\n"
            "sadalp v27.4s, v15.8h\n"

            "smull v12.8h, v3.8b, v4.8b\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"
            "sadalp v28.4s, v12.8h\n"
            "sadalp v29.4s, v13.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"
            "add %[a_ptr], %[a_ptr], #64\n"
            "add %[b_ptr], %[b_ptr], #64\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n"
            // reduction
            "addp v16.4s, v16.4s, v17.4s\n"
            "addp v17.4s, v18.4s, v19.4s\n"
            "addp v16.4s, v16.4s, v17.4s\n"
            "addp v18.4s, v20.4s, v21.4s\n"
            "addp v19.4s, v22.4s, v23.4s\n"
            "addp v17.4s, v18.4s, v19.4s\n"
            "addp v20.4s, v24.4s, v25.4s\n"
            "addp v21.4s, v26.4s, v27.4s\n"
            "addp v18.4s, v20.4s, v21.4s\n"
            "addp v22.4s, v28.4s, v29.4s\n"
            "addp v23.4s, v30.4s, v31.4s\n"
            "addp v19.4s, v22.4s, v23.4s\n"

            STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [output] "+r"(output), [m_remain] "+r"(m_remain),
              [n_remain] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "x0", "x1", "x2", "x3", "x4", "x5", "cc",
              "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_s8_4x4_pack_A_n(dt_int8* outptr, const dt_int8* inptr,
                                 int ldin, int y0, int ymax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 3 < ymax; y += 4) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        //! read 16 * 4 in each row
        for (; K > 15; K -= 16) {
            interleave_4x16_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 16, K);
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
                        inptr1 = zerobuff; MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff; MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            interleave_4x16_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff; MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff; MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 16, K);
        }
    }
}

static void gemm_s8_4x4_pack_B_n(dt_int8* out, const dt_int8* in, int ldin,
                                 int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 16) * 4;
    int8_t* outptr = out;

    int k = k0;
    for (; k < kmax; k += 16) {
        int ki = k;
        for (int cnt = 0; cnt < 2; ki += 8, cnt++) {
            const int8_t* inptr0 = in + ki * ldin + x0;
            const int8_t* inptr1 = inptr0 + ldin;
            const int8_t* inptr2 = inptr1 + ldin;
            const int8_t* inptr3 = inptr2 + ldin;
            const int8_t* inptr4 = inptr3 + ldin;
            const int8_t* inptr5 = inptr4 + ldin;
            const int8_t* inptr6 = inptr5 + ldin;
            const int8_t* inptr7 = inptr6 + ldin;
            int8_t* outptr_inner = outptr + ki - k;

            int remain = std::min(ki + 7 - kmax, 7);
            int x = x0;
            for (; x + 3 < xmax; x += 4) {
                if (remain >= 0) {
                    switch (remain) {
                        case 7:
                            inptr0 = zerobuff; MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff; MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff; MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff; MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff; MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff; MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff; MEGDNN_FALLTHRU
                        case 0:
                            inptr7 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                transpose_4x16_1_b_helper(inptr0, inptr1, inptr2, inptr3,
                                          inptr4, inptr5, inptr6, inptr7,
                                          outptr_inner);
                outptr_inner += ksize4;
            }

            if (x < xmax) {
                if (remain >= 0) {
                    switch (remain) {
                        case 7:
                            inptr0 = zerobuff; MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff; MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff; MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff; MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff; MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff; MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff; MEGDNN_FALLTHRU
                        case 0:
                            inptr7 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                for (; x < xmax; x++) {
                    *outptr_inner++ = *inptr0++;
                    *outptr_inner++ = *inptr1++;
                    *outptr_inner++ = *inptr2++;
                    *outptr_inner++ = *inptr3++;
                    *outptr_inner++ = *inptr4++;
                    *outptr_inner++ = *inptr5++;
                    *outptr_inner++ = *inptr6++;
                    *outptr_inner++ = *inptr7++;
                    outptr_inner += 8;
                }
            }
        }

        outptr += 16 * 4;
    }
}

}  // namespace matmul_4x4x16
}  // namespace aarch64
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
