/**
 * \file dnn/src/armv7/matrix_mul/int8x8x16/kernel_mk4_8x8x4.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <inttypes.h>
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_mk4_16x12x4_a53 {

//! optimize for A53

// clang-format off
/**
 * Overview of register layout:
 *
 * A 16x12x4 cell of Lhs is stored in 16bit in q0-q3
 * A 16x12x4 cell of Rhs is stored in 8bit in q4-q7
 * A 16x12 block of accumulators is stored in 16bit in q8-q31
 *
 *                     +------------------------------------------------------------------------+
 *                     | q4[0]|q4[1]|q4[2]|q4[3]|q4[4]|q4[5]|q4[6]|q4[7]|q5[0]|q5[1]|q5[2]|q5[3]|
 *                Rhs  +------------------------------------------------------------------------+
 *    Lhs              |      |     |     |     |     |     |     |     |     |     |     |     |
 *  +--------+ - - - - +------------------------------------------------------------------------+
 *  | q0 |             |  q8  | q9  | q10 | q11 | q12 | q13 | q14 | q15 | q16 | q17 | q18 | q19 |
 *  | q1 |             |  q20 | q21 | q22 | q23 | q24 | q25 | q26 | q27 | q28 | q29 | q30 | q31 |
 *  +--------+ - - - - +------------------------------------------------------------------------+
 *
 *                            Accumulator
 */
// clang-format on
static __attribute__((noinline)) void kern_16x12(const int16_t* packA,
                                                 const int8_t* packB, int K,
                                                 int16_t* output, int LDC,
                                                 bool is_first_k,
                                                 int remain_n) {
    K /= 4;
    const int16_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);

    // clang-format off
#define STORE_LINE(reg0, reg1)     \
    "cmp w10, #0 \n"                         \
    "beq 101f\n"                             \
    "st1 {v" reg0 ".4h},   [x0], #8\n"       \
    "st1 {v" reg0 ".d}[1], [x1], #8\n"       \
    "st1 {v" reg1 ".4h},   [x2], #8\n"       \
    "st1 {v" reg1 ".d}[1], [x3], #8\n"       \
    "subs w10, w10, #1\n"

#define STORE_C                 \
    "mov w10, %w[remain_n]\n"  \
    STORE_LINE("8", "20")      \
    STORE_LINE("9", "21")      \
    STORE_LINE("10", "22")      \
    STORE_LINE("11", "23")      \
    STORE_LINE("12", "24")      \
    STORE_LINE("13", "25")      \
    STORE_LINE("14", "26")      \
    STORE_LINE("15", "27")      \
    STORE_LINE("16", "28")      \
    STORE_LINE("17", "29")      \
    STORE_LINE("18", "30")      \
    STORE_LINE("19", "31")

    // clang-format on

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"
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

            "ldr d4, [%[b_ptr]]\n"
            "ldr d5, [%[b_ptr], #8]\n"
            "ldr q0, [%[a_ptr]]\n"
            "subs %w[K], %w[K], #1\n"
            "ldr q1, [%[a_ptr], #16]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "cmp %w[K], #0\n"
            "sshll v5.8h, v5.8b, #0\n"
            "beq 4f\n"

            "3: \n"

            //! k0

            "ldr d2, [%[a_ptr], #32]\n"
            "nop\n"

            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #40]\n"
            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d3, [%[a_ptr], #48]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr x9, [%[a_ptr], #56]\n"

            "mla v13.8h, v0.8h, v4.h[5]\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"

            "ldr d6, [%[b_ptr], #12]\n"
            "ins v3.d[1], x9\n"

            "mla v16.8h, v0.8h, v5.h[0]\n"
            "ldr d7, [%[b_ptr], #20]\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"
            "mla v19.8h, v0.8h, v5.h[3]\n"
            "mla v20.8h, v1.8h, v4.h[0]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v21.8h, v1.8h, v4.h[1]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v22.8h, v1.8h, v4.h[2]\n"
            "mla v23.8h, v1.8h, v4.h[3]\n"
            "mla v24.8h, v1.8h, v4.h[4]\n"
            "mla v25.8h, v1.8h, v4.h[5]\n"
            "mla v26.8h, v1.8h, v4.h[6]\n"
            "mla v27.8h, v1.8h, v4.h[7]\n"
            "mla v28.8h, v1.8h, v5.h[0]\n"
            "mla v29.8h, v1.8h, v5.h[1]\n"
            "mla v30.8h, v1.8h, v5.h[2]\n"
            "mla v31.8h, v1.8h, v5.h[3]\n"

            //! k1
            "ldr d0, [%[a_ptr], #64]\n"
            "nop\n"

            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "ldr x8, [%[a_ptr], #72]\n"

            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"

            "ldr d1, [%[a_ptr], #80]\n"
            "ins v0.d[1], x8\n"

            "mla v12.8h, v2.8h, v6.h[4]\n"
            "ldr x9, [%[a_ptr], #88]\n"

            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"

            "ldr d4, [%[b_ptr], #24]\n"
            "ins v1.d[1], x9\n"

            "mla v16.8h, v2.8h, v7.h[0]\n"
            "ldr d5, [%[b_ptr], #32]\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v20.8h, v3.8h, v6.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v21.8h, v3.8h, v6.h[1]\n"
            "mla v22.8h, v3.8h, v6.h[2]\n"
            "mla v23.8h, v3.8h, v6.h[3]\n"
            "mla v24.8h, v3.8h, v6.h[4]\n"
            "mla v25.8h, v3.8h, v6.h[5]\n"
            "mla v26.8h, v3.8h, v6.h[6]\n"
            "mla v27.8h, v3.8h, v6.h[7]\n"
            "mla v28.8h, v3.8h, v7.h[0]\n"
            "mla v29.8h, v3.8h, v7.h[1]\n"
            "mla v30.8h, v3.8h, v7.h[2]\n"
            "mla v31.8h, v3.8h, v7.h[3]\n"

            //! k2
            "ldr d2, [%[a_ptr], #96]\n"
            "nop\n"

            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #104]\n"

            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d3, [%[a_ptr], #112]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr x9, [%[a_ptr], #120]\n"
            "mla v13.8h, v0.8h, v4.h[5]\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"

            "ldr d6, [%[b_ptr], #36]\n"
            "ins v3.d[1], x9\n"

            "mla v16.8h, v0.8h, v5.h[0]\n"
            "ldr d7, [%[b_ptr], #44]\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"

            "mla v19.8h, v0.8h, v5.h[3]\n"
            "add %[a_ptr], %[a_ptr], #128\n"

            "mla v20.8h, v1.8h, v4.h[0]\n"
            "add %[b_ptr], %[b_ptr], #48\n"

            "mla v21.8h, v1.8h, v4.h[1]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v22.8h, v1.8h, v4.h[2]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v23.8h, v1.8h, v4.h[3]\n"
            "mla v24.8h, v1.8h, v4.h[4]\n"
            "mla v25.8h, v1.8h, v4.h[5]\n"
            "mla v26.8h, v1.8h, v4.h[6]\n"
            "mla v27.8h, v1.8h, v4.h[7]\n"
            "mla v28.8h, v1.8h, v5.h[0]\n"
            "mla v29.8h, v1.8h, v5.h[1]\n"
            "mla v30.8h, v1.8h, v5.h[2]\n"
            "mla v31.8h, v1.8h, v5.h[3]\n"

            //! k3
            "ldr d0, [%[a_ptr]]\n"
            "nop\n"

            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "ldr x8, [%[a_ptr], #8]\n"

            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"

            "ldr d1, [%[a_ptr], #16]\n"
            "ins v0.d[1], x8\n"

            "mla v12.8h, v2.8h, v6.h[4]\n"
            "ldr x9, [%[a_ptr], #24]\n"

            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"

            "ldr d4, [%[b_ptr]]\n"
            "ins v1.d[1], x9\n"

            "mla v16.8h, v2.8h, v7.h[0]\n"
            "ldr d5, [%[b_ptr], #8]\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"
            "mla v20.8h, v3.8h, v6.h[0]\n"
            "mla v21.8h, v3.8h, v6.h[1]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v22.8h, v3.8h, v6.h[2]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v23.8h, v3.8h, v6.h[3]\n"
            "mla v24.8h, v3.8h, v6.h[4]\n"
            "mla v25.8h, v3.8h, v6.h[5]\n"
            "mla v26.8h, v3.8h, v6.h[6]\n"
            "mla v27.8h, v3.8h, v6.h[7]\n"
            "mla v28.8h, v3.8h, v7.h[0]\n"
            "mla v29.8h, v3.8h, v7.h[1]\n"
            "mla v30.8h, v3.8h, v7.h[2]\n"
            "mla v31.8h, v3.8h, v7.h[3]\n"

            "subs %w[K], %w[K], #1\n"
            "bne 3b\n"

            "4:\n"  //! tail
            //! k0

            "ldr d2, [%[a_ptr], #32]\n"
            "nop\n"

            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #40]\n"
            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d3, [%[a_ptr], #48]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr x9, [%[a_ptr], #56]\n"

            "mla v13.8h, v0.8h, v4.h[5]\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"

            "ldr d6, [%[b_ptr], #12]\n"
            "ins v3.d[1], x9\n"

            "mla v16.8h, v0.8h, v5.h[0]\n"
            "ldr d7, [%[b_ptr], #20]\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"
            "mla v19.8h, v0.8h, v5.h[3]\n"
            "mla v20.8h, v1.8h, v4.h[0]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v21.8h, v1.8h, v4.h[1]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v22.8h, v1.8h, v4.h[2]\n"
            "mla v23.8h, v1.8h, v4.h[3]\n"
            "mla v24.8h, v1.8h, v4.h[4]\n"
            "mla v25.8h, v1.8h, v4.h[5]\n"
            "mla v26.8h, v1.8h, v4.h[6]\n"
            "mla v27.8h, v1.8h, v4.h[7]\n"
            "mla v28.8h, v1.8h, v5.h[0]\n"
            "mla v29.8h, v1.8h, v5.h[1]\n"
            "mla v30.8h, v1.8h, v5.h[2]\n"
            "mla v31.8h, v1.8h, v5.h[3]\n"

            //! k1
            "ldr d0, [%[a_ptr], #64]\n"
            "nop\n"

            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "ldr x8, [%[a_ptr], #72]\n"

            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"

            "ldr d1, [%[a_ptr], #80]\n"
            "ins v0.d[1], x8\n"

            "mla v12.8h, v2.8h, v6.h[4]\n"
            "ldr x9, [%[a_ptr], #88]\n"

            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"

            "ldr d4, [%[b_ptr], #24]\n"
            "ins v1.d[1], x9\n"

            "mla v16.8h, v2.8h, v7.h[0]\n"
            "ldr d5, [%[b_ptr], #32]\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v20.8h, v3.8h, v6.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v21.8h, v3.8h, v6.h[1]\n"
            "mla v22.8h, v3.8h, v6.h[2]\n"
            "mla v23.8h, v3.8h, v6.h[3]\n"
            "mla v24.8h, v3.8h, v6.h[4]\n"
            "mla v25.8h, v3.8h, v6.h[5]\n"
            "mla v26.8h, v3.8h, v6.h[6]\n"
            "mla v27.8h, v3.8h, v6.h[7]\n"
            "mla v28.8h, v3.8h, v7.h[0]\n"
            "mla v29.8h, v3.8h, v7.h[1]\n"
            "mla v30.8h, v3.8h, v7.h[2]\n"
            "mla v31.8h, v3.8h, v7.h[3]\n"

            //! k2
            "ldr d2, [%[a_ptr], #96]\n"
            "nop\n"

            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #104]\n"

            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d3, [%[a_ptr], #112]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr x9, [%[a_ptr], #120]\n"

            "mla v13.8h, v0.8h, v4.h[5]\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"

            "ldr d6, [%[b_ptr], #36]\n"
            "ins v3.d[1], x9\n"

            "mla v16.8h, v0.8h, v5.h[0]\n"
            "ldr d7, [%[b_ptr], #44]\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"
            "mla v19.8h, v0.8h, v5.h[3]\n"
            "mla v20.8h, v1.8h, v4.h[0]\n"
            "mla v21.8h, v1.8h, v4.h[1]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v22.8h, v1.8h, v4.h[2]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v23.8h, v1.8h, v4.h[3]\n"
            "mla v24.8h, v1.8h, v4.h[4]\n"
            "mla v25.8h, v1.8h, v4.h[5]\n"
            "mla v26.8h, v1.8h, v4.h[6]\n"
            "mla v27.8h, v1.8h, v4.h[7]\n"
            "mla v28.8h, v1.8h, v5.h[0]\n"
            "mla v29.8h, v1.8h, v5.h[1]\n"
            "mla v30.8h, v1.8h, v5.h[2]\n"
            "mla v31.8h, v1.8h, v5.h[3]\n"

            //! k3

            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"
            "mla v12.8h, v2.8h, v6.h[4]\n"
            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"
            "mla v16.8h, v2.8h, v7.h[0]\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"
            "mla v20.8h, v3.8h, v6.h[0]\n"
            "mla v21.8h, v3.8h, v6.h[1]\n"
            "mla v22.8h, v3.8h, v6.h[2]\n"
            "mla v23.8h, v3.8h, v6.h[3]\n"
            "mla v24.8h, v3.8h, v6.h[4]\n"
            "mla v25.8h, v3.8h, v6.h[5]\n"
            "mla v26.8h, v3.8h, v6.h[6]\n"
            "mla v27.8h, v3.8h, v6.h[7]\n"
            "mla v28.8h, v3.8h, v7.h[0]\n"
            "mla v29.8h, v3.8h, v7.h[1]\n"
            "mla v30.8h, v3.8h, v7.h[2]\n"
            "cmp %w[remain_n], #12\n"
            "mla v31.8h, v3.8h, v7.h[3]\n"

            "bne 6f\n"
            "5:\n"

            "st1 {v8.4h, v9.4h}, [x0], #16\n"
            "st1 {v10.4h, v11.4h}, [x0], #16\n"
            "st1 {v12.4h, v13.4h}, [x0], #16\n"
            "st1 {v14.4h, v15.4h}, [x0], #16\n"
            "st1 {v16.4h, v17.4h}, [x0], #16\n"
            "st1 {v18.4h, v19.4h}, [x0], #16\n"

            "st1 {v8.d} [1], [x1], #8\n"
            "st1 {v9.d} [1], [x1], #8\n"
            "st1 {v10.d}[1], [x1], #8\n"
            "st1 {v11.d}[1], [x1], #8\n"
            "st1 {v12.d}[1], [x1], #8\n"
            "st1 {v13.d}[1], [x1], #8\n"
            "st1 {v14.d}[1], [x1], #8\n"
            "st1 {v15.d}[1], [x1], #8\n"
            "st1 {v16.d}[1], [x1], #8\n"
            "st1 {v17.d}[1], [x1], #8\n"
            "st1 {v18.d}[1], [x1], #8\n"
            "st1 {v19.d}[1], [x1], #8\n"

            "st1 {v20.4h, v21.4h}, [x2], #16\n"
            "st1 {v22.4h, v23.4h}, [x2], #16\n"
            "st1 {v24.4h, v25.4h}, [x2], #16\n"
            "st1 {v26.4h, v27.4h}, [x2], #16\n"
            "st1 {v28.4h, v29.4h}, [x2], #16\n"
            "st1 {v30.4h, v31.4h}, [x2], #16\n"

            "st1 {v20.d}[1], [x3], #8\n"
            "st1 {v21.d}[1], [x3], #8\n"
            "st1 {v22.d}[1], [x3], #8\n"
            "st1 {v23.d}[1], [x3], #8\n"
            "st1 {v24.d}[1], [x3], #8\n"
            "st1 {v25.d}[1], [x3], #8\n"
            "st1 {v26.d}[1], [x3], #8\n"
            "st1 {v27.d}[1], [x3], #8\n"
            "st1 {v28.d}[1], [x3], #8\n"
            "st1 {v29.d}[1], [x3], #8\n"
            "st1 {v30.d}[1], [x3], #8\n"
            "st1 {v31.d}[1], [x3], #8\n"

            "b 101f\n"

            "6:\n" STORE_C

            "101:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [outptr] "+r"(outptr), [remain_n] "+r"(remain_n)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
              "x8", "x9", "x10", "cc", "memory");

#undef STORE_C
#undef STORE_LINE
}

// clang-format off
/**
 * Overview of register layout:
 *
 * A 8x12x4 cell of Lhs is stored in 16bit in q0-q3
 * A 8x12x4 cell of Rhs is stored in 8bit in q4-q7
 * A 8x12 block of accumulators is stored in 16bit in q8-q31
 *
 *                     +------------------------------------------------------------------------+
 *                     | q4[0]|q4[1]|q4[2]|q4[3]|q4[4]|q4[5]|q4[6]|q4[7]|q5[0]|q5[1]|q5[2]|q5[3]|
 *                Rhs  +------------------------------------------------------------------------+
 *    Lhs              |      |     |     |     |     |     |     |     |     |     |     |     |
 *  +--------+ - - - - +------------------------------------------------------------------------+
 *  | q0 |             |  q8  | q9  | q10 | q11 | q12 | q13 | q14 | q15 | q16 | q17 | q18 | q19 |
 *  +--------+ - - - - +------------------------------------------------------------------------+
 *
 *                            Accumulator
 */
// clang-format on
static __attribute__((noinline)) void kern_8x12(const int16_t* packA,
                                                const int8_t* packB, int K,
                                                int16_t* output, int LDC,
                                                bool is_first_k, int remain_n) {
    K /= 4;
    const int16_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);

    // clang-format off
#define STORE_LINE(reg0)     \
    "cmp w10, #0 \n"                         \
    "beq 101f\n"                             \
    "st1 {v" reg0 ".4h},   [x0], #8\n"       \
    "st1 {v" reg0 ".d}[1], [x1], #8\n"       \
    "subs w10, w10, #1\n"

#define STORE_C                 \
    "mov w10, %w[remain_n]\n"  \
    STORE_LINE("8" )      \
    STORE_LINE("9" )      \
    STORE_LINE("10")      \
    STORE_LINE("11")      \
    STORE_LINE("12")      \
    STORE_LINE("13")      \
    STORE_LINE("14")      \
    STORE_LINE("15")      \
    STORE_LINE("16")      \
    STORE_LINE("17")      \
    STORE_LINE("18")      \
    STORE_LINE("19")

    // clang-format on

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"
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

            "ldr d4, [%[b_ptr]]\n"
            "ldr d5, [%[b_ptr], #8]\n"
            "ldr q0, [%[a_ptr]]\n"
            "subs %w[K], %w[K], #1\n"
            "sshll v4.8h, v4.8b, #0\n"
            "cmp %w[K], #0\n"
            "sshll v5.8h, v5.8b, #0\n"
            "beq 4f\n"

            "3: \n"

            //! k0

            "ldr d2, [%[a_ptr], #16]\n"
            "nop\n"
            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #24]\n"
            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #12]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #20]\n"
            "mla v13.8h, v0.8h, v4.h[5]\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.8h, v0.8h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"
            "mla v19.8h, v0.8h, v5.h[3]\n"

            //! k1
            "ldr d0, [%[a_ptr], #32]\n"
            "nop\n"

            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "ldr x8, [%[a_ptr], #40]\n"
            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"

            "ldr d4, [%[b_ptr], #24]\n"
            "ins v0.d[1], x8\n"

            "mla v12.8h, v2.8h, v6.h[4]\n"
            "ldr d5, [%[b_ptr], #32]\n"

            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v16.8h, v2.8h, v7.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"

            //! k2
            "ldr d2, [%[a_ptr], #48]\n"
            "nop\n"

            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #56]\n"

            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #36]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #44]\n"

            "mla v13.8h, v0.8h, v4.h[5]\n"
            "add %[a_ptr], %[a_ptr], #64\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "add %[b_ptr], %[b_ptr], #48\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.8h, v0.8h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"
            "mla v19.8h, v0.8h, v5.h[3]\n"

            //! k3
            "ldr d0, [%[a_ptr]]\n"
            "nop\n"

            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "ldr x8, [%[a_ptr], #8]\n"

            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"

            "ldr d4, [%[b_ptr]]\n"
            "ins v0.d[1], x8\n"

            "mla v12.8h, v2.8h, v6.h[4]\n"
            "ldr d5, [%[b_ptr], #8]\n"
            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v16.8h, v2.8h, v7.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "subs %w[K], %w[K], #1\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"

            "bne 3b\n"

            "4:\n"  // tail
                    //! k0

            "ldr d2, [%[a_ptr], #16]\n"
            "nop\n"
            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #24]\n"
            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #12]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #20]\n"
            "mla v13.8h, v0.8h, v4.h[5]\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.8h, v0.8h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"
            "mla v19.8h, v0.8h, v5.h[3]\n"

            //! k1
            "ldr d0, [%[a_ptr], #32]\n"
            "nop\n"

            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "ldr x8, [%[a_ptr], #40]\n"
            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"

            "ldr d4, [%[b_ptr], #24]\n"
            "ins v0.d[1], x8\n"

            "mla v12.8h, v2.8h, v6.h[4]\n"
            "ldr d5, [%[b_ptr], #32]\n"

            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v16.8h, v2.8h, v7.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"

            //! k2
            "ldr d2, [%[a_ptr], #48]\n"
            "nop\n"

            "mla v8.8h,  v0.8h, v4.h[0]\n"
            "ldr x8, [%[a_ptr], #56]\n"

            "mla v9.8h,  v0.8h, v4.h[1]\n"
            "mla v10.8h, v0.8h, v4.h[2]\n"
            "mla v11.8h, v0.8h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #36]\n"
            "ins v2.d[1], x8\n"

            "mla v12.8h, v0.8h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #44]\n"

            "mla v13.8h, v0.8h, v4.h[5]\n"
            "add %[a_ptr], %[a_ptr], #64\n"
            "mla v14.8h, v0.8h, v4.h[6]\n"
            "add %[b_ptr], %[b_ptr], #48\n"
            "mla v15.8h, v0.8h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.8h, v0.8h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.8h, v0.8h, v5.h[1]\n"
            "mla v18.8h, v0.8h, v5.h[2]\n"
            "mla v19.8h, v0.8h, v5.h[3]\n"

            //! k3
            "mla v8.8h,  v2.8h, v6.h[0]\n"
            "mla v9.8h,  v2.8h, v6.h[1]\n"
            "mla v10.8h, v2.8h, v6.h[2]\n"
            "mla v11.8h, v2.8h, v6.h[3]\n"
            "mla v12.8h, v2.8h, v6.h[4]\n"
            "mla v13.8h, v2.8h, v6.h[5]\n"
            "mla v14.8h, v2.8h, v6.h[6]\n"
            "mla v15.8h, v2.8h, v6.h[7]\n"
            "mla v16.8h, v2.8h, v7.h[0]\n"
            "mla v17.8h, v2.8h, v7.h[1]\n"
            "cmp %w[remain_n], #12\n"
            "mla v18.8h, v2.8h, v7.h[2]\n"
            "mla v19.8h, v2.8h, v7.h[3]\n"

            "bne 6f\n"
            "5:\n"

            "st1 {v8.4h, v9.4h}, [x0], #16\n"
            "st1 {v10.4h, v11.4h}, [x0], #16\n"
            "st1 {v12.4h, v13.4h}, [x0], #16\n"
            "st1 {v14.4h, v15.4h}, [x0], #16\n"
            "st1 {v16.4h, v17.4h}, [x0], #16\n"
            "st1 {v18.4h, v19.4h}, [x0], #16\n"

            "st1 {v8.d} [1], [x1], #8\n"
            "st1 {v9.d} [1], [x1], #8\n"
            "st1 {v10.d}[1], [x1], #8\n"
            "st1 {v11.d}[1], [x1], #8\n"
            "st1 {v12.d}[1], [x1], #8\n"
            "st1 {v13.d}[1], [x1], #8\n"
            "st1 {v14.d}[1], [x1], #8\n"
            "st1 {v15.d}[1], [x1], #8\n"
            "st1 {v16.d}[1], [x1], #8\n"
            "st1 {v17.d}[1], [x1], #8\n"
            "st1 {v18.d}[1], [x1], #8\n"
            "st1 {v19.d}[1], [x1], #8\n"

            "b 101f\n"

            "6:\n" STORE_C

            "101:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [outptr] "+r"(outptr), [remain_n] "+r"(remain_n)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "cc",
              "memory");

#undef STORE_C
#undef STORE_LINE
}

// clang-format off
/**
 * Overview of register layout:
 *
 * A 4x12x4 cell of Lhs is stored in 16bit in q0-q3
 * A 4x12x4 cell of Rhs is stored in 8bit in q4-q7
 * A 4x12 block of accumulators is stored in 16bit in q8-q31
 *
 *                     +------------------------------------------------------------------------+
 *                     | q4[0]|q4[1]|q4[2]|q4[3]|q4[4]|q4[5]|q4[6]|q4[7]|q5[0]|q5[1]|q5[2]|q5[3]|
 *                Rhs  +------------------------------------------------------------------------+
 *    Lhs              |      |     |     |     |     |     |     |     |     |     |     |     |
 *  +--------+ - - - - +------------------------------------------------------------------------+
 *  | d0 |             |  d8  | d9  | d10 | d11 | d12 | d13 | d14 | d15 | d16 | d17 | d18 | d19 |
 *  +--------+ - - - - +------------------------------------------------------------------------+
 *
 *                            Accumulator
 */
// clang-format on
static __attribute__((noinline)) void kern_4x12(const int16_t* packA,
                                                const int8_t* packB, int K,
                                                int16_t* output, int LDC,
                                                bool is_first_k, int remain_n) {
    K /= 4;
    const int16_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);

    // clang-format off
#define STORE_LINE(reg0)     \
    "cmp w10, #0 \n"                         \
    "beq 101f\n"                             \
    "st1 {v" reg0 ".4h},   [x0], #8\n"       \
    "subs w10, w10, #1\n"

#define STORE_C                 \
    "mov w10, %w[remain_n]\n"  \
    STORE_LINE("8" )      \
    STORE_LINE("9" )      \
    STORE_LINE("10")      \
    STORE_LINE("11")      \
    STORE_LINE("12")      \
    STORE_LINE("13")      \
    STORE_LINE("14")      \
    STORE_LINE("15")      \
    STORE_LINE("16")      \
    STORE_LINE("17")      \
    STORE_LINE("18")      \
    STORE_LINE("19")

    // clang-format on

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"
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

            "ldr d4, [%[b_ptr]]\n"
            "ldr d5, [%[b_ptr], #8]\n"
            "ldr d0, [%[a_ptr]]\n"
            "subs %w[K], %w[K], #1\n"
            "sshll v4.8h, v4.8b, #0\n"
            "cmp %w[K], #0\n"
            "sshll v5.8h, v5.8b, #0\n"
            "beq 4f\n"

            "3: \n"

            //! k0

            "ldr d2, [%[a_ptr], #8]\n"
            "nop\n"
            "mla v8.4h,  v0.4h, v4.h[0]\n"
            "mla v9.4h,  v0.4h, v4.h[1]\n"
            "mla v10.4h, v0.4h, v4.h[2]\n"
            "mla v11.4h, v0.4h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #12]\n"

            "mla v12.4h, v0.4h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #20]\n"
            "mla v13.4h, v0.4h, v4.h[5]\n"
            "mla v14.4h, v0.4h, v4.h[6]\n"
            "mla v15.4h, v0.4h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.4h, v0.4h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.4h, v0.4h, v5.h[1]\n"
            "mla v18.4h, v0.4h, v5.h[2]\n"
            "mla v19.4h, v0.4h, v5.h[3]\n"

            //! k1
            "ldr d0, [%[a_ptr], #16]\n"
            "nop\n"

            "mla v8.4h,  v2.4h, v6.h[0]\n"
            "mla v9.4h,  v2.4h, v6.h[1]\n"
            "mla v10.4h, v2.4h, v6.h[2]\n"
            "mla v11.4h, v2.4h, v6.h[3]\n"

            "ldr d4, [%[b_ptr], #24]\n"

            "mla v12.4h, v2.4h, v6.h[4]\n"
            "ldr d5, [%[b_ptr], #32]\n"

            "mla v13.4h, v2.4h, v6.h[5]\n"
            "mla v14.4h, v2.4h, v6.h[6]\n"
            "mla v15.4h, v2.4h, v6.h[7]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v16.4h, v2.4h, v7.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v17.4h, v2.4h, v7.h[1]\n"
            "mla v18.4h, v2.4h, v7.h[2]\n"
            "mla v19.4h, v2.4h, v7.h[3]\n"

            //! k2
            "ldr d2, [%[a_ptr], #24]\n"
            "nop\n"

            "mla v8.4h,  v0.4h, v4.h[0]\n"
            "mla v9.4h,  v0.4h, v4.h[1]\n"
            "mla v10.4h, v0.4h, v4.h[2]\n"
            "mla v11.4h, v0.4h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #36]\n"

            "mla v12.4h, v0.4h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #44]\n"

            "mla v13.4h, v0.4h, v4.h[5]\n"
            "add %[a_ptr], %[a_ptr], #32\n"
            "mla v14.4h, v0.4h, v4.h[6]\n"
            "add %[b_ptr], %[b_ptr], #48\n"
            "mla v15.4h, v0.4h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.4h, v0.4h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.4h, v0.4h, v5.h[1]\n"
            "mla v18.4h, v0.4h, v5.h[2]\n"
            "mla v19.4h, v0.4h, v5.h[3]\n"

            //! k3
            "ldr d0, [%[a_ptr]]\n"
            "nop\n"

            "mla v8.4h,  v2.4h, v6.h[0]\n"
            "mla v9.4h,  v2.4h, v6.h[1]\n"
            "mla v10.4h, v2.4h, v6.h[2]\n"
            "mla v11.4h, v2.4h, v6.h[3]\n"

            "ldr d4, [%[b_ptr]]\n"

            "mla v12.4h, v2.4h, v6.h[4]\n"
            "ldr d5, [%[b_ptr], #8]\n"
            "mla v13.4h, v2.4h, v6.h[5]\n"
            "mla v14.4h, v2.4h, v6.h[6]\n"
            "mla v15.4h, v2.4h, v6.h[7]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v16.4h, v2.4h, v7.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v17.4h, v2.4h, v7.h[1]\n"
            "subs %w[K], %w[K], #1\n"
            "mla v18.4h, v2.4h, v7.h[2]\n"
            "mla v19.4h, v2.4h, v7.h[3]\n"

            "bne 3b\n"

            "4:\n"  // tail
            //! k0

            "ldr d2, [%[a_ptr], #8]\n"
            "nop\n"
            "mla v8.4h,  v0.4h, v4.h[0]\n"
            "mla v9.4h,  v0.4h, v4.h[1]\n"
            "mla v10.4h, v0.4h, v4.h[2]\n"
            "mla v11.4h, v0.4h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #12]\n"

            "mla v12.4h, v0.4h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #20]\n"
            "mla v13.4h, v0.4h, v4.h[5]\n"
            "mla v14.4h, v0.4h, v4.h[6]\n"
            "mla v15.4h, v0.4h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.4h, v0.4h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.4h, v0.4h, v5.h[1]\n"
            "mla v18.4h, v0.4h, v5.h[2]\n"
            "mla v19.4h, v0.4h, v5.h[3]\n"

            //! k1
            "ldr d0, [%[a_ptr], #16]\n"
            "nop\n"

            "mla v8.4h,  v2.4h, v6.h[0]\n"
            "mla v9.4h,  v2.4h, v6.h[1]\n"
            "mla v10.4h, v2.4h, v6.h[2]\n"
            "mla v11.4h, v2.4h, v6.h[3]\n"

            "ldr d4, [%[b_ptr], #24]\n"

            "mla v12.4h, v2.4h, v6.h[4]\n"
            "ldr d5, [%[b_ptr], #32]\n"

            "mla v13.4h, v2.4h, v6.h[5]\n"
            "mla v14.4h, v2.4h, v6.h[6]\n"
            "mla v15.4h, v2.4h, v6.h[7]\n"
            "sshll v4.8h, v4.8b, #0\n"
            "mla v16.4h, v2.4h, v7.h[0]\n"
            "sshll v5.8h, v5.8b, #0\n"
            "mla v17.4h, v2.4h, v7.h[1]\n"
            "mla v18.4h, v2.4h, v7.h[2]\n"
            "mla v19.4h, v2.4h, v7.h[3]\n"

            //! k2
            "ldr d2, [%[a_ptr], #24]\n"
            "nop\n"

            "mla v8.4h,  v0.4h, v4.h[0]\n"
            "mla v9.4h,  v0.4h, v4.h[1]\n"
            "mla v10.4h, v0.4h, v4.h[2]\n"
            "mla v11.4h, v0.4h, v4.h[3]\n"

            "ldr d6, [%[b_ptr], #36]\n"

            "mla v12.4h, v0.4h, v4.h[4]\n"
            "ldr d7, [%[b_ptr], #44]\n"

            "mla v13.4h, v0.4h, v4.h[5]\n"
            "add %[a_ptr], %[a_ptr], #32\n"
            "mla v14.4h, v0.4h, v4.h[6]\n"
            "add %[b_ptr], %[b_ptr], #48\n"
            "mla v15.4h, v0.4h, v4.h[7]\n"
            "sshll v6.8h, v6.8b, #0\n"
            "mla v16.4h, v0.4h, v5.h[0]\n"
            "sshll v7.8h, v7.8b, #0\n"
            "mla v17.4h, v0.4h, v5.h[1]\n"
            "mla v18.4h, v0.4h, v5.h[2]\n"
            "mla v19.4h, v0.4h, v5.h[3]\n"

            //! k3
            "mla v8.4h,  v2.4h, v6.h[0]\n"
            "mla v9.4h,  v2.4h, v6.h[1]\n"
            "mla v10.4h, v2.4h, v6.h[2]\n"
            "mla v11.4h, v2.4h, v6.h[3]\n"
            "mla v12.4h, v2.4h, v6.h[4]\n"
            "mla v13.4h, v2.4h, v6.h[5]\n"
            "mla v14.4h, v2.4h, v6.h[6]\n"
            "mla v15.4h, v2.4h, v6.h[7]\n"
            "mla v16.4h, v2.4h, v7.h[0]\n"
            "cmp %w[remain_n], #12\n"
            "mla v17.4h, v2.4h, v7.h[1]\n"
            "mla v18.4h, v2.4h, v7.h[2]\n"
            "mla v19.4h, v2.4h, v7.h[3]\n"

            "bne 6f\n"
            "5:\n"

            "st1 {v8.4h, v9.4h}, [x0], #16\n"
            "st1 {v10.4h, v11.4h}, [x0], #16\n"
            "st1 {v12.4h, v13.4h}, [x0], #16\n"
            "st1 {v14.4h, v15.4h}, [x0], #16\n"
            "st1 {v16.4h, v17.4h}, [x0], #16\n"
            "st1 {v18.4h, v19.4h}, [x0], #16\n"
            "b 101f\n"

            "6:\n" STORE_C

            "101:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [outptr] "+r"(outptr), [remain_n] "+r"(remain_n)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "cc",
              "memory");

#undef STORE_C
#undef STORE_LINE
}

static void gemm_s8x8x16_mk4_16x12_pack_A(dt_int16* outptr,
                                          const dt_int8* inptr, int ldin,
                                          int m0, int mmax, int k0, int kmax) {
    megdnn_assert(m0 % 4 == 0 && mmax % 4 == 0, "M must be time of 4");
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    constexpr int pack_m = 16;
    constexpr int pack_k = 4;
    constexpr int pack_size = 4;
    const int m_size = mmax - m0;
    const int m_end = m_size / pack_m * pack_m + m0;
    int remain_m = mmax - m_end;

    for (int m_idx = m0; m_idx < m_end; m_idx += pack_m) {
        const int8_t* inptr0 = inptr + m_idx / pack_size * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr0 + 2 * ldin;
        const int8_t* inptr3 = inptr0 + 3 * ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);

        for (int k_idx = k0; k_idx < kmax; k_idx += pack_size) {
            interleave_4x4_16x4_s8_s16(inptr0, inptr1, inptr2, inptr3, outptr);
            inptr0 += pack_size * pack_size;
            inptr1 += pack_size * pack_size;
            inptr2 += pack_size * pack_size;
            inptr3 += pack_size * pack_size;
            outptr += pack_m * pack_k;
        }
    }
    int m_idx = m_end;
    if (remain_m >= 8) {
        const int8_t* inptr0 = inptr + m_idx / pack_size * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        for (int k_idx = k0; k_idx < kmax; k_idx += pack_size) {
            interleave_4x4_8x4_s8_s16(inptr0, inptr1, outptr);
            inptr0 += pack_size * pack_size;
            inptr1 += pack_size * pack_size;
            outptr += 8 * pack_k;
        }
        remain_m -= 8;
        m_idx += 8;
    }
    if (remain_m == 4) {
        const int8_t* inptr0 = inptr + m_idx / pack_size * ldin + k0;
        const int k_size = kmax - k0;
        memcpy_s8_s16(inptr0, outptr, k_size * pack_size);
    }
}

static void gemm_s8x8x16_mk4_16x12_pack_B(dt_int8* out, const dt_int8* in,
                                          int ldin, int n0, int nmax, int k0,
                                          int kmax) {
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");

    constexpr int pack_n = 12;
    constexpr int pack_size = 4;
    int8_t tmpbuff[pack_n * pack_size] = {0};
    const int ksize = kmax - k0;
    const int nsize = nmax - n0;
    const int n_end = nsize / pack_n * pack_n + n0;
    const int remain_n = nsize % pack_n;
    int output_stride = ksize * pack_n;
    int8_t* outptr_base = out;

    for (int k_idx = k0; k_idx < kmax; k_idx += pack_size) {
        const int8_t* inptr = in + k_idx / pack_size * ldin + n0 * pack_size;
        prefetch_3x(inptr);

        auto outptr = outptr_base;
        for (int n_idx = n0; n_idx < n_end; n_idx += pack_n) {
            transpos_12x4_s8(inptr, outptr);
            inptr += pack_n * pack_size;
            outptr += output_stride;
        }
        if (remain_n > 0) {
            memcpy(tmpbuff, inptr, sizeof(int8_t) * remain_n * pack_size);
            transpos_12x4_s8(tmpbuff, outptr);
            outptr += output_stride;
        }
        outptr_base += pack_n * pack_size;
    }
}

}  // namespace matmul_mk4_16x12x4_a53
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
