/**
 * \file dnn/src/aarch64/matrix_mul/quint8_dot/kernel_8x8x4.h
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
namespace matmul_8x8x4 {

//! calc v0 = v0 - v1[lane1] - v2
#define SUB_LANE(v0, v1, lane1, v2, vtmp)   \
    "dup v" #vtmp ".4s, v" #v1 ".s[" #lane1 \
    "]\n"                                   \
    "sub v" #v0 ".4s, v" #v0 ".4s, v" #vtmp \
    ".4s\n"                                 \
    "sub v" #v0 ".4s, v" #v0 ".4s, v" #v2 ".4s\n"

// Overview of register layout:
//
// A 8x4 cell of Rhs is stored in 8bit in q2-q3.
// A 8x4x2 cell of Lhs is stored in 8bit in q0-q1,q4-q5
// A 8x12 block of accumulators is stored in 32bit in q6--q21.
//
//                            +--------+--------+
//                            |v2[0-16]|v3[0-16]|
//                       Rhs  +--------+--------+
//
//                            |        |        |
//
//    Lhs                     |        |        |
//
//  +-------+-------+ - - - - +--------+--------+
//  |v0[0-4]|v4[0-4]|         | v6[0-4]|v14[0-4]|
//  |v0[0-4]|v4[0-4]|         | v7[0-4]|v15[0-4]|
//  |v0[0-4]|v4[0-4]|         | v8[0-4]|v16[0-4]|
//  |v0[0-4]|v4[0-4]|         | v9[0-4]|v17[0-4]|
//  |v1[0-4]|v5[0-4]|         |v10[0-4]|v18[0-4]|
//  |v1[0-4]|v5[0-4]|         |v11[0-4]|v19[0-4]|
//  |v1[0-4]|v5[0-4]|         |v12[0-4]|v20[0-4]|
//  |v1[0-4]|v5[0-4]|         |v13[0-4]|v21[0-4]|
//  +-------+-------+ - - - - +--------+--------+
//
//                            Accumulator
//
//  C = sum((A - zA) * (B - zB)) = sum(A * B) - sum(A) * zB - sum(B) * zA + zA *
//      zB * k
//  A -> v27, v28 | B -> v29, v30 |  zA * zB * k -> v26

static void kern_8x8(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k,
                     uint8_t zero_point_A, uint8_t zero_point_B, uint32_t zAB) {
    K /= 4;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;

    int32x4_t a0;
    int32x4_t a1;
    int32x4_t b0;
    int32x4_t b1;
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
            "dup v24.16b, %w[zero_point_B] \n"
            "dup v25.16b, %w[zero_point_A] \n"
            "dup v26.4s, %w[zAB] \n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"

            "ldp q6, q14, [%[outptr0]]\n"
            "ldp q7, q15, [%[outptr1]]\n"
            "ldp q8, q16, [%[outptr2]]\n"
            "ldp q9, q17, [%[outptr3]]\n"
            "ldp q10, q18, [%[outptr4]]\n"
            "ldp q11, q19, [%[outptr5]]\n"
            "ldp q12, q20, [%[outptr6]]\n"
            "ldp q13, q21, [%[outptr7]]\n"
            "b 2f\n"

            "1:\n"
            "eor v6.16b,  v6.16b,  v6.16b\n"
            "eor v7.16b,  v7.16b,  v7.16b\n"
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
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"
            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "udot v27.4s, %[a0].16b, v24.16b\n"
            "udot v28.4s, %[a1].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot v30.4s, %[b1].16b, v25.16b\n"
            "udot  v6.4s, %[b0].16b, %[a0].4b[0]\n"
            "udot  v7.4s, %[b0].16b, %[a0].4b[1]\n"
            "udot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "udot  v10.4s, %[b0].16b, %[a1].4b[0]\n"
            "udot  v11.4s, %[b0].16b, %[a1].4b[1]\n"
            "udot  v12.4s, %[b0].16b, %[a1].4b[2]\n"
            "udot  v13.4s, %[b0].16b, %[a1].4b[3]\n"
            "udot  v14.4s, %[b1].16b, %[a0].4b[0]\n"
            "udot  v15.4s, %[b1].16b, %[a0].4b[1]\n"
            "udot  v16.4s, %[b1].16b, %[a0].4b[2]\n"
            "udot  v17.4s, %[b1].16b, %[a0].4b[3]\n"
            "udot  v18.4s, %[b1].16b, %[a1].4b[0]\n"
            "udot  v19.4s, %[b1].16b, %[a1].4b[1]\n"
            "udot  v20.4s, %[b1].16b, %[a1].4b[2]\n"
            "udot  v21.4s, %[b1].16b, %[a1].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[a1a], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "udot v27.4s, %[a0].16b, v24.16b\n"
            "udot v28.4s, %[a1].16b, v24.16b\n"
            "udot v27.4s, %[a0a].16b, v24.16b\n"
            "udot v28.4s, %[a1a].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot v30.4s, %[b1].16b, v25.16b\n"
            "udot  v6.4s, %[b0].16b, %[a0].4b[0]\n"
            "udot  v7.4s, %[b0].16b, %[a0].4b[1]\n"
            "udot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "udot  v10.4s, %[b0].16b, %[a1].4b[0]\n"
            "udot  v11.4s, %[b0].16b, %[a1].4b[1]\n"
            "udot  v12.4s, %[b0].16b, %[a1].4b[2]\n"
            "udot  v13.4s, %[b0].16b, %[a1].4b[3]\n"
            "udot  v14.4s, %[b1].16b, %[a0].4b[0]\n"
            "udot  v15.4s, %[b1].16b, %[a0].4b[1]\n"
            "udot  v16.4s, %[b1].16b, %[a0].4b[2]\n"
            "udot  v17.4s, %[b1].16b, %[a0].4b[3]\n"
            "udot  v18.4s, %[b1].16b, %[a1].4b[0]\n"
            "udot  v19.4s, %[b1].16b, %[a1].4b[1]\n"
            "udot  v20.4s, %[b1].16b, %[a1].4b[2]\n"
            "udot  v21.4s, %[b1].16b, %[a1].4b[3]\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot v30.4s, %[b1].16b, v25.16b\n"
            "udot  v6.4s, %[b0].16b, %[a0a].4b[0]\n"
            "udot  v7.4s, %[b0].16b, %[a0a].4b[1]\n"
            "udot  v8.4s, %[b0].16b, %[a0a].4b[2]\n"
            "udot  v9.4s, %[b0].16b, %[a0a].4b[3]\n"
            "udot  v10.4s, %[b0].16b, %[a1a].4b[0]\n"
            "udot  v11.4s, %[b0].16b, %[a1a].4b[1]\n"
            "udot  v12.4s, %[b0].16b, %[a1a].4b[2]\n"
            "udot  v13.4s, %[b0].16b, %[a1a].4b[3]\n"
            "udot  v14.4s, %[b1].16b, %[a0a].4b[0]\n"
            "udot  v15.4s, %[b1].16b, %[a0a].4b[1]\n"
            "udot  v16.4s, %[b1].16b, %[a0a].4b[2]\n"
            "udot  v17.4s, %[b1].16b, %[a0a].4b[3]\n"
            "udot  v18.4s, %[b1].16b, %[a1a].4b[0]\n"
            "udot  v19.4s, %[b1].16b, %[a1a].4b[1]\n"
            "udot  v20.4s, %[b1].16b, %[a1a].4b[2]\n"
            "udot  v21.4s, %[b1].16b, %[a1a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n"
            //! minus zAB
            "sub v27.4s, v27.4s, v26.4s\n"
            "sub v28.4s, v28.4s, v26.4s\n"

            // clang-format off
            SUB_LANE(6, 27, 0, 29, 23)
            SUB_LANE(14, 27, 0, 30, 23)
            SUB_LANE(7, 27, 1, 29, 23)
            SUB_LANE(15, 27, 1, 30, 23)
            SUB_LANE(8, 27, 2, 29, 23)
            SUB_LANE(16, 27, 2, 30, 23)
            SUB_LANE(9, 27, 3, 29, 23)
            SUB_LANE(17, 27, 3, 30, 23)
            SUB_LANE(10, 28, 0, 29, 23)
            SUB_LANE(18, 28, 0, 30, 23)
            SUB_LANE(11, 28, 1, 29, 23)
            SUB_LANE(19, 28, 1, 30, 23)
            SUB_LANE(12, 28, 2, 29, 23)
            SUB_LANE(20, 28, 2, 30, 23)
            SUB_LANE(13, 28, 3, 29, 23)
            SUB_LANE(21, 28, 3, 30, 23)
            // clang-format on

            "stp q6, q14, [%[outptr0]]\n"
            "stp q7, q15, [%[outptr1]]\n"
            "stp q8, q16, [%[outptr2]]\n"
            "stp q9, q17, [%[outptr3]]\n"
            "stp q10, q18, [%[outptr4]]\n"
            "stp q11, q19, [%[outptr5]]\n"
            "stp q12, q20, [%[outptr6]]\n"
            "stp q13, q21, [%[outptr7]]\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [a0] "+w"(a0),
              [a1] "+w"(a1), [a0a] "+w"(a0a), [a1a] "+w"(a1a), [b0] "+w"(b0),
              [b1] "+w"(b1), [k] "+r"(k), [LDC] "+r"(LDC), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [outptr0] "+r"(outptr0),
              [zero_point_A] "+r"(zero_point_A),
              [zero_point_B] "+r"(zero_point_B), [zAB] "+r"(zAB),
              [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [outptr4] "=r"(outptr4),
              [outptr5] "=r"(outptr5), [outptr6] "=r"(outptr6),
              [outptr7] "=r"(outptr7)
            :
            : "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "cc", "memory");
}

// Overview of register layout:
//
// A 8x4 cell of Rhs is stored in 8bit in q1-q2, q4-q5.
// A 8x4x2 cell of Lhs is stored in 8bit in q0,q3
// A 8x12 block of accumulators is stored in 8bit in q8--q31.
//
//                            +--------+--------+
//                            |v1[0-16]|v2[0-16]|
//                       Rhs  +--------+--------+
//                            |v4[0-16]|v5[0-16]|
//                            +--------+--------+
//
//                            |        |        |
//
//    Lhs                     |        |        |
//
//  +-------+-------+ - - - - +--------+--------+
//  |v0[0-4]|v3[0-4]|         | v6[0-4]|v10[0-4]|
//  |v0[0-4]|v3[0-4]|         | v7[0-4]|v11[0-4]|
//  |v0[0-4]|v3[0-4]|         | v8[0-4]|v12[0-4]|
//  |v0[0-4]|v3[0-4]|         | v9[0-4]|v13[0-4]|
//  +-------+-------+ - - - - +--------+--------+
//
//                            Accumulator
//
//  C = sum((A - zA) * (B - zB)) = sum(A * B) - sum(A) * zB - sum(B) * zA + zA *
//      zB * k
//  A -> v28 | B -> v29, v30 |  zA * zB * k -> v26

static void kern_4x8(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int m_remain,
                     uint8_t zero_point_A, uint8_t zero_point_B, uint32_t zAB) {
    K /= 4;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;
    int32x4_t a0;
    int32x4_t b0;
    int32x4_t b1;
    int32x4_t a0a;
    int32x4_t b0a;
    int32x4_t b1a;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    size_t x0;

// clang-format off
#define LOAD_LINE(v1, v2, m)                \
    "cbz %[x0], 100f\n"                     \
    "ldp " v1 "," v2 ", [%[outptr" m "]]\n" \
    "subs %[x0], %[x0], #1\n"

#define LOAD_C \
    "mov %[x0], %x[m_remain]\n" \
    LOAD_LINE("q6", "q10", "0") \
    LOAD_LINE("q7", "q11", "1") \
    LOAD_LINE("q8", "q12", "2") \
    LOAD_LINE("q9", "q13", "3") \
    "100:\n"

#define STORE_LINE(v1, v2, m)              \
    "cbz %[x0], 101f\n"                    \
    "stp " v1 "," v2", [%[outptr" m "]]\n" \
    "subs %[x0], %[x0], #1\n"

#define STORE_C \
    "mov %[x0], %x[m_remain]\n"  \
    STORE_LINE("q6", "q10", "0") \
    STORE_LINE("q7", "q11", "1") \
    STORE_LINE("q8", "q12", "2") \
    STORE_LINE("q9", "q13", "3") \
    "101:\n"

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "dup v24.16b, %w[zero_point_B] \n"
            "dup v25.16b, %w[zero_point_A] \n"
            "dup v26.4s, %w[zAB] \n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v6.16b,  v6.16b,  v6.16b\n"
            "eor v7.16b,  v7.16b,  v7.16b\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"

            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "udot v28.4s, %[a0].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot v30.4s, %[b1].16b, v25.16b\n"
            "udot  v6.4s, %[b0].16b, %[a0].4b[0]\n"
            "udot  v7.4s, %[b0].16b, %[a0].4b[1]\n"
            "udot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "udot  v10.4s, %[b1].16b, %[a0].4b[0]\n"
            "udot  v11.4s, %[b1].16b, %[a0].4b[1]\n"
            "udot  v12.4s, %[b1].16b, %[a0].4b[2]\n"
            "udot  v13.4s, %[b1].16b, %[a0].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[b1], [%[b_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "ldr  %q[b1a], [%[b_ptr]], #16\n"
            "udot v28.4s, %[a0].16b, v24.16b\n"
            "udot v28.4s, %[a0a].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot v30.4s, %[b1].16b, v25.16b\n"
            "udot v29.4s, %[b0a].16b, v25.16b\n"
            "udot v30.4s, %[b1a].16b, v25.16b\n"

            "udot  v6.4s, %[b0].16b, %[a0].4b[0]\n"
            "udot  v7.4s, %[b0].16b, %[a0].4b[1]\n"
            "udot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "udot  v10.4s, %[b1].16b, %[a0].4b[0]\n"
            "udot  v11.4s, %[b1].16b, %[a0].4b[1]\n"
            "udot  v12.4s, %[b1].16b, %[a0].4b[2]\n"
            "udot  v13.4s, %[b1].16b, %[a0].4b[3]\n"
            "udot  v6.4s , %[b0a].16b, %[a0a].4b[0]\n"
            "udot  v7.4s , %[b0a].16b, %[a0a].4b[1]\n"
            "udot  v8.4s, %[b0a].16b, %[a0a].4b[2]\n"
            "udot  v9.4s, %[b0a].16b, %[a0a].4b[3]\n"
            "udot  v10.4s, %[b1a].16b, %[a0a].4b[0]\n"
            "udot  v11.4s, %[b1a].16b, %[a0a].4b[1]\n"
            "udot  v12.4s, %[b1a].16b, %[a0a].4b[2]\n"
            "udot  v13.4s, %[b1a].16b, %[a0a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n"
            //! minus zAB
            "sub v29.4s, v29.4s, v26.4s\n"
            "sub v30.4s, v30.4s, v26.4s\n"

            // clang-format off
            SUB_LANE(6, 28, 0, 29, 23)
            SUB_LANE(10, 28, 0, 30, 23)
            SUB_LANE(7, 28, 1, 29, 23)
            SUB_LANE(11, 28, 1, 30, 23)
            SUB_LANE(8, 28, 2, 29, 23)
            SUB_LANE(12, 28, 2, 30, 23)
            SUB_LANE(9, 28, 3, 29, 23)
            SUB_LANE(13, 28, 3, 30, 23)
            // clang-format on

            STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [k] "+r"(k),
              [outptr0] "+r"(outptr0), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [m_remain] "+r"(m_remain),
              [zero_point_A] "+r"(zero_point_A),
              [zero_point_B] "+r"(zero_point_B), [zAB] "+r"(zAB),
              [LDC] "+r"(LDC), [a0] "=w"(a0), [a0a] "=w"(a0a), [b0] "=w"(b0),
              [b1] "=w"(b1), [b0a] "=w"(b0a), [b1a] "=w"(b1a),
              [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [x0] "=r"(x0)
            :
            : "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v23", "v24",
              "v25", "v26", "v28", "v29", "v30", "memory", "cc");

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
//
//  C = sum((A - zA) * (B - zB)) = sum(A * B) - sum(A) * zB - sum(B) * zA + zA *
//      zB * k
//  A -> v27, v28 | B -> v29 |  zA * zB * k -> v26

static void kern_8x4(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int n_remain,
                     uint8_t zero_point_A, uint8_t zero_point_B, uint32_t zAB) {
    K /= 4;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;
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
#define LOAD_LINE(reg_index, n)                \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                   \
    "blt 100" n "f\n"                          \
    "ldr q" reg_index ", [%[x0]] \n"           \
    "b 101" n "f\n"                            \
    "100" n ":\n"                              \
    "cmp %w[n_remain], #0\n"                   \
    "beq 101" n "f\n"                          \
    "ld1 {v" reg_index ".s}[0], [%[x0]], #4\n" \
    "cmp %w[n_remain], #1\n"                   \
    "beq 101" n "f\n"                          \
    "ld1 {v" reg_index ".s}[1], [%[x0]], #4\n" \
    "cmp %w[n_remain], #2\n"                   \
    "beq 101" n "f\n"                          \
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

#define STORE_LINE(reg_index, n)               \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                   \
    "blt 102" n "f\n"                          \
    "str q" reg_index ", [%[x0]]\n"            \
    "b 103" n "f\n"                            \
    "102" n ":\n"                              \
    "cmp %w[n_remain], #0\n"                   \
    "beq 103" n "f\n"                          \
    "st1 {v" reg_index ".s}[0], [%[x0]], #4\n" \
    "cmp %w[n_remain], #1\n"                   \
    "beq 103" n "f\n"                          \
    "st1 {v" reg_index ".s}[1], [%[x0]], #4\n" \
    "cmp %w[n_remain], #2\n"                   \
    "beq 103" n "f\n"                          \
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
            "dup v24.16b, %w[zero_point_B] \n"
            "dup v25.16b, %w[zero_point_A] \n"
            "dup v26.4s, %w[zAB] \n"
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
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"

            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "udot v27.4s, %[a0].16b, v24.16b\n"
            "udot v28.4s, %[a1].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot  v6.4s , %[b0].16b, %[a0].4b[0]\n"
            "udot  v7.4s , %[b0].16b, %[a0].4b[1]\n"
            "udot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "udot  v10.4s, %[b0].16b, %[a1].4b[0]\n"
            "udot  v11.4s, %[b0].16b, %[a1].4b[1]\n"
            "udot  v12.4s, %[b0].16b, %[a1].4b[2]\n"
            "udot  v13.4s, %[b0].16b, %[a1].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a1], [%[a_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[a1a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "udot v27.4s, %[a0].16b, v24.16b\n"
            "udot v28.4s, %[a1].16b, v24.16b\n"
            "udot v27.4s, %[a0a].16b, v24.16b\n"
            "udot v28.4s, %[a1a].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot v29.4s, %[b0a].16b, v25.16b\n"
            "udot  v6.4s , %[b0].16b, %[a0].4b[0]\n"
            "udot  v7.4s , %[b0].16b, %[a0].4b[1]\n"
            "udot  v8.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v9.4s, %[b0].16b, %[a0].4b[3]\n"
            "udot  v10.4s, %[b0].16b, %[a1].4b[0]\n"
            "udot  v11.4s, %[b0].16b, %[a1].4b[1]\n"
            "udot  v12.4s, %[b0].16b, %[a1].4b[2]\n"
            "udot  v13.4s, %[b0].16b, %[a1].4b[3]\n"
            "udot  v6.4s , %[b0a].16b, %[a0a].4b[0]\n"
            "udot  v7.4s , %[b0a].16b, %[a0a].4b[1]\n"
            "udot  v8.4s, %[b0a].16b, %[a0a].4b[2]\n"
            "udot  v9.4s, %[b0a].16b, %[a0a].4b[3]\n"
            "udot  v10.4s, %[b0a].16b, %[a1a].4b[0]\n"
            "udot  v11.4s, %[b0a].16b, %[a1a].4b[1]\n"
            "udot  v12.4s, %[b0a].16b, %[a1a].4b[2]\n"
            "udot  v13.4s, %[b0a].16b, %[a1a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n"
            //! minus zAB
            "sub v27.4s, v27.4s, v26.4s\n"
            "sub v28.4s, v28.4s, v26.4s\n"

            // clang-format off
            SUB_LANE(6, 27, 0, 29, 23)
            SUB_LANE(7, 27, 1, 29, 23)
            SUB_LANE(8, 27, 2, 29, 23)
            SUB_LANE(9, 27, 3, 29, 23)
            SUB_LANE(10, 28, 0, 29, 23)
            SUB_LANE(11, 28, 1, 29, 23)
            SUB_LANE(12, 28, 2, 29, 23)
            SUB_LANE(13, 28, 3, 29, 23)
            // clang-format on

            STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [LDC] "+r"(LDC),
              [oddk] "+r"(oddk), [is_first_k] "+r"(is_first_k),
              [n_remain] "+r"(n_remain), [k] "+r"(k), [outptr0] "+r"(outptr0),
              [zero_point_A] "+r"(zero_point_A),
              [zero_point_B] "+r"(zero_point_B), [zAB] "+r"(zAB), [a0] "=w"(a0),
              [a1] "=w"(a1), [a0a] "=w"(a0a), [a1a] "=w"(a1a), [b0] "=w"(b0),
              [b0a] "=w"(b0a), [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [outptr4] "=r"(outptr4),
              [outptr5] "=r"(outptr5), [outptr6] "=r"(outptr6),
              [outptr7] "=r"(outptr7), [x0] "=r"(x0)
            :
            : "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v23", "v24",
              "v25", "v26", "v27", "v28", "v29", "memory", "cc");

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
//
//  C = sum((A - zA) * (B - zB)) = sum(A * B) - sum(A) * zB - sum(B) * zA + zA *
//      zB * k
//  A -> v28 | B -> v29 |  zA * zB * k -> v26

static void kern_4x4(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain, uint8_t zero_point_A, uint8_t zero_point_B,
                     uint32_t zAB) {
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
#define LOAD_LINE(reg_index, n)                 \
    "cbz %[x1], 102f\n"                         \
    "mov %[x0], %[outptr" n "]\n"               \
    "cmp %w[n_remain], #4\n"                    \
    "blt 100" n "f\n"                           \
    "ldr q" reg_index ", [%[x0]]\n"             \
    "b 101" n "f\n"                             \
    "100" n ":\n"                               \
    "cmp %w[n_remain], #0\n"                    \
    "beq 101" n "f\n"                           \
    "ld1 {v" reg_index ".s}[0], [%[x0]], #4\n"  \
    "cmp %w[n_remain], #1\n"                    \
    "beq 101" n "f\n"                           \
    "ld1 {v" reg_index ".s}[1], [%[x0]], #4\n"  \
    "cmp %w[n_remain], #2\n"                    \
    "beq 101" n "f\n"                           \
    "ld1 {v" reg_index ".s}[2], [%[x0]], #4\n"  \
    "101" n ":\n" \
    "subs %[x1], %[x1], #1\n"

#define LOAD_C                  \
    "mov %[x1], %x[m_remain]\n" \
    LOAD_LINE("4", "0")         \
    LOAD_LINE("5", "1")         \
    LOAD_LINE("6", "2")         \
    LOAD_LINE("7", "3")         \
    "102:\n"

#define STORE_LINE(reg_index, n)               \
    "cbz %[x1], 105f\n"                        \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                   \
    "blt 103" n "f\n"                          \
    "str q" reg_index ", [%[x0]]\n"            \
    "b 104" n "f\n"                            \
    "103" n ":\n"                              \
    "cmp %w[n_remain], #0\n"                   \
    "beq 104" n "f\n"                          \
    "st1 {v" reg_index ".s}[0], [%[x0]], #4\n" \
    "cmp %w[n_remain], #1\n"                   \
    "beq 104" n "f\n"                          \
    "st1 {v" reg_index ".s}[1], [%[x0]], #4\n" \
    "cmp %w[n_remain], #2\n"                   \
    "beq 104" n "f\n"                          \
    "st1 {v" reg_index ".s}[2], [%[x0]], #4\n" \
    "104" n ":\n"                              \
    "subs %[x1], %[x1], #1\n"

#define STORE_C                 \
    "mov %[x1], %x[m_remain]\n" \
    STORE_LINE("4", "0")        \
    STORE_LINE("5", "1")        \
    STORE_LINE("6", "2")        \
    STORE_LINE("7", "3")        \
    "105:\n"

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "dup v24.16b, %w[zero_point_B] \n"
            "dup v25.16b, %w[zero_point_A] \n"
            "dup v26.4s, %w[zAB] \n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"  //
            LOAD_C      //

            "b 2f\n"

            "1:\n"
            "eor v4.16b,  v4.16b,  v4.16b\n"
            "eor v5.16b,  v5.16b,  v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"

            "2: \n"
            "cbz  %w[oddk], 3f\n"

            // parse the oddk
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "udot v28.4s, %[a0].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot  v4.4s , %[b0].16b, %[a0].4b[0]\n"
            "udot  v5.4s , %[b0].16b, %[a0].4b[1]\n"
            "udot  v6.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v7.4s, %[b0].16b, %[a0].4b[3]\n"

            "cbz %w[k], 4f\n"
            // Loop proper
            "3:\n"
            "ldr  %q[a0], [%[a_ptr]], #16\n"
            "ldr  %q[b0], [%[b_ptr]], #16\n"
            "ldr  %q[a0a], [%[a_ptr]], #16\n"
            "ldr  %q[b0a], [%[b_ptr]], #16\n"
            "udot v28.4s, %[a0].16b, v24.16b\n"
            "udot v28.4s, %[a0a].16b, v24.16b\n"
            "udot v29.4s, %[b0].16b, v25.16b\n"
            "udot v29.4s, %[b0a].16b, v25.16b\n"
            "udot  v4.4s , %[b0].16b, %[a0].4b[0]\n"
            "udot  v5.4s , %[b0].16b, %[a0].4b[1]\n"
            "udot  v6.4s, %[b0].16b, %[a0].4b[2]\n"
            "udot  v7.4s, %[b0].16b, %[a0].4b[3]\n"
            "udot  v4.4s , %[b0a].16b, %[a0a].4b[0]\n"
            "udot  v5.4s , %[b0a].16b, %[a0a].4b[1]\n"
            "udot  v6.4s, %[b0a].16b, %[a0a].4b[2]\n"
            "udot  v7.4s, %[b0a].16b, %[a0a].4b[3]\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n"
            //! minus zAB
            "sub v28.4s, v28.4s, v26.4s\n"

            // clang-format off
            SUB_LANE(4, 28, 0, 29, 23)
            SUB_LANE(5, 28, 1, 29, 23)
            SUB_LANE(6, 28, 2, 29, 23)
            SUB_LANE(7, 28, 3, 29, 23)
            // clang-format on

            STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [n_remain] "+r"(n_remain),
              [m_remain] "+r"(m_remain), [LDC] "+r"(LDC),
              [zero_point_A] "+r"(zero_point_A),
              [zero_point_B] "+r"(zero_point_B), [zAB] "+r"(zAB),
              [outptr0] "+r"(outptr0), [k] "+r"(k), [a0] "=w"(a0),
              [a0a] "=w"(a0a), [b0] "=w"(b0), [b0a] "=w"(b0a),
              [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [x0] "=r"(x0), [x1] "=r"(x1)
            :
            : "v4", "v5", "v6", "v7", "v23", "v24", "v25", "v26", "v28", "v29",
              "memory", "cc");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

#undef SUB_LANE

static void gemm_u8_8x8_transpose_pack_helper(uint8_t* out, const uint8_t* in,
                                              int ldin, int x0, int xmax,
                                              int k0, int kmax) {
    uint8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(uint8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 4) * 4;
    const int ksize8 = ksize4 * 2;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;
    //! 4x4 block output start pos
    uint8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 4) {
        const uint8_t* inptr0 = in + k * ldin + x0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;
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

static void gemm_u8_8x8_interleave_pack_helper(uint8_t* outptr,
                                               const uint8_t* inptr, int ldin,
                                               int y0, int ymax, int k0,
                                               int kmax) {
    uint8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(uint8_t) * 16);

    int y = y0;
    for (; y + 7 < ymax; y += 8) {
        const uint8_t* inptr0 = inptr + y * ldin + k0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;
        const uint8_t* inptr4 = inptr3 + ldin;
        const uint8_t* inptr5 = inptr4 + ldin;
        const uint8_t* inptr6 = inptr5 + ldin;
        const uint8_t* inptr7 = inptr6 + ldin;

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
        const uint8_t* inptr0 = inptr + y * ldin + k0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;

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

}  // namespace matmul_8x8x4
}  // namespace aarch64
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
