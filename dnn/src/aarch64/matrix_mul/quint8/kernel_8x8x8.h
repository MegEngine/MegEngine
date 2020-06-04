/**
 * \file dnn/src/aarch64/matrix_mul/quint8/kernel_8x8x8.h
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
namespace matmul_8x8x8 {

/**
 * Overview of register layout:
 *
 * A 8x8x8 cell of Rhs is stored in 8bit in q26-q27
 * A 8x8x8 cell of Lhs is stored in 8bit in q0-q7
 * A 8x8 block of accumulators is stored in 32bit in q8-q23
 * zero_point_A is stored in 8bit in q24
 * zero_point_B is stored in 8bit in q25.
 *
 *                     +--------+--------+
 *                     |v26[0-8]|v27[0-8]|
 *                Rhs  +--------+--------+
 *    Lhs              |        |        |
 *
 *  +--------+ - - - - +-----------------+
 *  |v0[0-8]|          | v8[0-4]| v9[0-4]|
 *  |v1[0-8]|          |v10[0-4]|v11[0-4]|
 *  |v2[0-8]|          |v12[0-4]|v13[0-4]|
 *  |v3[0-8]|          |v14[0-4]|v15[0-4]|
 *  |v4[0-8]|          |v16[0-4]|v17[0-4]|
 *  |v5[0-8]|          |v18[0-4]|v19[0-4]|
 *  |v6[0-8]|          |v20[0-4]|v21[0-4]|
 *  |v7[0-8]|          |v22[0-4]|v23[0-4]|
 *  +--------+ - - - - +-----------------+
 *
 *                            Accumulator
 */

static void kern_8x8(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, uint8_t za,
                     uint8_t zb) {
    K /= 8;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);

    asm volatile(
            // load accumulator C
            "add x1, %[output], %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"
            "add x4, x3, %x[LDC]\n"
            "add x5, x4, %x[LDC]\n"
            "add x6, x5, %x[LDC]\n"
            "add x7, x6, %x[LDC]\n"
            "dup v24.8b, %w[za]\n"
            "dup v25.8b, %w[zb]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"

            "ldp q8, q9, [%[output]]\n"
            "ldp q10, q11, [x1]\n"
            "ldp q12, q13, [x2]\n"
            "ldp q14, q15, [x3]\n"
            "ldp q16, q17, [x4]\n"
            "ldp q18, q19, [x5]\n"
            "ldp q20, q21, [x6]\n"
            "ldp q22, q23, [x7]\n"
            "b 2f\n"

            "1:\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"
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

            "2: \n"
            "ld1 {v26.8b}, [%[b_ptr]], 8\n"
            "ld1 {v0.8b}, [%[a_ptr]], 8\n"
            "ld1 {v1.8b}, [%[a_ptr]], 8\n"
            "ld1 {v2.8b}, [%[a_ptr]], 8\n"
            "ld1 {v3.8b}, [%[a_ptr]], 8\n"
            "ld1 {v4.8b}, [%[a_ptr]], 8\n"
            "ld1 {v5.8b}, [%[a_ptr]], 8\n"
            "ld1 {v6.8b}, [%[a_ptr]], 8\n"
            "ld1 {v7.8b}, [%[a_ptr]], 8\n"
            "usubl v26.8h, v26.8b, v25.8b\n"
            "usubl v0.8h, v0.8b, v24.8b\n"
            "usubl v1.8h, v1.8b, v24.8b\n"
            "usubl v2.8h, v2.8b, v24.8b\n"
            "usubl v3.8h, v3.8b, v24.8b\n"
            "usubl v4.8h, v4.8b, v24.8b\n"
            "usubl v5.8h, v5.8b, v24.8b\n"
            "usubl v6.8h, v6.8b, v24.8b\n"
            "usubl v7.8h, v7.8b, v24.8b\n"

            "ld1 {v27.8b}, [%[b_ptr]], 8\n"
            "smlal v8.4s, v26.4h, v0.h[0]\n"
            "smlal v10.4s, v26.4h, v1.h[0]\n"
            "smlal v12.4s, v26.4h, v2.h[0]\n"
            "smlal v14.4s, v26.4h, v3.h[0]\n"
            "smlal v16.4s, v26.4h, v4.h[0]\n"
            "smlal v18.4s, v26.4h, v5.h[0]\n"
            "smlal v20.4s, v26.4h, v6.h[0]\n"
            "smlal v22.4s, v26.4h, v7.h[0]\n"
            "usubl v27.8h, v27.8b, v25.8b\n"
            "smlal2 v9.4s, v26.8h, v0.h[0]\n"
            "smlal2 v11.4s, v26.8h, v1.h[0]\n"
            "smlal2 v13.4s, v26.8h, v2.h[0]\n"
            "smlal2 v15.4s, v26.8h, v3.h[0]\n"
            "smlal2 v17.4s, v26.8h, v4.h[0]\n"
            "smlal2 v19.4s, v26.8h, v5.h[0]\n"
            "smlal2 v21.4s, v26.8h, v6.h[0]\n"
            "smlal2 v23.4s, v26.8h, v7.h[0]\n"

            "ld1 {v26.8b}, [%[b_ptr]], 8\n"
            "smlal v8.4s, v27.4h, v0.h[1]\n"
            "smlal v10.4s, v27.4h, v1.h[1]\n"
            "smlal v12.4s, v27.4h, v2.h[1]\n"
            "smlal v14.4s, v27.4h, v3.h[1]\n"
            "smlal v16.4s, v27.4h, v4.h[1]\n"
            "smlal v18.4s, v27.4h, v5.h[1]\n"
            "smlal v20.4s, v27.4h, v6.h[1]\n"
            "smlal v22.4s, v27.4h, v7.h[1]\n"
            "usubl v26.8h, v26.8b, v25.8b\n"
            "smlal2 v9.4s, v27.8h, v0.h[1]\n"
            "smlal2 v11.4s, v27.8h, v1.h[1]\n"
            "smlal2 v13.4s, v27.8h, v2.h[1]\n"
            "smlal2 v15.4s, v27.8h, v3.h[1]\n"
            "smlal2 v17.4s, v27.8h, v4.h[1]\n"
            "smlal2 v19.4s, v27.8h, v5.h[1]\n"
            "smlal2 v21.4s, v27.8h, v6.h[1]\n"
            "smlal2 v23.4s, v27.8h, v7.h[1]\n"

            "ld1 {v27.8b}, [%[b_ptr]], 8\n"
            "smlal v8.4s, v26.4h, v0.h[2]\n"
            "smlal v10.4s, v26.4h, v1.h[2]\n"
            "smlal v12.4s, v26.4h, v2.h[2]\n"
            "smlal v14.4s, v26.4h, v3.h[2]\n"
            "smlal v16.4s, v26.4h, v4.h[2]\n"
            "smlal v18.4s, v26.4h, v5.h[2]\n"
            "smlal v20.4s, v26.4h, v6.h[2]\n"
            "smlal v22.4s, v26.4h, v7.h[2]\n"
            "usubl v27.8h, v27.8b, v25.8b\n"
            "smlal2 v9.4s, v26.8h, v0.h[2]\n"
            "smlal2 v11.4s, v26.8h, v1.h[2]\n"
            "smlal2 v13.4s, v26.8h, v2.h[2]\n"
            "smlal2 v15.4s, v26.8h, v3.h[2]\n"
            "smlal2 v17.4s, v26.8h, v4.h[2]\n"
            "smlal2 v19.4s, v26.8h, v5.h[2]\n"
            "smlal2 v21.4s, v26.8h, v6.h[2]\n"
            "smlal2 v23.4s, v26.8h, v7.h[2]\n"

            "ld1 {v26.8b}, [%[b_ptr]], 8\n"
            "smlal v8.4s, v27.4h, v0.h[3]\n"
            "smlal v10.4s, v27.4h, v1.h[3]\n"
            "smlal v12.4s, v27.4h, v2.h[3]\n"
            "smlal v14.4s, v27.4h, v3.h[3]\n"
            "smlal v16.4s, v27.4h, v4.h[3]\n"
            "smlal v18.4s, v27.4h, v5.h[3]\n"
            "smlal v20.4s, v27.4h, v6.h[3]\n"
            "smlal v22.4s, v27.4h, v7.h[3]\n"
            "usubl v26.8h, v26.8b, v25.8b\n"
            "smlal2 v9.4s, v27.8h, v0.h[3]\n"
            "smlal2 v11.4s, v27.8h, v1.h[3]\n"
            "smlal2 v13.4s, v27.8h, v2.h[3]\n"
            "smlal2 v15.4s, v27.8h, v3.h[3]\n"
            "smlal2 v17.4s, v27.8h, v4.h[3]\n"
            "smlal2 v19.4s, v27.8h, v5.h[3]\n"
            "smlal2 v21.4s, v27.8h, v6.h[3]\n"
            "smlal2 v23.4s, v27.8h, v7.h[3]\n"

            "ld1 {v27.8b}, [%[b_ptr]], 8\n"
            "smlal v8.4s, v26.4h, v0.h[4]\n"
            "smlal v10.4s, v26.4h, v1.h[4]\n"
            "smlal v12.4s, v26.4h, v2.h[4]\n"
            "smlal v14.4s, v26.4h, v3.h[4]\n"
            "smlal v16.4s, v26.4h, v4.h[4]\n"
            "smlal v18.4s, v26.4h, v5.h[4]\n"
            "smlal v20.4s, v26.4h, v6.h[4]\n"
            "smlal v22.4s, v26.4h, v7.h[4]\n"
            "usubl v27.8h, v27.8b, v25.8b\n"
            "smlal2 v9.4s, v26.8h, v0.h[4]\n"
            "smlal2 v11.4s, v26.8h, v1.h[4]\n"
            "smlal2 v13.4s, v26.8h, v2.h[4]\n"
            "smlal2 v15.4s, v26.8h, v3.h[4]\n"
            "smlal2 v17.4s, v26.8h, v4.h[4]\n"
            "smlal2 v19.4s, v26.8h, v5.h[4]\n"
            "smlal2 v21.4s, v26.8h, v6.h[4]\n"
            "smlal2 v23.4s, v26.8h, v7.h[4]\n"

            "ld1 {v26.8b}, [%[b_ptr]], 8\n"
            "smlal v8.4s, v27.4h, v0.h[5]\n"
            "smlal v10.4s, v27.4h, v1.h[5]\n"
            "smlal v12.4s, v27.4h, v2.h[5]\n"
            "smlal v14.4s, v27.4h, v3.h[5]\n"
            "smlal v16.4s, v27.4h, v4.h[5]\n"
            "smlal v18.4s, v27.4h, v5.h[5]\n"
            "smlal v20.4s, v27.4h, v6.h[5]\n"
            "smlal v22.4s, v27.4h, v7.h[5]\n"
            "usubl v26.8h, v26.8b, v25.8b\n"
            "smlal2 v9.4s, v27.8h, v0.h[5]\n"
            "smlal2 v11.4s, v27.8h, v1.h[5]\n"
            "smlal2 v13.4s, v27.8h, v2.h[5]\n"
            "smlal2 v15.4s, v27.8h, v3.h[5]\n"
            "smlal2 v17.4s, v27.8h, v4.h[5]\n"
            "smlal2 v19.4s, v27.8h, v5.h[5]\n"
            "smlal2 v21.4s, v27.8h, v6.h[5]\n"
            "smlal2 v23.4s, v27.8h, v7.h[5]\n"

            "ld1 {v27.8b}, [%[b_ptr]], 8\n"
            "smlal v8.4s, v26.4h, v0.h[6]\n"
            "smlal v10.4s, v26.4h, v1.h[6]\n"
            "smlal v12.4s, v26.4h, v2.h[6]\n"
            "smlal v14.4s, v26.4h, v3.h[6]\n"
            "smlal v16.4s, v26.4h, v4.h[6]\n"
            "smlal v18.4s, v26.4h, v5.h[6]\n"
            "smlal v20.4s, v26.4h, v6.h[6]\n"
            "smlal v22.4s, v26.4h, v7.h[6]\n"
            "usubl v27.8h, v27.8b, v25.8b\n"
            "smlal2 v9.4s, v26.8h, v0.h[6]\n"
            "smlal2 v11.4s, v26.8h, v1.h[6]\n"
            "smlal2 v13.4s, v26.8h, v2.h[6]\n"
            "smlal2 v15.4s, v26.8h, v3.h[6]\n"
            "smlal2 v17.4s, v26.8h, v4.h[6]\n"
            "smlal2 v19.4s, v26.8h, v5.h[6]\n"
            "smlal2 v21.4s, v26.8h, v6.h[6]\n"
            "smlal2 v23.4s, v26.8h, v7.h[6]\n"

            "smlal v8.4s, v27.4h, v0.h[7]\n"
            "smlal v10.4s, v27.4h, v1.h[7]\n"
            "smlal v12.4s, v27.4h, v2.h[7]\n"
            "smlal v14.4s, v27.4h, v3.h[7]\n"
            "smlal v16.4s, v27.4h, v4.h[7]\n"
            "smlal v18.4s, v27.4h, v5.h[7]\n"
            "smlal v20.4s, v27.4h, v6.h[7]\n"
            "smlal v22.4s, v27.4h, v7.h[7]\n"
            "smlal2 v9.4s, v27.8h, v0.h[7]\n"
            "smlal2 v11.4s, v27.8h, v1.h[7]\n"
            "smlal2 v13.4s, v27.8h, v2.h[7]\n"
            "smlal2 v15.4s, v27.8h, v3.h[7]\n"
            "smlal2 v17.4s, v27.8h, v4.h[7]\n"
            "smlal2 v19.4s, v27.8h, v5.h[7]\n"
            "smlal2 v21.4s, v27.8h, v6.h[7]\n"
            "smlal2 v23.4s, v27.8h, v7.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n"
            "stp q8, q9, [%[output]]\n"
            "stp q10, q11, [x1]\n"
            "stp q12, q13, [x2]\n"
            "stp q14, q15, [x3]\n"
            "stp q16, q17, [x4]\n"
            "stp q18, q19, [x5]\n"
            "stp q20, q21, [x6]\n"
            "stp q22, q23, [x7]\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [output] "+r"(output), [za] "+r"(za), [zb] "+r"(zb)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "x1",
              "x2", "x3", "x4", "x5", "x6", "x7", "cc", "memory");
}

/**
 * Overview of register layout:
 *
 * A 8x4x8 cell of Rhs is stored in 8bit in q16-q17
 * A 8x8x8 cell of Lhs is stored in 8bit in q0-q7
 * A 8x4 block of accumulators is stored in 32bit in q8-q15
 * zero_point_A is stored in 8bit in q18
 * zero_point_B is stored in 8bit in q19.
 *
 *                     +--------+
 *                     |v16[0-4]|
 *                Rhs  +--------+
 *                     |v17[0-4]|
 *     Lhs             +--------+
 *
 *  +--------+ - - - - +--------+
 *  |v0[0-8]|          | v8[0-4]|
 *  |v1[0-8]|          | v9[0-4]|
 *  |v2[0-8]|          |v10[0-4]|
 *  |v3[0-8]|          |v11[0-4]|
 *  |v4[0-8]|          |v12[0-4]|
 *  |v5[0-8]|          |v13[0-4]|
 *  |v6[0-8]|          |v14[0-4]|
 *  |v7[0-8]|          |v15[0-4]|
 *  +--------+ - - - - +--------+
 *
 *                            Accumulator
 */

static void kern_8x4(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, size_t n_remain,
                     uint8_t za, uint8_t zb) {
    K /= 8;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    int32_t* outptr4;
    int32_t* outptr5;
    int32_t* outptr6;
    int32_t* outptr7;
    size_t x0 = 0;

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

#define LOAD_C            \
    LOAD_LINE("8", "0")   \
    LOAD_LINE("9", "1")   \
    LOAD_LINE("10", "2")  \
    LOAD_LINE("11", "3")  \
    LOAD_LINE("12", "4")  \
    LOAD_LINE("13", "5")  \
    LOAD_LINE("14", "6")  \
    LOAD_LINE("15", "7")

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

#define STORE_C            \
    STORE_LINE("8", "0")   \
    STORE_LINE("9", "1")   \
    STORE_LINE("10", "2")  \
    STORE_LINE("11", "3")  \
    STORE_LINE("12", "4")  \
    STORE_LINE("13", "5")  \
    STORE_LINE("14", "6")  \
    STORE_LINE("15", "7")

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
            "dup v18.8b, %w[za]\n"
            "dup v19.8b, %w[zb]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v15.16b, v15.16b, v15.16b\n"

            "2: \n"
            "ld1 {v16.s}[0], [%[b_ptr]], 4\n"
            "ld1 {v0.8b}, [%[a_ptr]], 8\n"
            "ld1 {v1.8b}, [%[a_ptr]], 8\n"
            "ld1 {v2.8b}, [%[a_ptr]], 8\n"
            "ld1 {v3.8b}, [%[a_ptr]], 8\n"
            "ld1 {v4.8b}, [%[a_ptr]], 8\n"
            "ld1 {v5.8b}, [%[a_ptr]], 8\n"
            "ld1 {v6.8b}, [%[a_ptr]], 8\n"
            "ld1 {v7.8b}, [%[a_ptr]], 8\n"
            "usubl v16.8h, v16.8b, v19.8b\n"
            "usubl v0.8h, v0.8b, v18.8b\n"
            "usubl v1.8h, v1.8b, v18.8b\n"
            "usubl v2.8h, v2.8b, v18.8b\n"
            "usubl v3.8h, v3.8b, v18.8b\n"
            "usubl v4.8h, v4.8b, v18.8b\n"
            "usubl v5.8h, v5.8b, v18.8b\n"
            "usubl v6.8h, v6.8b, v18.8b\n"
            "usubl v7.8h, v7.8b, v18.8b\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "smlal v8.4s, v16.4h, v0.h[0]\n"
            "smlal v9.4s, v16.4h, v1.h[0]\n"
            "smlal v10.4s, v16.4h, v2.h[0]\n"
            "smlal v11.4s, v16.4h, v3.h[0]\n"
            "usubl v17.8h, v17.8b, v19.8b\n"
            "smlal v12.4s, v16.4h, v4.h[0]\n"
            "smlal v13.4s, v16.4h, v5.h[0]\n"
            "smlal v14.4s, v16.4h, v6.h[0]\n"
            "smlal v15.4s, v16.4h, v7.h[0]\n"

            "ld1 {v16.s}[0], [%[b_ptr]], 4\n"
            "smlal v8.4s, v17.4h, v0.h[1]\n"
            "smlal v9.4s, v17.4h, v1.h[1]\n"
            "smlal v10.4s, v17.4h, v2.h[1]\n"
            "smlal v11.4s, v17.4h, v3.h[1]\n"
            "usubl v16.8h, v16.8b, v19.8b\n"
            "smlal v12.4s, v17.4h, v4.h[1]\n"
            "smlal v13.4s, v17.4h, v5.h[1]\n"
            "smlal v14.4s, v17.4h, v6.h[1]\n"
            "smlal v15.4s, v17.4h, v7.h[1]\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "smlal v8.4s, v16.4h, v0.h[2]\n"
            "smlal v9.4s, v16.4h, v1.h[2]\n"
            "smlal v10.4s, v16.4h, v2.h[2]\n"
            "smlal v11.4s, v16.4h, v3.h[2]\n"
            "usubl v17.8h, v17.8b, v19.8b\n"
            "smlal v12.4s, v16.4h, v4.h[2]\n"
            "smlal v13.4s, v16.4h, v5.h[2]\n"
            "smlal v14.4s, v16.4h, v6.h[2]\n"
            "smlal v15.4s, v16.4h, v7.h[2]\n"

            "ld1 {v16.s}[0], [%[b_ptr]], 4\n"
            "smlal v8.4s, v17.4h, v0.h[3]\n"
            "smlal v9.4s, v17.4h, v1.h[3]\n"
            "smlal v10.4s, v17.4h, v2.h[3]\n"
            "smlal v11.4s, v17.4h, v3.h[3]\n"
            "usubl v16.8h, v16.8b, v19.8b\n"
            "smlal v12.4s, v17.4h, v4.h[3]\n"
            "smlal v13.4s, v17.4h, v5.h[3]\n"
            "smlal v14.4s, v17.4h, v6.h[3]\n"
            "smlal v15.4s, v17.4h, v7.h[3]\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "smlal v8.4s, v16.4h, v0.h[4]\n"
            "smlal v9.4s, v16.4h, v1.h[4]\n"
            "smlal v10.4s, v16.4h, v2.h[4]\n"
            "smlal v11.4s, v16.4h, v3.h[4]\n"
            "usubl v17.8h, v17.8b, v19.8b\n"
            "smlal v12.4s, v16.4h, v4.h[4]\n"
            "smlal v13.4s, v16.4h, v5.h[4]\n"
            "smlal v14.4s, v16.4h, v6.h[4]\n"
            "smlal v15.4s, v16.4h, v7.h[4]\n"

            "ld1 {v16.s}[0], [%[b_ptr]], 4\n"
            "smlal v8.4s, v17.4h, v0.h[5]\n"
            "smlal v9.4s, v17.4h, v1.h[5]\n"
            "smlal v10.4s, v17.4h, v2.h[5]\n"
            "smlal v11.4s, v17.4h, v3.h[5]\n"
            "usubl v16.8h, v16.8b, v19.8b\n"
            "smlal v12.4s, v17.4h, v4.h[5]\n"
            "smlal v13.4s, v17.4h, v5.h[5]\n"
            "smlal v14.4s, v17.4h, v6.h[5]\n"
            "smlal v15.4s, v17.4h, v7.h[5]\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "smlal v8.4s, v16.4h, v0.h[6]\n"
            "smlal v9.4s, v16.4h, v1.h[6]\n"
            "smlal v10.4s, v16.4h, v2.h[6]\n"
            "smlal v11.4s, v16.4h, v3.h[6]\n"
            "usubl v17.8h, v17.8b, v19.8b\n"
            "smlal v12.4s, v16.4h, v4.h[6]\n"
            "smlal v13.4s, v16.4h, v5.h[6]\n"
            "smlal v14.4s, v16.4h, v6.h[6]\n"
            "smlal v15.4s, v16.4h, v7.h[6]\n"

            "smlal v8.4s, v17.4h, v0.h[7]\n"
            "smlal v9.4s, v17.4h, v1.h[7]\n"
            "smlal v10.4s, v17.4h, v2.h[7]\n"
            "smlal v11.4s, v17.4h, v3.h[7]\n"
            "smlal v12.4s, v17.4h, v4.h[7]\n"
            "smlal v13.4s, v17.4h, v5.h[7]\n"
            "smlal v14.4s, v17.4h, v6.h[7]\n"
            "smlal v15.4s, v17.4h, v7.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [za] "+r"(za), [zb] "+r"(zb),
              [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [outptr4] "=r"(outptr4),
              [outptr5] "=r"(outptr5), [outptr6] "=r"(outptr6),
              [outptr7] "=r"(outptr7), [x0] "+r"(x0), [n_remain] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 8x8x8 cell of Rhs is stored in 8bit in q12-q13
 * A 8x8x4 cell of Lhs is stored in 8bit in q0-q3
 * A 4x8 block of accumulators is stored in 32bit in q4-q11
 * zero_point_A is stored in 8bit in q14
 * zero_point_B is stored in 8bit in q15.
 *
 *                     +--------+--------+
 *                     |v12[0-8]|v13[0-8]|
 *                Rhs  +--------+--------+
 *    Lhs              |        |        |
 *
 *  +--------+ - - - - +-----------------+
 *  |v0[0-8]|          | v4[0-4]| v5[0-4]|
 *  |v1[0-8]|          | v6[0-4]| v7[0-4]|
 *  |v2[0-8]|          | v8[0-4]| v9[0-4]|
 *  |v3[0-8]|          |v10[0-4]|v11[0-4]|
 *  +--------+ - - - - +-----------------+
 *
 *                            Accumulator
 */

static void kern_4x8(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, size_t m_remain,
                     uint8_t za, uint8_t zb) {
    K /= 8;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    size_t x0 = 0;

// clang-format off
#define LOAD_LINE(v1, v2, m)                \
    "cbz %[x0], 100f\n"                     \
    "ldp " v1 "," v2 ", [%[outptr" m "]]\n" \
    "subs %[x0], %[x0], #1\n"

#define LOAD_C                      \
    "mov %[x0], %x[m_remain]\n"     \
    LOAD_LINE("q4", "q5", "0")      \
    LOAD_LINE("q6", "q7", "1")      \
    LOAD_LINE("q8", "q9", "2")      \
    LOAD_LINE("q10", "q11", "3")    \
    "100:\n"

#define STORE_LINE(v1, v2, m)              \
    "cbz %[x0], 101f\n"                    \
    "stp " v1 "," v2", [%[outptr" m "]]\n" \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                     \
    "mov %[x0], %x[m_remain]\n"     \
    STORE_LINE("q4", "q5", "0")     \
    STORE_LINE("q6", "q7", "1")     \
    STORE_LINE("q8", "q9", "2")     \
    STORE_LINE("q10", "q11", "3")   \
    "101:\n"

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "dup v14.8b, %w[za]\n"
            "dup v15.8b, %w[zb]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v4.16b, v4.16b, v4.16b\n"
            "eor v5.16b, v5.16b, v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"

            "2: \n"
            "ld1 {v12.8b}, [%[b_ptr]], 8\n"
            "ld1 {v0.8b}, [%[a_ptr]], 8\n"
            "ld1 {v1.8b}, [%[a_ptr]], 8\n"
            "ld1 {v2.8b}, [%[a_ptr]], 8\n"
            "ld1 {v3.8b}, [%[a_ptr]], 8\n"
            "usubl v12.8h, v12.8b, v15.8b\n"
            "usubl v0.8h, v0.8b, v14.8b\n"
            "usubl v1.8h, v1.8b, v14.8b\n"
            "usubl v2.8h, v2.8b, v14.8b\n"
            "usubl v3.8h, v3.8b, v14.8b\n"

            "ld1 {v13.8b}, [%[b_ptr]], 8\n"
            "smlal v4.4s, v12.4h, v0.h[0]\n"
            "smlal v6.4s, v12.4h, v1.h[0]\n"
            "smlal v8.4s, v12.4h, v2.h[0]\n"
            "smlal v10.4s, v12.4h, v3.h[0]\n"
            "usubl v13.8h, v13.8b, v15.8b\n"
            "smlal2 v5.4s, v12.8h, v0.h[0]\n"
            "smlal2 v7.4s, v12.8h, v1.h[0]\n"
            "smlal2 v9.4s, v12.8h, v2.h[0]\n"
            "smlal2 v11.4s, v12.8h, v3.h[0]\n"

            "ld1 {v12.8b}, [%[b_ptr]], 8\n"
            "smlal v4.4s, v13.4h, v0.h[1]\n"
            "smlal v6.4s, v13.4h, v1.h[1]\n"
            "smlal v8.4s, v13.4h, v2.h[1]\n"
            "smlal v10.4s, v13.4h, v3.h[1]\n"
            "usubl v12.8h, v12.8b, v15.8b\n"
            "smlal2 v5.4s, v13.8h, v0.h[1]\n"
            "smlal2 v7.4s, v13.8h, v1.h[1]\n"
            "smlal2 v9.4s, v13.8h, v2.h[1]\n"
            "smlal2 v11.4s, v13.8h, v3.h[1]\n"

            "ld1 {v13.8b}, [%[b_ptr]], 8\n"
            "smlal v4.4s, v12.4h, v0.h[2]\n"
            "smlal v6.4s, v12.4h, v1.h[2]\n"
            "smlal v8.4s, v12.4h, v2.h[2]\n"
            "smlal v10.4s, v12.4h, v3.h[2]\n"
            "usubl v13.8h, v13.8b, v15.8b\n"
            "smlal2 v5.4s, v12.8h, v0.h[2]\n"
            "smlal2 v7.4s, v12.8h, v1.h[2]\n"
            "smlal2 v9.4s, v12.8h, v2.h[2]\n"
            "smlal2 v11.4s, v12.8h, v3.h[2]\n"

            "ld1 {v12.8b}, [%[b_ptr]], 8\n"
            "smlal v4.4s, v13.4h, v0.h[3]\n"
            "smlal v6.4s, v13.4h, v1.h[3]\n"
            "smlal v8.4s, v13.4h, v2.h[3]\n"
            "smlal v10.4s, v13.4h, v3.h[3]\n"
            "usubl v12.8h, v12.8b, v15.8b\n"
            "smlal2 v5.4s, v13.8h, v0.h[3]\n"
            "smlal2 v7.4s, v13.8h, v1.h[3]\n"
            "smlal2 v9.4s, v13.8h, v2.h[3]\n"
            "smlal2 v11.4s, v13.8h, v3.h[3]\n"

            "ld1 {v13.8b}, [%[b_ptr]], 8\n"
            "smlal v4.4s, v12.4h, v0.h[4]\n"
            "smlal v6.4s, v12.4h, v1.h[4]\n"
            "smlal v8.4s, v12.4h, v2.h[4]\n"
            "smlal v10.4s, v12.4h, v3.h[4]\n"
            "usubl v13.8h, v13.8b, v15.8b\n"
            "smlal2 v5.4s, v12.8h, v0.h[4]\n"
            "smlal2 v7.4s, v12.8h, v1.h[4]\n"
            "smlal2 v9.4s, v12.8h, v2.h[4]\n"
            "smlal2 v11.4s, v12.8h, v3.h[4]\n"

            "ld1 {v12.8b}, [%[b_ptr]], 8\n"
            "smlal v4.4s, v13.4h, v0.h[5]\n"
            "smlal v6.4s, v13.4h, v1.h[5]\n"
            "smlal v8.4s, v13.4h, v2.h[5]\n"
            "smlal v10.4s, v13.4h, v3.h[5]\n"
            "usubl v12.8h, v12.8b, v15.8b\n"
            "smlal2 v5.4s, v13.8h, v0.h[5]\n"
            "smlal2 v7.4s, v13.8h, v1.h[5]\n"
            "smlal2 v9.4s, v13.8h, v2.h[5]\n"
            "smlal2 v11.4s, v13.8h, v3.h[5]\n"

            "ld1 {v13.8b}, [%[b_ptr]], 8\n"
            "smlal v4.4s, v12.4h, v0.h[6]\n"
            "smlal v6.4s, v12.4h, v1.h[6]\n"
            "smlal v8.4s, v12.4h, v2.h[6]\n"
            "smlal v10.4s, v12.4h, v3.h[6]\n"
            "usubl v13.8h, v13.8b, v15.8b\n"
            "smlal2 v5.4s, v12.8h, v0.h[6]\n"
            "smlal2 v7.4s, v12.8h, v1.h[6]\n"
            "smlal2 v9.4s, v12.8h, v2.h[6]\n"
            "smlal2 v11.4s, v12.8h, v3.h[6]\n"

            "smlal v4.4s, v13.4h, v0.h[7]\n"
            "smlal v6.4s, v13.4h, v1.h[7]\n"
            "smlal v8.4s, v13.4h, v2.h[7]\n"
            "smlal v10.4s, v13.4h, v3.h[7]\n"
            "smlal2 v5.4s, v13.8h, v0.h[7]\n"
            "smlal2 v7.4s, v13.8h, v1.h[7]\n"
            "smlal2 v9.4s, v13.8h, v2.h[7]\n"
            "smlal2 v11.4s, v13.8h, v3.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [za] "+r"(za), [zb] "+r"(zb),
              [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [x0] "+r"(x0), [m_remain] "+r"(m_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 8x4x8 cell of Rhs is stored in 8bit in q8-q9
 * A 8x8x4 cell of Lhs is stored in 8bit in q0-q3
 * A 4x4 block of accumulators is stored in 32bit in q4-q7
 * zero_point_A is stored in 8bit in q10
 * zero_point_B is stored in 8bit in q11.
 *
 *                     +--------+
 *                     | v8[0-4]|
 *                Rhs  +--------+
 *                     | v9[0-4]|
 *     Lhs             +--------+
 *
 *  +--------+ - - - - +--------+
 *  |v0[0-8]|          | v4[0-4]|
 *  |v1[0-8]|          | v5[0-4]|
 *  |v2[0-8]|          | v6[0-4]|
 *  |v3[0-8]|          | v7[0-4]|
 *  +--------+ - - - - +--------+
 *
 *                            Accumulator
 */

static void kern_4x4(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, size_t m_remain,
                     size_t n_remain, uint8_t za, uint8_t zb) {
    K /= 8;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    size_t x0 = 0;
    size_t x1 = 0;

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
            "dup v10.8b, %w[za]\n"
            "dup v11.8b, %w[zb]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "eor v4.16b, v4.16b, v4.16b\n"
            "eor v5.16b, v5.16b, v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"

            "2: \n"
            "ld1 {v8.s}[0], [%[b_ptr]], 4\n"
            "ld1 {v0.8b}, [%[a_ptr]], 8\n"
            "ld1 {v1.8b}, [%[a_ptr]], 8\n"
            "ld1 {v2.8b}, [%[a_ptr]], 8\n"
            "ld1 {v3.8b}, [%[a_ptr]], 8\n"
            "usubl v8.8h, v8.8b, v11.8b\n"
            "usubl v0.8h, v0.8b, v10.8b\n"
            "usubl v1.8h, v1.8b, v10.8b\n"
            "usubl v2.8h, v2.8b, v10.8b\n"
            "usubl v3.8h, v3.8b, v10.8b\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "smlal v4.4s, v8.4h, v0.h[0]\n"
            "smlal v5.4s, v8.4h, v1.h[0]\n"
            "usubl v9.8h, v9.8b, v11.8b\n"
            "smlal v6.4s, v8.4h, v2.h[0]\n"
            "smlal v7.4s, v8.4h, v3.h[0]\n"

            "ld1 {v8.s}[0], [%[b_ptr]], 4\n"
            "smlal v4.4s, v9.4h, v0.h[1]\n"
            "smlal v5.4s, v9.4h, v1.h[1]\n"
            "usubl v8.8h, v8.8b, v11.8b\n"
            "smlal v6.4s, v9.4h, v2.h[1]\n"
            "smlal v7.4s, v9.4h, v3.h[1]\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "smlal v4.4s, v8.4h, v0.h[2]\n"
            "smlal v5.4s, v8.4h, v1.h[2]\n"
            "usubl v9.8h, v9.8b, v11.8b\n"
            "smlal v6.4s, v8.4h, v2.h[2]\n"
            "smlal v7.4s, v8.4h, v3.h[2]\n"

            "ld1 {v8.s}[0], [%[b_ptr]], 4\n"
            "smlal v4.4s, v9.4h, v0.h[3]\n"
            "smlal v5.4s, v9.4h, v1.h[3]\n"
            "usubl v8.8h, v8.8b, v11.8b\n"
            "smlal v6.4s, v9.4h, v2.h[3]\n"
            "smlal v7.4s, v9.4h, v3.h[3]\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "smlal v4.4s, v8.4h, v0.h[4]\n"
            "smlal v5.4s, v8.4h, v1.h[4]\n"
            "usubl v9.8h, v9.8b, v11.8b\n"
            "smlal v6.4s, v8.4h, v2.h[4]\n"
            "smlal v7.4s, v8.4h, v3.h[4]\n"

            "ld1 {v8.s}[0], [%[b_ptr]], 4\n"
            "smlal v4.4s, v9.4h, v0.h[5]\n"
            "smlal v5.4s, v9.4h, v1.h[5]\n"
            "usubl v8.8h, v8.8b, v11.8b\n"
            "smlal v6.4s, v9.4h, v2.h[5]\n"
            "smlal v7.4s, v9.4h, v3.h[5]\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "smlal v4.4s, v8.4h, v0.h[6]\n"
            "smlal v5.4s, v8.4h, v1.h[6]\n"
            "usubl v9.8h, v9.8b, v11.8b\n"
            "smlal v6.4s, v8.4h, v2.h[6]\n"
            "smlal v7.4s, v8.4h, v3.h[6]\n"

            "smlal v4.4s, v9.4h, v0.h[7]\n"
            "smlal v5.4s, v9.4h, v1.h[7]\n"
            "smlal v6.4s, v9.4h, v2.h[7]\n"
            "smlal v7.4s, v9.4h, v3.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [za] "+r"(za), [zb] "+r"(zb),
              [outptr1] "=r"(outptr1), [outptr2] "=r"(outptr2),
              [outptr3] "=r"(outptr3), [x0] "+r"(x0), [x1] "+r"(x1),
              [m_remain] "+r"(m_remain), [n_remain] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_u8_8x8_pack_A_n(dt_uint8* outptr, const dt_uint8* inptr,
                                 int ldin, int y0, int ymax, int k0, int kmax,
                                 uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);

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
        for (; K > 15; K -= 16) {
            interleave_8x8_2_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr);
        }

        if (K > 0) {
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr, 8, K, zero_point);
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

            interleave_4x8_2_b(inptr0, inptr1, inptr2, inptr3, outptr);
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 8, K,
                         zero_point);
        }
    }
}

static void gemm_u8_8x8_transpose_pack_A_n(dt_uint8* out, const dt_uint8* in,
                                           int ldin, int x0, int xmax, int k0,
                                           int kmax, uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 8) * 4;
    const int ksize8 = ksize4 * 2;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;
    //! 4x4 block output start pos
    uint8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 8) {
        const uint8_t* inptr0 = in + k * ldin + x0;
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

        int x = x0;
        outptr = outptr_base;

        for (; x + 7 < xmax; x += 8) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
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
            transpose_8x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                              inptr6, inptr7, outptr);
            outptr += ksize8;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
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

            transpose_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                        inptr7, outptr, 4, 4, zero_point);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
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

            transpose_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                        inptr7, outptr, 4, xmax - x, zero_point);
        }

        outptr_base += 8 * 8;
        outptr_base4 += 4 * 8;
    }
}

static void gemm_u8_8x8_pack_B_n(dt_uint8* out, const dt_uint8* in, int ldin,
                                 int x0, int xmax, int k0, int kmax,
                                 uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 8) * 4;
    const int ksize8 = ksize4 * 2;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;
    uint8_t* outptr_interleave = nullptr;
    //! 4x4 block output start pos
    uint8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 8) {
        const uint8_t* inptr0 = in + k * ldin + x0;
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

        int x = x0;
        outptr = outptr_base;

        for (; x + 7 < xmax; x += 8) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
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
            outptr_interleave = outptr;
            interleave_8x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr_interleave);
            outptr += ksize8;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
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

            outptr_interleave = outptr;
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr_interleave, 4, 4, zero_point);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
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

            outptr_interleave = outptr;
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr_interleave, 4, xmax - x, zero_point);
        }

        outptr_base += 8 * 8;
        outptr_base4 += 4 * 8;
    }
}

static void gemm_u8_8x8_transpose_pack_B_n(dt_uint8* outptr,
                                           const dt_uint8* inptr, int ldin,
                                           int y0, int ymax, int k0, int kmax,
                                           uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);
    constexpr int interleave4 = 32;
    constexpr int interleave8 = 64;

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
        for (; K > 7; K -= 8) {
            transpose_8x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                              inptr6, inptr7, outptr);
            outptr += interleave8;
        }

        if (K > 0) {
            transpose_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                        inptr7, outptr, 8, K, zero_point);
            outptr += interleave8;
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
        for (; K > 7; K -= 8) {
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

            transpose_8x4_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += interleave4;
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
            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 8, K,
                        zero_point);
            outptr += interleave4;
        }
    }
}

}  // namespace matmul_8x8x8
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
#endif
