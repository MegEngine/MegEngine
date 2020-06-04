/**
 * \file dnn/src/aarch64/matrix_mul/int8x8x16/kernel_8x8x8.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <inttypes.h>
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_8x8x8 {

/**
 * Overview of register layout:
 *
 * A 8x8x8 cell of Rhs is stored in 8bit in v16
 * A 8x8x8 cell of Lhs is stored in 8bit in v0-v7
 * A 8x8 block of accumulators is stored in 16bit in v8-v15
 *
 *                     +---------+
 *                     |v16[0-8] |
 *                Rhs  +---------+
 *    Lhs              |         |
 *
 *  +--------+ - - - - +---------+
 *  |v0[0-8]|          | v8[0-8] |
 *  |v1[0-8]|          | v9[0-8] |
 *  |v2[0-8]|          | v10[0-8]|
 *  |v3[0-8]|          | v11[0-8]|
 *  |v4[0-8]|          | v12[0-8]|
 *  |v5[0-8]|          | v13[0-8]|
 *  |v6[0-8]|          | v14[0-8]|
 *  |v7[0-8]|          | v15[0-8]|
 *  +--------+ - - - - +---------+
 *
 *                            Accumulator
 */
static void kern_8x8(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k) {
    K /= 8;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);

// clang-format off
#define LOAD_LINE(reg_index, n)  \
    "ld1 {v" reg_index ".8h}, [x" n "]\n"
#define LOAD_C           \
    LOAD_LINE("8", "0")  \
    LOAD_LINE("9", "1")  \
    LOAD_LINE("10", "2") \
    LOAD_LINE("11", "3") \
    LOAD_LINE("12", "4") \
    LOAD_LINE("13", "5") \
    LOAD_LINE("14", "6") \
    LOAD_LINE("15", "7")

#define STORE_LINE(reg_index, num)  \
    "st1 {v" reg_index ".8h}, [x" num "]\n"

#define STORE_C           \
    STORE_LINE("8", "0")  \
    STORE_LINE("9", "1")  \
    STORE_LINE("10", "2") \
    STORE_LINE("11", "3") \
    STORE_LINE("12", "4") \
    STORE_LINE("13", "5") \
    STORE_LINE("14", "6") \
    STORE_LINE("15", "7") 

#define CLEAR_8_INT16(reg) \
    "eor v" reg ".16b, v" reg ".16b, v" reg ".16b\n"
#define CLEAR_8_REGS    \
    CLEAR_8_INT16("8")  \
    CLEAR_8_INT16("9")  \
    CLEAR_8_INT16("10") \
    CLEAR_8_INT16("11") \
    CLEAR_8_INT16("12") \
    CLEAR_8_INT16("13") \
    CLEAR_8_INT16("14") \
    CLEAR_8_INT16("15")

    // clang-format on

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            // load accumulator C
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"
            "add x4, x3, %x[LDC]\n"
            "add x5, x4, %x[LDC]\n"
            "add x6, x5, %x[LDC]\n"
            "add x7, x6, %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C
            "b 2f\n"

            "1:\n" CLEAR_8_REGS

            "2: \n"
            "ld1 {v16.8b}, [%[b_ptr]], 8\n"
            "ld1 {v0.8b}, [%[a_ptr]], 8\n"
            "ld1 {v1.8b}, [%[a_ptr]], 8\n"
            "ld1 {v2.8b}, [%[a_ptr]], 8\n"
            "ld1 {v3.8b}, [%[a_ptr]], 8\n"
            "ld1 {v4.8b}, [%[a_ptr]], 8\n"
            "ld1 {v5.8b}, [%[a_ptr]], 8\n"
            "ld1 {v6.8b}, [%[a_ptr]], 8\n"
            "ld1 {v7.8b}, [%[a_ptr]], 8\n"

            "sshll v16.8h, v16.8b, #0\n"
            "sshll v0.8h, v0.8b, #0\n"
            "sshll v1.8h, v1.8b, #0\n"
            "sshll v2.8h, v2.8b, #0\n"
            "sshll v3.8h, v3.8b, #0\n"
            "sshll v4.8h, v4.8b, #0\n"
            "sshll v5.8h, v5.8b, #0\n"
            "sshll v6.8h, v6.8b, #0\n"
            "sshll v7.8h, v7.8b, #0\n"

            "ld1 {v17.8b}, [%[b_ptr]], 8\n"
            "mla v8.8h, v16.8h,  v0.h[0]\n"
            "mla v9.8h, v16.8h,  v1.h[0]\n"
            "mla v10.8h, v16.8h, v2.h[0]\n"
            "mla v11.8h, v16.8h, v3.h[0]\n"
            "mla v12.8h, v16.8h, v4.h[0]\n"
            "mla v13.8h, v16.8h, v5.h[0]\n"
            "mla v14.8h, v16.8h, v6.h[0]\n"
            "mla v15.8h, v16.8h, v7.h[0]\n"
            "sshll v17.8h, v17.8b, #0\n"

            "ld1 {v16.8b}, [%[b_ptr]], 8\n"
            "mla v8.8h,  v17.8h, v0.h[1]\n"
            "mla v9.8h,  v17.8h, v1.h[1]\n"
            "mla v10.8h, v17.8h, v2.h[1]\n"
            "mla v11.8h, v17.8h, v3.h[1]\n"
            "mla v12.8h, v17.8h, v4.h[1]\n"
            "mla v13.8h, v17.8h, v5.h[1]\n"
            "mla v14.8h, v17.8h, v6.h[1]\n"
            "mla v15.8h, v17.8h, v7.h[1]\n"
            "sshll v16.8h, v16.8b, #0\n"

            "ld1 {v17.8b}, [%[b_ptr]], 8\n"
            "mla v8.8h,  v16.8h, v0.h[2]\n"
            "mla v9.8h,  v16.8h, v1.h[2]\n"
            "mla v10.8h, v16.8h, v2.h[2]\n"
            "mla v11.8h, v16.8h, v3.h[2]\n"
            "mla v12.8h, v16.8h, v4.h[2]\n"
            "mla v13.8h, v16.8h, v5.h[2]\n"
            "mla v14.8h, v16.8h, v6.h[2]\n"
            "mla v15.8h, v16.8h, v7.h[2]\n"
            "sshll v17.8h, v17.8b, #0\n"

            "ld1 {v16.8b}, [%[b_ptr]], 8\n"
            "mla v8.8h,  v17.8h, v0.h[3]\n"
            "mla v9.8h,  v17.8h, v1.h[3]\n"
            "mla v10.8h, v17.8h, v2.h[3]\n"
            "mla v11.8h, v17.8h, v3.h[3]\n"
            "mla v12.8h, v17.8h, v4.h[3]\n"
            "mla v13.8h, v17.8h, v5.h[3]\n"
            "mla v14.8h, v17.8h, v6.h[3]\n"
            "mla v15.8h, v17.8h, v7.h[3]\n"
            "sshll v16.8h, v16.8b, #0\n"

            "ld1 {v17.8b}, [%[b_ptr]], 8\n"
            "mla v8.8h,  v16.8h, v0.h[4]\n"
            "mla v9.8h,  v16.8h, v1.h[4]\n"
            "mla v10.8h, v16.8h, v2.h[4]\n"
            "mla v11.8h, v16.8h, v3.h[4]\n"
            "mla v12.8h, v16.8h, v4.h[4]\n"
            "mla v13.8h, v16.8h, v5.h[4]\n"
            "mla v14.8h, v16.8h, v6.h[4]\n"
            "mla v15.8h, v16.8h, v7.h[4]\n"
            "sshll v17.8h, v17.8b, #0\n"

            "ld1 {v16.8b}, [%[b_ptr]], 8\n"
            "mla v8.8h,  v17.8h, v0.h[5]\n"
            "mla v9.8h,  v17.8h, v1.h[5]\n"
            "mla v10.8h, v17.8h, v2.h[5]\n"
            "mla v11.8h, v17.8h, v3.h[5]\n"
            "mla v12.8h, v17.8h, v4.h[5]\n"
            "mla v13.8h, v17.8h, v5.h[5]\n"
            "mla v14.8h, v17.8h, v6.h[5]\n"
            "mla v15.8h, v17.8h, v7.h[5]\n"
            "sshll v16.8h, v16.8b, #0\n"

            "ld1 {v17.8b}, [%[b_ptr]], 8\n"
            "mla v8.8h,  v16.8h, v0.h[6]\n"
            "mla v9.8h,  v16.8h, v1.h[6]\n"
            "mla v10.8h, v16.8h, v2.h[6]\n"
            "mla v11.8h, v16.8h, v3.h[6]\n"
            "mla v12.8h, v16.8h, v4.h[6]\n"
            "mla v13.8h, v16.8h, v5.h[6]\n"
            "mla v14.8h, v16.8h, v6.h[6]\n"
            "mla v15.8h, v16.8h, v7.h[6]\n"
            "sshll v17.8h, v17.8b, #0\n"

            "mla v8.8h,  v17.8h, v0.h[7]\n"
            "mla v9.8h,  v17.8h, v1.h[7]\n"
            "mla v10.8h, v17.8h, v2.h[7]\n"
            "mla v11.8h, v17.8h, v3.h[7]\n"
            "mla v12.8h, v17.8h, v4.h[7]\n"
            "mla v13.8h, v17.8h, v5.h[7]\n"
            "mla v14.8h, v17.8h, v6.h[7]\n"
            "mla v15.8h, v17.8h, v7.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "bne 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "x1", "x2", "x3",
              "x4", "x5", "x6", "x7", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
#undef CLEAR_8_INT16
#undef CLEAR_8_REGS
}

/**
 * Overview of register layout:
 *
 * A 8x4x8 cell of Rhs is stored in 8bit in v16-v17
 * A 8x8x8 cell of Lhs is stored in 8bit in v0-v7
 * A 8x4 block of accumulators is stored in 16bit in v8-v15
 *
 *                     +--------+
 *                     |v16[0-4]|
 *                Rhs  +--------+
 *                     |        |
 *     Lhs
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

static void kern_8x4(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k,
                     size_t n_remain) {
    K /= 8;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);
    int16_t* outptr0 = output;
    int16_t* outptr1;
    int16_t* outptr2;
    int16_t* outptr3;
    int16_t* outptr4;
    int16_t* outptr5;
    int16_t* outptr6;
    int16_t* outptr7;
    size_t x0 = 0;

// clang-format off
#define LOAD_LINE(reg_index, n)                \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                   \
    "blt 100" n "f\n"                          \
    "ld1 {v" reg_index ".4h}, [%[x0]]\n"       \
    "b 101" n "f\n"                            \
    "100" n ":\n"                              \
    "cmp %w[n_remain], #0\n"                   \
    "beq 101" n "f\n"                          \
    "ld1 {v" reg_index ".h}[0], [%[x0]], #2\n" \
    "cmp %w[n_remain], #1\n"                   \
    "beq 101" n "f\n"                          \
    "ld1 {v" reg_index ".h}[1], [%[x0]], #2\n" \
    "cmp %w[n_remain], #2\n"                   \
    "beq 101" n "f\n"                          \
    "ld1 {v" reg_index ".h}[2], [%[x0]], #2\n" \
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
    "st1 {v" reg_index ".4h}, [%[x0]]\n"       \
    "b 103" n "f\n"                            \
    "102" n ":\n"                              \
    "cmp %w[n_remain], #0\n"                   \
    "beq 103" n "f\n"                          \
    "st1 {v" reg_index ".h}[0], [%[x0]], #2\n" \
    "cmp %w[n_remain], #1\n"                   \
    "beq 103" n "f\n"                          \
    "st1 {v" reg_index ".h}[1], [%[x0]], #2\n" \
    "cmp %w[n_remain], #2\n"                   \
    "beq 103" n "f\n"                          \
    "st1 {v" reg_index ".h}[2], [%[x0]], #2\n" \
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

#define CLEAR_8_INT16(reg) \
    "eor v" reg ".16b, v" reg ".16b, v" reg ".16b\n"
#define CLEAR_8_REGS    \
    CLEAR_8_INT16("8")  \
    CLEAR_8_INT16("9")  \
    CLEAR_8_INT16("10") \
    CLEAR_8_INT16("11") \
    CLEAR_8_INT16("12") \
    CLEAR_8_INT16("13") \
    CLEAR_8_INT16("14") \
    CLEAR_8_INT16("15")
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

            "1:\n" CLEAR_8_REGS

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
            "sshll v16.8h, v16.8b, #0\n"
            "sshll v0.8h, v0.8b, #0\n"
            "sshll v1.8h, v1.8b, #0\n"
            "sshll v2.8h, v2.8b, #0\n"
            "sshll v3.8h, v3.8b, #0\n"
            "sshll v4.8h, v4.8b, #0\n"
            "sshll v5.8h, v5.8b, #0\n"
            "sshll v6.8h, v6.8b, #0\n"
            "sshll v7.8h, v7.8b, #0\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "mla  v8.4h, v16.4h, v0.h[0]\n"
            "mla  v9.4h, v16.4h, v1.h[0]\n"
            "mla v10.4h, v16.4h, v2.h[0]\n"
            "mla v11.4h, v16.4h, v3.h[0]\n"
            "sshll v17.8h, v17.8b, #0\n"
            "mla v12.4h, v16.4h, v4.h[0]\n"
            "mla v13.4h, v16.4h, v5.h[0]\n"
            "mla v14.4h, v16.4h, v6.h[0]\n"
            "mla v15.4h, v16.4h, v7.h[0]\n"

            "ld1 {v16.s}[0], [%[b_ptr]], 4\n"
            "mla  v8.4h, v17.4h, v0.h[1]\n"
            "mla  v9.4h, v17.4h, v1.h[1]\n"
            "mla v10.4h, v17.4h, v2.h[1]\n"
            "mla v11.4h, v17.4h, v3.h[1]\n"
            "sshll v16.8h, v16.8b, #0\n"
            "mla v12.4h, v17.4h, v4.h[1]\n"
            "mla v13.4h, v17.4h, v5.h[1]\n"
            "mla v14.4h, v17.4h, v6.h[1]\n"
            "mla v15.4h, v17.4h, v7.h[1]\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "mla  v8.4h, v16.4h, v0.h[2]\n"
            "mla  v9.4h, v16.4h, v1.h[2]\n"
            "mla v10.4h, v16.4h, v2.h[2]\n"
            "mla v11.4h, v16.4h, v3.h[2]\n"
            "sshll v17.8h, v17.8b, #0\n"
            "mla v12.4h, v16.4h, v4.h[2]\n"
            "mla v13.4h, v16.4h, v5.h[2]\n"
            "mla v14.4h, v16.4h, v6.h[2]\n"
            "mla v15.4h, v16.4h, v7.h[2]\n"

            "ld1 {v16.s}[0], [%[b_ptr]], 4\n"
            "mla  v8.4h, v17.4h, v0.h[3]\n"
            "mla  v9.4h, v17.4h, v1.h[3]\n"
            "mla v10.4h, v17.4h, v2.h[3]\n"
            "mla v11.4h, v17.4h, v3.h[3]\n"
            "sshll v16.8h, v16.8b, #0\n"
            "mla v12.4h, v17.4h, v4.h[3]\n"
            "mla v13.4h, v17.4h, v5.h[3]\n"
            "mla v14.4h, v17.4h, v6.h[3]\n"
            "mla v15.4h, v17.4h, v7.h[3]\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "mla  v8.4h, v16.4h, v0.h[4]\n"
            "mla  v9.4h, v16.4h, v1.h[4]\n"
            "mla v10.4h, v16.4h, v2.h[4]\n"
            "mla v11.4h, v16.4h, v3.h[4]\n"
            "sshll v17.8h, v17.8b, #0\n"
            "mla v12.4h, v16.4h, v4.h[4]\n"
            "mla v13.4h, v16.4h, v5.h[4]\n"
            "mla v14.4h, v16.4h, v6.h[4]\n"
            "mla v15.4h, v16.4h, v7.h[4]\n"

            "ld1 {v16.s}[0], [%[b_ptr]], 4\n"
            "mla  v8.4h, v17.4h, v0.h[5]\n"
            "mla  v9.4h, v17.4h, v1.h[5]\n"
            "mla v10.4h, v17.4h, v2.h[5]\n"
            "mla v11.4h, v17.4h, v3.h[5]\n"
            "sshll v16.8h, v16.8b, #0\n"
            "mla v12.4h, v17.4h, v4.h[5]\n"
            "mla v13.4h, v17.4h, v5.h[5]\n"
            "mla v14.4h, v17.4h, v6.h[5]\n"
            "mla v15.4h, v17.4h, v7.h[5]\n"

            "ld1 {v17.s}[0], [%[b_ptr]], 4\n"
            "mla  v8.4h, v16.4h, v0.h[6]\n"
            "mla  v9.4h, v16.4h, v1.h[6]\n"
            "mla v10.4h, v16.4h, v2.h[6]\n"
            "mla v11.4h, v16.4h, v3.h[6]\n"
            "sshll v17.8h, v17.8b, #0\n"
            "mla v12.4h, v16.4h, v4.h[6]\n"
            "mla v13.4h, v16.4h, v5.h[6]\n"
            "mla v14.4h, v16.4h, v6.h[6]\n"
            "mla v15.4h, v16.4h, v7.h[6]\n"

            "mla  v8.4h, v17.4h, v0.h[7]\n"
            "mla  v9.4h, v17.4h, v1.h[7]\n"
            "mla v10.4h, v17.4h, v2.h[7]\n"
            "mla v11.4h, v17.4h, v3.h[7]\n"
            "mla v12.4h, v17.4h, v4.h[7]\n"
            "mla v13.4h, v17.4h, v5.h[7]\n"
            "mla v14.4h, v17.4h, v6.h[7]\n"
            "mla v15.4h, v17.4h, v7.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3),
              [outptr4] "=r"(outptr4), [outptr5] "=r"(outptr5),
              [outptr6] "=r"(outptr6), [outptr7] "=r"(outptr7), [x0] "+r"(x0),
              [n_remain] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
#undef CLEAR_8_INT16
#undef CLEAR_8_REGS
}

/**
 * Overview of register layout:
 *
 * A 8x8x8 cell of Rhs is stored in 8bit in v8-v9
 * A 8x8x4 cell of Lhs is stored in 8bit in v0-v3
 * A 4x8 block of accumulators is stored in 16bit in v4-v7
 *
 *                     +--------+
 *                     | v8[0-8]|
 *                     +--------+
 *                     | v9[0-8]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +-------+ - - - - -+--------+
 *  |v0[0-8]|          | v4[0-8]|
 *  |v1[0-8]|          | v5[0-8]|
 *  |v2[0-8]|          | v6[0-8]|
 *  |v3[0-8]|          | v7[0-8]|
 *  +-------+ - - - - -+--------+
 *
 *                            Accumulator
 */

static void kern_4x8(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k,
                     size_t m_remain) {
    K /= 8;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);
    int16_t* outptr0 = output;
    int16_t* outptr1;
    int16_t* outptr2;
    int16_t* outptr3;
    size_t x0 = 0;

// clang-format off
#define LOAD_LINE(reg_index, m)                         \
    "cbz %[x0], 100f\n"                                 \
    "ld1 {v" reg_index ".8h}, [%[outptr" m "]], #16\n"  \
    "subs %[x0], %[x0], #1\n"

#define LOAD_C                      \
    "mov %[x0], %x[m_remain]\n"     \
    LOAD_LINE("4", "0")             \
    LOAD_LINE("5", "1")             \
    LOAD_LINE("6", "2")             \
    LOAD_LINE("7", "3")             \
    "100:\n"

#define STORE_LINE(reg_index, m)                    \
    "cbz %[x0], 101f\n"                             \
    "st1 {v" reg_index ".8h}, [%[outptr" m "]]\n"   \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                     \
    "mov %[x0], %x[m_remain]\n"     \
    STORE_LINE("4", "0")            \
    STORE_LINE("5", "1")            \
    STORE_LINE("6", "2")            \
    STORE_LINE("7", "3")            \
    "101:\n"

#define CLEAR_8_INT16(reg_index) \
    "eor v" reg_index ".16b, v" reg_index ".16b, v" reg_index ".16b\n"
#define CLEAR_4_REGS    \
    CLEAR_8_INT16("4")  \
    CLEAR_8_INT16("5")  \
    CLEAR_8_INT16("6")  \
    CLEAR_8_INT16("7")

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %x[LDC]\n"
            "add %[outptr2], %[outptr1], %x[LDC]\n"
            "add %[outptr3], %[outptr2], %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n" LOAD_C
            "b 2f\n"

            "1:\n" CLEAR_4_REGS

            "2: \n"
            "ld1 {v8.8b}, [%[b_ptr]], 8\n"
            "ld1 {v0.8b}, [%[a_ptr]], 8\n"
            "ld1 {v1.8b}, [%[a_ptr]], 8\n"
            "ld1 {v2.8b}, [%[a_ptr]], 8\n"
            "ld1 {v3.8b}, [%[a_ptr]], 8\n"
            "sshll v8.8h, v8.8b, #0\n"
            "sshll v0.8h, v0.8b, #0\n"
            "sshll v1.8h, v1.8b, #0\n"
            "sshll v2.8h, v2.8b, #0\n"
            "sshll v3.8h, v3.8b, #0\n"

            "ld1 {v9.8b}, [%[b_ptr]], 8\n"
            "mla v4.8h, v8.8h, v0.h[0]\n"
            "mla v5.8h, v8.8h, v1.h[0]\n"
            "mla v6.8h, v8.8h, v2.h[0]\n"
            "mla v7.8h, v8.8h, v3.h[0]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "ld1 {v8.8b}, [%[b_ptr]], 8\n"
            "mla v4.8h, v9.8h, v0.h[1]\n"
            "mla v5.8h, v9.8h, v1.h[1]\n"
            "mla v6.8h, v9.8h, v2.h[1]\n"
            "mla v7.8h, v9.8h, v3.h[1]\n"
            "sshll v8.8h, v8.8b, #0\n"

            "ld1 {v9.8b}, [%[b_ptr]], 8\n"
            "mla v4.8h, v8.8h, v0.h[2]\n"
            "mla v5.8h, v8.8h, v1.h[2]\n"
            "mla v6.8h, v8.8h, v2.h[2]\n"
            "mla v7.8h, v8.8h, v3.h[2]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "ld1 {v8.8b}, [%[b_ptr]], 8\n"
            "mla v4.8h, v9.8h, v0.h[3]\n"
            "mla v5.8h, v9.8h, v1.h[3]\n"
            "mla v6.8h, v9.8h, v2.h[3]\n"
            "mla v7.8h, v9.8h, v3.h[3]\n"
            "sshll v8.8h, v8.8b, #0\n"

            "ld1 {v9.8b}, [%[b_ptr]], 8\n"
            "mla v4.8h, v8.8h, v0.h[4]\n"
            "mla v5.8h, v8.8h, v1.h[4]\n"
            "mla v6.8h, v8.8h, v2.h[4]\n"
            "mla v7.8h, v8.8h, v3.h[4]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "ld1 {v8.8b}, [%[b_ptr]], 8\n"
            "mla v4.8h, v9.8h, v0.h[5]\n"
            "mla v5.8h, v9.8h, v1.h[5]\n"
            "mla v6.8h, v9.8h, v2.h[5]\n"
            "mla v7.8h, v9.8h, v3.h[5]\n"
            "sshll v8.8h, v8.8b, #0\n"

            "ld1 {v9.8b}, [%[b_ptr]], 8\n"
            "mla v4.8h, v8.8h, v0.h[6]\n"
            "mla v5.8h, v8.8h, v1.h[6]\n"
            "mla v6.8h, v8.8h, v2.h[6]\n"
            "mla v7.8h, v8.8h, v3.h[6]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "mla v4.8h, v9.8h, v0.h[7]\n"
            "mla v5.8h, v9.8h, v1.h[7]\n"
            "mla v6.8h, v9.8h, v2.h[7]\n"
            "mla v7.8h, v9.8h, v3.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3), [x0] "+r"(x0),
              [m_remain] "+r"(m_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "cc",
              "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
#undef CLEAR_8_INT16
#undef CLEAR_4_REGS
}

/**
 * Overview of register layout:
 *
 * A 8x4x8 cell of Rhs is stored in 8bit in v8-v9
 * A 4x8x8 cell of Lhs is stored in 8bit in q0-q3
 * A 8x8 block of accumulators is stored in 16bit in q4-q7
 *
 *                     +--------+
 *                     | q8[0-4]|
 *                     +--------+
 *                     | q9[0-4]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +--------+ - - - - +---------
 *  |q0[0-8]|          | q4[0-4]|
 *  |q1[0-8]|          | q5[0-4]|
 *  |q2[0-8]|          | q6[0-4]|
 *  |q3[0-8]|          | q7[0-4]|
 *  +--------+ - - - - +---------
 *
 *                            Accumulator
 */
static void kern_4x4(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, size_t m_remain,
                     size_t n_remain) {
    K /= 8;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);
    size_t x0 = 0;

// clang-format off
#define LOAD_LINE(reg_index, n)                  \
    "cmp %[x0], #0 \n"                           \
    "beq 102f\n"                                 \
    "cmp %[n_remain], #4\n"                      \
    "blt 100" n "f\n"                            \
    "ld1 {v" reg_index ".4h}, [x" n " ], #8\n"   \
    "b 101" n "f\n"                              \
    "100" n ":\n"                                \
    "cmp %[n_remain], #0\n"                      \
    "beq 101" n "f\n"                            \
    "ld1 {v" reg_index ".h}[0], [x" n "], #2\n"  \
    "cmp %[n_remain], #1\n"                      \
    "beq 101" n "f\n"                            \
    "ld1 {v" reg_index ".h}[1], [x" n "], #2\n"  \
    "cmp %[n_remain], #2\n"                      \
    "beq 101" n "f\n"                            \
    "ld1 {v" reg_index ".h}[2], [x" n "], #2\n"  \
    "101" n ":\n"                                \
    "subs %[x0], %[x0], #1\n"

#define LOAD_C                 \
    "mov %[x0], %[m_remain]\n" \
    "mov x1, x0\n"             \
    LOAD_LINE("4", "1")        \
    "add x1, x0, %x[LDC]\n"    \
    "add x0, x0, %x[LDC]\n"    \
    LOAD_LINE("5", "1")        \
    "add x1, x0, %x[LDC]\n"    \
    "add x0, x0, %x[LDC]\n"    \
    LOAD_LINE("6", "1")        \
    "add x1, x0, %x[LDC]\n"    \
    LOAD_LINE("7", "1")        \
    "102:\n"

#define STORE_LINE(reg_index, n)                 \
    "cmp %[x0], #0 \n"                           \
    "beq 105f\n"                                 \
    "cmp %[n_remain], #4\n"                      \
    "blt 103" n "f\n"                            \
    "st1 {v" reg_index ".4h}, [x" n "]\n"        \
    "b 104" n "f\n"                              \
    "103" n ":\n"                                \
    "cmp %[n_remain], #0\n"                      \
    "beq 104" n "f\n"                            \
    "st1 {v" reg_index ".h}[0], [x" n "], #2\n"  \
    "cmp %[n_remain], #1\n"                      \
    "beq 104" n "f\n"                            \
    "st1 {v" reg_index ".h}[1], [x" n "], #2\n"  \
    "cmp %[n_remain], #2\n"                      \
    "beq 104" n "f\n"                            \
    "st1 {v" reg_index ".h}[2], [x" n "], #2\n"  \
    "104" n ":\n"                                \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                \
    "mov %[x0], %[m_remain]\n" \
    "mov x1, x0\n"             \
    STORE_LINE("4", "1")       \
    "add x1, x0, %x[LDC]\n"    \
    "add x0, x0, %x[LDC]\n"    \
    STORE_LINE("5", "1")       \
    "add x1, x0, %x[LDC]\n"    \
    "add x0, x0, %x[LDC]\n"    \
    STORE_LINE("6", "1")       \
    "add x1, x0, %x[LDC]\n"    \
    STORE_LINE("7", "1")       \
    "105:\n"

    // clang-format on

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            // load accumulator C
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
            "sshll v8.8h, v8.8b, #0\n"
            "sshll v0.8h, v0.8b, #0\n"
            "sshll v1.8h, v1.8b, #0\n"
            "sshll v2.8h, v2.8b, #0\n"
            "sshll v3.8h, v3.8b, #0\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "mla v4.4h, v8.4h, v0.h[0]\n"
            "mla v5.4h, v8.4h, v1.h[0]\n"
            "mla v6.4h, v8.4h, v2.h[0]\n"
            "mla v7.4h, v8.4h, v3.h[0]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "ld1 {v8.s}[0], [%[b_ptr]], 4\n"
            "mla v4.4h, v9.4h, v0.h[1]\n"
            "mla v5.4h, v9.4h, v1.h[1]\n"
            "mla v6.4h, v9.4h, v2.h[1]\n"
            "mla v7.4h, v9.4h, v3.h[1]\n"
            "sshll v8.8h, v8.8b, #0\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "mla v4.4h, v8.4h, v0.h[2]\n"
            "mla v5.4h, v8.4h, v1.h[2]\n"
            "mla v6.4h, v8.4h, v2.h[2]\n"
            "mla v7.4h, v8.4h, v3.h[2]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "ld1 {v8.s}[0], [%[b_ptr]], 4\n"
            "mla v4.4h, v9.4h, v0.h[3]\n"
            "mla v5.4h, v9.4h, v1.h[3]\n"
            "mla v6.4h, v9.4h, v2.h[3]\n"
            "mla v7.4h, v9.4h, v3.h[3]\n"
            "sshll v8.8h, v8.8b, #0\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "mla v4.4h, v8.4h, v0.h[4]\n"
            "mla v5.4h, v8.4h, v1.h[4]\n"
            "mla v6.4h, v8.4h, v2.h[4]\n"
            "mla v7.4h, v8.4h, v3.h[4]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "ld1 {v8.s}[0], [%[b_ptr]], 4\n"
            "mla v4.4h, v9.4h, v0.h[5]\n"
            "mla v5.4h, v9.4h, v1.h[5]\n"
            "mla v6.4h, v9.4h, v2.h[5]\n"
            "mla v7.4h, v9.4h, v3.h[5]\n"
            "sshll v8.8h, v8.8b, #0\n"

            "ld1 {v9.s}[0], [%[b_ptr]], 4\n"
            "mla v4.4h, v8.4h, v0.h[6]\n"
            "mla v5.4h, v8.4h, v1.h[6]\n"
            "mla v6.4h, v8.4h, v2.h[6]\n"
            "mla v7.4h, v8.4h, v3.h[6]\n"
            "sshll v9.8h, v9.8b, #0\n"

            "mla v4.4h, v9.4h, v0.h[7]\n"
            "mla v5.4h, v9.4h, v1.h[7]\n"
            "mla v6.4h, v9.4h, v2.h[7]\n"
            "mla v7.4h, v9.4h, v3.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "bne 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [outptr] "+r"(outptr),
              [K] "+r"(K), [is_first_k] "+r"(is_first_k), [LDC] "+r"(LDC),
              [x0] "+r"(x0), [m_remain] "+r"(m_remain),
              [n_remain] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "x1",
              "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_s8x8x16_8x8_pack_A_n(dt_int8* outptr, const dt_int8* inptr,
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
        for (; K > 15; K -= 16) {
            interleave_8x8_2_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr);
        }

        if (K > 0) {
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr, 8, K);
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 8, K);
        }
    }
}

static void gemm_s8x8x16_8x8_transpose_pack_A_n(dt_int8* out, const dt_int8* in,
                                                int ldin, int x0, int xmax,
                                                int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 8) * 4;
    const int ksize8 = ksize4 * 2;
    int8_t* outptr = out;
    int8_t* outptr_base = out;
    //! 4x4 block output start pos
    int8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 8) {
        const int8_t* inptr0 = in + k * ldin + x0;
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
                        inptr7, outptr, 4, 4);
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
                        inptr7, outptr, 4, xmax - x);
        }

        outptr_base += 8 * 8;
        outptr_base4 += 4 * 8;
    }
}

static void gemm_s8x8x16_8x8_pack_B_n(dt_int8* out, const dt_int8* in, int ldin,
                                      int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 8) * 4;
    const int ksize8 = ksize4 * 2;
    int8_t* outptr = out;
    int8_t* outptr_base = out;
    int8_t* outptr_interleave = nullptr;
    int8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 8) {
        const int8_t* inptr0 = in + k * ldin + x0;
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
                         inptr7, outptr_interleave, 4, 4);
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
                         inptr7, outptr_interleave, 4, xmax - x);
        }

        outptr_base += 8 * 8;
        outptr_base4 += 4 * 8;
    }
}

static void gemm_s8x8x16_8x8_transpose_pack_B_n(dt_int8* outptr,
                                                const dt_int8* inptr, int ldin,
                                                int y0, int ymax, int k0,
                                                int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    constexpr int interleave4 = 32;
    constexpr int interleave8 = 64;

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
        for (; K > 7; K -= 8) {
            transpose_8x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                              inptr6, inptr7, outptr);
            outptr += interleave8;
        }

        if (K > 0) {
            transpose_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                        inptr7, outptr, 8, K);
            outptr += interleave8;
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
            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 8, K);
            outptr += interleave4;
        }
    }
}
}  // namespace matmul_8x8x8
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
