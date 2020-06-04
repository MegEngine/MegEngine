/**
 * \file dnn/src/aarch64/matrix_mul/int16/kernel_12x8x1.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_12x8x1 {

/**
 * Overview of register layout:
 *
 * A 1x8  cell of Rhs is stored in 16bit in q2
 * A 12x1 cell of Lhs is stored in 16bit in q0-q1
 * A 12x8 block of accumulators is stored in 32bit in q7-q30
 *
 *                     +--------+--------+
 *                     | v2[0-3]| v2[4-7]|
 *                Rhs  +--------+--------+
 *    Lhs              |        |        |
 *
 *  +--------+ - - - - +-----------------+
 *  |v0[0]|            | v7[0-3]| v8[0-3]|
 *  |v0[1]|            | v9[0-3]|v10[0-3]|
 *  |v0[2]|            |v11[0-3]|v12[0-3]|
 *  |v0[3]|            |v13[0-3]|v14[0-3]|
 *  |v0[4]|            |v15[0-3]|v16[0-3]|
 *  |v0[5]|            |v17[0-3]|v18[0-3]|
 *  |v0[6]|            |v19[0-3]|v20[0-3]|
 *  |v0[7]|            |v21[0-3]|v22[0-3]|
 *  |v1[0]|            |v23[0-3]|v24[0-3]|
 *  |v1[1]|            |v25[0-3]|v26[0-3]|
 *  |v1[2]|            |v27[0-3]|v28[0-3]|
 *  |v1[3]|            |v29[0-3]|v30[0-3]|
 *  +--------+ - - - - +-----------------+
 *
 *                         Accumulator
 */

static void kern_12x8(const int16_t* packA, const int16_t* packB, int K,
                      int32_t* output, int LDC, bool is_first_k) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

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
            "add x8, x7, %x[LDC]\n"
            "add x9, x8, %x[LDC]\n"
            "add x10, x9, %x[LDC]\n"
            "add x11, x10, %x[LDC]\n"
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"

            "ldp q7, q8, [%[output]]\n"
            "ldp q9, q10, [x1]\n"
            "ldp q11, q12, [x2]\n"
            "ldp q13, q14, [x3]\n"
            "ldp q15, q16, [x4]\n"
            "ldp q17, q18, [x5]\n"
            "ldp q19, q20, [x6]\n"
            "ldp q21, q22, [x7]\n"
            "ldp q23, q24, [x7]\n"
            "ldp q25, q26, [x7]\n"
            "ldp q27, q28, [x7]\n"
            "ldp q29, q30, [x7]\n"
            "b 2f\n"

            "1:\n"
            "eor v7.16b, v7.16b, v7.16b\n"
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
            "eor v24.16b, v24.16b, v24.16b\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"

            "2: \n"
            "ld1 {v2.8h}, [%[b_ptr]], 16\n"
            "ld1 {v0.8h}, [%[a_ptr]], 16\n"
            "ld1 {v1.4h}, [%[a_ptr]], 8\n"

            "smlal v7.4s, v2.4h, v0.h[0]\n"
            "smlal v9.4s, v2.4h, v0.h[1]\n"
            "smlal v11.4s, v2.4h, v0.h[2]\n"
            "smlal v13.4s, v2.4h, v0.h[3]\n"
            "smlal v15.4s, v2.4h, v0.h[4]\n"
            "smlal v17.4s, v2.4h, v0.h[5]\n"
            "smlal v19.4s, v2.4h, v0.h[6]\n"
            "smlal v21.4s, v2.4h, v0.h[7]\n"
            "smlal v23.4s, v2.4h, v1.h[0]\n"
            "smlal v25.4s, v2.4h, v1.h[1]\n"
            "smlal v27.4s, v2.4h, v1.h[2]\n"
            "smlal v29.4s, v2.4h, v1.h[3]\n"
            "smlal2 v8.4s, v2.8h, v0.h[0]\n"
            "smlal2 v10.4s, v2.8h, v0.h[1]\n"
            "smlal2 v12.4s, v2.8h, v0.h[2]\n"
            "smlal2 v14.4s, v2.8h, v0.h[3]\n"
            "smlal2 v16.4s, v2.8h, v0.h[4]\n"
            "smlal2 v18.4s, v2.8h, v0.h[5]\n"
            "smlal2 v20.4s, v2.8h, v0.h[6]\n"
            "smlal2 v22.4s, v2.8h, v0.h[7]\n"
            "smlal2 v24.4s, v2.8h, v1.h[0]\n"
            "smlal2 v26.4s, v2.8h, v1.h[1]\n"
            "smlal2 v28.4s, v2.8h, v1.h[2]\n"
            "smlal2 v30.4s, v2.8h, v1.h[3]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n"
            "stp q7, q8, [%[output]]\n"
            "stp q9, q10, [x1]\n"
            "stp q11, q12, [x2]\n"
            "stp q13, q14, [x3]\n"
            "stp q15, q16, [x4]\n"
            "stp q17, q18, [x5]\n"
            "stp q19, q20, [x6]\n"
            "stp q21, q22, [x7]\n"
            "stp q23, q24, [x8]\n"
            "stp q25, q26, [x9]\n"
            "stp q27, q28, [x10]\n"
            "stp q29, q30, [x11]\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [output] "+r"(output)
            :
            : "v0", "v1", "v2", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
              "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x1",
              "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11",
              "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 1x8  cell of Rhs is stored in 16bit in q2
 * A 8x1  cell of Lhs is stored in 16bit in q0
 * A 8x8  block of accumulators is stored in 32bit in q7-q22
 *
 *                     +--------+--------+
 *                     | v2[0-3]| v2[4-7]|
 *                Rhs  +--------+--------+
 *    Lhs              |        |        |
 *
 *  +--------+ - - - - +-----------------+
 *  |v0[0]|            | v7[0-3]| v8[0-3]|
 *  |v0[1]|            | v9[0-3]|v10[0-3]|
 *  |v0[2]|            |v11[0-3]|v12[0-3]|
 *  |v0[3]|            |v13[0-3]|v14[0-3]|
 *  |v0[4]|            |v15[0-3]|v16[0-3]|
 *  |v0[5]|            |v17[0-3]|v18[0-3]|
 *  |v0[6]|            |v19[0-3]|v20[0-3]|
 *  |v0[7]|            |v21[0-3]|v22[0-3]|
 *  +--------+ - - - - +-----------------+
 *
 *                         Accumulator
 */

static void kern_8x8(const int16_t* packA, const int16_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

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
            "cmp %w[is_first_k], #1\n"
            "beq 1f\n"

            "ldp q7, q8, [%[output]]\n"
            "ldp q9, q10, [x1]\n"
            "ldp q11, q12, [x2]\n"
            "ldp q13, q14, [x3]\n"
            "ldp q15, q16, [x4]\n"
            "ldp q17, q18, [x5]\n"
            "ldp q19, q20, [x6]\n"
            "ldp q21, q22, [x7]\n"
            "b 2f\n"

            "1:\n"
            "eor v7.16b, v7.16b, v7.16b\n"
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

            "2: \n"
            "ld1 {v2.8h}, [%[b_ptr]], 16\n"
            "ld1 {v0.8h}, [%[a_ptr]], 16\n"

            "smlal v7.4s, v2.4h, v0.h[0]\n"
            "smlal v9.4s, v2.4h, v0.h[1]\n"
            "smlal v11.4s, v2.4h, v0.h[2]\n"
            "smlal v13.4s, v2.4h, v0.h[3]\n"
            "smlal v15.4s, v2.4h, v0.h[4]\n"
            "smlal v17.4s, v2.4h, v0.h[5]\n"
            "smlal v19.4s, v2.4h, v0.h[6]\n"
            "smlal v21.4s, v2.4h, v0.h[7]\n"
            "smlal2 v8.4s, v2.8h, v0.h[0]\n"
            "smlal2 v10.4s, v2.8h, v0.h[1]\n"
            "smlal2 v12.4s, v2.8h, v0.h[2]\n"
            "smlal2 v14.4s, v2.8h, v0.h[3]\n"
            "smlal2 v16.4s, v2.8h, v0.h[4]\n"
            "smlal2 v18.4s, v2.8h, v0.h[5]\n"
            "smlal2 v20.4s, v2.8h, v0.h[6]\n"
            "smlal2 v22.4s, v2.8h, v0.h[7]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n"
            "stp q7, q8, [%[output]]\n"
            "stp q9, q10, [x1]\n"
            "stp q11, q12, [x2]\n"
            "stp q13, q14, [x3]\n"
            "stp q15, q16, [x4]\n"
            "stp q17, q18, [x5]\n"
            "stp q19, q20, [x6]\n"
            "stp q21, q22, [x7]\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [output] "+r"(output)
            :
            : "v0", "v2", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
              "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "x1",
              "x2", "x3", "x4", "x5", "x6", "x7", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 1x8  cell of Rhs is stored in 16bit in q2
 * A 4x1  cell of Lhs is stored in 16bit in q0
 * A 4x8  block of accumulators is stored in 32bit in q7-q14
 *
 *                     +--------+--------+
 *                     | v2[0-3]| v2[4-7]|
 *                Rhs  +--------+--------+
 *    Lhs              |        |        |
 *
 *  +--------+ - - - - +-----------------+
 *  |v0[0]|            | v7[0-3]| v8[0-3]|
 *  |v0[1]|            | v9[0-3]|v10[0-3]|
 *  |v0[2]|            |v11[0-3]|v12[0-3]|
 *  |v0[3]|            |v13[0-3]|v14[0-3]|
 *  +--------+ - - - - +-----------------+
 *
 *                         Accumulator
 */

static void kern_4x8(const int16_t* packA, const int16_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k,
                     size_t m_remain) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

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

#define LOAD_C                   \
    "mov %[x0], %x[m_remain]\n"  \
    LOAD_LINE("q7", "q8", "0")   \
    LOAD_LINE("q9", "q10", "1")  \
    LOAD_LINE("q11", "q12", "2") \
    LOAD_LINE("q13", "q14", "3") \
    "100:\n"

#define STORE_LINE(v1, v2, m)              \
    "cbz %[x0], 101f\n"                    \
    "stp " v1 "," v2", [%[outptr" m "]]\n" \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                   \
    "mov %[x0], %x[m_remain]\n"   \
    STORE_LINE("q7", "q8", "0")   \
    STORE_LINE("q9", "q10", "1")  \
    STORE_LINE("q11", "q12", "2") \
    STORE_LINE("q13", "q14", "3") \
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
            "eor v7.16b, v7.16b, v7.16b\n"
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"

            "2: \n"
            "ld1 {v2.8h}, [%[b_ptr]], 16\n"
            "ld1 {v0.4h}, [%[a_ptr]], 8\n"

            "smlal v7.4s, v2.4h, v0.h[0]\n"
            "smlal v9.4s, v2.4h, v0.h[1]\n"
            "smlal v11.4s, v2.4h, v0.h[2]\n"
            "smlal v13.4s, v2.4h, v0.h[3]\n"
            "smlal2 v8.4s, v2.8h, v0.h[0]\n"
            "smlal2 v10.4s, v2.8h, v0.h[1]\n"
            "smlal2 v12.4s, v2.8h, v0.h[2]\n"
            "smlal2 v14.4s, v2.8h, v0.h[3]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3), [x0] "+r"(x0),
              [m_remain] "+r"(m_remain)
            :
            : "v0", "v2", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
              "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 1x4  cell of Rhs is stored in 16bit in q2
 * A 12x1 cell of Lhs is stored in 16bit in q0-q1
 * A 12x4 block of accumulators is stored in 32bit in q7-q30
 *
 *                     +--------+
 *                     | v2[0-3]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +--------+ - - - - +---------
 *  |v0[0]|            | v8[0-3]|
 *  |v0[1]|            | v9[0-3]|
 *  |v0[2]|            |v10[0-3]|
 *  |v0[3]|            |v11[0-3]|
 *  |v0[4]|            |v12[0-3]|
 *  |v0[5]|            |v13[0-3]|
 *  |v0[6]|            |v14[0-3]|
 *  |v0[7]|            |v15[0-3]|
 *  |v1[0]|            |v16[0-3]|
 *  |v1[1]|            |v17[0-3]|
 *  |v1[2]|            |v18[0-3]|
 *  |v1[3]|            |v19[0-3]|
 *  +--------+ - - - - +---------
 *
 *                     Accumulator
 */

static void kern_12x4(const int16_t* packA, const int16_t* packB, int K,
                      int32_t* output, int LDC, bool is_first_k,
                      size_t n_remain) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    int32_t* outptr4;
    int32_t* outptr5;
    int32_t* outptr6;
    int32_t* outptr7;
    int32_t* outptr8;
    int32_t* outptr9;
    int32_t* outptr10;
    int32_t* outptr11;
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
    LOAD_LINE("15", "7")  \
    LOAD_LINE("16", "8")  \
    LOAD_LINE("17", "9")  \
    LOAD_LINE("18", "10") \
    LOAD_LINE("19", "11")

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
    STORE_LINE("15", "7")  \
    STORE_LINE("16", "8")  \
    STORE_LINE("17", "9")  \
    STORE_LINE("18", "10") \
    STORE_LINE("19", "11")
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
            "add %[outptr8], %[outptr7], %x[LDC]\n"
            "add %[outptr9], %[outptr8], %x[LDC]\n"
            "add %[outptr10], %[outptr9], %x[LDC]\n"
            "add %[outptr11], %[outptr10], %x[LDC]\n"
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
            "eor v16.16b, v16.16b, v16.16b\n"
            "eor v17.16b, v17.16b, v17.16b\n"
            "eor v18.16b, v18.16b, v18.16b\n"
            "eor v19.16b, v19.16b, v19.16b\n"

            "2: \n"
            "ld1 {v2.4h}, [%[b_ptr]], 8\n"
            "ld1 {v0.8h}, [%[a_ptr]], 16\n"
            "ld1 {v1.4h}, [%[a_ptr]], 8\n"

            "smlal v8.4s, v2.4h, v0.h[0]\n"
            "smlal v9.4s, v2.4h, v0.h[1]\n"
            "smlal v10.4s, v2.4h, v0.h[2]\n"
            "smlal v11.4s, v2.4h, v0.h[3]\n"
            "smlal v12.4s, v2.4h, v0.h[4]\n"
            "smlal v13.4s, v2.4h, v0.h[5]\n"
            "smlal v14.4s, v2.4h, v0.h[6]\n"
            "smlal v15.4s, v2.4h, v0.h[7]\n"
            "smlal v16.4s, v2.4h, v1.h[0]\n"
            "smlal v17.4s, v2.4h, v1.h[1]\n"
            "smlal v18.4s, v2.4h, v1.h[2]\n"
            "smlal v19.4s, v2.4h, v1.h[3]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3),
              [outptr4] "=r"(outptr4), [outptr5] "=r"(outptr5),
              [outptr6] "=r"(outptr6), [outptr7] "=r"(outptr7),
              [outptr8] "=r"(outptr8), [outptr9] "=r"(outptr9),
              [outptr10] "=r"(outptr10), [outptr11] "=r"(outptr11),
              [x0] "+r"(x0), [n_remain] "+r"(n_remain)
            :
            : "v0", "v1", "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
              "v15", "v16", "v17", "v18", "v19", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 1x4  cell of Rhs is stored in 16bit in q2
 * A 12x1 cell of Lhs is stored in 16bit in q0-q1
 * A 12x4 block of accumulators is stored in 32bit in q7-q30
 *
 *                     +--------+
 *                     | v2[0-3]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +--------+ - - - - +---------
 *  |v0[0]|            | v8[0-3]|
 *  |v0[1]|            | v9[0-3]|
 *  |v0[2]|            |v10[0-3]|
 *  |v0[3]|            |v11[0-3]|
 *  |v0[4]|            |v12[0-3]|
 *  |v0[5]|            |v13[0-3]|
 *  |v0[6]|            |v14[0-3]|
 *  |v0[7]|            |v15[0-3]|
 *  +--------+ - - - - +---------
 *
 *                     Accumulator
 */

static void kern_8x4(const int16_t* packA, const int16_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k,
                     size_t n_remain) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

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

#define LOAD_C           \
    LOAD_LINE("8", "0")  \
    LOAD_LINE("9", "1")  \
    LOAD_LINE("10", "2") \
    LOAD_LINE("11", "3") \
    LOAD_LINE("12", "4") \
    LOAD_LINE("13", "5") \
    LOAD_LINE("14", "6") \
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

#define STORE_C           \
    STORE_LINE("8", "0")  \
    STORE_LINE("9", "1")  \
    STORE_LINE("10", "2") \
    STORE_LINE("11", "3") \
    STORE_LINE("12", "4") \
    STORE_LINE("13", "5") \
    STORE_LINE("14", "6") \
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
            "ld1 {v2.4h}, [%[b_ptr]], 8\n"
            "ld1 {v0.8h}, [%[a_ptr]], 16\n"

            "smlal v8.4s, v2.4h, v0.h[0]\n"
            "smlal v9.4s, v2.4h, v0.h[1]\n"
            "smlal v10.4s, v2.4h, v0.h[2]\n"
            "smlal v11.4s, v2.4h, v0.h[3]\n"
            "smlal v12.4s, v2.4h, v0.h[4]\n"
            "smlal v13.4s, v2.4h, v0.h[5]\n"
            "smlal v14.4s, v2.4h, v0.h[6]\n"
            "smlal v15.4s, v2.4h, v0.h[7]\n"

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
            : "v0", "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 1x4  cell of Rhs is stored in 16bit in q2
 * A 12x1 cell of Lhs is stored in 16bit in q0-q1
 * A 12x4 block of accumulators is stored in 32bit in q7-q30
 *
 *                     +--------+
 *                     | v2[0-3]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +--------+ - - - - +---------
 *  |v0[0]|            | v8[0-3]|
 *  |v0[1]|            | v9[0-3]|
 *  |v0[2]|            |v10[0-3]|
 *  |v0[3]|            |v11[0-3]|
 *  +--------+ - - - - +---------
 *
 *                     Accumulator
 */

static void kern_4x4(const int16_t* packA, const int16_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, size_t m_remain,
                     size_t n_remain) {
    const int16_t* a_ptr = packA;
    const int16_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;
    int32_t* outptr2;
    int32_t* outptr3;
    size_t x0 = 0;
    size_t x1 = 0;

// clang-format off
#define LOAD_LINE(reg_index, n)                \
    "cbz %[x1], 102f\n"                        \
    "mov %[x0], %[outptr" n "]\n"              \
    "cmp %w[n_remain], #4\n"                   \
    "blt 100" n "f\n"                          \
    "ldr q" reg_index ", [%[x0]]\n"            \
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
    "101" n ":\n" \
    "subs %[x1], %[x1], #1\n"

#define LOAD_C                  \
    "mov %[x1], %x[m_remain]\n" \
    LOAD_LINE("8", "0")         \
    LOAD_LINE("9", "1")         \
    LOAD_LINE("10", "2")        \
    LOAD_LINE("11", "3")        \
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
    STORE_LINE("8", "0")        \
    STORE_LINE("9", "1")        \
    STORE_LINE("10", "2")       \
    STORE_LINE("11", "3")       \
    "105:\n"
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
            "eor v8.16b, v8.16b, v8.16b\n"
            "eor v9.16b, v9.16b, v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"

            "2: \n"
            "ld1 {v2.4h}, [%[b_ptr]], 8\n"
            "ld1 {v0.4h}, [%[a_ptr]], 8\n"

            "smlal v8.4s, v2.4h, v0.h[0]\n"
            "smlal v9.4s, v2.4h, v0.h[1]\n"
            "smlal v10.4s, v2.4h, v0.h[2]\n"
            "smlal v11.4s, v2.4h, v0.h[3]\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr0] "+r"(outptr0), [outptr1] "=r"(outptr1),
              [outptr2] "=r"(outptr2), [outptr3] "=r"(outptr3), [x0] "+r"(x0),
              [m_remain] "+r"(m_remain), [x1] "+r"(x1),
              [n_remain] "+r"(n_remain)
            :
            : "v0", "v2", "v8", "v9", "v10", "v11", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_s16_12x8x1_pack_A_n(int16_t* outptr, const int16_t* inptr,
                                     int ldin, int y0, int ymax, int k0,
                                     int kmax) {
    int16_t zerobuff[4];
    std::memset(zerobuff, 0, sizeof(int16_t) * 4);

    int y = y0;
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
        for (; K > 3; K -= 4) {
            interleave_12x1_4_h(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                                inptr6, inptr7, inptr8, inptr9, inptr10,
                                inptr11, outptr);
        }

        if (K > 0) {
            interleave_12(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                          inptr6, inptr7, inptr8, inptr9, inptr10, inptr11,
                          outptr, 1, K);
        }
    }

    for (; y + 7 < ymax; y += 8) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;
        const int16_t* inptr4 = inptr3 + ldin;
        const int16_t* inptr5 = inptr4 + ldin;
        const int16_t* inptr6 = inptr5 + ldin;
        const int16_t* inptr7 = inptr6 + ldin;

        int K = kmax - k0;
        for (; K > 7; K -= 8) {
            interleave_8x1_8_h(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr);
        }

        if (K > 0) {
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr, 1, K);
        }
    }

    for (; y < ymax; y += 4) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        for (; K > 3; K -= 4) {
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
            interleave_4x1_4_h(inptr0, inptr1, inptr2, inptr3, outptr);
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K);
        }
    }
}

static void gemm_s16_12x8x1_transpose_pack_A_n(int16_t* out, const int16_t* in,
                                               int ldin, int x0, int xmax,
                                               int k0, int kmax) {
    const int ksize = kmax - k0;
    const int ksize4 = ksize * 4;
    const int ksize8 = ksize4 * 2;
    const int ksize12 = ksize * 12;
    int16_t* outptr = out;
    int16_t* outptr_base = out;
    //! 1x8 block output start pos
    int16_t* outptr_base8 = out + ((xmax - x0) / 12) * ksize12;
    //! 1x4 block output start pos
    int16_t* outptr_base4 =
            outptr_base8 + (xmax - (x0 + (xmax - x0) / 12 * 12)) / 8 * ksize8;

    int k = k0;
    for (; k < kmax; k++) {
        const int16_t* inptr = in + k * ldin + x0;
        prefetch_2x(inptr);
        int x = x0;
        outptr = outptr_base;
        for (; x + 11 < xmax; x += 12) {
            transpose_12x1_1_h(inptr, outptr);
            outptr += ksize12;
        }
        outptr = outptr_base8;
        for (; x + 7 < xmax; x += 8) {
            transpose_8x1_1_h(inptr, outptr);
            outptr += ksize8;
        }
        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            transpose_4x1_1_h(inptr, outptr);
            outptr += ksize4;
        }
        int X = (4 - (xmax - x)) % 4;
        for (; x < xmax; x++) {
            *outptr++ = *inptr++;
        }
        memset(outptr, 0, sizeof(int16_t) * X);
        outptr += ksize4;
        outptr_base += 12;
        outptr_base8 += 8;
        outptr_base4 += 4;
    }
}

static void gemm_s16_12x8x1_pack_B_n(int16_t* out, const int16_t* in, int ldin,
                                     int x0, int xmax, int k0, int kmax) {
    const int ksize = kmax - k0;
    const int ksize4 = ksize * 4;
    const int ksize8 = ksize4 * 2;
    int16_t* outptr = out;
    int16_t* outptr_base = out;
    //! 1x4 block output start pos
    int16_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k++) {
        const int16_t* inptr = in + k * ldin + x0;
        prefetch_2x(inptr);
        int x = x0;
        outptr = outptr_base;
        for (; x + 7 < xmax; x += 8) {
            transpose_8x1_1_h(inptr, outptr);
            outptr += ksize8;
        }
        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            transpose_4x1_1_h(inptr, outptr);
            outptr += ksize4;
        }
        int X = (4 - (xmax - x)) % 4;
        for (; x < xmax; x++) {
            *outptr++ = *inptr++;
        }
        memset(outptr, 0, sizeof(int16_t) * X);
        outptr += ksize4;
        outptr_base += 8;
        outptr_base4 += 4;
    }
}

static void gemm_s16_12x8x1_transpose_pack_B_n(int16_t* outptr,
                                               const int16_t* inptr, int ldin,
                                               int y0, int ymax, int k0,
                                               int kmax) {
    int16_t zerobuff[4];
    std::memset(zerobuff, 0, sizeof(int16_t) * 4);

    int y = y0;
    for (; y + 7 < ymax; y += 8) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;
        const int16_t* inptr4 = inptr3 + ldin;
        const int16_t* inptr5 = inptr4 + ldin;
        const int16_t* inptr6 = inptr5 + ldin;
        const int16_t* inptr7 = inptr6 + ldin;

        int K = kmax - k0;
        for (; K > 7; K -= 8) {
            interleave_8x1_8_h(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr);
        }

        if (K > 0) {
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr, 1, K);
        }
    }

    for (; y < ymax; y += 4) {
        const int16_t* inptr0 = inptr + y * ldin + k0;
        const int16_t* inptr1 = inptr0 + ldin;
        const int16_t* inptr2 = inptr1 + ldin;
        const int16_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        for (; K > 3; K -= 4) {
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
            interleave_4x1_4_h(inptr0, inptr1, inptr2, inptr3, outptr);
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K);
        }
    }
}

}  // namespace matmul_12x8x1
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
