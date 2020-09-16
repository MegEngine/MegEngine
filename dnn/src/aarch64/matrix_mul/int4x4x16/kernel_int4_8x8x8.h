/**
 * \file dnn/src/aarch64/matrix_mul/int4x4x16/kernel_8x8x8.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <inttypes.h>
#include <cstring>
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_s4_4x4x16 {

/**
 * Overview of register layout:
 *
 *                     +---------+---------+---------+---------+
 *                     |v20[0-15]|v21[0-15]|v22[0-15]|v23[0-15]|
 *                Rhs  +---------+---------+---------+---------+
 *    Lhs              |         |         |
 *
 *  +--------+ - - - - +---------+---------+---------+---------+
 *  |v0[0-15]|         | v4[0-8] |  v8[0-8]| v12[0-8]| v16[0-8]|
 *  |v1[0-15]|         | v5[0-8] |  v9[0-8]| v13[0-8]| v17[0-8]|
 *  |v2[0-15]|         | v6[0-8] | v10[0-8]| v14[0-8]| v18[0-8]|
 *  |v3[0-15]|         | v7[0-8] | v11[0-8]| v15[0-8]| v19[0-8]|
 *  +--------+ - - - - +---------+---------+---------+---------+
 *
 *                            Accumulator
 */

static void s4_kern_8x8_remain(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 8;
    LDC = LDC * sizeof(int16_t);
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
// clang-format off
#define LOAD_LINE(reg_index, n)                    \
    "cmp x8, #0 \n"                                \
    "beq 105f\n"                                   \
    "cmp %w[n_remain], #4\n"                       \
    "blt 100" n "f\n"                              \
    "ld1 {v" reg_index ".8h}, [x" n "], #16\n"     \
    "b 101" n "f\n"                                \
    "100" n ":\n"                                  \
    "cmp %w[n_remain], #0\n"                       \
    "blt 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[0], [x" n "], #2\n"    \
    "cmp %w[n_remain], #1\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[1], [x" n "], #2\n"    \
    "cmp %w[n_remain], #2\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[2], [x" n "], #2\n"    \
    "cmp %w[n_remain], #3\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[3], [x" n "], #2\n"    \
    "cmp %w[n_remain], #4\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[4], [x" n "], #2\n"    \
    "cmp %w[n_remain], #5\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[5], [x" n "], #2\n"    \
    "cmp %w[n_remain], #6\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[6], [x" n "], #2\n"    \
    "101" n ":\n"                                  \
    "sub x8, x8, #1\n"

#define LOAD_C                     \
    "mov x8, %x[m_remain]\n"       \
    LOAD_LINE("24", "0")           \
    LOAD_LINE("25", "1")           \
    LOAD_LINE("26", "2")           \
    LOAD_LINE("27", "3")           \
    LOAD_LINE("28", "4")           \
    LOAD_LINE("29", "5")           \
    LOAD_LINE("30", "6")           \
    LOAD_LINE("31", "7")           \
    "105:\n"

#define STORE_LINE(reg_index, n)                \
    "cmp x8, #0 \n"                             \
    "beq 105f\n"                                \
    "cmp %w[n_remain], #8\n"                    \
    "blt 102" n "f\n"                           \
    "st1 {v" reg_index ".8h}, [x" n "], #16\n"  \
    "b 103" n "f\n"                             \
    "102" n ":\n"                               \
    "cmp %w[n_remain], #0\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[0], [x" n "], #2\n" \
    "cmp %w[n_remain], #1\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[1], [x" n "], #2\n" \
    "cmp %w[n_remain], #2\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[2], [x" n "], #2\n" \
    "cmp %w[n_remain], #3\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[3], [x" n "], #2\n" \
    "cmp %w[n_remain], #4\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[4], [x" n "], #2\n" \
    "cmp %w[n_remain], #5\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[5], [x" n "], #2\n" \
    "cmp %w[n_remain], #6\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[6], [x" n "], #2\n" \
    "103" n ":\n"                               \
    "sub x8, x8, #1\n"

#define STORE_C                     \
    "mov x8, %x[m_remain]\n"        \
    STORE_LINE("24", "0")           \
    STORE_LINE("25", "1")           \
    STORE_LINE("26", "2")           \
    STORE_LINE("27", "3")           \
    STORE_LINE("28", "4")           \
    STORE_LINE("29", "5")           \
    STORE_LINE("30", "6")           \
    STORE_LINE("31", "7")           \
    "105:\n"
    // clang-format on
    register int16_t* outptr asm("x0") = output;
    asm volatile(
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"
            "add x4, x3, %x[LDC]\n"
            "add x5, x4, %x[LDC]\n"
            "add x6, x5, %x[LDC]\n"
            "add x7, x6, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 2f\n" LOAD_C
            "b 1f\n"

            "2:\n"  // Clear the C regs.
            "eor v24.16b, v24.16b, v24.16b\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"
            "eor v31.16b, v31.16b, v31.16b\n"
            // General loop.
            "1:\n"
            "ld1  {v20.16b}, [%[a_ptr]],#16\n"
            "ld1  {v21.16b}, [%[a_ptr]],#16\n"
            "dup v0.8b,v20.b[0]\n"
            "dup v1.8b,v20.b[1]\n"
            "dup v2.8b,v20.b[2]\n"
            "dup v3.8b,v20.b[3]\n"
            "ld1  {v22.16b}, [%[a_ptr]],#16\n"
            "ld1  {v23.16b}, [%[a_ptr]],#16\n"
            "ld1  {v16.8b}, [%[b_ptr]], 8\n"
            "dup v4.8b,v20.b[4]\n"
            "dup v5.8b,v20.b[5]\n"
            "dup v6.8b,v20.b[6]\n"
            "dup v7.8b,v20.b[7]\n"
            
            "ld1  {v17.8b}, [%[b_ptr]], 8\n"

            "dup v8.8b,v20.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v16.8b\n"
            "dup v9.8b,v20.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v16.8b\n"
            "dup v10.8b,v20.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v16.8b\n"
            "dup v11.8b,v20.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v16.8b\n"
            "dup v12.8b,v20.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v16.8b\n"
            "dup v13.8b,v20.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v16.8b\n"
            "dup v14.8b,v20.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v16.8b\n"
            "dup v15.8b,v20.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v16.8b\n"

            "ld1  {v18.8b}, [%[b_ptr]], 8\n"

            "dup v0.8b,v21.b[0]\n"
            "smlal   v24.8h,  v8.8b,   v17.8b\n"
            "dup v1.8b,v21.b[1]\n"
            "smlal   v25.8h,  v9.8b,   v17.8b\n"
            "dup v2.8b,v21.b[2]\n"
            "smlal   v26.8h,  v10.8b,  v17.8b\n"
            "dup v3.8b,v21.b[3]\n"
            "smlal   v27.8h,  v11.8b,  v17.8b\n"
            "dup v4.8b,v21.b[4]\n"
            "smlal   v28.8h,  v12.8b,  v17.8b\n"
            "dup v5.8b,v21.b[5]\n"
            "smlal   v29.8h,  v13.8b,  v17.8b\n"
            "dup v6.8b,v21.b[6]\n"
            "smlal   v30.8h,  v14.8b,  v17.8b\n"
            "dup v7.8b,v21.b[7]\n"
            "smlal   v31.8h,  v15.8b,  v17.8b\n"

            "ld1  {v19.8b}, [%[b_ptr]], 8\n"

            "dup v8.8b,v21.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v18.8b\n"
            "dup v9.8b,v21.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v18.8b\n"
            "dup v10.8b,v21.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v18.8b\n"
            "dup v11.8b,v21.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v18.8b\n"
            "dup v12.8b,v21.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v18.8b\n"
            "dup v13.8b,v21.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v18.8b\n"
            "dup v14.8b,v21.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v18.8b\n"
            "dup v15.8b,v21.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v18.8b\n"

            "ld1  {v16.8b}, [%[b_ptr]], 8\n"
            "dup v0.8b,v22.b[0]\n"
            "smlal   v24.8h,  v8.8b,   v19.8b\n"
            "dup v1.8b,v22.b[1]\n"
            "smlal   v25.8h,  v9.8b,   v19.8b\n"
            "dup v2.8b,v22.b[2]\n"
            "smlal   v26.8h,  v10.8b,  v19.8b\n"
            "dup v3.8b,v22.b[3]\n"
            "smlal   v27.8h,  v11.8b,  v19.8b\n"
            "dup v4.8b,v22.b[4]\n"
            "smlal   v28.8h,  v12.8b,  v19.8b\n"
            "dup v5.8b,v22.b[5]\n"
            "smlal   v29.8h,  v13.8b,  v19.8b\n"
            "dup v6.8b,v22.b[6]\n"
            "smlal   v30.8h,  v14.8b,  v19.8b\n"
            "dup v7.8b,v22.b[7]\n"
            "smlal   v31.8h,  v15.8b,  v19.8b\n"

            "ld1  {v17.8b}, [%[b_ptr]], 8\n"

            "dup v8.8b,v22.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v16.8b\n"
            "dup v9.8b,v22.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v16.8b\n"
            "dup v10.8b,v22.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v16.8b\n"
            "dup v11.8b,v22.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v16.8b\n"
            "dup v12.8b,v22.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v16.8b\n"
            "dup v13.8b,v22.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v16.8b\n"
            "dup v14.8b,v22.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v16.8b\n"
            "dup v15.8b,v22.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v16.8b\n"

            "ld1  {v18.8b}, [%[b_ptr]], 8\n"
            "dup v0.8b,v23.b[0]\n"
            "smlal   v24.8h,  v8.8b,   v17.8b\n"
            "dup v1.8b,v23.b[1]\n"
            "smlal   v25.8h,  v9.8b,   v17.8b\n"
            "dup v2.8b,v23.b[2]\n"
            "smlal   v26.8h,  v10.8b,  v17.8b\n"
            "dup v3.8b,v23.b[3]\n"
            "smlal   v27.8h,  v11.8b,  v17.8b\n"
            "dup v4.8b,v23.b[4]\n"
            "smlal   v28.8h,  v12.8b,  v17.8b\n"
            "dup v5.8b,v23.b[5]\n"
            "smlal   v29.8h,  v13.8b,  v17.8b\n"
            "dup v6.8b,v23.b[6]\n"
            "smlal   v30.8h,  v14.8b,  v17.8b\n"
            "dup v7.8b,v23.b[7]\n"
            "smlal   v31.8h,  v15.8b,  v17.8b\n"

            "ld1  {v19.8b}, [%[b_ptr]], 8\n"
            "dup v8.8b,v23.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v18.8b\n"
            "dup v9.8b,v23.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v18.8b\n"
            "dup v10.8b,v23.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v18.8b\n"
            "dup v11.8b,v23.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v18.8b\n"
            "dup v12.8b,v23.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v18.8b\n"
            "dup v13.8b,v23.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v18.8b\n"
            "dup v14.8b,v23.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v18.8b\n"
            "dup v15.8b,v23.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v18.8b\n"

            "smlal   v24.8h,  v8.8b,   v19.8b\n"
            "smlal   v25.8h,  v9.8b,   v19.8b\n"
            "smlal   v26.8h,  v10.8b,  v19.8b\n"
            "smlal   v27.8h,  v11.8b,  v19.8b\n"
            "smlal   v28.8h,  v12.8b,  v19.8b\n"
            "smlal   v29.8h,  v13.8b,  v19.8b\n"
            "smlal   v30.8h,  v14.8b,  v19.8b\n"
            "smlal   v31.8h,  v15.8b,  v19.8b\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 1b\n"

            "3:\n"
            // Store back into memory
            STORE_C

            :
            [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
            [ is_first_k ] "+r"(is_first_k), [ K ] "+r"(K), [ LDC ] "+r"(LDC),
            [ outptr ] "+r"(outptr), [ m_remain ] "+r"(m_remain),
            [ n_remain ] "+r"(n_remain)  //,[tmp_packa1]"+r"(tmp_packa1),[tmp_packb1]"+r"(tmp_packb1)
            :
            : "cc", "memory", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void s4_kern_8x8(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 8;
    LDC = LDC * sizeof(int16_t);
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
// clang-format off

#define LOAD_C_8 \
    "ld1 {v24.8h}, [x0], #16\n"     \
    "ld1 {v25.8h}, [x1], #16\n"     \
    "ld1 {v26.8h}, [x2], #16\n"     \
    "ld1 {v27.8h}, [x3], #16\n"     \
    "ld1 {v28.8h}, [x4], #16\n"     \
    "ld1 {v29.8h}, [x5], #16\n"     \
    "ld1 {v30.8h}, [x6], #16\n"     \
    "ld1 {v31.8h}, [x7], #16\n"     \


#define STORE_C_8 \
    "st1 {v24.8h}, [x0], #16\n"     \
    "st1 {v25.8h}, [x1], #16\n"     \
    "st1 {v26.8h}, [x2], #16\n"     \
    "st1 {v27.8h}, [x3], #16\n"     \
    "st1 {v28.8h}, [x4], #16\n"     \
    "st1 {v29.8h}, [x5], #16\n"     \
    "st1 {v30.8h}, [x6], #16\n"     \
    "st1 {v31.8h}, [x7], #16\n"     \

// clang-format on
    register int16_t* outptr asm("x0") = output;
    asm volatile(
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"
            "add x4, x3, %x[LDC]\n"
            "add x5, x4, %x[LDC]\n"
            "add x6, x5, %x[LDC]\n"
            "add x7, x6, %x[LDC]\n"

            "cmp %w[is_first_k], #1\n"
            "beq 2f\n" LOAD_C_8
            "b 1f\n"

            "2:\n"  // Clear the C regs.
            "eor v24.16b, v24.16b, v24.16b\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"
            "eor v31.16b, v31.16b, v31.16b\n"
            // General loop.
            "ld1  {v20.16b}, [%[a_ptr]],#16\n"
            "ld1  {v21.16b}, [%[a_ptr]],#16\n"
            "PRFM PLDL1KEEP, [%[a_ptr], #512]\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #512]\n"
            "1:\n"
           // "ld1  {v20.16b}, [%[a_ptr]],#16\n"
           // "ld1  {v21.16b}, [%[a_ptr]],#16\n"
            "dup v0.8b,v20.b[0]\n"
            "ld1  {v22.16b}, [%[a_ptr]],#16\n"
            "dup v1.8b,v20.b[1]\n"
            "ld1  {v23.16b}, [%[a_ptr]],#16\n"
            "dup v2.8b,v20.b[2]\n"
            "ld1  {v16.8b}, [%[b_ptr]], 8\n"
            "dup v3.8b,v20.b[3]\n"
            "dup v4.8b,v20.b[4]\n"
            "ld1  {v17.8b}, [%[b_ptr]], 8\n"
            "dup v5.8b,v20.b[5]\n"
            "dup v6.8b,v20.b[6]\n"
            "dup v7.8b,v20.b[7]\n"
            

            "dup v8.8b,v20.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v16.8b\n"
            "dup v9.8b,v20.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v16.8b\n"
            "dup v10.8b,v20.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v16.8b\n"
            "dup v11.8b,v20.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v16.8b\n"
            "dup v12.8b,v20.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v16.8b\n"
            "dup v13.8b,v20.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v16.8b\n"
            "dup v14.8b,v20.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v16.8b\n"
            "dup v15.8b,v20.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v16.8b\n"

            "ld1  {v16.8b}, [%[b_ptr]], 8\n"

            "dup v0.8b,v21.b[0]\n"
            "smlal   v24.8h,  v8.8b,   v17.8b\n"
            "dup v1.8b,v21.b[1]\n"
            "smlal   v25.8h,  v9.8b,   v17.8b\n"
            "dup v2.8b,v21.b[2]\n"
            "smlal   v26.8h,  v10.8b,  v17.8b\n"
            "dup v3.8b,v21.b[3]\n"
            "smlal   v27.8h,  v11.8b,  v17.8b\n"
            "dup v4.8b,v21.b[4]\n"
            "smlal   v28.8h,  v12.8b,  v17.8b\n"
            "dup v5.8b,v21.b[5]\n"
            "smlal   v29.8h,  v13.8b,  v17.8b\n"
            "dup v6.8b,v21.b[6]\n"
            "smlal   v30.8h,  v14.8b,  v17.8b\n"
            "dup v7.8b,v21.b[7]\n"
            "smlal   v31.8h,  v15.8b,  v17.8b\n"

            "ld1  {v17.8b}, [%[b_ptr]], 8\n"

            "dup v8.8b,v21.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v16.8b\n"
            "dup v9.8b,v21.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v16.8b\n"
            "dup v10.8b,v21.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v16.8b\n"
            "dup v11.8b,v21.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v16.8b\n"
            "dup v12.8b,v21.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v16.8b\n"
            "dup v13.8b,v21.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v16.8b\n"
            "dup v14.8b,v21.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v16.8b\n"
            "dup v15.8b,v21.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v16.8b\n"

            "ld1  {v16.8b}, [%[b_ptr]], 8\n"
            "dup v0.8b,v22.b[0]\n"
            "smlal   v24.8h,  v8.8b,   v17.8b\n"
            "dup v1.8b,v22.b[1]\n"
            "smlal   v25.8h,  v9.8b,   v17.8b\n"
            "dup v2.8b,v22.b[2]\n"
            "smlal   v26.8h,  v10.8b,  v17.8b\n"
            "dup v3.8b,v22.b[3]\n"
            "smlal   v27.8h,  v11.8b,  v17.8b\n"
            "dup v4.8b,v22.b[4]\n"
            "smlal   v28.8h,  v12.8b,  v17.8b\n"
            "dup v5.8b,v22.b[5]\n"
            "smlal   v29.8h,  v13.8b,  v17.8b\n"
            "dup v6.8b,v22.b[6]\n"
            "smlal   v30.8h,  v14.8b,  v17.8b\n"
            "dup v7.8b,v22.b[7]\n"
            "smlal   v31.8h,  v15.8b,  v17.8b\n"

            "ld1  {v17.8b}, [%[b_ptr]], 8\n"

            "dup v8.8b,v22.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v16.8b\n"
            "dup v9.8b,v22.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v16.8b\n"
            "dup v10.8b,v22.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v16.8b\n"
            "dup v11.8b,v22.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v16.8b\n"
            "dup v12.8b,v22.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v16.8b\n"
            "dup v13.8b,v22.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v16.8b\n"
            "dup v14.8b,v22.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v16.8b\n"
            "dup v15.8b,v22.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v16.8b\n"

            "ld1  {v16.8b}, [%[b_ptr]], 8\n"
            "dup v0.8b,v23.b[0]\n"
            "smlal   v24.8h,  v8.8b,   v17.8b\n"
            "dup v1.8b,v23.b[1]\n"
            "smlal   v25.8h,  v9.8b,   v17.8b\n"
            "dup v2.8b,v23.b[2]\n"
            "smlal   v26.8h,  v10.8b,  v17.8b\n"
            "dup v3.8b,v23.b[3]\n"
            "smlal   v27.8h,  v11.8b,  v17.8b\n"
            "dup v4.8b,v23.b[4]\n"
            "smlal   v28.8h,  v12.8b,  v17.8b\n"
            "dup v5.8b,v23.b[5]\n"
            "smlal   v29.8h,  v13.8b,  v17.8b\n"
            "dup v6.8b,v23.b[6]\n"
            "smlal   v30.8h,  v14.8b,  v17.8b\n"
            "dup v7.8b,v23.b[7]\n"
            "smlal   v31.8h,  v15.8b,  v17.8b\n"

            "ld1  {v17.8b}, [%[b_ptr]], 8\n"
            "dup v8.8b,v23.b[8]\n"
            "smlal   v24.8h,  v0.8b,  v16.8b\n"
            "dup v9.8b,v23.b[9]\n"
            "smlal   v25.8h,  v1.8b,  v16.8b\n"
            "dup v10.8b,v23.b[10]\n"
            "smlal   v26.8h,  v2.8b,  v16.8b\n"
            "dup v11.8b,v23.b[11]\n"
            "smlal   v27.8h,  v3.8b,  v16.8b\n"
            "dup v12.8b,v23.b[12]\n"
            "smlal   v28.8h,  v4.8b,  v16.8b\n"
            "dup v13.8b,v23.b[13]\n"
            "smlal   v29.8h,  v5.8b,  v16.8b\n"
            "dup v14.8b,v23.b[14]\n"
            "smlal   v30.8h,  v6.8b,  v16.8b\n"
            "dup v15.8b,v23.b[15]\n"
            "smlal   v31.8h,  v7.8b,  v16.8b\n"

            "ld1  {v20.16b}, [%[a_ptr]],#16\n"
            "smlal   v24.8h,  v8.8b,   v17.8b\n"
            "smlal   v25.8h,  v9.8b,   v17.8b\n"
            "smlal   v26.8h,  v10.8b,  v17.8b\n"
            "smlal   v27.8h,  v11.8b,  v17.8b\n"
            "ld1  {v21.16b}, [%[a_ptr]],#16\n"
            "smlal   v28.8h,  v12.8b,  v17.8b\n"
            "smlal   v29.8h,  v13.8b,  v17.8b\n"
            "smlal   v30.8h,  v14.8b,  v17.8b\n"
            "smlal   v31.8h,  v15.8b,  v17.8b\n"
            //"ld1  {v20.16b}, [%[a_ptr]],#16\n"
            //"ld1  {v21.16b}, [%[a_ptr]],#16\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 1b\n"

            "3:\n"
            // Store back into memory
            STORE_C_8

            :
            [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
            [ is_first_k ] "+r"(is_first_k), [ K ] "+r"(K), [ LDC ] "+r"(LDC),
            [ outptr ] "+r"(outptr), [ m_remain ] "+r"(m_remain),
            [ n_remain ] "+r"(n_remain)  //,[tmp_packa1]"+r"(tmp_packa1),[tmp_packb1]"+r"(tmp_packb1)
            :
            : "cc", "memory", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}
//packa 
static void gemm_s4x4x16_8x8x8_transpose_pack(dt_int8* outptr, const dt_int8* inptr,
                                      int ldin, int y0, int ymax, int k0,
                                      int kmax) {
    int8_t zerobuff[8];
    int8_t tmpbuff0[8];
    int8_t tmpbuff1[8];
    int8_t tmpbuff2[8];
    int8_t tmpbuff3[8];
    int8_t tmpbuff4[8];
    int8_t tmpbuff5[8];
    int8_t tmpbuff6[8];
    int8_t tmpbuff7[8];
    std::memset(zerobuff, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff0, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff1, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff2, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff3, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff4, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff5, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff6, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff7, 0, sizeof(int8_t) * 8);
    ldin /= 2;
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
        int K = (kmax - k0)/2;
        //! read 4 * 16 in each row
        for (; K > 3; K -= 4) {
            transpose_4x8_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4,
                                         inptr5, inptr6, inptr7, outptr);
        }

        if (K > 0) {
            std::memcpy(tmpbuff0,inptr0,K);
            std::memcpy(tmpbuff1,inptr1,K);
            std::memcpy(tmpbuff2,inptr2,K);
            std::memcpy(tmpbuff3,inptr3,K);
            std::memcpy(tmpbuff4,inptr4,K);
            std::memcpy(tmpbuff5,inptr5,K);
            std::memcpy(tmpbuff6,inptr6,K);
            std::memcpy(tmpbuff7,inptr7,K);
            inptr0 = tmpbuff0;
            inptr1 = tmpbuff1;
            inptr2 = tmpbuff2;
            inptr3 = tmpbuff3;
            inptr4 = tmpbuff4;
            inptr5 = tmpbuff5;
            inptr6 = tmpbuff6;
            inptr7 = tmpbuff7;
            transpose_4x8_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4,
                                         inptr5, inptr6, inptr7, outptr);
        }
    }
    for (; y < ymax; y += 8) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        const int8_t* inptr6 = inptr5 + ldin;
        const int8_t* inptr7 = inptr6 + ldin;

        int K = (kmax - k0)/2;
        //! read 4 * 16 in each row
        for (; K > 3; K -= 4) {
            if (y + 7 >= ymax) {
                switch (y + 7 - ymax) {
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
            transpose_4x8_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4,
                                         inptr5, inptr6, inptr7, outptr);
        }
        if (K > 0) {
            if (y + 7 >= ymax) {
                switch (y + 7 - ymax) {
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

            std::memcpy(tmpbuff0,inptr0,K);
            std::memcpy(tmpbuff1,inptr1,K);
            std::memcpy(tmpbuff2,inptr2,K);
            std::memcpy(tmpbuff3,inptr3,K);
            std::memcpy(tmpbuff4,inptr4,K);
            std::memcpy(tmpbuff5,inptr5,K);
            std::memcpy(tmpbuff6,inptr6,K);
            std::memcpy(tmpbuff7,inptr7,K);
            inptr0 = tmpbuff0;
            inptr1 = tmpbuff1;
            inptr2 = tmpbuff2;
            inptr3 = tmpbuff3;
            inptr4 = tmpbuff4;
            inptr5 = tmpbuff5;
            inptr6 = tmpbuff6;
            inptr7 = tmpbuff7;
            transpose_4x8_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4,
                                         inptr5, inptr6, inptr7, outptr);
        }
    }
}
//packb
static void gemm_s4x4x16_8x8x8_interleave_pack(dt_int8* out, const dt_int8* in, int ldin,
                                      int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[8];
    int8_t tmpbuff0[8];
    int8_t tmpbuff1[8];
    int8_t tmpbuff2[8];
    int8_t tmpbuff3[8];
    int8_t tmpbuff4[8];
    int8_t tmpbuff5[8];
    int8_t tmpbuff6[8];
    int8_t tmpbuff7[8];
    std::memset(zerobuff, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff0, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff1, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff2, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff3, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff4, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff5, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff6, 0, sizeof(int8_t) * 8);
    std::memset(tmpbuff7, 0, sizeof(int8_t) * 8);
    const int ksize = kmax - k0;
    const int ksize8 = round_up(ksize, 8) * 8; //pack to int8 *8 packto s4 *4
    int8_t* outptr = out;
    int8_t* outptr_interleave = nullptr;

    int k = k0;
    ldin /= 2;
    xmax = xmax / 2;
    for (; k + 7 < kmax; k += 8) {
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
        int8_t* outptr_inner = outptr;
        for (; x + 3 < xmax; x += 4) {
            outptr_interleave = outptr_inner;
            interleave_8x4_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr_interleave);
            outptr_inner += ksize8;
        }

        if (x < xmax) {
            int remainx = xmax - x;
            std::memcpy(tmpbuff0,inptr0,remainx);
            std::memcpy(tmpbuff1,inptr1,remainx);
            std::memcpy(tmpbuff2,inptr2,remainx);
            std::memcpy(tmpbuff3,inptr3,remainx);
            std::memcpy(tmpbuff4,inptr4,remainx);
            std::memcpy(tmpbuff5,inptr5,remainx);
            std::memcpy(tmpbuff6,inptr6,remainx);
            std::memcpy(tmpbuff7,inptr7,remainx);
            inptr0 = tmpbuff0;
            inptr1 = tmpbuff1;
            inptr2 = tmpbuff2;
            inptr3 = tmpbuff3;
            inptr4 = tmpbuff4;
            inptr5 = tmpbuff5;
            inptr6 = tmpbuff6;
            inptr7 = tmpbuff7;

            outptr_interleave = outptr_inner;
            interleave_8x4_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr_interleave);
            outptr_inner += ksize8;
        }
        outptr += 64;
    }
    if (k < kmax) {
        const int8_t* inptr0 = in + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        const int8_t* inptr6 = inptr5 + ldin;
        const int8_t* inptr7 = inptr6 + ldin;
        int k_remain = kmax - k - 1;
        int x = x0;
        int8_t* outptr_inner = outptr;
        for (; x + 3 < xmax; x += 4) {
            switch (k_remain) {
                case 0:
                    inptr1 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 1:
                    inptr2 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 2:
                    inptr3 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 3:
                    inptr4 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 4:
                    inptr5 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 5:
                    inptr6 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 6:
                    inptr7 = zerobuff;
                    break;
                default:
                    megdnn_assert(0);
                    break;
            }
            outptr_interleave = outptr_inner;
            interleave_8x4_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr_interleave);
            outptr_inner += ksize8;
        }
        if (x < xmax) {
            switch (k_remain) {
                case 0:
                    inptr1 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 1:
                    inptr2 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 2:
                    inptr3 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 3:
                    inptr4 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 4:
                    inptr5 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 5:
                    inptr6 = zerobuff;
                    MEGDNN_FALLTHRU;
                case 6:
                    inptr7 = zerobuff;
                    break;
                default:
                    megdnn_assert(0);
                    break;
            }
            int remainx = xmax - x;
            outptr_interleave = outptr_inner;
            std::memcpy(tmpbuff0,inptr0,remainx);
            std::memcpy(tmpbuff1,inptr1,remainx);
            std::memcpy(tmpbuff2,inptr2,remainx);
            std::memcpy(tmpbuff3,inptr3,remainx);
            std::memcpy(tmpbuff4,inptr4,remainx);
            std::memcpy(tmpbuff5,inptr5,remainx);
            std::memcpy(tmpbuff6,inptr6,remainx);
            std::memcpy(tmpbuff7,inptr7,remainx);
            inptr0 = tmpbuff0;
            inptr1 = tmpbuff1;
            inptr2 = tmpbuff2;
            inptr3 = tmpbuff3;
            inptr4 = tmpbuff4;
            inptr5 = tmpbuff5;
            inptr6 = tmpbuff6;
            inptr7 = tmpbuff7;

            outptr_interleave = outptr_inner;
            interleave_8x4_1_b_with_shift(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr_interleave);
            outptr_inner += ksize8;
        }
    }
}

}  // namespace matmul_4x4x16
}  // namespace aarch64
}  // namespace megdnn


// vim: syntax=cpp.doxygen
