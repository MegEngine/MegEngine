/**
 * \file dnn/src/aarch64/matrix_mul/int8x8x16/kernel_mk4_8x8x8.h
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
namespace matmul_mk4_8x8x8 {


/**
 * Overview of register layout:
 *
 * A 8x8 cell of Lhs is stored in 8bit in v16-v17
 * B 8x8 cell of Rhs is stored in 8bit in v0-v15, v20-v23
 * C 8x8 block of accumulators is stored in 16bit in v24-v31
 *
 *                     +---------------------------------+
 *                     |  v0 ------------------------ v7 |
 *                     |  v8 ------------------------ v15|
 *                Rhs  +---------------------------------+
 *    Lhs              |                                 |
 *  +--------+ - - - - +---------------------------------+
 *  | v16 |             | v24                            |
 *  | v17 |             | v25                            |
 *  | v16 |             | v26                            |
 *  | v17 |             | v27                            |
 *  | v16 |             | v28                            |
 *  | v17 |             | v29                            |
 *  | v16 |             | v30                            |
 *  | v17 |             | v31                            |    
 *  +--------+ - - - - +---------------------------------+
 *
 *                            Accumulator
 */
static void kern_8x8(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 8;
    LDC = LDC * sizeof(int16_t);
    const int8_t* a_ptr = packB;//packA;
    const int8_t* b_ptr = packA;//packB;
// clang-format off
#define LOAD_C_8 \
    "ld1 {v0.8h}, [x0], #16\n"     \
    "ld1 {v1.8h}, [x0], #16\n"     \
    "ld1 {v2.8h}, [x0], #16\n"     \
    "ld1 {v3.8h}, [x0], #16\n"     \
    "ld1 {v4.8h}, [x1], #16\n"     \
    "ld1 {v5.8h}, [x1], #16\n"     \
    "ld1 {v6.8h}, [x1], #16\n"     \
    "ld1 {v7.8h}, [x1], #16\n"     \


#define STORE_C_8 \
    "st1 {v0.8h}, [x0], #16\n"     \
    "st1 {v1.8h}, [x0], #16\n"     \
    "st1 {v2.8h}, [x0], #16\n"     \
    "st1 {v3.8h}, [x0], #16\n"     \
    "st1 {v4.8h}, [x1], #16\n"     \
    "st1 {v5.8h}, [x1], #16\n"     \
    "st1 {v6.8h}, [x1], #16\n"     \
    "st1 {v7.8h}, [x1], #16\n"     \

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            "add x1, x0, %x[LDC]\n"

            "eor v24.16b, v24.16b, v24.16b\n"
            "PRFM PLDL1KEEP, [%[a_ptr], #512]\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #512]\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "ld1  {v20.16b}, [%[a_ptr]],#16\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "ld1  {v21.16b}, [%[a_ptr]],#16\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"
            "eor v31.16b, v31.16b, v31.16b\n"
            // General loop.
            "1:\n"
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

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 1b\n"

            "cmp %w[is_first_k], #1\n"
            "beq 2f\n" LOAD_C_8
            "b 3f                         \n"
            "2:                           \n"
            "eor v0.16b, v0.16b, v0.16b\n"
            "eor v1.16b, v1.16b, v1.16b\n"
            "eor v2.16b, v2.16b, v2.16b\n"
            "eor v3.16b, v3.16b, v3.16b\n"
            "eor v4.16b, v4.16b, v4.16b\n"
            "eor v5.16b, v5.16b, v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"
            "3:\n"
            "zip1 v8.2d,   v24.2d,  v25.2d\n"
            "zip2 v9.2d,   v24.2d,  v25.2d\n"
            "zip1 v10.2d,  v26.2d,  v27.2d\n"
            "zip2 v11.2d,  v26.2d,  v27.2d\n"
            "zip1 v12.2d,  v28.2d,  v29.2d\n"
            "zip2 v13.2d,  v28.2d,  v29.2d\n"
            "zip1 v14.2d,  v30.2d,  v31.2d\n"
            "zip2 v15.2d,  v30.2d,  v31.2d\n"
            "add v0.8h,    v0.8h,    v8.8h\n"
            "add v1.8h,    v1.8h,   v10.8h\n"
            "add v2.8h,    v2.8h,   v12.8h\n"
            "add v3.8h,    v3.8h,   v14.8h\n"
            "add v4.8h,    v4.8h,    v9.8h\n"
            "add v5.8h,    v5.8h,   v11.8h\n"
            "add v6.8h,    v6.8h,   v13.8h\n"
            "add v7.8h,    v7.8h,   v15.8h\n"

            // Store back into memory
            STORE_C_8

            :
            [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
            [ is_first_k ] "+r"(is_first_k), [ K ] "+r"(K), [ LDC ] "+r"(LDC),
            [ outptr ] "+r"(outptr), [ m_remain ] "+r"(m_remain),
            [ n_remain ] "+r"(n_remain)
            :
            : "cc", "memory", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31");
// clang-format on
}

static void kern_8x8_remain(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 8;
    LDC = LDC * sizeof(int16_t);
    const int8_t* a_ptr = packB;
    const int8_t* b_ptr = packA;
//  clang-format off
    register int16_t* outptr asm("x0") = output;
    asm volatile(
            "add x1, x0, %x[LDC]\n"

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

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 1b\n"

            "cmp %w[is_first_k], #1\n"
            "beq 2f\n" 
            "cmp  %x[m_remain], #8     \n"
            "beq  8f                   \n"
            "cmp  %x[m_remain], #4     \n"
            "beq  9f                   \n"
            "8:                        \n"
            "cmp %x[n_remain],       #8\n"
            "beq  200f                 \n"
            "cmp %x[n_remain],       #7\n"
            "beq  201f                 \n"
            "cmp %x[n_remain],       #6\n"
            "beq  202f                 \n"
            "cmp %x[n_remain],       #5\n"
            "beq  203f                 \n"
            "cmp %x[n_remain],       #4\n"
            "beq  204f                 \n"
            "cmp %x[n_remain],       #3\n"
            "beq  205f                 \n"
            "cmp %x[n_remain],       #2\n"
            "beq  206f                 \n"
            "cmp %x[n_remain],       #1\n"
            "beq  207f                 \n"
            "200:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.8h}, [x0], #16\n"
            "ld1 {v3.8h}, [x0], #16\n"
            "ld1 {v4.8h}, [x1], #16\n"
            "ld1 {v5.8h}, [x1], #16\n"
            "ld1 {v6.8h}, [x1], #16\n"
            "ld1 {v7.8h}, [x1], #16\n"
            "b 3f                   \n"
            "201:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.8h}, [x0], #16\n"
            "ld1 {v3.d}[0], [x0], #8\n"
            "ld1 {v4.8h}, [x1], #16\n"
            "ld1 {v5.8h}, [x1], #16\n"
            "ld1 {v6.8h}, [x1], #16\n"
            "ld1 {v7.d}[0], [x1], #8\n"
            "b 3f                   \n"
            "202:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.8h}, [x0], #16\n"
            "ld1 {v4.8h}, [x1], #16\n"
            "ld1 {v5.8h}, [x1], #16\n"
            "ld1 {v6.8h}, [x1], #16\n"
            "b 3f                   \n"
            "203:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.d}[0], [x0], #8\n"
            "ld1 {v4.8h}, [x1], #16\n"
            "ld1 {v5.8h}, [x1], #16\n"
            "ld1 {v6.d}[0], [x1], #8\n"
            "b 3f                   \n"
            "204:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v4.8h}, [x1], #16\n"
            "ld1 {v5.8h}, [x1], #16\n"
            "b 3f                   \n"
            "205:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.d}[0], [x0], #8\n"
            "ld1 {v4.8h}, [x1], #16\n"
            "ld1 {v5.d}[0], [x1], #8\n"
            "b 3f                   \n"
            "206:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v4.8h}, [x1], #16\n"
            "b 3f                   \n"
            "207:                      \n"
            "ld1 {v0.d}[0], [x0], #8\n"
            "ld1 {v4.d}[0], [x1], #8\n"
            "b 3f                   \n"
            "9:                        \n"
            "cmp %x[n_remain],       #8\n"
            "beq  300f                 \n"
            "cmp %x[n_remain],       #7\n"
            "beq  301f                 \n"
            "cmp %x[n_remain],       #6\n"
            "beq  302f                 \n"
            "cmp %x[n_remain],       #5\n"
            "beq  303f                 \n"
            "cmp %x[n_remain],       #4\n"
            "beq  304f                 \n"
            "cmp %x[n_remain],       #3\n"
            "beq  305f                 \n"
            "cmp %x[n_remain],       #2\n"
            "beq  306f                 \n"
            "cmp %x[n_remain],       #1\n"
            "beq  307f                 \n"
            "300:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.8h}, [x0], #16\n"
            "ld1 {v3.8h}, [x0], #16\n"
            "b 3f                   \n"
            "301:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.8h}, [x0], #16\n"
            "ld1 {v3.d}[0], [x0], #8\n"
            "b 3f                   \n"
            "302:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.8h}, [x0], #16\n"
            "b 3f                   \n"
            "303:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "ld1 {v2.d}[0], [x0], #8\n"
            "b 3f                   \n"
            "304:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.8h}, [x0], #16\n"
            "b 3f                   \n"
            "305:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "ld1 {v1.d}[0], [x0], #8\n"
            "b 3f                   \n"
            "306:                      \n"
            "ld1 {v0.8h}, [x0], #16\n"
            "b 3f                   \n"
            "307:                      \n"
            "ld1 {v0.d}[0], [x0], #8\n"
            "b 3f                   \n"
            "2:                           \n"
            "eor v0.16b, v0.16b, v0.16b\n"
            "eor v1.16b, v1.16b, v1.16b\n"
            "eor v2.16b, v2.16b, v2.16b\n"
            "eor v3.16b, v3.16b, v3.16b\n"
            "eor v4.16b, v4.16b, v4.16b\n"
            "eor v5.16b, v5.16b, v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"
            "3:\n"
            "zip1 v8.2d,   v24.2d,  v25.2d\n"
            "zip1 v10.2d,  v26.2d,  v27.2d\n"
            "add v0.8h,  v0.8h,  v8.8h    \n"
            "zip1 v12.2d,  v28.2d,  v29.2d\n"
            "add v1.8h,  v1.8h,  v10.8h   \n"
            "zip1 v14.2d,  v30.2d,  v31.2d\n"
            "add v2.8h,  v2.8h,  v12.8h   \n"
            "add v3.8h,  v3.8h,  v14.8h   \n"
            "zip2 v9.2d,   v24.2d,  v25.2d\n"
            "zip2 v11.2d,  v26.2d,  v27.2d  \n"
            "add v4.8h,  v4.8h,  v9.8h      \n"
            "zip2 v13.2d,  v28.2d,  v29.2d  \n"
            "add v5.8h,  v5.8h,  v11.8h     \n"
            "zip2 v15.2d,  v30.2d,  v31.2d  \n"
            "add v6.8h,  v6.8h,  v13.8h     \n"
            "add v7.8h,  v7.8h,  v15.8h     \n"
//save to memory
            "cmp  %x[m_remain], #8     \n"
            "beq  4f                   \n"
            "cmp  %x[m_remain], #4     \n"
            "beq  5f                   \n"
            "4:                        \n"
            "cmp %x[n_remain],       #8\n"
            "beq  100f                 \n"
            "cmp %x[n_remain],       #7\n"
            "beq  101f                 \n"
            "cmp %x[n_remain],       #6\n"
            "beq  102f                 \n"
            "cmp %x[n_remain],       #5\n"
            "beq  103f                 \n"
            "cmp %x[n_remain],       #4\n"
            "beq  104f                 \n"
            "cmp %x[n_remain],       #3\n"
            "beq  105f                 \n"
            "cmp %x[n_remain],       #2\n"
            "beq  106f                 \n"
            "cmp %x[n_remain],       #1\n"
            "beq  107f                 \n"
            "100:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.8h}, [x0], #16\n"
            "st1 {v3.8h}, [x0], #16\n"
            "st1 {v4.8h}, [x1], #16\n"
            "st1 {v5.8h}, [x1], #16\n"
            "st1 {v6.8h}, [x1], #16\n"
            "st1 {v7.8h}, [x1], #16\n"
            "b 1000f                   \n"
            "101:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.8h}, [x0], #16\n"
            "st1 {v3.d}[0], [x0], #8\n"
            "st1 {v4.8h}, [x1], #16\n"
            "st1 {v5.8h}, [x1], #16\n"
            "st1 {v6.8h}, [x1], #16\n"
            "st1 {v7.d}[0], [x1], #8\n"
            "b 1000f                   \n"
            "102:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.8h}, [x0], #16\n"
            "st1 {v4.8h}, [x1], #16\n"
            "st1 {v5.8h}, [x1], #16\n"
            "st1 {v6.8h}, [x1], #16\n"
            "b 1000f                   \n"
            "103:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.d}[0], [x0], #8\n"
            "st1 {v4.8h}, [x1], #16\n"
            "st1 {v5.8h}, [x1], #16\n"
            "st1 {v6.d}[0], [x1], #8\n"
            "b 1000f                   \n"
            "104:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v4.8h}, [x1], #16\n"
            "st1 {v5.8h}, [x1], #16\n"
            "b 1000f                   \n"
            "105:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.d}[0], [x0], #8\n"
            "st1 {v4.8h}, [x1], #16\n"
            "st1 {v5.d}[0], [x1], #8\n"
            "b 1000f                   \n"
            "106:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v4.8h}, [x1], #16\n"
            "b 1000f                   \n"
            "107:                      \n"
            "st1 {v0.d}[0], [x0], #8\n"
            "st1 {v4.d}[0], [x1], #8\n"
            "b 1000f                   \n"
            "5:                        \n"
            "cmp %x[n_remain],       #8\n"
            "beq  200f                 \n"
            "cmp %x[n_remain],       #7\n"
            "beq  201f                 \n"
            "cmp %x[n_remain],       #6\n"
            "beq  202f                 \n"
            "cmp %x[n_remain],       #5\n"
            "beq  203f                 \n"
            "cmp %x[n_remain],       #4\n"
            "beq  204f                 \n"
            "cmp %x[n_remain],       #3\n"
            "beq  205f                 \n"
            "cmp %x[n_remain],       #2\n"
            "beq  206f                 \n"
            "cmp %x[n_remain],       #1\n"
            "beq  207f                 \n"
            "200:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.8h}, [x0], #16\n"
            "st1 {v3.8h}, [x0], #16\n"
            "b 1000f                   \n"
            "201:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.8h}, [x0], #16\n"
            "st1 {v3.d}[0], [x0], #8\n"
            "b 1000f                   \n"
            "202:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.8h}, [x0], #16\n"
            "b 1000f                   \n"
            "203:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "st1 {v2.d}[0], [x0], #8\n"
            "b 1000f                   \n"
            "204:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.8h}, [x0], #16\n"
            "b 1000f                   \n"
            "205:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "st1 {v1.d}[0], [x0], #8\n"
            "b 1000f                   \n"
            "206:                      \n"
            "st1 {v0.8h}, [x0], #16\n"
            "b 1000f                   \n"
            "207:                      \n"
            "st1 {v0.d}[0], [x0], #8\n"
            "b 1000f                   \n"

            "1000:                     \n"
            :
            [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
            [ is_first_k ] "+r"(is_first_k), [ K ] "+r"(K), [ LDC ] "+r"(LDC),
            [ outptr ] "+r"(outptr), [ m_remain ] "+r"(m_remain),
            [ n_remain ] "+r"(n_remain)
            :
            : "cc", "memory", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31");
// clang-format on

#undef LOAD_C_8
#undef STORE_C_8
}


static void kern_4x8(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 8;
    LDC = LDC * sizeof(int16_t);
    const int8_t* a_ptr = packB;//packA;
    const int8_t* b_ptr = packA;//packB;
// clang-format off
#define LOAD_C_4 \
    "ld1 {v0.8h}, [x0], #16\n"     \
    "ld1 {v1.8h}, [x0], #16\n"     \
    "ld1 {v2.8h}, [x0], #16\n"     \
    "ld1 {v3.8h}, [x0], #16\n"     \


#define STORE_C_4 \
    "st1 {v0.8h}, [x0], #16\n"     \
    "st1 {v1.8h}, [x0], #16\n"     \
    "st1 {v2.8h}, [x0], #16\n"     \
    "st1 {v3.8h}, [x0], #16\n"     \

    register int16_t* outptr asm("x0") = output;
    asm volatile(

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

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 1b\n"

            "cmp %w[is_first_k], #1\n"
            "beq 2f\n" LOAD_C_4
            "b 3f                         \n"
            "2:                           \n"
            "eor v0.16b, v0.16b, v0.16b\n"
            "eor v1.16b, v1.16b, v1.16b\n"
            "eor v2.16b, v2.16b, v2.16b\n"
            "eor v3.16b, v3.16b, v3.16b\n"
            "eor v4.16b, v4.16b, v4.16b\n"
            "eor v5.16b, v5.16b, v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"
            "3:\n"
            "zip1 v8.2d,   v24.2d,  v25.2d\n"
            "zip1 v10.2d,  v26.2d,  v27.2d\n"
            "add v0.8h,  v0.8h,  v8.8h\n"
            "zip1 v12.2d,  v28.2d,  v29.2d\n"
            "add v1.8h,  v1.8h,  v10.8h\n"
            "zip1 v14.2d,  v30.2d,  v31.2d\n"
            "add v2.8h,  v2.8h,  v12.8h\n"
            "add v3.8h,  v3.8h,  v14.8h\n"

            // Store back into memory
            STORE_C_4

            :
            [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
            [ is_first_k ] "+r"(is_first_k), [ K ] "+r"(K), [ LDC ] "+r"(LDC),
            [ outptr ] "+r"(outptr), [ m_remain ] "+r"(m_remain),
            [ n_remain ] "+r"(n_remain)
            :
            : "cc", "memory", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31");
// clang-format on
#undef LOAD_C_4
#undef STORE_C_4
}
static void kern_4x8_remain(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 8;
    LDC = LDC * sizeof(int16_t);
    const int8_t* a_ptr = packB;//packA;
    const int8_t* b_ptr = packA;//packB;
// clang-format off
    register int16_t* outptr asm("x0") = output;
    asm volatile(

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

            "subs %w[K], %w[K], #1     \n"
            "cbnz %w[K], 1b            \n"
            "cmp %w[is_first_k], #1    \n"
            "beq 2f                    \n"
            "cmp %w[n_remain],#7       \n"
            "beq 200f                  \n"
            "cmp %w[n_remain],#6       \n"
            "beq 201f                  \n"
            "cmp %w[n_remain],#5       \n"
            "beq 202f                  \n"
            "cmp %w[n_remain],#4       \n"
            "beq 203f                  \n"
            "cmp %w[n_remain],#3       \n"
            "beq 204f                  \n"
            "cmp %w[n_remain],#2       \n"
            "beq 205f                  \n"
            "cmp %w[n_remain],#1       \n"
            "beq 206f                  \n"
            "200:                      \n"
            "ld1 {v0.8h}, [x0],#16     \n"
            "ld1 {v1.8h}, [x0],#16     \n"
            "ld1 {v2.8h}, [x0],#16     \n"
            "ld1 {v3.d}[0], [x0],#8    \n"
            "b 3f                      \n"
            "201:                      \n"
            "ld1 {v0.8h}, [x0],#16     \n"
            "ld1 {v1.8h}, [x0],#16     \n"
            "ld1 {v2.8h}, [x0],#16     \n"
            "b 3f                      \n"
            "202:                      \n"
            "ld1 {v0.8h}, [x0],#16     \n"
            "ld1 {v1.8h}, [x0],#16     \n"
            "ld1 {v2.d}[0], [x0],#8    \n"
            "b 3f                      \n"
            "203:                      \n"
            "ld1 {v0.8h}, [x0],#16     \n"
            "ld1 {v1.8h}, [x0],#16     \n"
            "b 3f                      \n"
            "204:                      \n"
            "ld1 {v0.8h}, [x0],#16     \n"
            "ld1 {v1.d}[0], [x0],#8    \n"
            "b 3f                      \n"
            "205:                      \n"
            "ld1 {v0.8h}, [x0],#16     \n"
            "b 3f                      \n"
            "206:                      \n"
            "ld1 {v0.d}[0], [x0],#8    \n"
            "b 3f                      \n"
            "2:                        \n"
            "eor v0.16b, v0.16b, v0.16b\n"
            "eor v1.16b, v1.16b, v1.16b\n"
            "eor v2.16b, v2.16b, v2.16b\n"
            "eor v3.16b, v3.16b, v3.16b\n"
            "eor v4.16b, v4.16b, v4.16b\n"
            "eor v5.16b, v5.16b, v5.16b\n"
            "eor v6.16b, v6.16b, v6.16b\n"
            "eor v7.16b, v7.16b, v7.16b\n"
            "3:                           \n"
            "zip1 v8.2d,  v24.2d, v25.2d\n"
            "zip1 v10.2d, v26.2d, v27.2d\n"
            "add  v0.8h,  v0.8h,  v8.8h \n"
            "zip1 v12.2d, v28.2d, v29.2d\n"
            "add  v1.8h,  v1.8h,  v10.8h\n"
            "zip1 v14.2d, v30.2d, v31.2d\n"
            "add  v2.8h,  v2.8h,  v12.8h\n"
            "add  v3.8h,  v3.8h,  v14.8h\n"

            // Store back into memory
            "cmp %w[n_remain],#7      \n"
            "beq 100f                 \n"
            "cmp %w[n_remain],#6      \n"
            "beq 101f                 \n"
            "cmp %w[n_remain],#5      \n"
            "beq 102f                 \n"
            "cmp %w[n_remain],#4      \n"
            "beq 103f                 \n"
            "cmp %w[n_remain],#3      \n"
            "beq 104f                 \n"
            "cmp %w[n_remain],#2      \n"
            "beq 105f                 \n"
            "cmp %w[n_remain],#1      \n"
            "beq 106f                 \n"
            "100:                     \n"
            "st1 {v0.8h}, [x0],#16    \n"
            "st1 {v1.8h}, [x0],#16    \n"
            "st1 {v2.8h}, [x0],#16    \n"
            "st1 {v3.d}[0], [x0],#8   \n"
            "b 1000f                  \n"
            "101:                     \n"
            "st1 {v0.8h}, [x0],#16    \n"
            "st1 {v1.8h}, [x0],#16    \n"
            "st1 {v2.8h}, [x0],#16    \n"
            "b 1000f                  \n"
            "102:                     \n"
            "st1 {v0.8h}, [x0],#16    \n"
            "st1 {v1.8h}, [x0],#16    \n"
            "st1 {v2.d}[0], [x0],#8   \n"
            "b 1000f                  \n"
            "103:                     \n"
            "st1 {v0.8h}, [x0],#16    \n"
            "st1 {v1.8h}, [x0],#16    \n"
            "b 1000f                  \n"
            "104:                     \n"
            "st1 {v0.8h}, [x0],#16    \n"
            "st1 {v1.d}[0], [x0],#8   \n"
            "b 1000f                  \n"
            "105:                     \n"
            "st1 {v0.8h}, [x0],#16    \n"
            "b 1000f                  \n"
            "106:                     \n"
            "st1 {v0.d}[0], [x0],#8   \n"
            "b 1000f                  \n"
            "1000:                    \n"
            :
            [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr),
            [ is_first_k ] "+r"(is_first_k), [ K ] "+r"(K), [ LDC ] "+r"(LDC),
            [ outptr ] "+r"(outptr), [ m_remain ] "+r"(m_remain),
            [ n_remain ] "+r"(n_remain)
            :
            : "cc", "memory", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31");
// clang-format on
#undef LOAD_C_4
#undef STORE_C_4
}


//! pack to icxoc
//! (M/4,K/4,4(K),4(M)) pack to (M/8,k/8,8(K_ic_0~3_ic_4~7),8(M_oc0~3_OC_4~7))
//! if M K is not times of 8,pack 0 instead 
static void gemm_s8x8x16_mk4_8x8x8_pack_A(dt_int8* outptr,
                                          const dt_int8* inptr, int ldin,
                                          int m0, int mmax, int k0, int kmax) {
    megdnn_assert(m0 % 4 == 0 && mmax % 4 == 0, "M must be time of 4");
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    constexpr int pack_m = 8;
    constexpr int pack_k = 8;
    constexpr int pack_size = 4;
    int8_t tmpbuff0[pack_m * pack_size] = {0};
    int8_t tmpbuff1[pack_m * pack_size] = {0};
    int8_t zerobuff[pack_m * pack_size] = {0};
    const int m_size = mmax - m0;
    const int m_end = m_size / pack_m * pack_m + m0;
    int remain_m = mmax - m_end;

    for (int m_idx = m0; m_idx < m_end; m_idx += pack_m) {
        const int8_t* inptr0 = inptr + m_idx / pack_size * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        int k_idx = k0;
        for ( ; k_idx + 7 < kmax; k_idx += pack_k) {
            interleave_8x8_mk4_b(inptr0,inptr1,outptr);
        }

        if (k_idx < kmax) {
            memcpy(tmpbuff0, inptr0, sizeof(int8_t) * (kmax - k_idx) * pack_size);
            memcpy(tmpbuff1, inptr1, sizeof(int8_t) * (kmax - k_idx) * pack_size);
            inptr0 = tmpbuff0;
            inptr1 = tmpbuff1;
            interleave_8x8_mk4_b(inptr0, inptr1, outptr);
        }
    }
    int m_idx = m_end;
    if (remain_m == 4) {
        const int8_t* inptr0 = inptr + m_idx / pack_size * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        int k_idx = k0;
        for ( ; k_idx + 7 < kmax; k_idx += pack_k) {
            inptr1 = zerobuff;
            interleave_8x8_mk4_b(inptr0,inptr1,outptr);
        }

        if (k_idx < kmax) {
            memcpy(tmpbuff0, inptr0, sizeof(int8_t) * (kmax - k_idx) * pack_size);
            inptr0 = tmpbuff0;
            inptr1 = zerobuff;
            interleave_8x8_mk4_b(inptr0, inptr1, outptr);
        }
    }
}
//! pack to nxic
//! (K/4,N,4) pack to K/8,N,8(ic0~7) ,K is not times of 8 ,pack 0 instead.
static void gemm_s8x8x16_mk4_8x8x8_pack_B(dt_int8* out, const dt_int8* in,
                                          int ldin, int n0, int nmax, int k0,
                                          int kmax) {
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");

    constexpr int pack_n = 8;
    constexpr int pack_k = 8;
    constexpr int pack_size = 4;
    int8_t tmpbuff0[pack_n * pack_size] = {0};
    int8_t tmpbuff1[pack_n * pack_size] = {0};
    int8_t zerobuff[pack_n * pack_size] = {0};
    const int ksize = round_up<int>((kmax - k0),8);
    const int nsize = nmax - n0;
    const int n_end = nsize / pack_n * pack_n + n0;
    const int remain_n = nsize % pack_n;
    int output_stride = ksize * pack_n;
    int8_t* outptr_base = out;
    int k_idx = k0;
    for ( ; k_idx + 7 < kmax; k_idx += pack_k) {
        const int8_t* inptr0 = in + k_idx / pack_size * ldin + n0 * pack_size;
        const int8_t* inptr1 = inptr0 + ldin;
        prefetch_3x(inptr0);
        prefetch_3x(inptr1);

        auto outptr = outptr_base;
        for (int n_idx = n0; n_idx < n_end; n_idx += pack_n) {
            transpose_8x8_mk4_b(inptr0, inptr1, outptr);
           outptr += output_stride;
        }
        if (remain_n > 0) {
            memcpy(tmpbuff0, inptr0, sizeof(int8_t) * remain_n * pack_size);
            memcpy(tmpbuff1, inptr1, sizeof(int8_t) * remain_n * pack_size);
            inptr0 = tmpbuff0;
            inptr1 = tmpbuff1;
            transpose_8x8_mk4_b(inptr0, inptr1, outptr);
            outptr += output_stride;
        }
        outptr_base += pack_n * pack_k;
    }
    
    if(k_idx < kmax){
        const int8_t* inptr0 = in + k_idx / pack_size * ldin + n0 * pack_size;
        const int8_t* inptr1 = nullptr;
        prefetch_3x(inptr0);
        auto outptr = outptr_base;
        for (int n_idx = n0; n_idx < n_end; n_idx += pack_n) {
            inptr1 = zerobuff;
            transpose_8x8_mk4_b(inptr0, inptr1, outptr);
            outptr += output_stride;
        }
        if (remain_n > 0) {
            memcpy(tmpbuff0, inptr0, sizeof(int8_t) * remain_n * pack_size);
            inptr1 = zerobuff;
            inptr0 = tmpbuff0;
            transpose_8x8_mk4_b(inptr0, inptr1, outptr);
            outptr += output_stride;
        }
        outptr_base += pack_n * pack_size;
    }
}

}  // namespace matmul_mk4_16x12x4_a53
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
