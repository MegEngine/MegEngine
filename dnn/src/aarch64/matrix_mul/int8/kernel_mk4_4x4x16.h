/**
 * \file dnn/src/aarch64/matrix_mul/int8/kernel_mk4_4x4x16.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <cstring>
#if !(__ARM_FEATURE_DOTPROD)
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_mk4_4x4x16 {

/**
 * Overview of register layout:
 *
 * A 16x4 cell of Rhs is stored in 8bit in v0-q3.
 * B 16x4 cell of Lhs is stored in 8bit in q4-q7
 * C 8x16 block of accumulators is stored in 8bit in q8--q31.
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
                     int32_t* output, bool is_first_k) {
    K = div_ceil(K, 16);
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    asm volatile(
            // load accumulator C
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "eor v16.16b,  v16.16b,  v16.16b\n"
            "eor v17.16b,  v17.16b,  v17.16b\n"
            "eor v18.16b,  v18.16b,  v18.16b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "eor v19.16b,  v19.16b,  v19.16b\n"
            "eor v20.16b,  v19.16b,  v19.16b\n"
            "eor v21.16b,  v19.16b,  v19.16b\n"
            "ld1 {v4.16b, v5.16b}, [%[b_ptr]], #32\n"
            "eor v22.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[a_ptr], #32]\n"
            "eor v23.16b,  v19.16b,  v19.16b\n"
            "eor v24.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #32]\n"
            "eor v25.16b,  v19.16b,  v19.16b\n"
            "eor v26.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #64]\n"
            "eor v27.16b,  v19.16b,  v19.16b\n"
            "eor v28.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[a_ptr], #64]\n"
            "eor v29.16b,  v19.16b,  v19.16b\n"
            "eor v30.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #128]\n"
            "eor v31.16b,  v19.16b,  v19.16b\n"

            //! if K==1 jump to compute last K
            "cmp %w[k], #2\n"
            "beq 2f\n"
            "blt 3f\n"

            //! K>2
            "1:\n"
            //! First k
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "ld1 {v4.16b}, [%[b_ptr]], #16\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "ld1 {v5.16b}, [%[b_ptr]], #16\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "sadalp v25.4s, v9.8h\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v28.4s, v12.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"

            //! Second k
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "sadalp v27.4s, v11.8h\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "sadalp v30.4s, v14.8h\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"
            "sadalp v31.4s, v15.8h\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "sub %w[k], %w[k], #2\n"
            "cmp %w[k], #2\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "ld1 {v4.16b}, [%[b_ptr]], #16\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "ld1 {v5.16b}, [%[b_ptr]], #16\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v25.4s, v9.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "sadalp v28.4s, v12.8h\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"

            "sadalp v27.4s, v11.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"

            "bgt 1b\n"
            "blt 3f\n"

            //! K==2
            "2:\n"
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "ld1 {v4.16b}, [%[b_ptr]], #16\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "ld1 {v5.16b}, [%[b_ptr]], #16\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "sadalp v25.4s, v9.8h\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v28.4s, v12.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "sadalp v27.4s, v11.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"

            //! K==1
            "3:\n"
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "sadalp v25.4s, v9.8h\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "sadalp v28.4s, v12.8h\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v27.4s, v11.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"

            "addp v4.4s, v16.4s, v20.4s\n"
            "addp v5.4s, v24.4s, v28.4s\n"
            "addp v6.4s, v17.4s, v21.4s\n"
            "addp v7.4s, v25.4s, v29.4s\n"
            "addp v8.4s, v18.4s, v22.4s\n"
            "addp v9.4s, v26.4s, v30.4s\n"
            "addp v10.4s, v19.4s, v23.4s\n"
            "addp v11.4s, v27.4s, v31.4s\n"

            "cmp %w[is_first_k], #1\n"

            "addp v0.4s, v4.4s, v5.4s\n"
            "addp v1.4s, v6.4s, v7.4s\n"
            "addp v2.4s, v8.4s, v9.4s\n"
            "addp v3.4s, v10.4s, v11.4s\n"

            "beq 6f\n"

            "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output]]\n"
            "add v0.4s, v0.4s, v8.4s\n"
            "add v1.4s, v1.4s, v9.4s\n"
            "add v2.4s, v2.4s, v10.4s\n"
            "add v3.4s, v3.4s, v11.4s\n"

            "6:\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[output]], #64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [k] "+r"(K), [output] "+r"(output)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
}

static void kern_4x4_remain(const int8_t* packA, const int8_t* packB, int K,
                            int32_t* output, bool is_first_k, size_t remain_n) {
    K = div_ceil(K, 16);
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    asm volatile(
            // load accumulator C
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "eor v16.16b,  v16.16b,  v16.16b\n"
            "eor v17.16b,  v17.16b,  v17.16b\n"
            "eor v18.16b,  v18.16b,  v18.16b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "eor v19.16b,  v19.16b,  v19.16b\n"
            "eor v20.16b,  v19.16b,  v19.16b\n"
            "eor v21.16b,  v19.16b,  v19.16b\n"
            "ld1 {v4.16b, v5.16b}, [%[b_ptr]], #32\n"
            "eor v22.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[a_ptr], #32]\n"
            "eor v23.16b,  v19.16b,  v19.16b\n"
            "eor v24.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #32]\n"
            "eor v25.16b,  v19.16b,  v19.16b\n"
            "eor v26.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #64]\n"
            "eor v27.16b,  v19.16b,  v19.16b\n"
            "eor v28.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[a_ptr], #64]\n"
            "eor v29.16b,  v19.16b,  v19.16b\n"
            "eor v30.16b,  v19.16b,  v19.16b\n"
            "PRFM PLDL1KEEP, [%[b_ptr], #128]\n"
            "eor v31.16b,  v19.16b,  v19.16b\n"

            //! if K==1 jump to compute last K
            "cmp %w[k], #2\n"
            "beq 2f\n"
            "blt 3f\n"

            //! K>2
            "1:\n"
            //! First k
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "ld1 {v4.16b}, [%[b_ptr]], #16\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "ld1 {v5.16b}, [%[b_ptr]], #16\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "sadalp v25.4s, v9.8h\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v28.4s, v12.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"

            //! Second k
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "sadalp v27.4s, v11.8h\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "sadalp v30.4s, v14.8h\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"
            "sadalp v31.4s, v15.8h\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "sub %w[k], %w[k], #2\n"
            "cmp %w[k], #2\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "ld1 {v4.16b}, [%[b_ptr]], #16\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "ld1 {v5.16b}, [%[b_ptr]], #16\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v25.4s, v9.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "sadalp v28.4s, v12.8h\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"

            "sadalp v27.4s, v11.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"

            "bgt 1b\n"
            "blt 3f\n"

            //! K==2
            "2:\n"
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "ld1 {v0.16b}, [%[a_ptr]], #16\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "ld1 {v1.16b}, [%[a_ptr]], #16\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "ld1 {v4.16b}, [%[b_ptr]], #16\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "ld1 {v5.16b}, [%[b_ptr]], #16\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "sadalp v25.4s, v9.8h\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v28.4s, v12.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "sadalp v27.4s, v11.8h\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"

            //! K==1
            "3:\n"
            "smull v8.8h, v0.8b, v4.8b\n"
            "smull v9.8h, v0.8b, v5.8b\n"
            "ld1 {v6.16b}, [%[b_ptr]], #16\n"
            "smull v12.8h, v1.8b, v4.8b\n"
            "smull v13.8h, v1.8b, v5.8b\n"
            "ld1 {v7.16b}, [%[b_ptr]], #16\n"
            "smlal2 v8.8h, v0.16b, v4.16b\n"
            "smlal2 v9.8h, v0.16b, v5.16b\n"
            "smlal2 v12.8h, v1.16b, v4.16b\n"
            "smlal2 v13.8h, v1.16b, v5.16b\n"

            "smull v10.8h, v0.8b, v6.8b\n"
            "ld1 {v2.16b}, [%[a_ptr]], #16\n"
            "smull v11.8h, v0.8b, v7.8b\n"
            "smull v14.8h, v1.8b, v6.8b\n"
            "ld1 {v3.16b}, [%[a_ptr]], #16\n"
            "smull v15.8h, v1.8b, v7.8b\n"
            "sadalp v16.4s, v8.8h\n"
            "smlal2 v10.8h, v0.16b, v6.16b\n"
            "sadalp v17.4s, v9.8h\n"
            "smlal2 v11.8h, v0.16b, v7.16b\n"
            "sadalp v20.4s, v12.8h\n"
            "smlal2 v14.8h, v1.16b, v6.16b\n"
            "sadalp v21.4s, v13.8h\n"
            "smlal2 v15.8h, v1.16b, v7.16b\n"

            "smull v8.8h, v2.8b, v4.8b\n"
            "smull v9.8h, v2.8b, v5.8b\n"
            "smull v12.8h, v3.8b, v4.8b\n"
            "smull v13.8h, v3.8b, v5.8b\n"
            "sadalp v18.4s, v10.8h\n"
            "smlal2 v8.8h, v2.16b, v4.16b\n"
            "sadalp v19.4s, v11.8h\n"
            "smlal2 v9.8h, v2.16b, v5.16b\n"
            "sadalp v22.4s, v14.8h\n"
            "smlal2 v12.8h, v3.16b, v4.16b\n"
            "sadalp v23.4s, v15.8h\n"
            "smlal2 v13.8h, v3.16b, v5.16b\n"

            "smull v10.8h, v2.8b, v6.8b\n"
            "sadalp v24.4s, v8.8h\n"
            "smull v11.8h, v2.8b, v7.8b\n"
            "sadalp v25.4s, v9.8h\n"
            "smull v14.8h, v3.8b, v6.8b\n"
            "sadalp v28.4s, v12.8h\n"
            "smull v15.8h, v3.8b, v7.8b\n"
            "sadalp v29.4s, v13.8h\n"
            "smlal2 v10.8h, v2.16b, v6.16b\n"
            "smlal2 v11.8h, v2.16b, v7.16b\n"
            "sadalp v26.4s, v10.8h\n"
            "smlal2 v14.8h, v3.16b, v6.16b\n"
            "sadalp v27.4s, v11.8h\n"
            "smlal2 v15.8h, v3.16b, v7.16b\n"
            "sadalp v30.4s, v14.8h\n"
            "sadalp v31.4s, v15.8h\n"

            "addp v4.4s, v16.4s, v20.4s\n"
            "addp v5.4s, v24.4s, v28.4s\n"
            "addp v6.4s, v17.4s, v21.4s\n"
            "addp v7.4s, v25.4s, v29.4s\n"
            "addp v8.4s, v18.4s, v22.4s\n"
            "addp v9.4s, v26.4s, v30.4s\n"
            "addp v10.4s, v19.4s, v23.4s\n"
            "addp v11.4s, v27.4s, v31.4s\n"

            "addp v0.4s, v4.4s, v5.4s\n"
            "addp v1.4s, v6.4s, v7.4s\n"
            "addp v2.4s, v8.4s, v9.4s\n"
            "addp v3.4s, v10.4s, v11.4s\n"

            "cmp %w[is_first_k], #1\n"
            "beq 6f\n"

            "cmp %w[remain_n], #3\n"
            "beq 1003f\n"
            "cmp %w[remain_n], #2\n"
            "beq 1002f\n"
            "cmp %w[remain_n], #1\n"
            "beq 1001f\n"
            "1003:\n"
            "ld1 {v8.4s, v9.4s, v10.4s}, [%[output]]\n"
            "add v0.4s, v0.4s, v8.4s\n"
            "add v1.4s, v1.4s, v9.4s\n"
            "add v2.4s, v2.4s, v10.4s\n"
            "b 6f\n"
            "1002:\n"
            "ld1 {v8.4s, v9.4s}, [%[output]]\n"
            "add v0.4s, v0.4s, v8.4s\n"
            "add v1.4s, v1.4s, v9.4s\n"
            "b 6f\n"
            "1001:\n"
            "ld1 {v8.4s}, [%[output]]\n"
            "add v0.4s, v0.4s, v8.4s\n"

            "6:\n"
            "cmp %w[remain_n], #3\n"
            "beq 10003f\n"
            "cmp %w[remain_n], #2\n"
            "beq 10002f\n"
            "cmp %w[remain_n], #1\n"
            "beq 10001f\n"
            "10003:\n"
            "str q2, [%[output], #32]\n"
            "10002:\n"
            "str q1, [%[output], #16]\n"
            "10001:\n"
            "str q0, [%[output]]\n"

            "7:\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [remain_n] "+r"(remain_n), [is_first_k] "+r"(is_first_k),
              [k] "+r"(K), [output] "+r"(output)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
}

static void gemm_mk4_s8_4x4_pack_A(dt_int8* outptr, const dt_int8* inptr,
                                   int ldin, int y0, int ymax, int k0,
                                   int kmax) {
    //! pack form {oc/4, ic/4, 4(ic), 4(oc)} to {oc/4, ic/16, 4(oc), 16(ic)}
    int8_t zerobuff[4][64];
    std::memset(zerobuff, 0, sizeof(int8_t) * 64 * 4);
    megdnn_assert(ymax % 4 == 0 && y0 % 4 == 0 && (ymax - y0) % 4 == 0,
                  "mk4 matmul with m is not times of 4");
    megdnn_assert(kmax % 4 == 0 && k0 % 4 == 0 && (kmax - k0) % 4 == 0,
                  "mk4 matmul with k is not times of 4");
    size_t roundk = round_up(kmax - k0, 16);
    size_t out_offset = roundk * 4;
    int y = y0;
    int start_y = y0 / 4;
    for (; y + 15 < ymax; y += 16, start_y += 4) {
        const int8_t* inptr0 = inptr + start_y * ldin + k0 * 4;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        int8_t* output = outptr + (y - y0) / 4 * out_offset;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            transpose_interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, output,
                                         out_offset);
            output += 64;
        }
        if (K > 0) {
            std::memcpy(zerobuff[0], inptr0, sizeof(int8_t) * K * 4);
            std::memcpy(zerobuff[1], inptr1, sizeof(int8_t) * K * 4);
            std::memcpy(zerobuff[2], inptr2, sizeof(int8_t) * K * 4);
            std::memcpy(zerobuff[3], inptr3, sizeof(int8_t) * K * 4);
            inptr0 = zerobuff[0];
            inptr1 = zerobuff[1];
            inptr2 = zerobuff[2];
            inptr3 = zerobuff[3];
            transpose_interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, output,
                                         out_offset);
            output += 64;
        }
    }
    for (; y + 3 < ymax; y += 4, start_y++) {
        const int8_t* inptr0 = inptr + start_y * ldin + k0 * 4;
        int8_t* output = outptr + (y - y0) / 4 * out_offset;
        prefetch_2x(inptr0);
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            transpose_interleave_1x4_4_b(inptr0, output);
            output += 64;
        }
        if (K > 0) {
            std::memcpy(zerobuff[0], inptr0, sizeof(int8_t) * K * 4);
            inptr0 = zerobuff[0];
            transpose_interleave_1x4_4_b(inptr0, output);
            output += 64;
        }
    }
}

static void gemm_mk4_s8_4x4_pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                   int x0, int xmax, int k0, int kmax) {
    int32_t zerobuff[4];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ICB = (ksize) / 4;
    const int ksize4 = round_up<int>(ICB, 4) * 4;
    int32_t* outptr = reinterpret_cast<int32_t*>(out);
    megdnn_assert(kmax % 4 == 0 && k0 % 4 == 0 && ksize % 4 == 0,
                  "mk4 matmul with k is not times of 4");

    int k = k0 / 4;
    for (; k + 3 < ICB; k += 4) {
        const int32_t* inptr0 =
                reinterpret_cast<const int32_t*>(in + k * ldin + x0);
        const int32_t* inptr1 =
                reinterpret_cast<const int32_t*>(in + (k + 1) * ldin + x0);
        const int32_t* inptr2 =
                reinterpret_cast<const int32_t*>(in + (k + 2) * ldin + x0);
        const int32_t* inptr3 =
                reinterpret_cast<const int32_t*>(in + (k + 3) * ldin + x0);
        int32_t* outptr_inner = outptr;

        int x = x0;
        for (; x + 3 < xmax; x += 4) {
            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner);
            outptr_inner += ksize4;
        }
        if (x < xmax) {
            for (; x < xmax; x++) {
                *outptr_inner++ = *inptr0++;
                *outptr_inner++ = *inptr1++;
                *outptr_inner++ = *inptr2++;
                *outptr_inner++ = *inptr3++;
            }
        }
        outptr += 4 * 4;
    }
    if (k < ICB) {
        const int32_t* inptr0 =
                reinterpret_cast<const int32_t*>(in + k * ldin + x0);
        const int32_t* inptr1 =
                reinterpret_cast<const int32_t*>(in + (k + 1) * ldin + x0);
        const int32_t* inptr2 =
                reinterpret_cast<const int32_t*>(in + (k + 2) * ldin + x0);
        const int32_t* inptr3 =
                reinterpret_cast<const int32_t*>(in + (k + 3) * ldin + x0);
        int32_t* outptr_inner = outptr;

        int x = x0;
        for (; x + 3 < xmax; x += 4) {
            if (k + 3 >= ICB) {
                switch (k + 3 - ICB) {
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
            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner);
            outptr_inner += ksize4;
        }
        if (x < xmax) {
            if (k + 3 >= ICB) {
                switch (k + 3 - ICB) {
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
            for (; x < xmax; x++) {
                *outptr_inner++ = *inptr0++;
                *outptr_inner++ = *inptr1++;
                *outptr_inner++ = *inptr2++;
                *outptr_inner++ = *inptr3++;
            }
        }
        outptr += 4 * 4;
    }
}

}  // namespace matmul_4x4x16
}  // namespace aarch64
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
