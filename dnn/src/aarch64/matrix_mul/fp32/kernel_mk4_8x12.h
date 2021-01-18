/**
 * \file dnn/src/aarch64/matrix_mul/fp32/kernel_mk4_8x12.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
struct matmul_mk4_8x12 {
    // Overview of register layout:
    //
    // A 1x12 cell of Rhs is stored in 32bit in v2-v7
    // A 8x1 cell of Lhs is stored in 32bit in (v0-v1)
    // A 8x12 block of accumulators is stored in 32bit in v8-v31.
    //
    //                 +--------+--------+--------+
    //                 | v2[0-3]| v3[0-3]| v4[0-3]|
    //                 | v5[0-3]| v6[0-3]| v7[0-3]|
    //           Rhs   +--------+--------+--------+
    //
    //                 |        |        |        |
    //
    //    Lhs          |        |        |        |
    //
    //  +--+   ---  -  +--------+--------+--------+
    //  |v0|           | v8[0-3]| v9[0-3]|v10[0-3]|
    //  |v0|           |v11[0-3]|v12[0-3]|v13[0-3]|
    //  |v0|           |v14[0-3]|v15[0-3]|v16[0-3]|
    //  |v0|           |v17[0-3]|v18[0-3]|v19[0-3]|
    //  |v1|           |v20[0-3]|v21[0-3]|v22[0-3]|
    //  |v1|           |v23[0-3]|v24[0-3]|v25[0-3]|
    //  |v1|           |v26[0-3]|v27[0-3]|v28[0-3]|
    //  |v1|           |v29[0-3]|v30[0-3]|v31[0-3]|
    //  +--+   ---  -  +--------+--------+--------+
    //
    //                        Accumulator
    static void kern_8x12(const float* packA, const float* packB, int K,
                          float* output, int LDC, bool is_first_k) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        float* output0 = output;
        float* output1 = output0 + LDC;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile(
                "cmp %w[is_first_k], #1\n"
                "beq 1f\n"
                "mov x1, %[output0]\n"
                "mov x2, %[output1]\n"
                "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64\n"
                "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64\n"
                "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64\n"
                "ld1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x2], #64\n"
                "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64\n"
                "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64\n"

                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], #48\n"
                "b 2f\n"

                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "prfm pstl1keep, [%[output0]]\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v12.16b, v12.16b, v12.16b\n"
                "eor v13.16b, v13.16b, v13.16b\n"
                "prfm pstl1keep, [%[output1]]\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v15.16b, v15.16b, v15.16b\n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"
                "eor v16.16b, v16.16b, v16.16b\n"
                "ld1 {v3.4s}, [%[b_ptr]], #16\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "ld1 {v4.4s}, [%[b_ptr]], #16\n"
                "eor v18.16b, v18.16b, v18.16b\n"
                "eor v19.16b, v19.16b, v19.16b\n"
                "eor v20.16b, v20.16b, v20.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
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
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ld1 {v5.4s}, [%[b_ptr]], #16\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "ld1 {v6.4s}, [%[b_ptr]], #16\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "ld1 {v7.4s}, [%[b_ptr]], #16\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"

                "fmla v20.4s, v1.4s, v2.s[0]\n"
                "fmla v21.4s, v1.4s, v2.s[1]\n"
                "fmla v22.4s, v1.4s, v2.s[2]\n"
                "fmla v23.4s, v1.4s, v2.s[3]\n"
                "fmla v24.4s, v1.4s, v3.s[0]\n"
                "fmla v25.4s, v1.4s, v3.s[1]\n"
                "fmla v26.4s, v1.4s, v3.s[2]\n"
                "fmla v27.4s, v1.4s, v3.s[3]\n"
                "fmla v28.4s, v1.4s, v4.s[0]\n"
                "fmla v29.4s, v1.4s, v4.s[1]\n"
                "fmla v30.4s, v1.4s, v4.s[2]\n"
                "fmla v31.4s, v1.4s, v4.s[3]\n"

                "fmla v8.4s,  v0.4s, v5.s[0]\n"
                "fmla v9.4s,  v0.4s, v5.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v5.s[2]\n"
                "fmla v11.4s, v0.4s, v5.s[3]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "fmla v12.4s, v0.4s, v6.s[0]\n"
                "fmla v13.4s, v0.4s, v6.s[1]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "fmla v14.4s, v0.4s, v6.s[2]\n"
                "fmla v15.4s, v0.4s, v6.s[3]\n"
                "ld1 {v4.4s}, [%[b_ptr]], 16\n"
                "fmla v16.4s, v0.4s, v7.s[0]\n"
                "fmla v17.4s, v0.4s, v7.s[1]\n"
                "fmla v18.4s, v0.4s, v7.s[2]\n"
                "fmla v19.4s, v0.4s, v7.s[3]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"

                "fmla v20.4s, v1.4s, v5.s[0]\n"
                "fmla v21.4s, v1.4s, v5.s[1]\n"
                "fmla v22.4s, v1.4s, v5.s[2]\n"
                "fmla v23.4s, v1.4s, v5.s[3]\n"
                "fmla v24.4s, v1.4s, v6.s[0]\n"
                "subs %w[K], %w[K], #1\n"
                "fmla v25.4s, v1.4s, v6.s[1]\n"
                "fmla v26.4s, v1.4s, v6.s[2]\n"
                "fmla v27.4s, v1.4s, v6.s[3]\n"
                "fmla v28.4s, v1.4s, v7.s[0]\n"
                "fmla v29.4s, v1.4s, v7.s[1]\n"
                "fmla v30.4s, v1.4s, v7.s[2]\n"
                "fmla v31.4s, v1.4s, v7.s[3]\n"

                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "fmla v20.4s, v1.4s, v2.s[0]\n"
                "ld1 {v5.4s}, [%[b_ptr]], #16\n"
                "fmla v21.4s, v1.4s, v2.s[1]\n"
                "fmla v22.4s, v1.4s, v2.s[2]\n"
                "ld1 {v6.4s}, [%[b_ptr]], #16\n"
                "fmla v23.4s, v1.4s, v2.s[3]\n"
                "fmla v24.4s, v1.4s, v3.s[0]\n"
                "ld1 {v7.4s}, [%[b_ptr]], #16\n"
                "fmla v25.4s, v1.4s, v3.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v1.4s, v3.s[2]\n"
                "fmla v27.4s, v1.4s, v3.s[3]\n"
                "fmla v28.4s, v1.4s, v4.s[0]\n"
                "fmla v29.4s, v1.4s, v4.s[1]\n"
                "fmla v30.4s, v1.4s, v4.s[2]\n"
                "fmla v31.4s, v1.4s, v4.s[3]\n"

                "fmla v8.4s,  v0.4s, v5.s[0]\n"
                "fmla v9.4s,  v0.4s, v5.s[1]\n"
                "fmla v10.4s, v0.4s, v5.s[2]\n"
                "fmla v11.4s, v0.4s, v5.s[3]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v12.4s, v0.4s, v6.s[0]\n"
                "fmla v13.4s, v0.4s, v6.s[1]\n"

                "fmla v14.4s, v0.4s, v6.s[2]\n"
                "fmla v15.4s, v0.4s, v6.s[3]\n"
                "st1 {v8.4s}, [%[output0]], #16\n"
                "fmla v16.4s, v0.4s, v7.s[0]\n"
                "st1 {v9.4s}, [%[output0]], #16\n"
                "fmla v17.4s, v0.4s, v7.s[1]\n"
                "st1 {v10.4s}, [%[output0]], #16\n"
                "fmla v18.4s, v0.4s, v7.s[2]\n"
                "st1 {v11.4s}, [%[output0]], #16\n"
                "fmla v19.4s, v0.4s, v7.s[3]\n"
                "st1 {v12.4s}, [%[output0]], #16\n"

                "fmla v20.4s, v1.4s, v5.s[0]\n"
                "st1 {v13.4s}, [%[output0]], #16\n"
                "fmla v21.4s, v1.4s, v5.s[1]\n"
                "st1 {v14.4s}, [%[output0]], #16\n"
                "fmla v22.4s, v1.4s, v5.s[2]\n"
                "st1 {v15.4s}, [%[output0]], #16\n"
                "fmla v23.4s, v1.4s, v5.s[3]\n"
                "st1 {v16.4s}, [%[output0]], #16\n"
                "fmla v24.4s, v1.4s, v6.s[0]\n"
                "st1 {v17.4s}, [%[output0]], #16\n"
                "fmla v25.4s, v1.4s, v6.s[1]\n"
                "st1 {v18.4s}, [%[output0]], #16\n"
                "fmla v26.4s, v1.4s, v6.s[2]\n"
                "st1 {v19.4s}, [%[output0]], #16\n"
                "fmla v27.4s, v1.4s, v6.s[3]\n"
                "st1 {v20.4s}, [%[output1]], #16\n"
                "fmla v28.4s, v1.4s, v7.s[0]\n"
                "st1 {v21.4s}, [%[output1]], #16\n"
                "fmla v29.4s, v1.4s, v7.s[1]\n"
                "st1 {v22.4s}, [%[output1]], #16\n"
                "fmla v30.4s, v1.4s, v7.s[2]\n"
                "st1 {v23.4s}, [%[output1]], #16\n"
                "fmla v31.4s, v1.4s, v7.s[3]\n"
                "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output1]], #64\n"
                "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output1]], #64\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "st1 {v8.4s}, [%[output0]], #16\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "st1 {v9.4s}, [%[output0]], #16\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "st1 {v10.4s}, [%[output0]], #16\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "st1 {v11.4s}, [%[output0]], #16\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "st1 {v12.4s}, [%[output0]], #16\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"
                "st1 {v13.4s}, [%[output0]], #16\n"

                "fmla v20.4s, v1.4s, v2.s[0]\n"
                "st1 {v14.4s}, [%[output0]], #16\n"
                "fmla v21.4s, v1.4s, v2.s[1]\n"
                "st1 {v15.4s}, [%[output0]], #16\n"
                "fmla v22.4s, v1.4s, v2.s[2]\n"
                "st1 {v16.4s}, [%[output0]], #16\n"
                "fmla v23.4s, v1.4s, v2.s[3]\n"
                "st1 {v17.4s}, [%[output0]], #16\n"
                "fmla v24.4s, v1.4s, v3.s[0]\n"
                "st1 {v18.4s}, [%[output0]], #16\n"
                "fmla v25.4s, v1.4s, v3.s[1]\n"
                "st1 {v19.4s}, [%[output0]], #16\n"
                "fmla v26.4s, v1.4s, v3.s[2]\n"
                "st1 {v20.4s}, [%[output1]], #16\n"
                "fmla v27.4s, v1.4s, v3.s[3]\n"
                "st1 {v21.4s}, [%[output1]], #16\n"
                "fmla v28.4s, v1.4s, v4.s[0]\n"
                "st1 {v22.4s}, [%[output1]], #16\n"
                "fmla v29.4s, v1.4s, v4.s[1]\n"
                "st1 {v23.4s}, [%[output1]], #16\n"
                "fmla v30.4s, v1.4s, v4.s[2]\n"
                "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output1]], #64\n"
                "fmla v31.4s, v1.4s, v4.s[3]\n"
                "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output1]], #64\n"

                "6:\n"
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
                  [output0] "+r"(output0), [output1] "+r"(output1)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30", "v31", "x1", "x2", "cc", "memory");
    }

    // Overview of register layout:
    //
    // A 1x12 cell of Rhs is stored in 32bit in v2-v7
    // A 8x1 cell of Lhs is stored in 32bit in (v0-v1)
    // A 8x12 block of accumulators is stored in 32bit in v8-v31.
    //
    //                 +--------+
    //                 | v2[0-3]|
    //                 | v3[0-3]|
    //           Rhs   +--------+
    //
    //                 |        |
    //
    //    Lhs          |        |
    //
    //  +--+   ---  -  +--------+
    //  |v0|           | v8[0-3]|
    //  |v0|           |v11[0-3]|
    //  |v0|           |v14[0-3]|
    //  |v0|           |v17[0-3]|
    //  |v1|           |v20[0-3]|
    //  |v1|           |v23[0-3]|
    //  |v1|           |v26[0-3]|
    //  |v1|           |v29[0-3]|
    //  +--+   ---  -  +--------+
    //
    //                        Accumulator
    static void kern_8x4(const float* packA, const float* packB, int K,
                         float* output, int LDC, bool is_first_k,
                         int n_remain) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        float* output0 = output;
        float* output1 = output0 + LDC;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        //clang-format off
#define LOAD_C                                            \
    "cmp %w[n_remain], #4\n"                              \
    "blt 11f\n"                                           \
    "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]]\n"  \
    "ld1 {v12.4s, v13.4s, v14.4s, v15.4s},[%[output1]]\n" \
    "b 14f\n"                                             \
    "11:\n"                                               \
    "cmp %w[n_remain], #3\n"                              \
    "blt 12f\n"                                           \
    "ld1 {v8.4s, v9.4s, v10.4s}, [%[output0]]\n"          \
    "ld1 {v12.4s, v13.4s, v14.4s},[%[output1]]\n"         \
    "b 14f\n"                                             \
    "12:\n"                                               \
    "cmp %w[n_remain], #2\n"                              \
    "blt 13f\n"                                           \
    "ld1 {v8.4s, v9.4s}, [%[output0]]\n"                  \
    "ld1 {v12.4s, v13.4s},[%[output1]]\n"                 \
    "b 14f\n"                                             \
    "13:\n"                                               \
    "ld1 {v8.4s}, [%[output0]]\n"                         \
    "ld1 {v12.4s},[%[output1]]\n"                         \
    "14:\n"

#define STORE_C                                           \
    "cmp %w[n_remain], #4\n"                              \
    "blt 21f\n"                                           \
    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]]\n"  \
    "st1 {v12.4s, v13.4s, v14.4s, v15.4s},[%[output1]]\n" \
    "b 24f\n"                                             \
    "21:\n"                                               \
    "cmp %w[n_remain], #3\n"                              \
    "blt 22f\n"                                           \
    "st1 {v8.4s, v9.4s, v10.4s}, [%[output0]]\n"          \
    "st1 {v12.4s, v13.4s, v14.4s},[%[output1]]\n"         \
    "b 23f\n"                                             \
    "22:\n"                                               \
    "cmp %w[n_remain], #2\n"                              \
    "blt 23f\n"                                           \
    "st1 {v8.4s, v9.4s}, [%[output0]]\n"                  \
    "st1 {v12.4s, v13.4s},[%[output1]]\n"                 \
    "b 24f\n"                                             \
    "23:\n"                                               \
    "st1 {v8.4s}, [%[output0]]\n"                         \
    "st1 {v12.4s},[%[output1]]\n"                         \
    "24:\n"
        //clang-format on

        asm volatile(
                // load accumulator C
                "cmp %w[is_first_k], #1\n"
                "beq 1f\n" LOAD_C

                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"
                "b 2f\n"

                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "prfm pstl1keep, [%[output0]]\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v12.16b, v12.16b, v12.16b\n"
                "prfm pstl1keep, [%[output1]]\n"
                "eor v13.16b, v13.16b, v13.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v15.16b, v15.16b, v15.16b\n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], #16\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], #16\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v1.4s, v2.s[0]\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "fmla v13.4s, v1.4s, v2.s[1]\n"
                "fmla v14.4s, v1.4s, v2.s[2]\n"
                "fmla v15.4s, v1.4s, v2.s[3]\n"

                "fmla v8.4s,  v0.4s, v3.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], #16\n"
                "fmla v9.4s,  v0.4s, v3.s[1]\n"
                "fmla v10.4s, v0.4s, v3.s[2]\n"
                "fmla v11.4s, v0.4s, v3.s[3]\n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"
                "fmla v12.4s, v1.4s, v3.s[0]\n"
                "subs %w[K], %w[K], #1\n"
                "fmla v13.4s, v1.4s, v3.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "fmla v14.4s, v1.4s, v3.s[2]\n"
                "fmla v15.4s, v1.4s, v3.s[3]\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], #16\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], #16\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v1.4s, v2.s[0]\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "fmla v13.4s, v1.4s, v2.s[1]\n"
                "fmla v14.4s, v1.4s, v2.s[2]\n"
                "fmla v15.4s, v1.4s, v2.s[3]\n"

                "fmla v8.4s,  v0.4s, v3.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], #16\n"
                "fmla v9.4s,  v0.4s, v3.s[1]\n"
                "fmla v10.4s, v0.4s, v3.s[2]\n"
                "fmla v11.4s, v0.4s, v3.s[3]\n"
                "fmla v12.4s, v1.4s, v3.s[0]\n"
                "fmla v13.4s, v1.4s, v3.s[1]\n"
                "fmla v14.4s, v1.4s, v3.s[2]\n"
                "fmla v15.4s, v1.4s, v3.s[3]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], #16\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v1.4s, v2.s[0]\n"
                "fmla v13.4s, v1.4s, v2.s[1]\n"
                "fmla v14.4s, v1.4s, v2.s[2]\n"
                "fmla v15.4s, v1.4s, v2.s[3]\n"

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
                  [output0] "+r"(output0), [output1] "+r"(output1),
                  [n_remain] "+r"(n_remain)
                :
                : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12",
                  "v13", "v14", "v15", "cc", "memory");

#undef LOAD_C
#undef STORE_C
    }

    // Overview of register layout:
    //
    // A 1x12 cell of Rhs is stored in 32bit in v2-v7
    // A 8x1 cell of Lhs is stored in 32bit in (v0-v1)
    // A 8x12 block of accumulators is stored in 32bit in v8-v31.
    //
    //                 +--------+--------+--------+
    //                 | v2[0-3]| v3[0-3]| v4[0-3]|
    //                 | v5[0-3]| v6[0-3]| v7[0-3]|
    //           Rhs   +--------+--------+--------+
    //
    //                 |        |        |        |
    //
    //    Lhs          |        |        |        |
    //
    //  +--+   ---  -  +--------+--------+--------+
    //  |v0|           | v8[0-3]| v9[0-3]|v10[0-3]|
    //  |v0|           |v11[0-3]|v12[0-3]|v13[0-3]|
    //  |v0|           |v14[0-3]|v15[0-3]|v16[0-3]|
    //  |v0|           |v17[0-3]|v18[0-3]|v19[0-3]|
    //  +--+   ---  -  +--------+--------+--------+
    //
    //                        Accumulator

    static void kern_4x12(const float* packA, const float* packB, int K,
                          float* output, int LDC, bool is_first_k) {
        MEGDNN_MARK_USED_VAR(LDC);
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        float* output0 = output;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        asm volatile(
                "cmp %w[is_first_k], #1\n"
                "beq 1f\n"
                "mov x1, %[output0]\n"
                "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x1], #64\n"
                "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x1], #64\n"
                "ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x1], #64\n"

                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], #48\n"
                "b 2f\n"

                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "prfm pstl1keep, [%[output0]]\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v12.16b, v12.16b, v12.16b\n"
                "eor v13.16b, v13.16b, v13.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v15.16b, v15.16b, v15.16b\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], #48\n"
                "eor v16.16b, v16.16b, v16.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "eor v18.16b, v18.16b, v18.16b\n"
                "eor v19.16b, v19.16b, v19.16b\n"

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], #48\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "fmla v8.4s,  v1.4s, v5.s[0]\n"
                "fmla v9.4s,  v1.4s, v5.s[1]\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "fmla v10.4s, v1.4s, v5.s[2]\n"
                "fmla v11.4s, v1.4s, v5.s[3]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v12.4s, v1.4s, v6.s[0]\n"
                "fmla v13.4s, v1.4s, v6.s[1]\n"
                "subs %w[K], %w[K], #1\n"
                "fmla v14.4s, v1.4s, v6.s[2]\n"
                "fmla v15.4s, v1.4s, v6.s[3]\n"
                "fmla v16.4s, v1.4s, v7.s[0]\n"
                "fmla v17.4s, v1.4s, v7.s[1]\n"
                "fmla v18.4s, v1.4s, v7.s[2]\n"
                "fmla v19.4s, v1.4s, v7.s[3]\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], #48\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "fmla v8.4s,  v1.4s, v5.s[0]\n"
                "fmla v9.4s,  v1.4s, v5.s[1]\n"
                "fmla v10.4s, v1.4s, v5.s[2]\n"
                "fmla v11.4s, v1.4s, v5.s[3]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v12.4s, v1.4s, v6.s[0]\n"
                "fmla v13.4s, v1.4s, v6.s[1]\n"
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]], #64\n"
                "fmla v14.4s, v1.4s, v6.s[2]\n"
                "fmla v15.4s, v1.4s, v6.s[3]\n"
                "fmla v16.4s, v1.4s, v7.s[0]\n"
                "fmla v17.4s, v1.4s, v7.s[1]\n"
                "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[output0]], #64\n"
                "fmla v18.4s, v1.4s, v7.s[2]\n"
                "fmla v19.4s, v1.4s, v7.s[3]\n"
                "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output0]], #64\n"

                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "fmla v12.4s, v0.4s, v3.s[0]\n"
                "fmla v13.4s, v0.4s, v3.s[1]\n"
                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]], #64\n"
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[output0]], #64\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"
                "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output0]], #64\n"

                "6:\n"
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
                  [output0] "+r"(output0)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "x1", "cc", "memory");
    }

    // Overview of register layout:
    //
    // A 2x4 cell of Rhs is stored in 32bit in v2 - v3
    // A 4x2 cell of Lhs is stored in 32bit in v0 - v1
    // A 4x4 block of accumulators is stored in 32bit in v4-v6
    //
    //                 +--------+
    //                 | v2[0-3]|
    //                 | v5[0-3]|
    //           Rhs   +--------+
    //
    //                 |        |
    //
    //    Lhs          |        |
    //
    //  +--+   ---  -  +--------+
    //  |v0|           | v8[0-3]|
    //  |v0|           |v11[0-3]|
    //  |v0|           |v14[0-3]|
    //  |v0|           |v17[0-3]|
    //  +--+   ---  -  +--------+
    //
    //                        Accumulator
    static void kern_4x4(const float* packA, const float* packB, int K,
                         float* output, int LDC, bool is_first_k,
                         int n_remain) {
        MEGDNN_MARK_USED_VAR(LDC);
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        float* output0 = output;

        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        //clang-format off
#define LOAD_C                                           \
    "cmp %w[n_remain], #4\n"                             \
    "blt 11f\n"                                          \
    "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]]\n" \
    "b 14f\n"                                            \
    "11:\n"                                              \
    "cmp %w[n_remain], #3\n"                             \
    "blt 12f\n"                                          \
    "ld1 {v8.4s, v9.4s, v10.4s}, [%[output0]]\n"         \
    "b 14f\n"                                            \
    "12:\n"                                              \
    "cmp %w[n_remain], #2\n"                             \
    "blt 13f\n"                                          \
    "ld1 {v8.4s, v9.4s}, [%[output0]]\n"                 \
    "b 14f\n"                                            \
    "13:\n"                                              \
    "ld1 {v8.4s}, [%[output0]]\n"                        \
    "14:\n"

#define STORE_C                                          \
    "cmp %w[n_remain], #4\n"                             \
    "blt 21f\n"                                          \
    "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[output0]]\n" \
    "b 24f\n"                                            \
    "21:\n"                                              \
    "cmp %w[n_remain], #3\n"                             \
    "blt 22f\n"                                          \
    "st1 {v8.4s, v9.4s, v10.4s}, [%[output0]]\n"         \
    "b 24f\n"                                            \
    "22:\n"                                              \
    "cmp %w[n_remain], #2\n"                             \
    "blt 23f\n"                                          \
    "st1 {v8.4s, v9.4s}, [%[output0]]\n"                 \
    "b 24f\n"                                            \
    "23:\n"                                              \
    "st1 {v8.4s}, [%[output0]]\n"                        \
    "24:\n"
        //clang-format on

        asm volatile(
                // load accumulator C
                "cmp %w[is_first_k], #1\n"
                "beq 1f\n" LOAD_C

                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"
                "b 2f\n"

                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "ld1 {v2.4s}, [%[b_ptr]], #16\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], #16\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "prfm pstl1keep, [%[output0]]\n"
                "eor v11.16b, v11.16b, v11.16b\n"

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"

                "fmla v8.4s,  v1.4s, v3.s[0]\n"
                "fmla v9.4s,  v1.4s, v3.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v10.4s, v1.4s, v3.s[2]\n"
                "fmla v11.4s, v1.4s, v3.s[3]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"

                "fmla v8.4s,  v1.4s, v3.s[0]\n"
                "fmla v9.4s,  v1.4s, v3.s[1]\n"
                "fmla v10.4s, v1.4s, v3.s[2]\n"
                "fmla v11.4s, v1.4s, v3.s[3]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
                  [output0] "+r"(output0), [n_remain] "+r"(n_remain)
                :
                : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "cc",
                  "memory");

#undef LOAD_C
#undef STORE_C
    }

    static void sgemm_8x12_pack_A(float* outptr, const float* inptr, int ldin,
                                  int y0, int ymax, int k0, int kmax) {
        megdnn_assert(y0 % 4 == 0 && ymax % 4 == 0, "M must be time of 4");
        megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
        constexpr int PACK_SIZE_32 = 4 * 8;
        constexpr int PACK_SIZE_16 = 4 * 4;
        constexpr int PACK_C_SIZE = 4;
        int y = y0;
        for (; y + 7 < ymax; y += 8) {
            const float* inptr0 = inptr + y / PACK_C_SIZE * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            int k = (kmax - k0);
            for (; k > 3; k -= 4) {
                interleave_2x4_4_s(inptr0, inptr1, outptr);
                outptr += PACK_SIZE_32;
            }
        }
        for (; y < ymax; y += 4) {
            const float* inptr0 = inptr + y / PACK_C_SIZE * ldin + k0;
            prefetch_2x(inptr0);
            int K = (kmax - k0);
            for (; K > 3; K -= 4) {
                interleave_1x4_4_s(inptr0, outptr);
                outptr += PACK_SIZE_16;
            }
        }
    }

    static void sgemm_8x12_pack_B(float* out, const float* in, int ldin, int x0,
                                  int xmax, int k0, int kmax) {
        megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
        float tmpbuff[16] = {0.0f};

        constexpr int PACK_C_SIZE = 4;
        int ksize = kmax - k0;
        int ksize12 = ksize * 12;
        int ksize4 = (ksize << 2);
        float* outptr_base = out;
        float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* inptr = in + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;
            prefetch_3x(inptr);

            int x = x0;
            auto outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                auto outptr_interleave = outptr;
                transpose_1x12_4_s(inptr, outptr_interleave);
                outptr += ksize12;
            }
            outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                auto outptr_interleave = outptr;
                transpose_1x4_4_s(inptr, outptr_interleave);
                outptr += ksize4;
            }
            if (x < xmax) {
                std::memcpy(tmpbuff, inptr,
                            sizeof(float) * (xmax - x) * PACK_C_SIZE);
                auto outptr_interleave = outptr;
                const float* tmp_ptr = &tmpbuff[0];
                transpose_1x4_4_s<float>(tmp_ptr, outptr_interleave);
                outptr += ksize4;
            }
            outptr_base += 12 * 4;
            outptr_base4 += 4 * 4;
        }
    }
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
