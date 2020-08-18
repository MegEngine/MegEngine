/**
 * \file dnn/src/aarch64/matrix_mul/fp32/kernel_mk4_8x12_a53.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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
struct matmul_mk4_8x12_a53 {
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
                "ldr d2, [%[b_ptr]]\n"

                "eor v9.16b, v9.16b, v9.16b\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "eor v10.16b, v10.16b, v10.16b\n"
                "ldr d3, [%[b_ptr], #16]\n"

                "eor v11.16b, v11.16b, v11.16b\n"
                "ldr x11, [%[b_ptr], #24]\n"

                "eor v12.16b, v12.16b, v12.16b\n"
                "ldr d4, [%[b_ptr], #32]\n"

                "eor v13.16b, v13.16b, v13.16b\n"
                "ldr x12, [%[b_ptr], #40]\n"

                "eor v14.16b, v14.16b, v14.16b\n"
                "ldr d0, [%[a_ptr]]\n"

                "eor v15.16b, v15.16b, v15.16b\n"
                "ldr x9, [%[a_ptr], #8]\n"

                "eor v16.16b, v16.16b, v16.16b\n"
                "add %[b_ptr], %[b_ptr], #48\n"

                "eor v17.16b, v17.16b, v17.16b\n"
                "add %[a_ptr], %[a_ptr], #16\n"

                "eor v18.16b, v18.16b, v18.16b\n"
                "ins v2.d[1], x10\n"

                "eor v19.16b, v19.16b, v19.16b\n"
                "ins v3.d[1], x11\n"

                "eor v20.16b, v20.16b, v20.16b\n"
                "ins v4.d[1], x12\n"

                "eor v21.16b, v21.16b, v21.16b\n"
                "ins v0.d[1], x9\n"

                "eor v22.16b, v22.16b, v22.16b\n"
                "prfm pldl1keep, [%[a_ptr], #384]\n"

                "eor v23.16b, v23.16b, v23.16b\n"
                "prfm pldl1keep, [%[b_ptr]]\n"

                "eor v24.16b, v24.16b, v24.16b\n"
                "prfm pldl1keep, [%[b_ptr], #64]\n"

                "eor v25.16b, v25.16b, v25.16b\n"
                "prfm pldl1keep, [%[b_ptr], #128]\n"

                "eor v26.16b, v26.16b, v26.16b\n"
                "prfm pldl1keep, [%[b_ptr], #192]\n"

                "eor v27.16b, v27.16b, v27.16b\n"
                "prfm pldl1keep, [%[b_ptr], #256]\n"

                "eor v28.16b, v28.16b, v28.16b\n"
                "prfm pldl1keep, [%[b_ptr], #320]\n"

                "eor v29.16b, v29.16b, v29.16b\n"
                "prfm pldl1keep, [%[b_ptr], #384]\n"

                "eor v30.16b, v30.16b, v30.16b\n"
                "prfm pldl1keep, [%[b_ptr], #448]\n"

                "eor v31.16b, v31.16b, v31.16b\n"
                "prfm pldl1keep, [%[b_ptr], #512]\n"

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "ldr d1, [%[a_ptr]]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "subs %w[K], %w[K], #1\n"

                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "ldr d5, [%[b_ptr]]\n"
                "ins v1.d[1], x8\n"

                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v12.4s, v0.4s, v3.s[0]\n"

                "fmla v13.4s, v0.4s, v3.s[1]\n"

                "ldr d6, [%[b_ptr], #16]\n"
                "ins v5.d[1], x10\n"

                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "ldr x11, [%[b_ptr], #24]\n"

                "fmla v15.4s, v0.4s, v3.s[3]\n"

                "fmla v16.4s, v0.4s, v4.s[0]\n"

                "ldr d7, [%[b_ptr], #32]\n"
                "ins v6.d[1], x11\n"

                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "ldr x12, [%[b_ptr], #40]\n"

                "fmla v18.4s, v0.4s, v4.s[2]\n"

                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "ldr d0, [%[a_ptr], #16]\n"
                "ins v7.d[1], x12\n"

                "fmla v20.4s, v1.4s, v2.s[0]\n"
                "ldr x9, [%[a_ptr], #24]\n"

                "fmla v21.4s, v1.4s, v2.s[1]\n"

                "fmla v22.4s, v1.4s, v2.s[2]\n"

                "prfm pldl1keep, [%[a_ptr], #448]\n"
                "ins v0.d[1], x9\n"

                "fmla v23.4s, v1.4s, v2.s[3]\n"

                "fmla v24.4s, v1.4s, v3.s[0]\n"

                "fmla v25.4s, v1.4s, v3.s[1]\n"

                "prfm pldl1keep, [%[b_ptr], #576]\n"
                "nop\n"

                "fmla v26.4s, v1.4s, v3.s[2]\n"

                "fmla v27.4s, v1.4s, v3.s[3]\n"

                "fmla v28.4s, v1.4s, v4.s[0]\n"

                "nop\n"
                "nop\n"

                "fmla v29.4s, v1.4s, v4.s[1]\n"

                "fmla v30.4s, v1.4s, v4.s[2]\n"

                "fmla v31.4s, v1.4s, v4.s[3]\n"
                //! UNROLL
                "ldr d1, [%[a_ptr], #32]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v5.s[0]\n"
                "ldr x8, [%[a_ptr], #40]\n"
                "fmla v9.4s,  v0.4s, v5.s[1]\n"

                "fmla v10.4s, v0.4s, v5.s[2]\n"

                "ldr d2, [%[b_ptr], #48]\n"
                "ins v1.d[1], x8\n"

                "fmla v11.4s, v0.4s, v5.s[3]\n"
                "ldr x10, [%[b_ptr], #56]\n"

                "fmla v12.4s, v0.4s, v6.s[0]\n"

                "fmla v13.4s, v0.4s, v6.s[1]\n"

                "ldr d3, [%[b_ptr], #64]\n"
                "ins v2.d[1], x10\n"

                "fmla v14.4s, v0.4s, v6.s[2]\n"
                "ldr x11, [%[b_ptr], #72]\n"
                "fmla v15.4s, v0.4s, v6.s[3]\n"

                "fmla v16.4s, v0.4s, v7.s[0]\n"

                "ldr d4, [%[b_ptr], #80]\n"
                "ins v3.d[1], x11\n"

                "fmla v17.4s, v0.4s, v7.s[1]\n"
                "ldr x12, [%[b_ptr], #88]\n"
                "fmla v18.4s, v0.4s, v7.s[2]\n"

                "fmla v19.4s, v0.4s, v7.s[3]\n"

                "ldr d0, [%[a_ptr], #48]\n"
                "ins v4.d[1], x12\n"

                "fmla v20.4s, v1.4s, v5.s[0]\n"
                "ldr x9, [%[a_ptr], #56]\n"

                "fmla v21.4s, v1.4s, v5.s[1]\n"
                "add %[b_ptr], %[b_ptr], #96\n"

                "fmla v22.4s, v1.4s, v5.s[2]\n"

                "nop\n"
                "ins v0.d[1], x9\n"

                "fmla v23.4s, v1.4s, v5.s[3]\n"
                "add %[a_ptr], %[a_ptr], #64\n"

                "fmla v24.4s, v1.4s, v6.s[0]\n"

                "fmla v25.4s, v1.4s, v6.s[1]\n"

                "prfm pldl1keep, [%[b_ptr], #640]\n"
                "nop\n"

                "fmla v26.4s, v1.4s, v6.s[2]\n"

                "fmla v27.4s, v1.4s, v6.s[3]\n"

                "fmla v28.4s, v1.4s, v7.s[0]\n"

                "nop\n"
                "nop\n"

                "fmla v29.4s, v1.4s, v7.s[1]\n"

                "fmla v30.4s, v1.4s, v7.s[2]\n"

                "fmla v31.4s, v1.4s, v7.s[3]\n"

                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "ldr d1, [%[a_ptr]] \n"
                "prfm pstl1keep, [%[output0]]\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x8, [%[a_ptr], #8] \n"

                "fmla v9.4s,  v0.4s, v2.s[1]\n"

                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "ldr d5, [%[b_ptr]]\n"
                "prfm pstl1keep, [%[output1]]\n"

                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v12.4s, v0.4s, v3.s[0]\n"

                "fmla v13.4s, v0.4s, v3.s[1]\n"

                "ldr d6, [%[b_ptr], #16]\n"
                "ins v1.d[1], x8\n"

                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "ldr x11, [%[b_ptr], #24]\n"

                "fmla v15.4s, v0.4s, v3.s[3]\n"

                "fmla v16.4s, v0.4s, v4.s[0]\n"

                "nop\n"
                "ins v5.d[1], x10\n"

                "fmla v17.4s, v0.4s, v4.s[1]\n"

                "fmla v18.4s, v0.4s, v4.s[2]\n"

                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "ldr d0, [%[a_ptr], #16]\n"
                "ins v6.d[1], x11\n"

                "fmla v20.4s, v1.4s, v2.s[0]\n"
                "ldr x9, [%[a_ptr], #24]\n"

                "fmla v21.4s, v1.4s, v2.s[1]\n"

                "fmla v22.4s, v1.4s, v2.s[2]\n"

                "ldr d7, [%[b_ptr], #32]\n"
                "ins v0.d[1], x9\n"

                "fmla v23.4s, v1.4s, v2.s[3]\n"
                "ldr x12, [%[b_ptr], #40]\n"

                "fmla v24.4s, v1.4s, v3.s[0]\n"

                "fmla v25.4s, v1.4s, v3.s[1]\n"

                "nop\n"
                "ins v7.d[1], x12\n"

                "fmla v26.4s, v1.4s, v3.s[2]\n"

                "fmla v27.4s, v1.4s, v3.s[3]\n"

                "fmla v28.4s, v1.4s, v4.s[0]\n"

                "nop\n"
                "nop\n"

                "fmla v29.4s, v1.4s, v4.s[1]\n"

                "fmla v30.4s, v1.4s, v4.s[2]\n"

                "fmla v31.4s, v1.4s, v4.s[3]\n"

                "ldr d1, [%[a_ptr], #32]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v5.s[0]\n"
                "ldr x8, [%[a_ptr], #40]\n"

                "fmla v9.4s,  v0.4s, v5.s[1]\n"

                "fmla v10.4s, v0.4s, v5.s[2]\n"

                "nop\n"
                "ins v1.d[1], x8\n"

                "fmla v11.4s, v0.4s, v5.s[3]\n"

                "fmla v12.4s, v0.4s, v6.s[0]\n"

                "fmla v13.4s, v0.4s, v6.s[1]\n"

                "fmla v14.4s, v0.4s, v6.s[2]\n"

                "fmla v15.4s, v0.4s, v6.s[3]\n"
                "str q8, [%[output0]]\n"

                "fmla v16.4s, v0.4s, v7.s[0]\n"
                "str q9, [%[output0], #16]\n"

                "fmla v17.4s, v0.4s, v7.s[1]\n"
                "str q10, [%[output0], #32]\n"

                "fmla v18.4s, v0.4s, v7.s[2]\n"

                "fmla v19.4s, v0.4s, v7.s[3]\n"
                "str q11, [%[output0], #48]\n"

                "fmla v20.4s, v1.4s, v5.s[0]\n"
                "str q12, [%[output0], #64]\n"

                "fmla v21.4s, v1.4s, v5.s[1]\n"
                "str q13, [%[output0], #80]\n"

                "fmla v22.4s, v1.4s, v5.s[2]\n"
                "str q14, [%[output0], #96]\n"

                "fmla v23.4s, v1.4s, v5.s[3]\n"
                "str q15, [%[output0], #112]\n"

                "fmla v24.4s, v1.4s, v6.s[0]\n"
                "str q16, [%[output0], #128]\n"

                "fmla v25.4s, v1.4s, v6.s[1]\n"
                "str q17, [%[output0], #144]\n"

                "fmla v26.4s, v1.4s, v6.s[2]\n"
                "str q18, [%[output0], #160]\n"

                "fmla v27.4s, v1.4s, v6.s[3]\n"
                "str q19, [%[output0], #176]\n"

                "fmla v28.4s, v1.4s, v7.s[0]\n"
                "str q20, [%[output1]]\n"

                "fmla v29.4s, v1.4s, v7.s[1]\n"
                "str q21, [%[output1], #16]\n"

                "fmla v30.4s, v1.4s, v7.s[2]\n"
                "str q22, [%[output1], #32]\n"

                "fmla v31.4s, v1.4s, v7.s[3]\n"
                "str q23, [%[output1], #48]\n"

                "str q24, [%[output1], #64]\n"
                "str q25, [%[output1], #80]\n"
                "str q26, [%[output1], #96]\n"
                "str q27, [%[output1], #112]\n"
                "str q28, [%[output1], #128]\n"
                "str q29, [%[output1], #144]\n"
                "str q30, [%[output1], #160]\n"
                "str q31, [%[output1], #176]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "ldr d1, [%[a_ptr]]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x8, [%[a_ptr], #8]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"

                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "nop\n"
                "ins v1.d[1], x8\n"

                "fmla v11.4s, v0.4s, v2.s[3]\n"

                "fmla v12.4s, v0.4s, v3.s[0]\n"

                "fmla v13.4s, v0.4s, v3.s[1]\n"

                "fmla v14.4s, v0.4s, v3.s[2]\n"

                "fmla v15.4s, v0.4s, v3.s[3]\n"

                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "str q8, [%[output0]]\n"

                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "str q9, [%[output0], #16]\n"

                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "str q10, [%[output0], #32]\n"

                "fmla v19.4s, v0.4s, v4.s[3]\n"
                "str q11, [%[output0], #48]\n"

                "fmla v20.4s, v1.4s, v2.s[0]\n"
                "str q12, [%[output0], #64]\n"

                "fmla v21.4s, v1.4s, v2.s[1]\n"
                "str q13, [%[output0], #80]\n"

                "fmla v22.4s, v1.4s, v2.s[2]\n"
                "str q14, [%[output0], #96]\n"

                "fmla v23.4s, v1.4s, v2.s[3]\n"
                "str q15, [%[output0], #112]\n"

                "fmla v24.4s, v1.4s, v3.s[0]\n"
                "str q16, [%[output0], #128]\n"

                "fmla v25.4s, v1.4s, v3.s[1]\n"
                "str q17, [%[output0], #144]\n"

                "fmla v26.4s, v1.4s, v3.s[2]\n"
                "str q18, [%[output0], #160]\n"

                "fmla v27.4s, v1.4s, v3.s[3]\n"
                "str q19, [%[output0], #176]\n"

                "fmla v28.4s, v1.4s, v4.s[0]\n"
                "str q20, [%[output1]]\n"

                "fmla v29.4s, v1.4s, v4.s[1]\n"
                "str q21, [%[output1], #16]\n"

                "fmla v30.4s, v1.4s, v4.s[2]\n"
                "str q22, [%[output1], #32]\n"

                "fmla v31.4s, v1.4s, v4.s[3]\n"
                "str q23, [%[output1], #48]\n"

                "str q24, [%[output1], #64]\n"
                "str q25, [%[output1], #80]\n"
                "str q26, [%[output1], #96]\n"
                "str q27, [%[output1], #112]\n"
                "str q28, [%[output1], #128]\n"
                "str q29, [%[output1], #144]\n"
                "str q30, [%[output1], #160]\n"
                "str q31, [%[output1], #176]\n"

                "6:\n"
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
                  [output0] "+r"(output0), [output1] "+r"(output1)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30", "v31", "x1", "x2", "x8", "x9", "x10",
                  "x11", "x12", "x13", "cc", "memory");
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
                "ldr d0, [%[a_ptr]]\n"

                "eor v9.16b, v9.16b, v9.16b\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "eor v10.16b, v10.16b, v10.16b\n"
                "ldr d2, [%[b_ptr]]\n"

                "eor v11.16b, v11.16b, v11.16b\n"
                "ldr x9, [%[b_ptr], #8]\n"

                "eor v12.16b, v12.16b, v12.16b\n"
                "ins v0.d[1], x8\n"

                "eor v13.16b, v13.16b, v13.16b\n"
                "add %[a_ptr], %[a_ptr], #16\n"

                "eor v14.16b, v14.16b, v14.16b\n"
                "ins v2.d[1], x9\n"

                "eor v15.16b, v15.16b, v15.16b\n"
                "add %[b_ptr], %[b_ptr], #16\n"

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "ldr q1, [%[a_ptr]]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v9.4s,  v0.4s, v2.s[1]\n"

                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "ldr d3, [%[b_ptr]]\n"
                "ins v1.d[1], x8\n"

                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v12.4s, v1.4s, v2.s[0]\n"

                "fmla v13.4s, v1.4s, v2.s[1]\n"

                "ldr d0, [%[a_ptr], #16]\n"
                "ins v3.d[1], x10\n"

                "fmla v14.4s, v1.4s, v2.s[2]\n"
                "ldr x9, [%[a_ptr], #24]\n"

                "fmla v15.4s, v1.4s, v2.s[3]\n"

                "ldr d1, [%[a_ptr], #32]\n"
                "ldr x8, [%[a_ptr], #40]\n"

                "ldr d2, [%[b_ptr], #16]\n"
                "ins v0.d[1], x9\n"

                "fmla v8.4s,  v0.4s, v3.s[0]\n"
                "ldr x10, [%[b_ptr], #24]\n"

                "fmla v9.4s,  v0.4s, v3.s[1]\n"

                "fmla v10.4s, v0.4s, v3.s[2]\n"

                "ins v1.d[1], x8\n"
                "ins v2.d[1], x10\n"

                "fmla v11.4s, v0.4s, v3.s[3]\n"

                "fmla v12.4s, v1.4s, v3.s[0]\n"

                "fmla v13.4s, v1.4s, v3.s[1]\n"

                "ldr d0, [%[a_ptr], #48]\n"
                "nop\n"

                "fmla v14.4s, v1.4s, v3.s[2]\n"
                "ldr x9, [%[a_ptr], #56]\n"

                "fmla v15.4s, v1.4s, v3.s[3]\n"
                "add %[b_ptr], %[b_ptr], #32\n"

                "add %[a_ptr], %[a_ptr], #64\n"
                "subs %w[K], %w[K], #1\n"

                "ins v0.d[1], x9\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "ldr d1, [%[a_ptr]]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "prfm pstl1keep, [%[output1]]\n"

                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "ldr d3, [%[b_ptr]]\n"
                "ins v1.d[1], x8\n"

                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v12.4s, v1.4s, v2.s[0]\n"

                "fmla v13.4s, v1.4s, v2.s[1]\n"

                "ldr d0, [%[a_ptr], #16]\n"
                "ins v3.d[1], x10\n"

                "fmla v14.4s, v1.4s, v2.s[2]\n"
                "ldr x9, [%[a_ptr], #24]\n"

                "fmla v15.4s, v1.4s, v2.s[3]\n"
                "prfm pstl1keep, [%[output1]]\n"

                "ldr d1, [%[a_ptr], #32]\n"
                "ins v0.d[1], x9\n"

                "fmla v8.4s,  v0.4s, v3.s[0]\n"
                "ldr x8, [%[a_ptr], #40]\n"
                "fmla v9.4s,  v0.4s, v3.s[1]\n"

                "fmla v10.4s, v0.4s, v3.s[2]\n"

                "nop\n"
                "ins v1.d[1], x8\n"

                "fmla v11.4s, v0.4s, v3.s[3]\n"
                "fmla v12.4s, v1.4s, v3.s[0]\n"
                "fmla v13.4s, v1.4s, v3.s[1]\n"
                "fmla v14.4s, v1.4s, v3.s[2]\n"
                "fmla v15.4s, v1.4s, v3.s[3]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "ldr q1, [%[a_ptr]]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "nop\n"
                "ins v1.d[1], x8\n"

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
                  "v13", "v14", "v15", "x8", "x9", "x10", "cc", "memory");

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
                "ldr d5, [%[b_ptr]]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v9.4s,  v0.4s, v2.s[1]\n"

                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "ldr d1, [%[a_ptr]]\n"
                "ins v5.d[1], x10\n"

                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v12.4s, v0.4s, v3.s[0]\n"

                "fmla v13.4s, v0.4s, v3.s[1]\n"

                "ldr d6, [%[b_ptr], #16]\n"
                "ins v1.d[1], x8\n"

                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "ldr x11, [%[b_ptr], #24]\n"

                "fmla v15.4s, v0.4s, v3.s[3]\n"

                "fmla v16.4s, v0.4s, v4.s[0]\n"

                "ldr d7, [%[b_ptr], #32]\n"
                "ins v6.d[1], x11\n"

                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "ldr x12, [%[b_ptr], #40]\n"

                "fmla v18.4s, v0.4s, v4.s[2]\n"

                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "ldr d2, [%[b_ptr], #48]\n"
                "ins v7.d[1], x12\n"

                "fmla v8.4s,  v1.4s, v5.s[0]\n"
                "ldr x10, [%[b_ptr], #56]\n"

                "fmla v9.4s,  v1.4s, v5.s[1]\n"

                "fmla v10.4s, v1.4s, v5.s[2]\n"

                "ldr d3, [%[b_ptr], #64]\n"
                "ins v2.d[1], x10\n"

                "fmla v11.4s, v1.4s, v5.s[3]\n"
                "ldr x11, [%[b_ptr], #72]\n"

                "fmla v12.4s, v1.4s, v6.s[0]\n"
                "subs %w[K], %w[K], #1\n"

                "fmla v13.4s, v1.4s, v6.s[1]\n"

                "ldr d4, [%[b_ptr], #80]\n"
                "ins v3.d[1], x11\n"

                "fmla v14.4s, v1.4s, v6.s[2]\n"
                "ldr x12, [%[b_ptr], #88]\n"

                "fmla v15.4s, v1.4s, v6.s[3]\n"

                "fmla v16.4s, v1.4s, v7.s[0]\n"

                "ldr d0, [%[a_ptr], #16]\n"
                "ins v4.d[1], x12\n"

                "fmla v17.4s, v1.4s, v7.s[1]\n"
                "ldr x10, [%[a_ptr], #24]\n"

                "fmla v18.4s, v1.4s, v7.s[2]\n"
                "add %[b_ptr], %[b_ptr], #96\n"

                "fmla v19.4s, v1.4s, v7.s[3]\n"
                "add %[a_ptr], %[a_ptr], #32\n"

                "ins v0.d[1], x10\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "ldr d5, [%[b_ptr]]\n"
                "nop\n"

                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v9.4s,  v0.4s, v2.s[1]\n"

                "fmla v10.4s, v0.4s, v2.s[2]\n"

                "ldr d6, [%[b_ptr], #16]\n"
                "ins v5.d[1], x10\n"

                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ldr x11, [%[b_ptr], #24]\n"

                "fmla v12.4s, v0.4s, v3.s[0]\n"

                "fmla v13.4s, v0.4s, v3.s[1]\n"

                "ldr d1, [%[a_ptr]]\n"
                "ins v6.d[1], x11\n"

                "fmla v14.4s, v0.4s, v3.s[2]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v15.4s, v0.4s, v3.s[3]\n"

                "fmla v16.4s, v0.4s, v4.s[0]\n"

                "ldr d7, [%[b_ptr], #32]\n"
                "ins v1.d[1], x8\n"

                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "ldr x12, [%[b_ptr], #40]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"

                "fmla v19.4s, v0.4s, v4.s[3]\n"

                "nop\n"
                "ins v7.d[1], x12\n"

                "fmla v8.4s,  v1.4s, v5.s[0]\n"
                "fmla v9.4s,  v1.4s, v5.s[1]\n"
                "fmla v10.4s, v1.4s, v5.s[2]\n"
                "fmla v11.4s, v1.4s, v5.s[3]\n"
                "fmla v12.4s, v1.4s, v6.s[0]\n"
                "fmla v13.4s, v1.4s, v6.s[1]\n"
                "fmla v14.4s, v1.4s, v6.s[2]\n"
                "fmla v15.4s, v1.4s, v6.s[3]\n"
                "fmla v16.4s, v1.4s, v7.s[0]\n"
                "fmla v17.4s, v1.4s, v7.s[1]\n"
                "str q8, [%[output0]]\n"
                "fmla v18.4s, v1.4s, v7.s[2]\n"
                "str q9, [%[output0], #16]\n"
                "fmla v19.4s, v1.4s, v7.s[3]\n"
                "str q10, [%[output0], #32]\n"
                "str q11, [%[output0], #48]\n"
                "str q12, [%[output0], #64]\n"
                "str q13, [%[output0], #80]\n"
                "str q14, [%[output0], #96]\n"
                "str q15, [%[output0], #112]\n"
                "str q16, [%[output0], #128]\n"
                "str q17, [%[output0], #144]\n"
                "str q18, [%[output0], #160]\n"
                "str q19, [%[output0], #176]\n"

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
                "fmla v15.4s, v0.4s, v3.s[3]\n"
                "fmla v16.4s, v0.4s, v4.s[0]\n"
                "fmla v17.4s, v0.4s, v4.s[1]\n"
                "str q8, [%[output0]]\n"
                "fmla v18.4s, v0.4s, v4.s[2]\n"
                "str q9, [%[output0], #16]\n"
                "fmla v19.4s, v0.4s, v4.s[3]\n"
                "str q10, [%[output0], #32]\n"
                "str q11, [%[output0], #48]\n"
                "str q12, [%[output0], #64]\n"
                "str q13, [%[output0], #80]\n"
                "str q14, [%[output0], #96]\n"
                "str q15, [%[output0], #112]\n"
                "str q16, [%[output0], #128]\n"
                "str q17, [%[output0], #144]\n"
                "str q18, [%[output0], #160]\n"
                "str q19, [%[output0], #176]\n"

                "6:\n"
                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
                  [output0] "+r"(output0)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "x1", "x8", "x9", "x10", "x11", "x12", "cc", "memory");
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
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
                "fmla v11.4s, v0.4s, v2.s[3]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v8.4s,  v1.4s, v3.s[0]\n"
                "fmla v9.4s,  v1.4s, v3.s[1]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "fmla v10.4s, v1.4s, v3.s[2]\n"
                "fmla v11.4s, v1.4s, v3.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v8.4s,  v0.4s, v2.s[0]\n"
                "fmla v9.4s,  v0.4s, v2.s[1]\n"
                "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                "fmla v10.4s, v0.4s, v2.s[2]\n"
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
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
