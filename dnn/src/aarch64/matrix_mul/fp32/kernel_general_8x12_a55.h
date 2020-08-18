/**
 * \file dnn/src/aarch64/matrix_mul/fp32/kernel_general_8x12_a55.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
struct matmul_general_8x12_a55 {
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
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;
        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = reinterpret_cast<float*>(output);
// clang-format off
#define LOAD_LINE(v0, v1, v2, n)                            \
    "ld1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \

#define LOAD_C                        \
    LOAD_LINE("8", "9", "10", "0")    \
    LOAD_LINE("11", "12", "13", "1")  \
    LOAD_LINE("14", "15", "16", "2")  \
    LOAD_LINE("17", "18", "19", "3")  \
    LOAD_LINE("20", "21", "22", "4")  \
    LOAD_LINE("23", "24", "25", "5")  \
    LOAD_LINE("26", "27", "28", "6")  \
    LOAD_LINE("29", "30", "31", "7")

        // clang-format on
        asm volatile(
                // load accumulator C
                "add x1, x0, %x[LDC]\n"
                "prfm pldl1keep, [%[a_ptr]]\n"
                "add x2, x1, %x[LDC]\n"
                "prfm pldl1keep, [%[b_ptr]]\n"
                "add x3, x2, %x[LDC]\n"
                "prfm pldl1keep, [%[a_ptr], #64]\n"
                "add x4, x3, %x[LDC]\n"
                "prfm pldl1keep, [%[a_ptr], #128]\n"
                "add x5, x4, %x[LDC]\n"
                "prfm pldl1keep, [%[a_ptr], #192]\n"
                "add x6, x5, %x[LDC]\n"
                "prfm pldl1keep, [%[a_ptr], #256]\n"
                "add x7, x6, %x[LDC]\n"

                "cmp %w[is_first_k], #1\n"
                "beq 1f\n" LOAD_C
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
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

                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ldr d1, [%[a_ptr]]\n"

                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "subs %w[K], %w[K], #1\n"

                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v11.4s, v2.4s, v0.s[1]\n"

                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "ldr d5, [%[b_ptr]]\n"

                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ins v1.d[1], x8\n"

                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v15.4s, v3.4s, v0.s[2]\n"

                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "ldr d6, [%[b_ptr], #16]\n"

                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "ins v5.d[1], x10\n"

                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "ldr x11, [%[b_ptr], #24]\n"

                "fmla v19.4s, v4.4s, v0.s[3]\n"

                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "ldr d7, [%[b_ptr], #32]\n"

                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "ins v6.d[1], x11\n"

                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "ldr d0, [%[a_ptr], #16]\n"

                "fmla v23.4s, v2.4s, v1.s[1]\n"

                "fmla v24.4s, v3.4s, v1.s[1]\n"
                "ldr x12, [%[b_ptr], #40]\n"

                "fmla v25.4s, v4.4s, v1.s[1]\n"

                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "ldr x9, [%[a_ptr], #24]\n"

                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "ins v7.d[1], x12\n"

                "fmla v28.4s, v4.4s, v1.s[2]\n"
                "prfm pldl1keep, [%[a_ptr], #448]\n"

                "fmla v29.4s, v2.4s, v1.s[3]\n"
                "ins v0.d[1], x9\n"

                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "prfm pldl1keep, [%[b_ptr], #576]\n"

                "fmla v31.4s, v4.4s, v1.s[3]\n"

                //! UNROLL
                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "ldr d1, [%[a_ptr], #32]\n"

                "fmla v9.4s,  v6.4s, v0.s[0]\n"

                "fmla v10.4s, v7.4s, v0.s[0]\n"
                "ldr x8, [%[a_ptr], #40]\n"

                "fmla v11.4s, v5.4s, v0.s[1]\n"

                "fmla v12.4s, v6.4s, v0.s[1]\n"
                "ldr d2, [%[b_ptr], #48]\n"

                "fmla v13.4s, v7.4s, v0.s[1]\n"
                "ins v1.d[1], x8\n"

                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "ldr x10, [%[b_ptr], #56]\n"

                "fmla v15.4s, v6.4s, v0.s[2]\n"

                "fmla v16.4s, v7.4s, v0.s[2]\n"
                "ldr d3, [%[b_ptr], #64]\n"

                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "ins v2.d[1], x10\n"

                "fmla v18.4s, v6.4s, v0.s[3]\n"
                "ldr x11, [%[b_ptr], #72]\n"

                "fmla v19.4s, v7.4s, v0.s[3]\n"

                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "ldr d4, [%[b_ptr], #80]\n"

                "fmla v21.4s, v6.4s, v1.s[0]\n"
                "ins v3.d[1], x11\n"

                "fmla v22.4s, v7.4s, v1.s[0]\n"
                "ldr x12, [%[b_ptr], #88]\n"

                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "add %[b_ptr], %[b_ptr], #96\n"

                "fmla v24.4s, v6.4s, v1.s[1]\n"
                "ldr d0, [%[a_ptr], #48]\n"

                "fmla v25.4s, v7.4s, v1.s[1]\n"
                "ins v4.d[1], x12\n"

                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "ldr x9, [%[a_ptr], #56]\n"

                "fmla v27.4s, v6.4s, v1.s[2]\n"
                "add %[a_ptr], %[a_ptr], #64\n"

                "fmla v28.4s, v7.4s, v1.s[2]\n"
                "prfm pldl1keep, [%[b_ptr], #640]\n"

                "fmla v29.4s, v5.4s, v1.s[3]\n"
                "ins v0.d[1], x9\n"

                "fmla v30.4s, v6.4s, v1.s[3]\n"

                "fmla v31.4s, v7.4s, v1.s[3]\n"

                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "prfm pstl1keep, [x0]\n"

                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "ldr d1, [%[a_ptr]] \n"

                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "prfm pstl1keep, [x1]\n"

                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ldr x8, [%[a_ptr], #8] \n"

                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "prfm pstl1keep, [x2]\n"

                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ldr d5, [%[b_ptr]]\n"

                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "ins v1.d[1], x8\n"

                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "ldr x10, [%[b_ptr], #8]\n"

                "fmla v16.4s, v4.4s, v0.s[2]\n"

                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "ldr d6, [%[b_ptr], #16]\n"

                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "ins v5.d[1], x10\n"

                "fmla v19.4s, v4.4s, v0.s[3]\n"
                "ldr x11, [%[b_ptr], #24]\n"

                "fmla v20.4s, v2.4s, v1.s[0]\n"

                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "ldr d0, [%[a_ptr], #16]\n"

                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "ins v6.d[1], x11\n"

                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "ldr x9, [%[a_ptr], #24]\n"

                "fmla v24.4s, v3.4s, v1.s[1]\n"

                "fmla v25.4s, v4.4s, v1.s[1]\n"
                "ldr d7, [%[b_ptr], #32]\n"

                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "ins v0.d[1], x9\n"

                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "ldr x12, [%[b_ptr], #40]\n"

                "fmla v28.4s, v4.4s, v1.s[2]\n"

                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "ins v7.d[1], x12\n"

                "fmla v31.4s, v4.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "ldr d1, [%[a_ptr], #32]\n"

                "fmla v9.4s,  v6.4s, v0.s[0]\n"

                "fmla v10.4s, v7.4s, v0.s[0]\n"
                "ldr x8, [%[a_ptr], #40]\n"

                "fmla v11.4s, v5.4s, v0.s[1]\n"

                "fmla v12.4s, v6.4s, v0.s[1]\n"
                "str q8, [x0]\n"

                "fmla v13.4s, v7.4s, v0.s[1]\n"
                "ins v1.d[1], x8\n"

                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "str q9, [x0, #16]\n"

                "fmla v15.4s, v6.4s, v0.s[2]\n"
                "str q10, [x0, #32]\n"

                "fmla v16.4s, v7.4s, v0.s[2]\n"
                "str q11, [x1]\n"

                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "str q12, [x1, #16]\n"

                "fmla v18.4s, v6.4s, v0.s[3]\n"
                "str q13, [x1, #32]\n"

                "fmla v19.4s, v7.4s, v0.s[3]\n"
                "str q14, [x2]\n"

                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "str q15, [x2, #16]\n"

                "fmla v21.4s, v6.4s, v1.s[0]\n"
                "str q16, [x2, #32]\n"

                "fmla v22.4s, v7.4s, v1.s[0]\n"
                "str q17, [x3]\n"

                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "str q18, [x3, #16]\n"

                "fmla v24.4s, v6.4s, v1.s[1]\n"
                "str q19, [x3, #32]\n"

                "fmla v25.4s, v7.4s, v1.s[1]\n"
                "str q20, [x4]\n"

                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "str q21, [x4, #16]\n"

                "fmla v27.4s, v6.4s, v1.s[2]\n"
                "str q22, [x4, #32]\n"

                "fmla v28.4s, v7.4s, v1.s[2]\n"
                "str q23, [x5]\n"

                "fmla v29.4s, v5.4s, v1.s[3]\n"
                "str q24, [x5, #16]\n"

                "fmla v30.4s, v6.4s, v1.s[3]\n"
                "str q25, [x5, #32]\n"

                "fmla v31.4s, v7.4s, v1.s[3]\n"

                "st1 {v26.4s, v27.4s, v28.4s}, [x6]\n"
                "st1 {v29.4s, v30.4s, v31.4s}, [x7]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ldr d1, [%[a_ptr]]\n"

                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "ldr x8, [%[a_ptr], #8]\n"

                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "str q8, [x0]\n"

                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "str q9, [x0, #16]\n"

                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "str q10, [x0, #32]\n"

                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ins v1.d[1], x8\n"

                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "str q11, [x1]\n"

                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "str q12, [x1, #16]\n"

                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "str q13, [x1, #32]\n"

                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "str q14, [x2]\n"

                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "str q15, [x2, #16]\n"

                "fmla v19.4s, v4.4s, v0.s[3]\n"
                "str q16, [x2, #32]\n"

                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "str q17, [x3]\n"

                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "str q18, [x3, #16]\n"

                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "str q19, [x3, #32]\n"

                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "str q20, [x4]\n"

                "fmla v24.4s, v3.4s, v1.s[1]\n"
                "str q21, [x4, #16]\n"

                "fmla v25.4s, v4.4s, v1.s[1]\n"
                "str q22, [x4, #32]\n"

                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "str q23, [x5]\n"

                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "str q24, [x5, #16]\n"

                "fmla v28.4s, v4.4s, v1.s[2]\n"
                "str q25, [x5, #32]\n"

                "fmla v29.4s, v2.4s, v1.s[3]\n"
                "str q26, [x6]\n"

                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "str q27, [x6, #16]\n"

                "fmla v31.4s, v4.4s, v1.s[3]\n"
                "str q28, [x6, #32]\n"

                "st1 {v29.4s, v30.4s, v31.4s}, [x7]\n"

                "6:\n"

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                  "v28", "v29", "v30", "v31", "x1", "x2", "x3", "x4", "x5",
                  "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "cc",
                  "memory");
#undef LOAD_LINE
#undef LOAD_C
    }

    // Overview of register layout:
    //
    // A 1x12 cell of Rhs is stored in 32bit in v2-v7
    // A 8x1 cell of Lhs is stored in 32bit in (v0-v1)
    // A 8x12 block of accumulators is stored in 32bit in v8-v31.
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
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = reinterpret_cast<float*>(output);

// clang-format off
#define LOAD_LINE(v0, n)                \
    "cmp %w[n_remain], #4\n"            \
    "blt 100" n "f\n"                   \
    "ld1 {v" v0 ".4s}, [x" n "],#16\n"  \
    "b 101" n "f\n"                     \
    "100" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[0], [x" n "],#4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[1], [x" n "],#4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[2], [x" n "],#4\n" \
    "101" n ":\n"                       \

#define LOAD_C                   \
    LOAD_LINE("8", "0")          \
    LOAD_LINE("11", "1")         \
    LOAD_LINE("14", "2")         \
    LOAD_LINE("17", "3")         \
    LOAD_LINE("20", "4")         \
    LOAD_LINE("23", "5")         \
    LOAD_LINE("26", "6")         \
    LOAD_LINE("29", "7")         \


#define STORE_LINE(v0, n)               \
    "cmp %w[n_remain], #4\n"            \
    "blt 103" n "f\n"                   \
    "st1 {v" v0 ".4s}, [x" n " ],#16\n" \
    "b 104" n "f\n"                     \
    "103" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[0], [x" n "],#4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[1], [x" n "],#4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[2], [x" n "],#4\n" \
    "104" n ":\n"                       \


#define STORE_C                  \
    STORE_LINE("8", "0")         \
    STORE_LINE("11", "1")        \
    STORE_LINE("14", "2")        \
    STORE_LINE("17", "3")        \
    STORE_LINE("20", "4")        \
    STORE_LINE("23", "5")        \
    STORE_LINE("26", "6")        \
    STORE_LINE("29", "7") \
    // clang-format on

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

                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "eor v20.16b, v20.16b, v20.16b\n"
                "eor v23.16b, v23.16b, v23.16b\n"
                "eor v26.16b, v26.16b, v26.16b\n"
                "eor v29.16b, v29.16b, v29.16b\n"

                "2: \n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ldr d5, [%[b_ptr]]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "ldr x10, [%[b_ptr], #8]\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "ins v5.d[1], x10\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "ldr d2, [%[b_ptr], #16]\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "ldr x10, [%[b_ptr], #24]\n"
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "ins v2.d[1], x10\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "add %[b_ptr], %[b_ptr], #32\n"
                "fmla v29.4s, v5.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "fmla v29.4s, v5.4s, v1.s[3]\n"

                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr),
                  [n_remain] "+r"(n_remain)
                :
                : "v0", "v1", "v2", "v5", "v8", "v11", "v14", "v17", "v20",
                  "v23", "v26", "v29", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
                  "x10", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
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
                          float* output, int LDC, bool is_first_k,
                          int m_remain) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = output;

// clang-format off
#define LOAD_LINE(v0, v1, v2, n)                            \
    "cmp x10, #0\n"                                         \
    "beq 102f\n"                                            \
    "ld1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \
    "subs x10, x10, #1\n"

#define LOAD_C                      \
    "mov x10, %x[m_remain]\n"       \
    LOAD_LINE("8","9","10", "0")    \
    LOAD_LINE("11","12","13", "1")  \
    LOAD_LINE("14","15","16", "2")  \
    LOAD_LINE("17","18","19", "3")  \
    "102:\n"

#define STORE_LINE(v0, v1, v2, n)                           \
    "cmp x10, #0 \n"                                        \
    "beq 105f\n"                                            \
    "st1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \
    "subs x10, x10, #1\n"


#define STORE_C                          \
    "mov x10, %x[m_remain]\n"            \
    STORE_LINE("8","9","10", "0")        \
    STORE_LINE("11","12","13", "1")      \
    STORE_LINE("14","15","16", "2")      \
    STORE_LINE("17","18","19", "3")      \
    "105:\n"
        // clang-format on

        asm volatile(
                // load accumulator C
                "add x1, x0, %x[LDC]\n"
                "add x2, x1, %x[LDC]\n"
                "add x3, x2, %x[LDC]\n"

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
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ldr d5, [%[b_ptr]]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "ldr x20, [%[b_ptr], #8]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "ldr d6, [%[b_ptr], #16]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ldr x21, [%[b_ptr], #24]\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "ins v5.d[1], x20\n"

                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ldr d7, [%[b_ptr], #32]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "ldr x22, [%[b_ptr], #40]\n"

                "ld1 {v1.4s}, [%[a_ptr]], 16\n"

                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "ins v6.d[1], x21\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "ins v7.d[1], x22\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"

                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "ldr d2, [%[b_ptr], #48]\n"
                "fmla v9.4s,  v6.4s, v1.s[0]\n"
                "ldr x20, [%[b_ptr], #56]\n"
                "fmla v10.4s, v7.4s, v1.s[0]\n"
                "ldr d3, [%[b_ptr], #64]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "ldr x21, [%[b_ptr], #72]\n"
                "fmla v12.4s, v6.4s, v1.s[1]\n"
                "ldr d4, [%[b_ptr], #80]\n"
                "fmla v13.4s, v7.4s, v1.s[1]\n"
                "ldr x22, [%[b_ptr], #88]\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "ins v2.d[1], x20\n"
                "fmla v15.4s, v6.4s, v1.s[2]\n"
                "ins v3.d[1], x21\n"

                "ld1 {v0.4s}, [%[a_ptr]], 16\n"

                "fmla v16.4s, v7.4s, v1.s[2]\n"
                "ins v4.d[1], x22\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"
                "add %[b_ptr], %[b_ptr], #96\n"
                "fmla v18.4s, v6.4s, v1.s[3]\n"
                "subs %w[K], %w[K], #1\n"
                "fmla v19.4s, v7.4s, v1.s[3]\n"

                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "ldr d5, [%[b_ptr]]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "ldr x20, [%[b_ptr], #8]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "ldr d6, [%[b_ptr], #16]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ldr x21, [%[b_ptr], #24]\n"

                "ld1 {v1.4s}, [%[a_ptr]], 16\n"

                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "ldr d7, [%[b_ptr], #32]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ins v5.d[1], x20\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "ldr x22, [%[b_ptr], #40]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "ins v6.d[1], x21\n"

                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "ins v7.d[1], x22\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"

                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v9.4s,  v6.4s, v1.s[0]\n"
                "fmla v10.4s, v7.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "fmla v12.4s, v6.4s, v1.s[1]\n"
                "fmla v13.4s, v7.4s, v1.s[1]\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "fmla v15.4s, v6.4s, v1.s[2]\n"
                "fmla v16.4s, v7.4s, v1.s[2]\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"
                "fmla v18.4s, v6.4s, v1.s[3]\n"
                "fmla v19.4s, v7.4s, v1.s[3]\n"

                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr),
                  [m_remain] "+r"(m_remain)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                  "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
                  "v19", "x1", "x2", "x3", "x10", "x20", "x21", "x22", "cc",
                  "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
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
                         float* output, int LDC, bool is_first_k, int m_remain,
                         int n_remain) {
        const float* a_ptr = packA;
        const float* b_ptr = packB;
        int oddk = (K & 1);
        K = ((K + 1) / 2) - 1;

        LDC = LDC * sizeof(float);
        register float* outptr asm("x0") = output;

// clang-format off
#define LOAD_LINE(v0, n)                \
    "cmp x10, #0\n"                     \
    "beq 102f\n"                        \
    "cmp %w[n_remain], #4\n"            \
    "blt 100" n "f\n"                   \
    "ld1 {v" v0 ".4s}, [x" n "], 16\n"  \
    "b 101" n "f\n"                     \
    "100" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[0], [x" n "], 4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[1], [x" n "], 4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 101" n "f\n"                   \
    "ld1 {v" v0 ".s}[2], [x" n "], 4\n" \
    "101" n ":\n"                       \
    "subs x10, x10, #1\n"

#define LOAD_C                  \
    "mov x10, %x[m_remain]\n"   \
    LOAD_LINE("8", "0")         \
    LOAD_LINE("11", "1")        \
    LOAD_LINE("14", "2")        \
    LOAD_LINE("17", "3")        \
    "102:\n"

#define STORE_LINE(v0, n)               \
    "cmp x10, #0 \n"                    \
    "beq 105f\n"                        \
    "cmp %w[n_remain], #4\n"            \
    "blt 103" n "f\n"                   \
    "st1 {v" v0 ".4s}, [x" n " ], 16\n" \
    "b 104" n "f\n"                     \
    "103" n ":\n"                       \
    "cmp %w[n_remain], #0\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[0], [x" n "], 4\n" \
    "cmp %w[n_remain], #1\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[1], [x" n "], 4\n" \
    "cmp %w[n_remain], #2\n"            \
    "beq 104" n "f\n"                   \
    "st1 {v" v0 ".s}[2], [x" n "], 4\n" \
    "104" n ":\n"                       \
    "subs x10, x10, #1\n"


#define STORE_C                 \
    "mov x10, %x[m_remain]\n"   \
    STORE_LINE("8", "0")        \
    STORE_LINE("11", "1")       \
    STORE_LINE("14", "2")       \
    STORE_LINE("17", "3")       \
    "105:\n"
        // clang-format on

        asm volatile(
                // load accumulator C
                "add x1, x0, %x[LDC]\n"
                "add x2, x1, %x[LDC]\n"
                "add x3, x2, %x[LDC]\n"

                "cmp %w[is_first_k], #1\n"
                "beq 1f\n" LOAD_C

                "b 2f\n"

                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"

                "2: \n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"

                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"

                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"

                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "6:\n" STORE_C

                : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                  [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
                  [oddk] "+r"(oddk), [outptr] "+r"(outptr),
                  [n_remain] "+r"(n_remain), [m_remain] "+r"(m_remain)
                :
                : "v0", "v1", "v2", "v5", "v8", "v11", "v14", "v17", "x1", "x2",
                  "x3", "x10", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
    }
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
