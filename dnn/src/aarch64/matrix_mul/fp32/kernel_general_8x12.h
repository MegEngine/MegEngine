/**
 * \file dnn/src/aarch64/matrix_mul/fp32/kernel_general_8x12.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
struct matmul_general_8x12 {
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
    LOAD_LINE("29", "30", "31", "7")  \

#define STORE_LINE(v0, v1, v2, n)                           \
    "st1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s}, [x" n "]\n"  \

#define STORE_C                        \
    STORE_LINE("8", "9", "10", "0")    \
    STORE_LINE("11", "12", "13", "1")  \
    STORE_LINE("14", "15", "16", "2")  \
    STORE_LINE("17", "18", "19", "3")  \
    STORE_LINE("20", "21", "22", "4")  \
    STORE_LINE("23", "24", "25", "5")  \
    STORE_LINE("26", "27", "28", "6")  \
    STORE_LINE("29", "30", "31", "7") \
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
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "b 2f\n"

                "1:\n"
                "eor v8.16b, v8.16b, v8.16b\n"
                "eor v9.16b, v9.16b, v9.16b\n"
                "eor v10.16b, v10.16b, v10.16b\n"
                "prfm pstl1keep, [x0]\n"
                "eor v11.16b, v11.16b, v11.16b\n"
                "eor v12.16b, v12.16b, v12.16b\n"
                "eor v13.16b, v13.16b, v13.16b\n"
                "prfm pstl1keep, [x1]\n"
                "eor v14.16b, v14.16b, v14.16b\n"
                "eor v15.16b, v15.16b, v15.16b\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "eor v16.16b, v16.16b, v16.16b\n"
                "eor v17.16b, v17.16b, v17.16b\n"
                "prfm pstl1keep, [x2]\n"
                "eor v18.16b, v18.16b, v18.16b\n"
                "eor v19.16b, v19.16b, v19.16b\n"
                "eor v20.16b, v20.16b, v20.16b\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "eor v21.16b, v21.16b, v21.16b\n"
                "eor v22.16b, v22.16b, v22.16b\n"
                "prfm pstl1keep, [x3]\n"
                "eor v23.16b, v23.16b, v23.16b\n"
                "eor v24.16b, v24.16b, v24.16b\n"
                "eor v25.16b, v25.16b, v25.16b\n"
                "prfm pstl1keep, [x4]\n"
                "eor v26.16b, v26.16b, v26.16b\n"
                "eor v27.16b, v27.16b, v27.16b\n"
                "eor v28.16b, v28.16b, v28.16b\n"
                "prfm pstl1keep, [x5]\n"
                "eor v29.16b, v29.16b, v29.16b\n"
                "eor v30.16b, v30.16b, v30.16b\n"
                "eor v31.16b, v31.16b, v31.16b\n"
                "prfm pstl1keep, [x6]\n"

                "2: \n"
                "cmp %w[K], #0\n"
                "beq 4f\n"

                "3:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "prfm pldl1keep, [%[a_ptr], #64]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "fmla v24.4s, v3.4s, v1.s[1]\n"
                "fmla v25.4s, v4.4s, v1.s[1]\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "fmla v28.4s, v4.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"
                "prfm pldl1keep, [%[b_ptr], #64]\n"
                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "fmla v31.4s, v4.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "fmla v9.4s,  v6.4s, v0.s[0]\n"
                "fmla v10.4s, v7.4s, v0.s[0]\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                "fmla v12.4s, v6.4s, v0.s[1]\n"
                "fmla v13.4s, v7.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "fmla v15.4s, v6.4s, v0.s[2]\n"
                "fmla v16.4s, v7.4s, v0.s[2]\n"
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "fmla v18.4s, v6.4s, v0.s[3]\n"
                "fmla v19.4s, v7.4s, v0.s[3]\n"
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "fmla v21.4s, v6.4s, v1.s[0]\n"
                "fmla v22.4s, v7.4s, v1.s[0]\n"
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v24.4s, v6.4s, v1.s[1]\n"
                "fmla v25.4s, v7.4s, v1.s[1]\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "fmla v27.4s, v6.4s, v1.s[2]\n"
                "fmla v28.4s, v7.4s, v1.s[2]\n"
                "fmla v29.4s, v5.4s, v1.s[3]\n"
                "fmla v30.4s, v6.4s, v1.s[3]\n"
                "prfm pldl1keep, [%[b_ptr], #64]\n"
                "fmla v31.4s, v7.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "fmla v24.4s, v3.4s, v1.s[1]\n"
                "fmla v25.4s, v4.4s, v1.s[1]\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "fmla v28.4s, v4.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"
                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "fmla v31.4s, v4.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "fmla v9.4s,  v6.4s, v0.s[0]\n"
                "fmla v10.4s, v7.4s, v0.s[0]\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                "fmla v12.4s, v6.4s, v0.s[1]\n"
                "fmla v13.4s, v7.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "fmla v15.4s, v6.4s, v0.s[2]\n"
                "fmla v16.4s, v7.4s, v0.s[2]\n"
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "st1 {v8.4s, v9.4s, v10.4s}, [x0]\n"
                "fmla v18.4s, v6.4s, v0.s[3]\n"
                "fmla v19.4s, v7.4s, v0.s[3]\n"
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "fmla v21.4s, v6.4s, v1.s[0]\n"
                "st1 {v11.4s, v12.4s, v13.4s}, [x1]\n"
                "fmla v22.4s, v7.4s, v1.s[0]\n"
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "fmla v24.4s, v6.4s, v1.s[1]\n"
                "fmla v25.4s, v7.4s, v1.s[1]\n"
                "st1 {v14.4s, v15.4s, v16.4s}, [x2]\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "fmla v27.4s, v6.4s, v1.s[2]\n"
                "fmla v28.4s, v7.4s, v1.s[2]\n"
                "fmla v29.4s, v5.4s, v1.s[3]\n"
                "fmla v30.4s, v6.4s, v1.s[3]\n"
                "fmla v31.4s, v7.4s, v1.s[3]\n"
                "st1 {v17.4s, v18.4s, v19.4s}, [x3]\n"
                "st1 {v20.4s, v21.4s, v22.4s}, [x4]\n"
                "st1 {v23.4s, v24.4s, v25.4s}, [x5]\n"
                "st1 {v26.4s, v27.4s, v28.4s}, [x6]\n"
                "st1 {v29.4s, v30.4s, v31.4s}, [x7]\n"
                "b 6f\n"

                // odd tail
                "5:\n"
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "st1 {v8.4s, v9.4s, v10.4s}, [x0]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "st1 {v11.4s, v12.4s, v13.4s}, [x1]\n"
                "fmla v21.4s, v3.4s, v1.s[0]\n"
                "fmla v22.4s, v4.4s, v1.s[0]\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "fmla v24.4s, v3.4s, v1.s[1]\n"
                "st1 {v14.4s, v15.4s, v16.4s}, [x2]\n"
                "fmla v25.4s, v4.4s, v1.s[1]\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "fmla v27.4s, v3.4s, v1.s[2]\n"
                "fmla v28.4s, v4.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"
                "st1 {v17.4s, v18.4s, v19.4s}, [x3]\n"
                "fmla v30.4s, v3.4s, v1.s[3]\n"
                "fmla v31.4s, v4.4s, v1.s[3]\n"
                "st1 {v20.4s, v21.4s, v22.4s}, [x4]\n"
                "st1 {v23.4s, v24.4s, v25.4s}, [x5]\n"
                "st1 {v26.4s, v27.4s, v28.4s}, [x6]\n"
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
                  "x6", "x7", "cc", "memory");

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
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "ld1 {v5.4s}, [%[b_ptr]], 16\n"
                "fmla v20.4s, v2.4s, v1.s[0]\n"
                "fmla v23.4s, v2.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v2.4s, v1.s[2]\n"
                "fmla v29.4s, v2.4s, v1.s[3]\n"

                "fmla v8.4s,  v5.4s, v0.s[0]\n"
                "fmla v11.4s, v5.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v5.4s, v0.s[2]\n"
                "fmla v17.4s, v5.4s, v0.s[3]\n"
                "fmla v20.4s, v5.4s, v1.s[0]\n"
                "fmla v23.4s, v5.4s, v1.s[1]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v26.4s, v5.4s, v1.s[2]\n"
                "ld1 {v2.4s}, [%[b_ptr]], 16\n"
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
                  "cc", "memory");

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
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
                "fmla v19.4s, v4.4s, v0.s[3]\n"

                "fmla v8.4s,  v5.4s, v1.s[0]\n"
                "fmla v9.4s,  v6.4s, v1.s[0]\n"
                "fmla v10.4s, v7.4s, v1.s[0]\n"
                "fmla v11.4s, v5.4s, v1.s[1]\n"
                "ld1 {v2.4s, v3.4s, v4.4s}, [%[b_ptr]], 48\n"
                "fmla v12.4s, v6.4s, v1.s[1]\n"
                "fmla v13.4s, v7.4s, v1.s[1]\n"
                "fmla v14.4s, v5.4s, v1.s[2]\n"
                "fmla v15.4s, v6.4s, v1.s[2]\n"
                "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                "fmla v16.4s, v7.4s, v1.s[2]\n"
                "fmla v17.4s, v5.4s, v1.s[3]\n"
                "fmla v18.4s, v6.4s, v1.s[3]\n"
                "fmla v19.4s, v7.4s, v1.s[3]\n"

                "subs %w[K], %w[K], #1\n"
                "bne 3b\n"

                "4:\n"
                "cmp %w[oddk], #1\n"
                "beq 5f\n"

                // Even tail
                "fmla v8.4s,  v2.4s, v0.s[0]\n"
                "fmla v9.4s,  v3.4s, v0.s[0]\n"
                "fmla v10.4s, v4.4s, v0.s[0]\n"
                "fmla v11.4s, v2.4s, v0.s[1]\n"
                "ld1 {v5.4s, v6.4s, v7.4s}, [%[b_ptr]], 48\n"
                "fmla v12.4s, v3.4s, v0.s[1]\n"
                "fmla v13.4s, v4.4s, v0.s[1]\n"
                "fmla v14.4s, v2.4s, v0.s[2]\n"
                "fmla v15.4s, v3.4s, v0.s[2]\n"
                "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                "fmla v16.4s, v4.4s, v0.s[2]\n"
                "fmla v17.4s, v2.4s, v0.s[3]\n"
                "fmla v18.4s, v3.4s, v0.s[3]\n"
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
                  "v19", "x1", "x2", "x3", "x10", "cc", "memory");

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

    static void sgemm_8x12_pack_A_n(float* outptr, const float* inptr, int ldin,
                                    int y0, int ymax, int k0, int kmax) {
        float zerobuff[8];
        std::memset(zerobuff, 0, sizeof(float) * 8);
        constexpr int PACK_SIZE_32 = 4 * 8;
        constexpr int PACK_SIZE_16 = 4 * 4;
        int y = y0;
        for (; y + 7 < ymax; y += 8) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;
            const float* inptr4 = inptr3 + ldin;
            const float* inptr5 = inptr4 + ldin;
            const float* inptr6 = inptr5 + ldin;
            const float* inptr7 = inptr6 + ldin;
            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);
            prefetch_2x(inptr4);
            prefetch_2x(inptr5);
            prefetch_2x(inptr6);
            prefetch_2x(inptr7);
            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_8x4_1_s(inptr0, inptr1, inptr2, inptr3, inptr4,
                                  inptr5, inptr6, inptr7, outptr);
                outptr += PACK_SIZE_32;
            }
            for (; x > 0; x--) {
                *outptr++ = *inptr0++;
                *outptr++ = *inptr1++;
                *outptr++ = *inptr2++;
                *outptr++ = *inptr3++;
                *outptr++ = *inptr4++;
                *outptr++ = *inptr5++;
                *outptr++ = *inptr6++;
                *outptr++ = *inptr7++;
            }
        }

        for (; y < ymax; y += 4) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);

            int K = (kmax - k0);
            for (; K > 3; K -= 4) {
                if ((y + 3) >= ymax) {
                    switch ((y + 3) - ymax) {
                        /* Everything falls through in here */
                        case 2:
                            inptr1 = zerobuff;
                            MEGDNN_FALLTHRU
                        case 1:
                            inptr2 = zerobuff;
                            MEGDNN_FALLTHRU
                        case 0:
                            inptr3 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
                outptr += PACK_SIZE_16;
            }

            if (K > 0) {
                if (y + 3 >= ymax) {
                    switch (y + 3 - ymax) {
                        case 2:
                            inptr1 = zerobuff;
                            MEGDNN_FALLTHRU
                        case 1:
                            inptr2 = zerobuff;
                            MEGDNN_FALLTHRU
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

    static void sgemm_8x12_pack_A_t(float* out, const float* in, int ldin,
                                    int x0, int xmax, int k0, int kmax) {
        int ksize = kmax - k0;
        int ksize8 = (ksize << 3);
        int ksize4 = (ksize << 2);
        float* outptr_base = out;
        float* outptr_base4 = outptr_base + (xmax - x0) / 8 * ksize8;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* inptr = in + k * ldin + x0;
            const float* inptr1 = inptr + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_3x(inptr);
            prefetch_3x(inptr1);
            prefetch_3x(inptr2);
            prefetch_3x(inptr3);

            int x = x0;
            auto outptr = outptr_base;
            for (; x + 8 <= xmax; x += 8) {
                auto outptr_interleave = outptr;
                interleave_4x8_1_s(inptr, inptr1, inptr2, inptr3,
                                   outptr_interleave);
                outptr += ksize8;
            }
            outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                auto outptr_interleave = outptr;
                interleave_4x4_1_s(inptr, inptr1, inptr2, inptr3,
                                   outptr_interleave);
                outptr += ksize4;
            }
            if (x < xmax) {
                interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4,
                             xmax - x);
            }
            outptr_base += 4 * 8;
            outptr_base4 += 4 * 4;
        }

        for (; k < kmax; k++) {
            const float* inptr = in + k * ldin + x0;
            prefetch_3x(inptr);
            int x = x0;
            auto outptr = outptr_base;
            for (; x + 8 <= xmax; x += 8) {
                auto outptr_interleave = outptr;
                interleave_1x8_1_s(inptr, outptr_interleave);
                outptr += ksize8;
            }
            outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                auto outptr_interleave = outptr;
                interleave_1x4_1_s(inptr, outptr_interleave);
                outptr += ksize4;
            }
            if (x < xmax) {
                interleave_1(inptr, outptr, 4, xmax - x);
            }
            outptr_base += 8;
            outptr_base4 += 4;
        }
    }

    static void sgemm_8x12_pack_B_n(float* out, const float* in, int ldin,
                                    int x0, int xmax, int k0, int kmax) {
        int ksize = kmax - k0;
        int ksize12 = ksize * 12;
        int ksize4 = (ksize << 2);
        float* outptr_base = out;
        float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

        int k = k0;
        for (; k + 3 < kmax; k += 4) {
            const float* inptr = in + k * ldin + x0;
            const float* inptr1 = inptr + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_3x(inptr);
            prefetch_3x(inptr1);
            prefetch_3x(inptr2);
            prefetch_3x(inptr3);

            int x = x0;
            auto outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                auto outptr_interleave = outptr;
                interleave_4x12_1_s(inptr, inptr1, inptr2, inptr3,
                                    outptr_interleave);
                outptr += ksize12;
            }
            outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                auto outptr_interleave = outptr;
                interleave_4x4_1_s(inptr, inptr1, inptr2, inptr3,
                                   outptr_interleave);
                outptr += ksize4;
            }
            if (x < xmax) {
                interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4,
                             xmax - x);
            }
            outptr_base += 12 * 4;
            outptr_base4 += 4 * 4;
        }

        for (; k < kmax; k++) {
            const float* inptr = in + k * ldin + x0;
            prefetch_3x(inptr);
            int x = x0;
            auto outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                auto outptr_interleave = outptr;
                interleave_1x12_1_s(inptr, outptr_interleave);
                outptr += ksize12;
            }
            outptr = outptr_base4;
            for (; x + 4 <= xmax; x += 4) {
                auto outptr_interleave = outptr;
                interleave_1x4_1_s(inptr, outptr_interleave);
                outptr += ksize4;
            }
            if (x < xmax) {
                interleave_1(inptr, outptr, 4, xmax - x);
            }
            outptr_base += 12;
            outptr_base4 += 4;
        }
    }

    static void sgemm_8x12_pack_B_t(float* out, const float* in, int ldin,
                                    int y0, int ymax, int k0, int kmax) {
        float* outptr = out;
        const float* inptr = in;
        float zerobuff[12];
        std::memset(zerobuff, 0, sizeof(float) * 12);
        int y = y0;
        for (; y + 12 <= ymax; y += 12) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;
            const float* inptr4 = inptr3 + ldin;
            const float* inptr5 = inptr4 + ldin;
            const float* inptr6 = inptr5 + ldin;
            const float* inptr7 = inptr6 + ldin;
            const float* inptr8 = inptr7 + ldin;
            const float* inptr9 = inptr8 + ldin;
            const float* inptr10 = inptr9 + ldin;
            const float* inptr11 = inptr10 + ldin;
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
            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_12x4_1_s(inptr0, inptr1, inptr2, inptr3, inptr4,
                                   inptr5, inptr6, inptr7, inptr8, inptr9,
                                   inptr10, inptr11, outptr);
                outptr += 48;
            }
            for (; x > 0; x--) {
                *outptr++ = *inptr0++;
                *outptr++ = *inptr1++;
                *outptr++ = *inptr2++;
                *outptr++ = *inptr3++;
                *outptr++ = *inptr4++;
                *outptr++ = *inptr5++;
                *outptr++ = *inptr6++;
                *outptr++ = *inptr7++;
                *outptr++ = *inptr8++;
                *outptr++ = *inptr9++;
                *outptr++ = *inptr10++;
                *outptr++ = *inptr11++;
            }
        }

        for (; y < ymax; y += 4) {
            const float* inptr0 = inptr + y * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);

            /* Cope with ragged cases by copying from a buffer of zeroes instead
             */
            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                if ((y + 3) >= ymax) {
                    switch ((y + 3) - ymax) {
                        /* Everything falls through in here */
                        case 2:
                            inptr1 = zerobuff;
                            MEGDNN_FALLTHRU
                        case 1:
                            inptr2 = zerobuff;
                            MEGDNN_FALLTHRU
                        case 0:
                            inptr3 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
                outptr += 16;
            }

            if (x > 0) {
                if ((y + 3) >= ymax) {
                    switch ((y + 3) - ymax) {
                        /* Everything falls through in here */
                        case 2:
                            inptr1 = zerobuff;
                            MEGDNN_FALLTHRU
                        case 1:
                            inptr2 = zerobuff;
                            MEGDNN_FALLTHRU
                        case 0:
                            inptr3 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }
                interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, x);
            }
        }
    }
};
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
