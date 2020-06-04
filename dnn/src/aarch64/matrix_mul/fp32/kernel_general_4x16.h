/**
 * \file dnn/src/aarch64/matrix_mul/fp32/kernel_general_4x16.h
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
namespace matmul_general_4x16 {

// Overview of register layout:
//
// A 1x16 cell of Rhs is stored in 32bit in v1-v4
// A 4x1 cell of Lhs is stored in 32bit in v0
// A 4x16 block of accumulators is stored in 32bit in v10-v25.
//
//                +--------+--------+--------+--------+
//                | v2[0-3]| v3[0-3]| v4[0-3]| v5[0-3]|
//           Rhs  +--------+--------+--------+--------+
//
//                |        |        |        |        |
//
//    Lhs         |        |        |        |        |
//
//  +--+ - - - -  +--------+--------+--------+--------+
//  |v0|          |v10[0-3]|v11[0-3]|v12[0-3]|v13[0-3]|
//  |v0|          |v14[0-3]|v15[0-3]|v16[0-3]|v17[0-3]|
//  |v0|          |v18[0-3]|v19[0-3]|v20[0-3]|v21[0-3]|
//  |v0|          |v22[0-3]|v23[0-3]|v24[0-3]|v25[0-3]|
//  +--+ - - - -  +--------+--------+--------+--------+
//
//                        Accumulator
void kern_4x16(const float* packA, const float* packB, int K,
               float* output, int LDC, bool is_first_k, int m_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(float);
    register float* outptr asm("x0") = reinterpret_cast<float*>(output);

// clang-format off
#define LOAD_LINE(v0, v1, v2, v3, n)                                    \
    "cmp x10, #0\n"                                                     \
    "beq 100f\n"                                                        \
    "mov x9, x" n "\n"                                                  \
    "ld1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s, v" v3 ".4s}, [x9], 64\n"  \
    "subs x10, x10, #1\n"

#define LOAD_C                              \
    "mov x10, %x[m_remain]\n"               \
    LOAD_LINE("10", "11", "12", "13", "0")  \
    LOAD_LINE("14", "15", "16", "17", "1")  \
    LOAD_LINE("18", "19", "20", "21", "2")  \
    LOAD_LINE("22", "23", "24", "25", "3")  \
    "100:\n"

#define STORE_LINE(v0, v1, v2, v3, n)                                   \
    "cmp x10, #0\n"                                                     \
    "beq 101f\n"                                                        \
    "mov x9, x" n "\n"                                                  \
    "st1 {v" v0 ".4s, v" v1 ".4s, v" v2 ".4s, v" v3 ".4s}, [x9], 64\n"  \
    "subs x10, x10, #1\n"

#define STORE_C                             \
    "mov x10, %x[m_remain]\n"               \
    STORE_LINE("10", "11", "12", "13", "0") \
    STORE_LINE("14", "15", "16", "17", "1") \
    STORE_LINE("18", "19", "20", "21", "2") \
    STORE_LINE("22", "23", "24", "25", "3") \
    "101:\n"
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

            "2: \n"
            "ld1 {v2.4s, v3.4s, v4.4s, v5.4s}, [%[b_ptr]], 64\n"

            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v10.4s, v2.4s, v0.s[0]\n"
            "fmla v11.4s, v3.4s, v0.s[0]\n"
            "fmla v12.4s, v4.4s, v0.s[0]\n"
            "fmla v13.4s, v5.4s, v0.s[0]\n"
            "ld1 {v6.4s, v7.4s, v8.4s, v9.4s}, [%[b_ptr]], 64\n"
            "fmla v14.4s, v2.4s, v0.s[1]\n"
            "fmla v15.4s, v3.4s, v0.s[1]\n"
            "fmla v16.4s, v4.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v0.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v18.4s, v2.4s, v0.s[2]\n"
            "fmla v19.4s, v3.4s, v0.s[2]\n"
            "fmla v20.4s, v4.4s, v0.s[2]\n"
            "fmla v21.4s, v5.4s, v0.s[2]\n"
            "fmla v22.4s, v2.4s, v0.s[3]\n"
            "fmla v23.4s, v3.4s, v0.s[3]\n"
            "fmla v24.4s, v4.4s, v0.s[3]\n"
            "fmla v25.4s, v5.4s, v0.s[3]\n"

            "ld1 {v2.4s, v3.4s, v4.4s, v5.4s}, [%[b_ptr]], 64\n"
            "fmla v10.4s, v6.4s, v1.s[0]\n"
            "fmla v11.4s, v7.4s, v1.s[0]\n"
            "fmla v12.4s, v8.4s, v1.s[0]\n"
            "fmla v13.4s, v9.4s, v1.s[0]\n"
            "fmla v14.4s, v6.4s, v1.s[1]\n"
            "fmla v15.4s, v7.4s, v1.s[1]\n"
            "fmla v16.4s, v8.4s, v1.s[1]\n"
            "fmla v17.4s, v9.4s, v1.s[1]\n"
            "fmla v18.4s, v6.4s, v1.s[2]\n"
            "fmla v19.4s, v7.4s, v1.s[2]\n"
            "fmla v20.4s, v8.4s, v1.s[2]\n"
            "fmla v21.4s, v9.4s, v1.s[2]\n"
            "fmla v22.4s, v6.4s, v1.s[3]\n"
            "fmla v23.4s, v7.4s, v1.s[3]\n"
            "fmla v24.4s, v8.4s, v1.s[3]\n"
            "fmla v25.4s, v9.4s, v1.s[3]\n"

            "subs %w[K], %w[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v10.4s, v2.4s, v0.s[0]\n"
            "fmla v11.4s, v3.4s, v0.s[0]\n"
            "fmla v12.4s, v4.4s, v0.s[0]\n"
            "fmla v13.4s, v5.4s, v0.s[0]\n"
            "ld1 {v6.4s, v7.4s, v8.4s, v9.4s}, [%[b_ptr]], 64\n"
            "fmla v14.4s, v2.4s, v0.s[1]\n"
            "fmla v15.4s, v3.4s, v0.s[1]\n"
            "fmla v16.4s, v4.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v0.s[1]\n"
            "ld1 {v1.4s}, [%[a_ptr]], 16\n"
            "fmla v18.4s, v2.4s, v0.s[2]\n"
            "fmla v19.4s, v3.4s, v0.s[2]\n"
            "fmla v20.4s, v4.4s, v0.s[2]\n"
            "fmla v21.4s, v5.4s, v0.s[2]\n"
            "fmla v22.4s, v2.4s, v0.s[3]\n"
            "fmla v23.4s, v3.4s, v0.s[3]\n"
            "fmla v24.4s, v4.4s, v0.s[3]\n"
            "fmla v25.4s, v5.4s, v0.s[3]\n"

            "fmla v10.4s, v6.4s, v1.s[0]\n"
            "fmla v11.4s, v7.4s, v1.s[0]\n"
            "fmla v12.4s, v8.4s, v1.s[0]\n"
            "fmla v13.4s, v9.4s, v1.s[0]\n"
            "fmla v14.4s, v6.4s, v1.s[1]\n"
            "fmla v15.4s, v7.4s, v1.s[1]\n"
            "fmla v16.4s, v8.4s, v1.s[1]\n"
            "fmla v17.4s, v9.4s, v1.s[1]\n"
            "fmla v18.4s, v6.4s, v1.s[2]\n"
            "fmla v19.4s, v7.4s, v1.s[2]\n"
            "fmla v20.4s, v8.4s, v1.s[2]\n"
            "fmla v21.4s, v9.4s, v1.s[2]\n"
            "fmla v22.4s, v6.4s, v1.s[3]\n"
            "fmla v23.4s, v7.4s, v1.s[3]\n"
            "fmla v24.4s, v8.4s, v1.s[3]\n"
            "fmla v25.4s, v9.4s, v1.s[3]\n"

            "b 6f\n"

            // odd tail
            "5:\n"
            "ld1 {v0.4s}, [%[a_ptr]], 16\n"
            "fmla v10.4s, v2.4s, v0.s[0]\n"
            "fmla v11.4s, v3.4s, v0.s[0]\n"
            "fmla v12.4s, v4.4s, v0.s[0]\n"
            "fmla v13.4s, v5.4s, v0.s[0]\n"
            "fmla v14.4s, v2.4s, v0.s[1]\n"
            "fmla v15.4s, v3.4s, v0.s[1]\n"
            "fmla v16.4s, v4.4s, v0.s[1]\n"
            "fmla v17.4s, v5.4s, v0.s[1]\n"
            "fmla v18.4s, v2.4s, v0.s[2]\n"
            "fmla v19.4s, v3.4s, v0.s[2]\n"
            "fmla v20.4s, v4.4s, v0.s[2]\n"
            "fmla v21.4s, v5.4s, v0.s[2]\n"
            "fmla v22.4s, v2.4s, v0.s[3]\n"
            "fmla v23.4s, v3.4s, v0.s[3]\n"
            "fmla v24.4s, v4.4s, v0.s[3]\n"
            "fmla v25.4s, v5.4s, v0.s[3]\n"

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
              [m_remain] "+r"(m_remain), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "x1", "x2", "x3", "x9",
              "x10", "cc", "memory");

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
//                   +--------+
//                   | v2[0-3]|
//              Rhs  +--------+
//                   | v3[0-3]|
//                   +--------+
//
//                   |        |
//
//    Lhs            |        |
//
//  +--+--+ - - - -  +--------+
//  |v0|v1|          | v4[0-3]|
//  |v0|v1|          | v5[0-3]|
//  |v0|v1|          | v6[0-3]|
//  |v0|v1|          | v7[0-3]|
//  +--+--+ - - - -  +--------+
//
//                        Accumulator
void kern_4x4(const float* packA, const float* packB, int K, float* output,
              int LDC, bool is_first_k, int m_remain, int n_remain) {
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
    LOAD_LINE("4", "0")         \
    LOAD_LINE("5", "1")         \
    LOAD_LINE("6", "2")         \
    LOAD_LINE("7", "3")         \
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
    STORE_LINE("4", "0")        \
    STORE_LINE("5", "1")        \
    STORE_LINE("6", "2")        \
    STORE_LINE("7", "3")        \
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
                    "eor v4.16b, v4.16b, v4.16b\n"
                    "eor v5.16b, v5.16b, v5.16b\n"
                    "eor v6.16b, v6.16b, v6.16b\n"
                    "eor v7.16b, v7.16b, v7.16b\n"

                    "2: \n"
                    "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                    "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                    "cmp %w[K], #0\n"
                    "beq 4f\n"

                    "3:\n"
                    "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                    "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                    "fmla v4.4s, v2.4s, v0.s[0]\n"
                    "fmla v5.4s, v2.4s, v0.s[1]\n"
                    "fmla v6.4s, v2.4s, v0.s[2]\n"
                    "fmla v7.4s, v2.4s, v0.s[3]\n"

                    "ld1 {v0.4s}, [%[a_ptr]], 16\n"
                    "ld1 {v2.4s}, [%[b_ptr]], 16\n"
                    "fmla v4.4s, v3.4s, v1.s[0]\n"
                    "fmla v5.4s, v3.4s, v1.s[1]\n"
                    "fmla v6.4s, v3.4s, v1.s[2]\n"
                    "fmla v7.4s, v3.4s, v1.s[3]\n"

                    "subs %w[K], %w[K], #1\n"
                    "bne 3b\n"

                    "4:\n"
                    "cmp %w[oddk], #1\n"
                    "beq 5f\n"

                    // Even tail
                    "ld1 {v1.4s}, [%[a_ptr]], 16\n"
                    "ld1 {v3.4s}, [%[b_ptr]], 16\n"
                    "fmla v4.4s, v2.4s, v0.s[0]\n"
                    "fmla v5.4s, v2.4s, v0.s[1]\n"
                    "fmla v6.4s, v2.4s, v0.s[2]\n"
                    "fmla v7.4s, v2.4s, v0.s[3]\n"

                    "fmla v4.4s, v3.4s, v1.s[0]\n"
                    "fmla v5.4s, v3.4s, v1.s[1]\n"
                    "fmla v6.4s, v3.4s, v1.s[2]\n"
                    "fmla v7.4s, v3.4s, v1.s[3]\n"

                    "b 6f\n"

                    // odd tail
                    "5:\n"
                    "fmla v4.4s, v2.4s, v0.s[0]\n"
                    "fmla v5.4s, v2.4s, v0.s[1]\n"
                    "fmla v6.4s, v2.4s, v0.s[2]\n"
                    "fmla v7.4s, v2.4s, v0.s[3]\n"

                    "6:\n" STORE_C

                    : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                      [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
                      [oddk] "+r"(oddk), [m_remain] "+r"(m_remain),
                      [n_remain] "+r"(n_remain), [outptr] "+r"(outptr)
                    :
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "x1",
                      "x2", "x3", "x10", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

void sgemm_4x16_pack_A_n(float * outptr, const float * inptr, int ldin, int y0,
                         int ymax, int k0, int kmax) {
    float zerobuff[4];
    std::memset(zerobuff, 0, sizeof(float) * 4);
    constexpr int PACK_SIZE = 4*4;

    int y = y0;
    for (; y + 3 < ymax; y += 4) {
       // printf("main loop pack_a_n %p \n",outptr);
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
            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += PACK_SIZE;
        }

        interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K);
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

            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += PACK_SIZE;
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

void sgemm_4x16_pack_A_t(float* out, const float* in, int ldin, int x0,
                         int xmax, int k0, int kmax) {
    int ksize = kmax - k0;
    int ksize4 = (ksize << 2);
    float* outptr_base = out;

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
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_4x4_1_s(inptr, inptr1, inptr2, inptr3,
                               outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 4 * 4;
    }

    for (; k < kmax; k++) {
        const float* inptr = in + k * ldin + x0;
        prefetch_3x(inptr);
        int x = x0;
        auto outptr = outptr_base;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_1x4_1_s(inptr, outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_1(inptr, outptr, 4, xmax - x);
        }

        outptr_base += 4;
    }
}

void sgemm_4x16_pack_B_n(float* out, const float* in, int ldin,
                         int x0, int xmax, int k0, int kmax) {
    int ksize = kmax - k0;
    int ksize16 = ksize * 16;
    int ksize4 = (ksize << 2);
    float* outptr_base = out;
    float* outptr_base4 = outptr_base + (xmax - x0) / 16 * ksize16;

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
        for (; x + 16 <= xmax; x += 16) {
            auto outptr_interleave = outptr;
            interleave_4x16_1_s(inptr, inptr1, inptr2, inptr3,
                                outptr_interleave);
            outptr += ksize16;
        }
        outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_4x4_1_s(inptr, inptr1, inptr2, inptr3,
                               outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 16 * 4;
        outptr_base4 += 4 * 4;
    }

    for (; k < kmax; k++) {
        const float* inptr = in + k * ldin + x0;
        prefetch_3x(inptr);
        int x = x0;
        auto outptr = outptr_base;
        for (; x + 16 <= xmax; x += 16) {
            auto outptr_interleave = outptr;
            interleave_1x16_1_s(inptr, outptr_interleave);
            outptr += ksize16;
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

        outptr_base += 16;
        outptr_base4 += 4;
    }
}

void sgemm_4x16_pack_B_t(float* out, const float* in, int ldin,
                         int y0, int ymax, int k0, int kmax) {
    float* outptr = out;
    const float* inptr = in;
    float zerobuff[4];
    std::memset(zerobuff, 0, sizeof(float) * 4);
    int K16 = 16 * (kmax - k0);

    int y = y0;

    for (; y + 16 <= ymax; y += 16) {
        int yi = y;
        for (; yi < y + 16; yi += 4) {
            const float* inptr0 = inptr + yi * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;
            float* outptr_inner = outptr + yi - y;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);

            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner,
                                  64);
                outptr_inner += 64;
            }
            for (; x > 0; x--) {
                *outptr_inner++ = *inptr0++;
                *outptr_inner++ = *inptr1++;
                *outptr_inner++ = *inptr2++;
                *outptr_inner++ = *inptr3++;
                outptr_inner += 12;
            }
        }
        outptr += K16;
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

            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += 16;
        }

        if (x > 0) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, x);
        }
    }
}

} // matmul_general_4x16
} // aarch64
} // megdnn

// vim: syntax=cpp.doxygen
