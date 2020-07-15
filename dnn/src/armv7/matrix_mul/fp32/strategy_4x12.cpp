/**
 * \file dnn/src/armv7/matrix_mul/fp32/strategy_4x12.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/armv7/matrix_mul/fp32/strategy.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

namespace {

// Overview of register layout:
//
// A 1x12 cell of Rhs is stored in 32bit in q1-q3
// A 4x1 cell of Lhs is stored in 132bit in q0
// A 4x12 block of accumulators is stored in 32bit in q4-q15.
//
//                +--------+--------+--------+
//                | v1[0-3]| v2[0-3]| v3[0-3]|
//           Rhs  +--------+--------+--------+
//
//                |        |        |        |
//
//    Lhs         |        |        |        |
//
//  +--+ - - - -  +--------+--------+--------+
//  |v0|          | v4[0-3]| v5[0-3]| v6[0-3]|
//  |v0|          | v7[0-3]| v8[0-3]| v9[0-3]|
//  |v0|          |v10[0-3]|v11[0-3]|v12[0-3]|
//  |v0|          |v13[0-3]|v14[0-3]|v15[0-3]|
//  +--+ - - - -  +--------+--------+--------+
//
//                        Accumulator
void kern_4x12(const float* packA, const float* packB, int K, float* output,
               int LDC, bool is_first_k, int m_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(float);
    register float* outptr asm("r0") = reinterpret_cast<float*>(output);

// clang-format off
#define LOAD_LINE(d0, d1, d2, d3, d4, d5, n)             \
    "cmp r10, #0\n"                                      \
    "beq 100f\n"                                         \
    "mov r9, r" n "\n"                                   \
    "vld1.32 {d" d0 ",d" d1 ",d" d2 ",d" d3 "}, [r9]!\n" \
    "vld1.32 {d" d4 ",d" d5 "}, [r9]\n"                  \
    "subs r10, r10, #1\n"

#define LOAD_C                                         \
    "mov r10, %[m_remain]\n"                           \
    LOAD_LINE("8", "9", "10", "11", "12", "13", "0")   \
    LOAD_LINE("14", "15", "16", "17", "18", "19", "1") \
    LOAD_LINE("20", "21", "22", "23", "24", "25", "2") \
    LOAD_LINE("26", "27", "28", "29", "30", "31", "3") \
    "100:\n"

#define STORE_LINE(d0, d1, d2, d3, d4, d5, n)            \
    "cmp r10, #0\n"                                      \
    "beq 101f\n"                                         \
    "mov r9, r" n "\n"                                   \
    "vst1.32 {d" d0 ",d" d1 ",d" d2 ",d" d3 "}, [r9]!\n" \
    "vst1.32 {d" d4 ",d" d5 "}, [r9]\n"                  \
    "subs r10, r10, #1\n"

#define STORE_C                                         \
    "mov r10, %[m_remain]\n"                            \
    STORE_LINE("8", "9", "10", "11", "12", "13", "0")   \
    STORE_LINE("14", "15", "16", "17", "18", "19", "1") \
    STORE_LINE("20", "21", "22", "23", "24", "25", "2") \
    STORE_LINE("26", "27", "28", "29", "30", "31", "3") \
    "101:\n"
    // clang-format on

    asm volatile(
            // load accumulator C
            "add r1, r0, %[LDC]\n"
            "add r2, r1, %[LDC]\n"
            "add r3, r2, %[LDC]\n"

            "cmp %[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "veor.32 q4, q4, q4\n"
            "veor.32 q5, q5, q5\n"
            "veor.32 q6, q6, q6\n"
            "veor.32 q7, q7, q7\n"
            "veor.32 q8, q8, q8\n"
            "veor.32 q9, q9, q9\n"
            "veor.32 q10, q10, q10\n"
            "veor.32 q11, q11, q11\n"
            "veor.32 q12, q12, q12\n"
            "veor.32 q13, q13, q13\n"
            "veor.32 q14, q14, q14\n"
            "veor.32 q15, q15, q15\n"

            "2: \n"
            "vld1.32 {d2, d3, d4, d5}, [%[b_ptr]]!\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"

            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vmla.f32 q15, q3, d1[1]\n"

            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vld1.32 {d2, d3, d4, d5}, [%[b_ptr]]!\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vmla.f32 q15, q3, d1[1]\n"

            "vld1.32 {d2, d3, d4, d5}, [%[b_ptr]]!\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            "subs %[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vmla.f32 q15, q3, d1[1]\n"

            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vld1.32 {d2, d3, d4, d5}, [%[b_ptr]]!\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vmla.f32 q15, q3, d1[1]\n"
            "b 6f\n"

            // odd tail
            "5:\n"
            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vmla.f32 q4, q1, d0[0]\n"
            "vmla.f32 q5, q2, d0[0]\n"
            "vmla.f32 q6, q3, d0[0]\n"
            "vmla.f32 q7, q1, d0[1]\n"
            "vmla.f32 q8, q2, d0[1]\n"
            "vmla.f32 q9, q3, d0[1]\n"
            "vmla.f32 q10, q1, d1[0]\n"
            "vmla.f32 q11, q2, d1[0]\n"
            "vmla.f32 q12, q3, d1[0]\n"
            "vmla.f32 q13, q1, d1[1]\n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vmla.f32 q15, q3, d1[1]\n"

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
              [m_remain] "+r"(m_remain), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31", "r1", "r2", "r3", "r9", "r10", "cc",
              "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A 2x4 cell of Rhs is stored in 32bit in q2 - q3
// A 4x2 cell of Lhs is stored in 32bit in q0 - q1
// A 4x4 block of accumulators is stored in 32bit in q4-q6
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
    register float* outptr asm("r0") = output;

// clang-format off
#define LOAD_LINE(d0, d1, n)                 \
    "cmp r10, #0\n"                          \
    "beq 102f\n"                             \
    "cmp %[n_remain], #4\n"                  \
    "blt 100" n "f\n"                        \
    "vld1.32 {d" d0 ", d" d1 "}, [r" n "]\n" \
    "b 101" n "f\n"                          \
    "100" n ":\n"                            \
    "cmp %[n_remain], #0\n"                  \
    "beq 101" n "f\n"                        \
    "vld1.32 {d" d0 "[0]}, [r" n "]!\n"      \
    "cmp %[n_remain], #1\n"                  \
    "beq 101" n "f\n"                        \
    "vld1.32 {d" d0 "[1]}, [r" n "]!\n"      \
    "cmp %[n_remain], #2\n"                  \
    "beq 101" n "f\n"                        \
    "vld1.32 {d" d1 "[0]}, [r" n "]!\n"      \
    "101" n ":\n"                            \
    "subs r10, r10, #1\n"

#define LOAD_C                 \
    "mov r10, %[m_remain]\n"   \
    LOAD_LINE("8", "9", "0")   \
    LOAD_LINE("10", "11", "1") \
    LOAD_LINE("12", "13", "2") \
    LOAD_LINE("14", "15", "3") \
    "102:\n"

#define STORE_LINE(d0, d1, n)                  \
    "cmp r10, #0 \n"                           \
    "beq 105f\n"                               \
    "cmp %[n_remain], #4\n"                    \
    "blt 103" n "f\n"                          \
    "vst1.32 {d" d0 ", d" d1 "}, [r" n " ]!\n" \
    "b 104" n "f\n"                            \
    "103" n ":\n"                              \
    "cmp %[n_remain], #0\n"                    \
    "beq 104" n "f\n"                          \
    "vst1.32 {d" d0 "[0]}, [r" n "]!\n"        \
    "cmp %[n_remain], #1\n"                    \
    "beq 104" n "f\n"                          \
    "vst1.32 {d" d0 "[1]}, [r" n "]!\n"        \
    "cmp %[n_remain], #2\n"                    \
    "beq 104" n "f\n"                          \
    "vst1.32 {d" d1 "[0]}, [r" n "]!\n"        \
    "104" n ":\n"                              \
    "subs r10, r10, #1\n"


#define STORE_C                 \
    "mov r10, %[m_remain]\n"    \
    STORE_LINE("8", "9", "0")   \
    STORE_LINE("10", "11", "1") \
    STORE_LINE("12", "13", "2") \
    STORE_LINE("14", "15", "3") \
    "105:\n"
    // clang-format on

    asm volatile(
            // load accumulator C
            "add r1, r0, %[LDC]\n"
            "add r2, r1, %[LDC]\n"
            "add r3, r2, %[LDC]\n"

            "cmp %[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "veor.32 q4, q4, q4\n"
            "veor.32 q5, q5, q5\n"
            "veor.32 q6, q6, q6\n"
            "veor.32 q7, q7, q7\n"

            "2: \n"
            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
            "cmp %[K], #0\n"
            "beq 4f\n"

            "3:\n"
            "vld1.32 {d2, d3}, [%[a_ptr]]!\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            "vmla.f32 q4, q2, d0[0]\n"
            "vmla.f32 q5, q2, d0[1]\n"
            "vmla.f32 q6, q2, d1[0]\n"
            "vmla.f32 q7, q2, d1[1]\n"

            "vld1.32 {d0, d1}, [%[a_ptr]]!\n"
            "vld1.32 {d4, d5}, [%[b_ptr]]!\n"
            "vmla.f32 q4, q3, d2[0]\n"
            "vmla.f32 q5, q3, d2[1]\n"
            "vmla.f32 q6, q3, d3[0]\n"
            "vmla.f32 q7, q3, d3[1]\n"

            "subs %[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %[oddk], #1\n"
            "beq 5f\n"

            // Even tail
            "vld1.32 {d2, d3}, [%[a_ptr]]!\n"
            "vld1.32 {d6, d7}, [%[b_ptr]]!\n"
            "vmla.f32 q4, q2, d0[0]\n"
            "vmla.f32 q5, q2, d0[1]\n"
            "vmla.f32 q6, q2, d1[0]\n"
            "vmla.f32 q7, q2, d1[1]\n"

            "vmla.f32 q4, q3, d2[0]\n"
            "vmla.f32 q5, q3, d2[1]\n"
            "vmla.f32 q6, q3, d3[0]\n"
            "vmla.f32 q7, q3, d3[1]\n"

            "b 6f\n"

            // odd tail
            "5:\n"
            "vmla.f32 q4, q2, d0[0]\n"
            "vmla.f32 q5, q2, d0[1]\n"
            "vmla.f32 q6, q2, d1[0]\n"
            "vmla.f32 q7, q2, d1[1]\n"

            "6:\n" STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k), [oddk] "+r"(oddk),
              [m_remain] "+r"(m_remain), [n_remain] "+r"(n_remain),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "r1", "r2", "r3", "r10", "cc",
              "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

void sgemm_4x12_pack_A_n(float* outptr, const float* inptr, int ldin, int y0,
                         int ymax, int k0, int kmax) {
    float zerobuff[4];
    std::memset(zerobuff, 0, sizeof(float) * 4);

    int y = y0;
    for (; y + 3 < ymax; y += 4) {
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
        }

        if (K > 0) {
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K);
        }
    }
}

void sgemm_4x12_pack_A_t(float* out, const float* in, int ldin, int x0,
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

void sgemm_4x12_pack_B_n(float* out, const float* in, int ldin, int x0,
                         int xmax, int k0, int kmax) {
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
            interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
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

void sgemm_4x12_pack_B_t(float* out, const float* in, int ldin, int y0,
                         int ymax, int k0, int kmax) {
    float* outptr = out;
    const float* inptr = in;
    float zerobuff[4];
    std::memset(zerobuff, 0, sizeof(float) * 4);
    int K12 = 12 * (kmax - k0);

    int y = y0;

    for (; y + 12 <= ymax; y += 12) {
        int yi = y;
        for (; yi < y + 12; yi += 4) {
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
                                  48);
            }
            for (; x > 0; x--) {
                *outptr_inner++ = *inptr0++;
                *outptr_inner++ = *inptr1++;
                *outptr_inner++ = *inptr2++;
                *outptr_inner++ = *inptr3++;
                outptr_inner += 8;
            }
        }
        outptr += K12;
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

}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL(sgemm_4x12);

void sgemm_4x12::pack_A(float* out, const float* in, int ldin, int y0, int ymax,
                        int k0, int kmax, bool transpose_A) const {
    if (transpose_A) {
        sgemm_4x12_pack_A_t(out, in, ldin, y0, ymax, k0, kmax);
    } else {
        sgemm_4x12_pack_A_n(out, in, ldin, y0, ymax, k0, kmax);
    }
}

void sgemm_4x12::pack_B(float* out, const float* in, int ldin, int x0, int xmax,
                        int k0, int kmax, bool transpose_B) const {
    if (transpose_B) {
        sgemm_4x12_pack_B_t(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        sgemm_4x12_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

void sgemm_4x12::kern(const float* packA, const float* packB, size_t M,
                      size_t N, size_t K, float* C, size_t LDC, bool is_first_k,
                      const float*, float*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == C_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Float32);
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 12;
    const int K12 = K * 12;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m < M; m += A_INTERLEAVE) {
        float* output = C + (m * LDC);

        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x12(packA, cur_packB, K, output, LDC, is_first_k,
                      std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K12;
        }

        for (; n < N; n += 4) {
            kern_4x4(packA, cur_packB, K, output, LDC, is_first_k,
                     std::min<size_t>(M - m, 4), std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }

        packA += K4;
    }
}

// vim: syntax=cpp.doxygen
