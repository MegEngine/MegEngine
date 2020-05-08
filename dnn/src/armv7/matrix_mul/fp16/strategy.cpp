/**
 * \file dnn/src/armv7/matrix_mul/fp16/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/matrix_mul/fp16/strategy.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

namespace {

// Overview of register layout:
//
// A 2x16 cell of Rhs is stored in 16bit in q1-q4
// A 4x2 cell of Lhs is stored in 16bit in q0
// A 4x16 block of accumulators is stored in 16bit in q5-q12.
//
//                   +--------+--------+
//                   | v1[0-7]| v2[0-7]|
//              Rhs  +--------+--------+
//                   | v3[0-7]| v4[0-7]|
//                   +--------+--------+
//
//                   |        |        |
//
//    Lhs            |        |        |
//
//  +--+--+ - - - -  +--------+--------+
//  |v0|v0|          | v5[0-7]| v6[0-7]|
//  |v0|v0|          | v7[0-7]| v8[0-7]|
//  |v0|v0|          | v9[0-7]|v10[0-7]|
//  |v0|v0|          |v11[0-7]|v12[0-7]|
//  +--+--+ - - - -  +--------+--------+
//
//                        Accumulator
void kern_4x16(const dt_float16* packA, const dt_float16* packB, int K,
               dt_float16* output, int LDC, bool is_first_k, int m_remain) {
    const __fp16* a_ptr = reinterpret_cast<const __fp16*>(packA);
    const __fp16* b_ptr = reinterpret_cast<const __fp16*>(packB);
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(__fp16);
    register __fp16* outptr asm("r0") = reinterpret_cast<__fp16*>(output);

// clang-format off
#define LOAD_LINE(d0, d1, d2, d3, n)                        \
    "cmp r10, #0\n"                                         \
    "beq 100f\n"                                            \
    "vld1.16 {d" d0 ",d" d1 ",d" d2 ",d" d3 "}, [r" n "]\n" \
    "subs r10, r10, #1\n"

#define LOAD_C                             \
    "mov r10, %[m_remain]\n"               \
    LOAD_LINE("10", "11", "12", "13", "0") \
    LOAD_LINE("14", "15", "16", "17", "1") \
    LOAD_LINE("18", "19", "20", "21", "2") \
    LOAD_LINE("22", "23", "24", "25", "3") \
    "100:\n"

#define STORE_LINE(d0, d1, d2, d3, n)                       \
    "cmp r10, #0\n"                                         \
    "beq 101f\n"                                            \
    "vst1.16 {d" d0 ",d" d1 ",d" d2 ",d" d3 "}, [r" n "]\n" \
    "subs r10, r10, #1\n"

#define STORE_C                             \
    "mov r10, %[m_remain]\n"                \
    STORE_LINE("10", "11", "12", "13", "0") \
    STORE_LINE("14", "15", "16", "17", "1") \
    STORE_LINE("18", "19", "20", "21", "2") \
    STORE_LINE("22", "23", "24", "25", "3") \
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
                    "veor.32 q5, q5, q5\n"
                    "veor.32 q6, q6, q6\n"
                    "veor.32 q7, q7, q7\n"
                    "veor.32 q8, q8, q8\n"
                    "veor.32 q9, q9, q9\n"
                    "veor.32 q10, q10, q10\n"
                    "veor.32 q11, q11, q11\n"
                    "veor.32 q12, q12, q12\n"

                    "2: \n"
                    "vld1.16 {d2, d3, d4, d5}, [%[b_ptr]]!\n"

                    "cmp %[K], #0\n"
                    "beq 4f\n"

                    "3:\n"
                    "vld1.16 {d0, d1}, [%[a_ptr]]!\n"
                    "vld1.16 {d6, d7, d8, d9}, [%[b_ptr]]!\n"
                    "vmla.f16 q5, q1, d0[0]\n"
                    "vmla.f16 q6, q2, d0[0]\n"
                    "vmla.f16 q7, q1, d0[1]\n"
                    "vmla.f16 q8, q2, d0[1]\n"
                    "vmla.f16 q9, q1, d0[2]\n"
                    "vmla.f16 q10, q2, d0[2]\n"
                    "vmla.f16 q11, q1, d0[3]\n"
                    "vmla.f16 q12, q2, d0[3]\n"

                    "vmla.f16 q5, q3, d1[0]\n"
                    "vmla.f16 q6, q4, d1[0]\n"
                    "vmla.f16 q7, q3, d1[1]\n"
                    "vmla.f16 q8, q4, d1[1]\n"
                    "vmla.f16 q9, q3, d1[2]\n"
                    "vmla.f16 q10, q4, d1[2]\n"
                    "vmla.f16 q11, q3, d1[3]\n"
                    "vmla.f16 q12, q4, d1[3]\n"

                    "vld1.16 {d2, d3, d4, d5}, [%[b_ptr]]!\n"
                    "subs %[K], #1\n"
                    "bne 3b\n"

                    "4:\n"
                    "cmp %[oddk], #1\n"
                    "beq 5f\n"

                    // Even tail
                    "vld1.16 {d0, d1}, [%[a_ptr]]!\n"
                    "vld1.16 {d6, d7, d8, d9}, [%[b_ptr]]!\n"
                    "vmla.f16 q5, q1, d0[0]\n"
                    "vmla.f16 q6, q2, d0[0]\n"
                    "vmla.f16 q7, q1, d0[1]\n"
                    "vmla.f16 q8, q2, d0[1]\n"
                    "vmla.f16 q9, q1, d0[2]\n"
                    "vmla.f16 q10, q2, d0[2]\n"
                    "vmla.f16 q11, q1, d0[3]\n"
                    "vmla.f16 q12, q2, d0[3]\n"

                    "vmla.f16 q5, q3, d1[0]\n"
                    "vmla.f16 q6, q4, d1[0]\n"
                    "vmla.f16 q7, q3, d1[1]\n"
                    "vmla.f16 q8, q4, d1[1]\n"
                    "vmla.f16 q9, q3, d1[2]\n"
                    "vmla.f16 q10, q4, d1[2]\n"
                    "vmla.f16 q11, q3, d1[3]\n"
                    "vmla.f16 q12, q4, d1[3]\n"
                    "b 6f\n"

                    // odd tail
                    "5:\n"
                    "vld1.16 {d0}, [%[a_ptr]]!\n"
                    "vmla.f16 q5, q1, d0[0]\n"
                    "vmla.f16 q6, q2, d0[0]\n"
                    "vmla.f16 q7, q1, d0[1]\n"
                    "vmla.f16 q8, q2, d0[1]\n"
                    "vmla.f16 q9, q1, d0[2]\n"
                    "vmla.f16 q10, q2, d0[2]\n"
                    "vmla.f16 q11, q1, d0[3]\n"
                    "vmla.f16 q12, q2, d0[3]\n"

                    "6:\n" STORE_C

                    : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                      [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
                      [oddk] "+r"(oddk), [m_remain] "+r"(m_remain),
                      [outptr] "+r"(outptr)
                    :
                    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                      "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16",
                      "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24",
                      "d25", "r1", "r2", "r3", "r10", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}


// Overview of register layout:
//
// A 2x4 cell of Rhs is stored in 16bit in q1
// A 4x2 cell of Lhs is stored in 16bit in q0
// A 4x4 block of accumulators is stored in 16bit in q2-q5
//
//                   +--------+
//                   | v1[0-3]|
//              Rhs  +--------+
//                   | v1[4-7]|
//                   +--------+
//
//                   |        |
//
//    Lhs            |        |
//
//  +--+--+ - - - -  +--------+
//  |v0|v0|          | v2[0-3]|
//  |v0|v0|          | v3[0-3]|
//  |v0|v0|          | v4[0-3]|
//  |v0|v0|          | v5[0-3]|
//  +--+--+ - - - -  +--------+
//
//                        Accumulator
void kern_4x4(const dt_float16* packA, const dt_float16* packB, int K,
               dt_float16* output, int LDC, bool is_first_k, int m_remain,
               int n_remain) {
    const __fp16* a_ptr = reinterpret_cast<const __fp16*>(packA);
    const __fp16* b_ptr = reinterpret_cast<const __fp16*>(packB);
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    LDC = LDC * sizeof(__fp16);
    register __fp16* outptr asm("r0") = reinterpret_cast<__fp16*>(output);

// clang-format off
#define LOAD_LINE(d0, n)                 \
    "cmp r10, #0\n"                      \
    "beq 102f\n"                         \
    "cmp %[n_remain], #4\n"              \
    "blt 100" n "f\n"                    \
    "vld1.16 {d" d0 "}, [r" n "]\n"      \
    "b 101" n "f\n"                      \
    "100" n ":\n"                        \
    "cmp %[n_remain], #0\n"              \
    "beq 101" n "f\n"                    \
    "vld1.16 {d" d0 "[0]}, [r" n " ]!\n" \
    "cmp %[n_remain], #1\n"              \
    "beq 101" n "f\n"                    \
    "vld1.16 {d" d0 "[1]}, [r" n " ]!\n" \
    "cmp %[n_remain], #2\n"              \
    "beq 101" n "f\n"                    \
    "vld1.16 {d" d0 "[2]}, [r" n " ]!\n" \
    "101" n ":\n"                        \
    "subs r10, r10, #1\n"

#define LOAD_C               \
    "mov r10, %[m_remain]\n" \
    LOAD_LINE("4", "0")      \
    LOAD_LINE("6", "1")      \
    LOAD_LINE("8", "2")      \
    LOAD_LINE("10", "3")     \
    "102:\n"

#define STORE_LINE(d0, n)                \
    "cmp r10, #0 \n"                     \
    "beq 105f\n"                         \
    "cmp %[n_remain], #4\n"              \
    "blt 103" n "f\n"                    \
    "vst1.16 {d" d0 "}, [r" n " ]!\n"    \
    "b 104" n "f\n"                      \
    "103" n ":\n"                        \
    "cmp %[n_remain], #0\n"              \
    "beq 104" n "f\n"                    \
    "vst1.16 {d" d0 "[0]}, [r" n " ]!\n" \
    "cmp %[n_remain], #1\n"              \
    "beq 104" n "f\n"                    \
    "vst1.16 {d" d0 "[1]}, [r" n " ]!\n" \
    "cmp %[n_remain], #2\n"              \
    "beq 104" n "f\n"                    \
    "vst1.16 {d" d0 "[2]}, [r" n " ]!\n" \
    "104" n ":\n"                        \
    "subs r10, r10, #1\n"



#define STORE_C                \
    "mov r10, %[m_remain]\n"   \
    STORE_LINE("4", "0")       \
    STORE_LINE("6", "1")       \
    STORE_LINE("8", "2")       \
    STORE_LINE("10", "3")      \
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
                    "veor.32 q2, q2, q2\n"
                    "veor.32 q3, q3, q3\n"
                    "veor.32 q4, q4, q4\n"
                    "veor.32 q5, q5, q5\n"

                    "2: \n"
                    "cmp %[K], #0\n"
                    "beq 4f\n"

                    "3:\n"
                    "vld1.16 {d0, d1}, [%[a_ptr]]!\n"
                    "vld1.16 {d2, d3}, [%[b_ptr]]!\n"
                    "vmla.f16 d4, d2, d0[0]\n"
                    "vmla.f16 d6, d2, d0[1]\n"
                    "vmla.f16 d8, d2, d0[2]\n"
                    "vmla.f16 d10, d2, d0[3]\n"

                    "vmla.f16 d4, d3, d1[0]\n"
                    "vmla.f16 d6, d3, d1[1]\n"
                    "vmla.f16 d8, d3, d1[2]\n"
                    "vmla.f16 d10, d3, d1[3]\n"

                    "subs %[K], #1\n"
                    "bne 3b\n"

                    "4:\n"
                    "cmp %[oddk], #1\n"
                    "beq 5f\n"

                    // Even tail
                    "vld1.16 {d0, d1}, [%[a_ptr]]!\n"
                    "vld1.16 {d2, d3}, [%[b_ptr]]!\n"
                    "vmla.f16 d4, d2, d0[0]\n"
                    "vmla.f16 d6, d2, d0[1]\n"
                    "vmla.f16 d8, d2, d0[2]\n"
                    "vmla.f16 d10, d2, d0[3]\n"

                    "vmla.f16 d4, d3, d1[0]\n"
                    "vmla.f16 d6, d3, d1[1]\n"
                    "vmla.f16 d8, d3, d1[2]\n"
                    "vmla.f16 d10, d3, d1[3]\n"

                    "b 6f\n"

                    // odd tail
                    "5:\n"
                    "vld1.16 {d0}, [%[a_ptr]]!\n"
                    "vld1.16 {d2}, [%[b_ptr]]!\n"
                    "vmla.f16 d4, d2, d0[0]\n"
                    "vmla.f16 d6, d2, d0[1]\n"
                    "vmla.f16 d8, d2, d0[2]\n"
                    "vmla.f16 d10, d2, d0[3]\n"

                    "6:\n" STORE_C

                    : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
                      [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
                      [oddk] "+r"(oddk), [m_remain] "+r"(m_remain),
                      [n_remain] "+r"(n_remain), [outptr] "+r"(outptr)
                    :
                    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                      "d9", "d10", "r1", "r2", "r3", "r10", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

void hgemm_4x16_pack_A_n(__fp16* outptr, const __fp16* inptr, int ldin, int y0,
                         int ymax, int k0, int kmax) {
    __fp16 zerobuff[16];
    std::memset(zerobuff, 0, sizeof(__fp16) * 8);

    int y = y0;
    for (; y + 3 < ymax; y += 4) {
        const __fp16* inptr0 = inptr + y * ldin + k0;
        const __fp16* inptr1 = inptr0 + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = (kmax - k0);
        for (; K > 3; K -= 4) {
            transpose_4x4_1_h(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K);
    }

    for (; y < ymax; y += 4) {
        const __fp16* inptr0 = inptr + y * ldin + k0;
        const __fp16* inptr1 = inptr0 + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;

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
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4x4_1_h(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
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

void hgemm_4x16_pack_A_t(__fp16* out, const __fp16* in, int ldin, int x0,
                         int xmax, int k0, int kmax) {
    int ksize = kmax - k0;
    int ksize4 = (ksize << 2);
    __fp16* outptr_base = reinterpret_cast<__fp16*>(out);

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const __fp16* inptr = in + k * ldin + x0;
        const __fp16* inptr1 = inptr + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);
        prefetch_3x(inptr3);

        int x = x0;
        auto outptr = outptr_base;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_4x4_1_h(inptr, inptr1, inptr2, inptr3,
                               outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 4 * 4;
    }

    for (; k < kmax; k++) {
        const __fp16* inptr =
                reinterpret_cast<const __fp16*>(in + k * ldin + x0);
        prefetch_3x(inptr);
        int x = x0;
        auto outptr = outptr_base;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_1x4_1_h(inptr, outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_1(inptr, outptr, 4, xmax - x);
        }

        outptr_base += 4;
    }

}


void hgemm_4x16_pack_B_n(__fp16* out, const __fp16* in, int ldin,
                         int x0, int xmax, int k0, int kmax) {
    int ksize = kmax - k0;
    int ksize16 = (ksize << 4);
    int ksize4 = (ksize << 2);
    __fp16* outptr_base = reinterpret_cast<__fp16*>(out);
    __fp16* outptr_base4 = outptr_base + (xmax - x0) / 16 * ksize16;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const __fp16* inptr = in + k * ldin + x0;
        const __fp16* inptr1 = inptr + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);
        prefetch_3x(inptr3);

        int x = x0;
        auto outptr = outptr_base;
        for (; x + 16 <= xmax; x += 16) {
            auto outptr_interleave = outptr;
            interleave_4x16_1_h(inptr, inptr1, inptr2, inptr3,
                                outptr_interleave);
            outptr += ksize16;
        }
        outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_4x4_1_h(inptr, inptr1, inptr2, inptr3,
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
        const __fp16* inptr =
                reinterpret_cast<const __fp16*>(in + k * ldin + x0);
        prefetch_3x(inptr);
        int x = x0;
        auto outptr = outptr_base;
        for (; x + 16 <= xmax; x += 16) {
            auto outptr_interleave = outptr;
            interleave_1x16_1_h(inptr, outptr_interleave);
            outptr += ksize16;
        }
        outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_1x4_1_h(inptr, outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_1(inptr, outptr, 4, xmax - x);
        }

        outptr_base += 16;
        outptr_base4 += 4;
    }
}

void hgemm_4x16_pack_B_t(__fp16* out, const __fp16* in, int ldin,
                         int y0, int ymax, int k0, int kmax) {
    __fp16* outptr = out;
    const __fp16* inptr = in;
    __fp16 zerobuff[16];
    std::memset(zerobuff, 0, sizeof(__fp16) * 16);
    int K16 = 16 * (kmax - k0);

    int y = y0;

    for (; y + 16 <= ymax; y += 16) {
        int yi = y;
        for (; yi < y + 16; yi += 4) {
            const __fp16* inptr0 = inptr + yi * ldin + k0;
            const __fp16* inptr1 = inptr0 + ldin;
            const __fp16* inptr2 = inptr1 + ldin;
            const __fp16* inptr3 = inptr2 + ldin;
            __fp16* outptr_inner = outptr + yi - y;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);

            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_4x4_1_h(inptr0, inptr1, inptr2, inptr3, outptr_inner,
                                  32);
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
        const __fp16* inptr0 = inptr + y * ldin + k0;
        const __fp16* inptr1 = inptr0 + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;

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
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4x4_1_h(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (x > 0) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
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

MEGDNN_REG_GEMM_STRATEGY_IMPL(hgemm_4x16);

void hgemm_4x16::pack_A(dt_float16* out, const dt_float16* in, int ldin, int y0,
                        int ymax, int k0, int kmax, bool transpose_A) const {
    if (transpose_A) {
        hgemm_4x16_pack_A_t(reinterpret_cast<__fp16*>(out),
                            reinterpret_cast<const __fp16*>(in), ldin, y0, ymax,
                            k0, kmax);
    } else {
        hgemm_4x16_pack_A_n(reinterpret_cast<__fp16*>(out),
                            reinterpret_cast<const __fp16*>(in), ldin, y0, ymax,
                            k0, kmax);
    }
}

void hgemm_4x16::pack_B(dt_float16* out, const dt_float16* in, int ldin, int x0,
                        int xmax, int k0, int kmax, bool transpose_B) const {
    if (transpose_B) {
        hgemm_4x16_pack_B_t(reinterpret_cast<__fp16*>(out),
                            reinterpret_cast<const __fp16*>(in), ldin, x0, xmax,
                            k0, kmax);
    } else {
        hgemm_4x16_pack_B_n(reinterpret_cast<__fp16*>(out),
                            reinterpret_cast<const __fp16*>(in), ldin, x0, xmax,
                            k0, kmax);
    }
}

void hgemm_4x16::kern(const dt_float16* packA, const dt_float16* packB,
                      size_t M, size_t N, size_t K, dt_float16* C, size_t LDC,
                      bool is_first_k, const dt_float16*, dt_float16*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == C_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Float16);
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 16;
    const int K16 = K * 16;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m < M; m += A_INTERLEAVE) {
        dt_float16* output = C + (m * LDC);

        size_t n = 0;
        const dt_float16* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x16(packA, cur_packB, K, output, LDC, is_first_k,
                      std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K16;
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
#endif

// vim: syntax=cpp.doxygen
