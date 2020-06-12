/**
 * \file dnn/src/armv7/matrix_mul/int8/kernel_mk4_dot_8x4x4.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#if __ARM_FEATURE_DOTPROD

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"

namespace megdnn {
namespace armv7 {
namespace matmul_mk4_dot_8x4x4 {

// Overview of register layout:
//
// A 2x4x4 cell of Rhs is stored in 8bit in q1, q3.
// A 2x2x4x4 ping-pong cell of Lhs is stored in 8bit in q5, q7, q9, q11
// A 2x4x4 block of accumulators is stored in 8bit in q0, q2, q4, q6, q8, q10,
// q12, q14
//
//                              +--------+
//                        Rhs   |q1[0-16]|
//                              |q3[0-16]|
//                              +--------+
//    Lhs                       |        |
//  +-------+-------+ - - - -   +--------+
//  | q5[0-16]|                 | q0[0-4]|
//  | q7[0-16]|                 | q2[0-4]|
//  | q9[0-16]|                 | q4[0-4]|
//  |q11[0-16]|                 | q6[0-4]|
//  +---------+                 | q8[0-4]|
//                              |q10[0-4]|
//                              |q12[0-4]|
//                              |q14[0-4]|
//                              +--------+
//                              Accumulator

static void kern_8x4(const int8_t* packA, const int8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int n_remain) {
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;

    LDC = LDC * sizeof(int32_t);
    int32_t* outptr0 = output;
    int32_t* outptr1;

    size_t x0;

// clang-format off
#define LOAD_LINE(dr0, dr1, dr2, dr3, dr4, dr5, dr6, dr7, n)        \
	"mov %[x0], %[outptr" n "]\n"  \
    "cmp %[n_remain], #4\n"                \
    "blt 100" n "f\n"                       \
    "vld1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "vld1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "vld1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "vld1.32 {d" dr6 ", d" dr7 "}, [%[x0]]!\n"  \
    "b 101" n "f\n"                         \
    "100" n ":\n"                           \
    "cmp %[n_remain], #0\n"                \
    "beq 101" n "f\n"                       \
    "vld1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #1\n"                \
    "beq 101" n "f\n"                       \
    "vld1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #2\n"                \
    "beq 101" n "f\n"                       \
    "vld1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "101" n ":\n"

#define LOAD_C                              \
    LOAD_LINE("0", "1", "4", "5", "8", "9", "12", "13", "0")      \
    LOAD_LINE("16", "17", "20", "21", "24", "25", "28", "29", "1")  \
    
#define STORE_LINE(dr0, dr1, dr2, dr3, dr4, dr5, dr6, dr7, n)        \
    "mov %[x0], %[outptr" n "]\n"           \
    "cmp %[n_remain], #4\n"                \
    "blt 102" n "f\n"                       \
    "vst1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "vst1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "vst1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "vst1.32 {d" dr6 ", d" dr7 "}, [%[x0]]!\n"  \
    "b 103" n "f\n"                         \
    "102" n ":\n"                           \
    "cmp %[n_remain], #0\n"                \
    "beq 103" n "f\n"                       \
    "vst1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #1\n"                \
    "beq 103" n "f\n"                       \
    "vst1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #2\n"                \
    "beq 103" n "f\n"                       \
    "vst1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "103" n ":\n"

#define STORE_C                             \
    STORE_LINE("0", "1", "4", "5", "8", "9", "12", "13", "0")      \
    STORE_LINE("16", "17", "20", "21", "24", "25", "28", "29", "1")

    // clang-format on

    asm volatile(
            // load accumulator C
            "add %[outptr1], %[outptr0], %[LDC]\n"
            "cmp %[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "veor.s32 q0, q0, q0\n"
            "veor.s32 q2, q2, q2\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q8, q8, q8\n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q12, q12, q12\n"
            "veor.s32 q14, q14, q14\n"

            "2: \n"
            "cmp %[oddk], #0      \n"
            "beq 3f            \n"

            // parse the oddk
            "vld1.s8  {q1}, [%[b_ptr]]!\n"
            "vld1.s8  {q3}, [%[a_ptr]]!\n"
            "vld1.s8  {q5}, [%[a_ptr]]!\n"
            "vsdot.s8 q0 , q3, d2[0]\n"
            "vsdot.s8 q2 , q3, d2[1]\n"
            "vsdot.s8 q4 , q3, d3[0]\n"
            "vsdot.s8 q6 , q3, d3[1]\n"
            "vsdot.s8 q8 , q5, d2[0]\n"
            "vsdot.s8 q10 , q5, d2[1]\n"
            "vsdot.s8 q12 , q5, d3[0]\n"
            "vsdot.s8 q14 , q5, d3[1]\n"

            "cmp %[k], #0      \n"
            "beq 4f            \n"
            // Loop proper
            "3:\n"
            "vld1.s8  {q1}, [%[b_ptr]]!\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vld1.s8  {q5}, [%[a_ptr]]!\n"
            "vld1.s8  {q7}, [%[a_ptr]]!\n"
            "vsdot.s8 q0 , q5, d2[0]\n"
            "vsdot.s8 q2 , q5, d2[1]\n"
            "vsdot.s8 q4 , q5, d3[0]\n"
            "vsdot.s8 q6 , q5, d3[1]\n"
            "vld1.s8  {q9}, [%[a_ptr]]!\n"
            "vld1.s8  {q11}, [%[a_ptr]]!\n"
            "vsdot.s8 q8 , q7, d2[0]\n"
            "vsdot.s8 q10 , q7, d2[1]\n"
            "vsdot.s8 q12 , q7, d3[0]\n"
            "vsdot.s8 q14 , q7, d3[1]\n"

            "vsdot.s8 q0 , q9, d6[0]\n"
            "vsdot.s8 q2 , q9, d6[1]\n"
            "vsdot.s8 q4 , q9, d7[0]\n"
            "vsdot.s8 q6 , q9, d7[1]\n"
            "vsdot.s8 q8 , q11, d6[0]\n"
            "vsdot.s8 q10 , q11, d6[1]\n"
            "vsdot.s8 q12 , q11, d7[0]\n"
            "vsdot.s8 q14 , q11, d7[1]\n"

            "subs %[k], %[k], #1\n"
            "bne 3b\n"

            "4:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [LDC] "+r"(LDC),
              [oddk] "+r"(oddk), [is_first_k] "+r"(is_first_k),
              [n_remain] "+r"(n_remain), [k] "+r"(k), [outptr0] "+r"(outptr0),
              [outptr1] "+r"(outptr1), [x0] "+r"(x0)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
              "q11", "q12", "q14", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A 2x4x4 cell of Rhs is stored in 8bit in q1, q3.
// A 1x2x4x4 ping-pong cell of Lhs is stored in 8bit in q5, q7
// A 1x4x4 block of accumulators is stored in 8bit in q0, q2, q4, q6
//
//                              +--------+
//                        Rhs   |q1[0-16]|
//                              |q3[0-16]|
//                              +--------+
//    Lhs                       |        |
//  +-------+-------+ - - - -   +--------+
//  | q5[0-16]|                 | q0[0-4]|
//  | q7[0-16]|                 | q2[0-4]|
//  +---------+                 | q4[0-4]|
//                              | q6[0-4]|
//                              +--------+
//                              Accumulator

static void kern_4x4(const int8_t* packA, const int8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, int n_remain) {
    K /= 4;
    const int32_t* a_ptr = reinterpret_cast<const int32_t*>(packA);
    const int32_t* b_ptr = reinterpret_cast<const int32_t*>(packB);
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;
    LDC = LDC * sizeof(int32_t);

    int32_t* outptr0 = output;
    size_t x0;

// clang-format off
#define LOAD_LINE(dr0, dr1, dr2, dr3, dr4, dr5, dr6, dr7, n)        \
    "mov %[x0], %[outptr" n "]\n"           \
    "cmp %[n_remain], #4\n"                \
    "blt 100" n "f\n"                       \
    "vld1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "vld1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "vld1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "vld1.32 {d" dr6 ", d" dr7 "}, [%[x0]]!\n"  \
    "b 101" n "f\n"                         \
    "100" n ":\n"                           \
    "cmp %[n_remain], #0\n"                \
    "beq 101" n "f\n"                       \
    "vld1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #1\n"                \
    "beq 101" n "f\n"                       \
    "vld1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #2\n"                \
    "beq 101" n "f\n"                       \
    "vld1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "101" n ":\n"

#define LOAD_C                              \
    LOAD_LINE("0", "1", "4", "5", "8", "9", "12", "13", "0")      \

#define STORE_LINE(dr0, dr1, dr2, dr3, dr4, dr5, dr6, dr7, n)        \
    "mov %[x0], %[outptr" n "]\n"           \
    "cmp %[n_remain], #4\n"                \
    "blt 102" n "f\n"                       \
    "vst1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "vst1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "vst1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "vst1.32 {d" dr6 ", d" dr7 "}, [%[x0]]!\n"  \
    "b 103" n "f\n"                         \
    "102" n ":\n"                           \
    "cmp %[n_remain], #0\n"                \
    "beq 103" n "f\n"                       \
    "vst1.32 {d" dr0 ", d" dr1 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #1\n"                \
    "beq 103" n "f\n"                       \
    "vst1.32 {d" dr2 ", d" dr3 "}, [%[x0]]!\n"  \
    "cmp %[n_remain], #2\n"                \
    "beq 103" n "f\n"                       \
    "vst1.32 {d" dr4 ", d" dr5 "}, [%[x0]]!\n"  \
    "103" n ":\n"

#define STORE_C                             \
    STORE_LINE("0", "1", "4", "5", "8", "9", "12", "13", "0") \
    // clang-format on

    asm volatile(
            // load accumulator C
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"  //
            LOAD_C      //

            "b 2f\n"

            "1:\n"
            "veor.s32 q0, q0, q0\n"
            "veor.s32 q2, q2, q2\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q6, q6, q6\n"

            "2: \n"
            "cmp %[oddk], #0      \n"
            "beq 3f            \n"

            // parse the oddk
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vsdot.s8 q0 , q1, d6[0]\n"
            "vsdot.s8 q2 , q1, d6[1]\n"
            "vsdot.s8 q4 , q1, d7[0]\n"
            "vsdot.s8 q6 , q1, d7[1]\n"

            "cmp %[k], #0      \n"
            "beq 4f            \n"
            // Loop proper
            "3:\n"
            "vld1.s8  {q1}, [%[b_ptr]]!\n"
            "vld1.s8  {q5}, [%[a_ptr]]!\n"
            "vsdot.s8 q0 , q5, d2[0]\n"
            "vsdot.s8 q2 , q5, d2[1]\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vld1.s8  {q7}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q5, d3[0]\n"
            "vsdot.s8 q6 , q5, d3[1]\n"
            "vsdot.s8 q0 , q7, d6[0]\n"
            "vsdot.s8 q2 , q7, d6[1]\n"
            "vsdot.s8 q4 , q7, d7[0]\n"
            "vsdot.s8 q6 , q7, d7[1]\n"

            "subs %[k], %[k], #1\n"
            "bne 3b\n"

            "4:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [oddk] "+r"(oddk),
              [is_first_k] "+r"(is_first_k), [n_remain] "+r"(n_remain),
              [LDC] "+r"(LDC), [outptr0] "+r"(outptr0), [k] "+r"(k),
              [x0] "+r"(x0)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_dots8_8x4_pack_A(dt_int8* outptr, const dt_int8* inptr,
                                  int ldin, int y0, int ymax, int k0,
                                  int kmax) {
    int y = y0, y_start = y0 / 4;
    for (; y + 7 < ymax; y += 8, y_start += 2) {
        const int8_t* inptr0 = inptr + y_start * ldin + k0 * 4;
        const int8_t* inptr1 = inptr0 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        int K = kmax - k0;
        for (; K > 3; K -= 4) {
            interleave_2x4_4_b(inptr0, inptr1, outptr);
        }
    }
    for (; y + 3 < ymax; y += 4, ++y_start) {
        int K = kmax - k0;
        const int8_t* inptr0 = inptr + y_start * ldin + k0 * 4;
        std::memcpy(outptr, inptr0, sizeof(dt_int8) * K * 4);
    }
}

static void gemm_dots8_8x4_pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                  int x0, int xmax, int k0, int kmax) {
    const int ksize = kmax - k0;
    const int ksize4 = ksize * 4;
    int8_t* outptr = out;
    int8_t* outptr_base = out;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const int8_t* inptr = in + k / 4 * ldin + x0 * 4;
        prefetch_2x(inptr);

        outptr = outptr_base;
        int x = x0;
        for (; x + 3 < xmax; x += 4) {
            memcpy(outptr, inptr, sizeof(int8_t) * 16);
            outptr += ksize4;
            inptr += 16;
        }
        if (x < xmax) {
            int i = 0;
            for (; i < xmax - x; i++) {
                *outptr++ = *inptr++;
                *outptr++ = *inptr++;
                *outptr++ = *inptr++;
                *outptr++ = *inptr++;
            }
            for (; i < 4; i++) {
                *outptr++ = 0;
                *outptr++ = 0;
                *outptr++ = 0;
                *outptr++ = 0;
            }
        }
        outptr_base += 16;
    }
}

}  // namespace matmul_mk4_dot_8x4x4
}  // namespace armv7
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
