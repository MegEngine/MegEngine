/**
 * \file dnn/src/armv7/matrix_mul/quint8/kernel_dot_4x8x4.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if __ARM_FEATURE_DOTPROD

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"

namespace megdnn {
namespace armv7 {
namespace matmul_dot_4x8x4 {

// Overview of register layout:
//
// A 8x4 cell of Rhs is stored in 8bit in q2-q3.
// A 4x4 ping-pong cell of Lhs is stored in 8bit in q0-q1
// A 4x8 block of accumulators is stored in 8bit in q4-q11
// A 3x4 sum zero point ZA,ZB,ZAB stroed in q1,q12,q13
//
//                            +--------+--------+
//                            |q2[0-16]|q3[0-16]|
//                       Rhs  +--------+--------+
//
//                            |        |        |
//
//    Lhs                     |        |        |
//
//  +-------+-------+ - - - - +--------+--------+
//  |q0[0-4]|                 | q4[0-4]| q5[0-4]|
//  |q0[0-4]|                 | q6[0-4]| q7[0-4]|
//  |q0[0-4]|                 | q8[0-4]| q9[0-4]|
//  |q0[0-4]|                 |q10[0-4]|q11[0-4]|
//  +-------+-------+ - - - - +--------+--------+--------+
//
//                            Accumulator

static void kern_4x8(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, uint8_t zA,
                     uint8_t zB, uint32_t zAB, size_t m_remain = 4) {
    K /= 4;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;

    register uint8x16_t za asm("q14") = vdupq_n_u8(zA);
    register uint8x16_t zb asm("q15") = vdupq_n_u8(zB);

    register int32_t* outptr0 asm("r0") = output;
    register int32_t* outptr1 asm("r1") = outptr0 + LDC;
    register int32_t* outptr2 asm("r2") = outptr1 + LDC;
    register int32_t* outptr3 asm("r3") = outptr2 + LDC;

// clang-format off
#define LOAD_LINE(reg_index1, reg_index2, reg_index3, reg_index4, n)    \
    "cmp r12, #0 \n"                                                    \
    "beq 100f\n"                                                        \
    "vld1.32 {d" reg_index1 ", d" reg_index2 ", d" reg_index3 ", d"     \
    reg_index4 "}, [r" n "]\n"                                          \
    "subs r12, r12, #1\n"

#define LOAD_C                                   \
    "mov r12, %[m_remain]\n"                     \
    LOAD_LINE("8", "9", "10", "11", "0")         \
    LOAD_LINE("12", "13", "14", "15", "1")       \
    LOAD_LINE("16", "17", "18", "19", "2")       \
    LOAD_LINE("20", "21", "22", "23", "3")       \
    "100:\n"

#define STORE_LINE(reg_index1, reg_index2, reg_index3, reg_index4, n)   \
    "cmp r12, #0 \n"                                                    \
    "beq 101f\n"                                                        \
    "vst1.32 {d" reg_index1 ", d" reg_index2 ", d" reg_index3 ", d"     \
    reg_index4 "}, [r" n "]!\n"                                         \
    "subs r12, r12, #1\n"

#define STORE_C                                    \
    "mov r12, %[m_remain]\n"                       \
    STORE_LINE("8", "9", "10", "11", "0")          \
    STORE_LINE("12", "13", "14", "15", "1")        \
    STORE_LINE("16", "17", "18", "19", "2")        \
    STORE_LINE("20", "21", "22", "23", "3")        \
    "101:\n"

    // clang-format on

    asm volatile(
            "pld [%[a_ptr]]  \n"
            "pld [%[b_ptr]]  \n"

            "cmp %[is_first_k], #1      \n"
            "beq 5f                     \n"
            "cmp %[m_remain], #4        \n"
            "beq 7f                     \n" LOAD_C
            "b 6f                       \n"

            "7:\n"
            "vld1.s32 {q4, q5}, [%[outptr0]]\n"
            "vld1.s32 {q6, q7}, [%[outptr1]]\n"
            "vld1.s32 {q8, q9}, [%[outptr2]]\n"
            "vld1.s32 {q10, q11}, [%[outptr3]]\n"
            "b 6f                       \n"

            "5:\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"
            "veor.s32 q8, q8, q8\n"
            "veor.s32 q9, q9, q9\n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q11, q11, q11\n"

            "6: \n"
            "veor.s32 q12, q12, q12\n"
            "veor.s32 q13, q13, q13\n"
            "veor.s32 q1, q1, q1\n"

            "vld1.u8  {q0}, [%[a_ptr]]!\n"
            "vld1.u8  {q2}, [%[b_ptr]]!\n"

            // Skip loop if we are doing zero iterations of it.
            "cmp %[k], #0      \n"
            "beq 4f            \n"

            // Loop proper
            "1:\n"
            "vudot.u8 q12, q2, %[za] \n"
            "vld1.u8  {q3}, [%[b_ptr]]!\n"
            "vudot.u8 q1, q0, %[zb] \n"
            "vudot.u8 q4 , q2, d0[0]\n"
            "vudot.u8 q6 , q2, d0[1]\n"
            "vudot.u8 q8 , q2, d1[0]\n"
            "vudot.u8 q10 , q2, d1[1]\n"

            "vudot.u8 q5, q3, d0[0]\n"
            "vudot.u8 q7, q3, d0[1]\n"
            "vld1.u8  {q2}, [%[b_ptr]]!\n"
            "vudot.u8 q9, q3, d1[0]\n"
            "vudot.u8 q11, q3, d1[1]\n"

            "vld1.u8  {q0}, [%[a_ptr]]!\n"
            "vudot.u8 q13, q3, %[za] \n"
            ///////////////////////////////////////
            "vudot.u8 q12, q2, %[za] \n"
            "vld1.u8  {q3}, [%[b_ptr]]!\n"
            "vudot.u8 q1, q0, %[zb] \n"
            "vudot.u8 q4 , q2, d0[0]\n"
            "vudot.u8 q6 , q2, d0[1]\n"
            "vudot.u8 q8 , q2, d1[0]\n"
            "vudot.u8 q10 , q2, d1[1]\n"

            "vudot.u8 q5, q3, d0[0]\n"
            "vudot.u8 q7, q3, d0[1]\n"
            "vld1.u8  {q2}, [%[b_ptr]]!\n"
            "vudot.u8 q9, q3, d1[0]\n"
            "vudot.u8 q11, q3, d1[1]\n"

            "pld [%[b_ptr]]  \n"
            "subs  %[k], %[k], #1\n"
            "pld [%[a_ptr]]  \n"

            "vld1.u8  {q0}, [%[a_ptr]]!\n"
            "vudot.u8 q13, q3, %[za] \n"
            "bne  1b\n"

            "4:\n"
            // Branch to alternative tail for even K
            "cmp %[oddk], #0      \n"
            "beq 2f            \n"

            "vudot.u8 q12, q2, %[za] \n"
            "vld1.u8  {q3}, [%[b_ptr]]!\n"
            "vudot.u8 q1, q0, %[zb] \n"
            "vudot.u8 q4 , q2, d0[0]\n"
            "vudot.u8 q6 , q2, d0[1]\n"
            "vudot.u8 q8 , q2, d1[0]\n"
            "vudot.u8 q10 , q2, d1[1]\n"

            "vudot.u8 q5, q3, d0[0]\n"
            "vudot.u8 q7, q3, d0[1]\n"
            "vudot.u8 q9, q3, d1[0]\n"
            "vudot.u8 q11, q3, d1[1]\n"
            "vudot.u8 q13, q3, %[za] \n"

            "2:\n"
            "vdup.s32 q2, %[zab]\n"
            "vsub.s32 q1, q1, q2 \n"  // sub zab

            "vdup.s32 q3, d2[1]\n"
            "vdup.s32 q2, d2[0]\n"
            "vsub.s32 q6, q6, q3\n"
            "vsub.s32 q7, q7, q3\n"
            "vsub.s32 q4, q4, q2\n"
            "vsub.s32 q5, q5, q2\n"
            "vsub.s32 q6, q6, q12\n"
            "vsub.s32 q7, q7, q13\n"
            "vsub.s32 q4, q4, q12\n"
            "vsub.s32 q5, q5, q13\n"

            "vdup.s32 q2, d3[0]\n"
            "vdup.s32 q3, d3[1]\n"
            "vsub.s32 q8, q8, q2\n"
            "vsub.s32 q9, q9, q2\n"
            "vsub.s32 q10, q10, q3\n"
            "vsub.s32 q11, q11, q3\n"
            "vsub.s32 q8, q8, q12\n"
            "vsub.s32 q9, q9, q13\n"
            "vsub.s32 q10, q10, q12\n"
            "vsub.s32 q11, q11, q13\n"

            STORE_C

            : [k] "+r"(k), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [oddk] "+r"(oddk), [is_first_k] "+r"(is_first_k),
              [m_remain] "+r"(m_remain), [za] "+w"(za), [zb] "+w"(zb),
              [zab] "+r"(zAB), [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
              [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
              "q11", "q12", "q13", "cc", "r12", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}
// Overview of register layout:
//
// A 4x4 cell of Rhs is stored in 8bit in q2-q3.
// A 4x4 ping-pong cell of Lhs is stored in 8bit in q0-q1
// A 4x8 block of accumulators is stored in 8bit in q4-q10
// A 2x4 sum zero point ZA,ZB,ZAB stroed in q1,q12
//
//                            +--------+
//                            |q2[0-16]|
//                       Rhs  +--------+
//
//                            |        |
//
//    Lhs                     |        |
//
//  +-------+-------+ - - - - +--------+
//  |q0[0-4]|                 | q4[0-4]|
//  |q0[0-4]|                 | q6[0-4]|
//  |q0[0-4]|                 | q8[0-4]|
//  |q0[0-4]|                 |q10[0-4]|
//  +-------+-------+ - - - - +--------+--------+--------+
//
//                            Accumulator
static void kern_4x4(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, uint8_t zA,
                     uint8_t zB, uint32_t zAB, size_t m_remain = 4,
                     size_t n_remain = 4) {
    K /= 4;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = K / 2;
    LDC = LDC * sizeof(int32_t);

    register uint8x16_t za asm("q14") = vdupq_n_u8(zA);
    register uint8x16_t zb asm("q15") = vdupq_n_u8(zB);

    register int32_t* outptr0 asm("r2") = output;
    size_t x0 = 0;

    // clang-format off
#define LOAD_LINE(reg_index1, reg_index2, n)                    \
    "cmp %[x0], #0 \n"                                          \
    "beq 102f\n"                                                \
    "cmp %[n_remain], #4\n"                                     \
    "blt 100" n "f\n"                                           \
    "vld1.32 {d" reg_index1 ", d" reg_index2 "}, [r" n " ]!\n"  \
    "b 101" n "f\n"                                             \
    "100" n ":\n"                                               \
    "cmp %[n_remain], #0\n"                                     \
    "beq 101" n "f\n"                                           \
    "vld1.32 {d" reg_index1 "[0]}, [r" n " ]!\n"                \
    "cmp %[n_remain], #1\n"                                     \
    "beq 101" n "f\n"                                           \
    "vld1.32 {d" reg_index1 "[1]}, [r" n " ]!\n"                \
    "cmp %[n_remain], #2\n"                                     \
    "beq 101" n "f\n"                                           \
    "vld1.32 {d" reg_index2 "[0]}, [r" n " ]!\n"                \
    "101" n ":\n"                                               \
    "subs %[x0], %[x0], #1\n"

#define LOAD_C                      \
    "mov %[x0], %[m_remain]\n"      \
    "mov r0, r2\n"                  \
    "mov r1, r0\n"                  \
    LOAD_LINE("8", "9", "1")        \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    LOAD_LINE("12", "13", "1")      \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    LOAD_LINE("16", "17", "1")      \
    "add r1, r0, %[LDC]\n"          \
    LOAD_LINE("20", "21", "1")      \
    "102:\n"

#define STORE_LINE(reg_index1, reg_index2, n)                   \
    "cmp %[x0], #0 \n"                                          \
    "beq 105f\n"                                                \
    "cmp %[n_remain], #4\n"                                     \
    "blt 103" n "f\n"                                           \
    "vst1.32 {d" reg_index1 ", d" reg_index2 "}, [r" n " ]!\n"  \
    "b 104" n "f\n"                                             \
    "103" n ":\n"                                               \
    "cmp %[n_remain], #0\n"                                     \
    "beq 104" n "f\n"                                           \
    "vst1.32 {d" reg_index1 "[0]}, [r" n " ]!\n"                \
    "cmp %[n_remain], #1\n"                                     \
    "beq 104" n "f\n"                                           \
    "vst1.32 {d" reg_index1 "[1]}, [r" n " ]!\n"                \
    "cmp %[n_remain], #2\n"                                     \
    "beq 104" n "f\n"                                           \
    "vst1.32 {d" reg_index2 "[0]}, [r" n " ]!\n"                \
    "104" n ":\n"                                               \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                     \
    "mov %[x0], %[m_remain]\n"      \
    "mov r1, r2\n"                  \
    "mov r0, r2\n"                  \
    STORE_LINE("8", "9", "1")       \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    STORE_LINE("12", "13", "1")     \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    STORE_LINE("16", "17", "1")     \
    "add r1, r0, %[LDC]\n"          \
    STORE_LINE("20", "21", "1")     \
    "105:\n"

    // clang-format on

    asm volatile(
            "pld [%[a_ptr]]  \n"
            "pld [%[b_ptr]]  \n"

            "cmp %[is_first_k], #1      \n"
            "beq 5f                     \n" LOAD_C
            "b 6f                       \n"

            "5:\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q8, q8, q8\n"
            "veor.s32 q10, q10, q10\n"

            "6: \n"
            "veor.s32 q12, q12, q12\n"
            "veor.s32 q1, q1, q1\n"

            "vld1.u8  {q0}, [%[a_ptr]]!\n"
            "vld1.u8  {q2}, [%[b_ptr]]!\n"

            // Skip loop if we are doing zero iterations of it.
            "cmp %[k], #0      \n"
            "beq 4f            \n"

            // Loop proper
            "1:\n"
            "vudot.u8 q12, q2, %[za]\n"
            "vld1.u8  {q3}, [%[b_ptr]]!\n"
            "vudot.u8 q1, q0, %[zb] \n"
            "vudot.u8 q4, q2, d0[0]\n"
            "vld1.u8  {q5}, [%[a_ptr]]!\n"
            "vudot.u8 q6 , q2, d0[1]\n"
            "vudot.u8 q8 , q2, d1[0]\n"
            "vudot.u8 q10 , q2, d1[1]\n"

            ///////////////////////////////////////
            "vudot.u8 q12, q3, %[za] \n"
            "vudot.u8 q1, q5, %[zb] \n"
            "vld1.u8  {q2}, [%[b_ptr]]!\n"
            "vudot.u8 q4 , q3, d10[0]\n"
            "vudot.u8 q6 , q3, d10[1]\n"
            "vudot.u8 q8 , q3, d11[0]\n"
            "vudot.u8 q10 , q3, d11[1]\n"
            "vld1.u8  {q0}, [%[a_ptr]]!\n"

            "pld [%[b_ptr]]  \n"
            "subs  %[k], %[k], #1\n"
            "pld [%[a_ptr]]  \n"
            "bne  1b\n"

            "4:\n"
            // Branch to alternative tail for even K
            "cmp %[oddk], #0      \n"
            "beq 2f            \n"

            "vudot.u8 q12, q2, %[za]\n"
            "vudot.u8 q1, q0, %[zb] \n"
            "vudot.u8 q4, q2, d0[0]\n"
            "vudot.u8 q6 , q2, d0[1]\n"
            "vudot.u8 q8 , q2, d1[0]\n"
            "vudot.u8 q10 , q2, d1[1]\n"

            "2:\n"
            "vdup.s32 q2, %[zab]\n"
            "vsub.s32 q1, q1, q2 \n"  // sub zab

            "vdup.s32 q3, d2[1]\n"
            "vdup.s32 q2, d2[0]\n"
            "vdup.s32 q5, d3[0]\n"
            "vdup.s32 q7, d3[1]\n"

            "vsub.s32 q4, q4, q2\n"
            "vsub.s32 q6, q6, q3\n"
            "vsub.s32 q8, q8, q5\n"
            "vsub.s32 q10, q10, q7\n"

            "vsub.s32 q4, q4, q12\n"
            "vsub.s32 q6, q6, q12\n"
            "vsub.s32 q8, q8, q12\n"
            "vsub.s32 q10, q10, q12\n" STORE_C

            : [k] "+r"(k), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [oddk] "+r"(oddk), [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [za] "+w"(za), [zb] "+w"(zb), [zab] "+r"(zAB),
              [outptr0] "+r"(outptr0), [m_remain] "+r"(m_remain),
              [n_remain] "+r"(n_remain), [x0] "+r"(x0)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
              "q11", "q12", "q13", "r0", "r1", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}


static void gemm_quint8_4x8_pack_A_n(dt_uint8* outptr, const dt_uint8* inptr,
                                 int ldin, int y0, int ymax, int k0, int kmax) {
    uint8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(uint8_t) * 16);

    int y = y0;
    for (; y < ymax; y += 4) {
        const uint8_t* inptr0 = inptr + y * ldin + k0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
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
            interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }
        if (K > 0) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, K);
        }
    }
}

static void gemm_quint8_4x8_pack_A_t(dt_uint8* out, const dt_uint8* in, int ldin,
                                 int x0, int xmax, int k0, int kmax) {
    uint8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(uint8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize4 = round_up<int>(ksize, 4) * 4;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;

    int k = k0;
    for (; k < kmax; k += 4) {
        const uint8_t* inptr0 = in + k * ldin + x0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = x0;
        outptr = outptr_base;
        for (; x + 4 < xmax; x += 4) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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

            transpose_4x4_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += ksize4;
        }
        if (x < xmax) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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
            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }
        outptr_base += 4 * 4;
    }
}

static void gemm_quint8_4x8_pack_B_n(dt_uint8* out, const dt_uint8* in,
                                     int ldin, int x0, int xmax, int k0,
                                     int kmax) {
    uint8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(uint8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize8 = round_up<int>(ksize, 4) * 8;
    const int ksize4 = round_up(ksize, 4) * 4;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;
    //! 4x4 block output start pos
    uint8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 4) {
        const uint8_t* inptr0 = in + k * ldin + x0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = x0;
        outptr = outptr_base;
        for (; x + 7 < xmax; x += 8) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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

            transpose_8x4_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += ksize8;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, 4);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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

            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 8 * 4;
        outptr_base4 += 4 * 4;
    }
}

static void gemm_quint8_4x8_pack_B_t(dt_uint8* outptr, const dt_uint8* inptr,
                                     int ldin, int y0, int ymax, int k0,
                                     int kmax) {
    uint8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(uint8_t) * 16);

    int y = y0;
    for (; y + 7 < ymax; y += 8) {
        const uint8_t* inptr0 = inptr + y * ldin + k0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;
        const uint8_t* inptr4 = inptr3 + ldin;
        const uint8_t* inptr5 = inptr4 + ldin;
        const uint8_t* inptr6 = inptr5 + ldin;
        const uint8_t* inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int K = kmax - k0;
        //! read 12 * 4 in each row
        for (; K > 15; K -= 16) {
            interleave_8x4_4_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr);
        }
        if (K > 0) {
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr, 4, K);
        }
    }
    for (; y < ymax; y += 4) {
        const uint8_t* inptr0 = inptr + y * ldin + k0;
        const uint8_t* inptr1 = inptr0 + ldin;
        const uint8_t* inptr2 = inptr1 + ldin;
        const uint8_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        //! read 4 * 4 in each row
        for (; K > 15; K -= 16) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
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

            interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 4, K);
        }
    }
}

}  // namespace matmul_dot_4x8x4
}  // namespace armv7
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
