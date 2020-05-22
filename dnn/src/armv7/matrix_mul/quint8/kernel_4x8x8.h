/**
 * \file dnn/src/armv7/matrix_mul/quint8/kernel_4x8x8.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"

namespace megdnn {
namespace armv7 {
namespace matmul_4x8x8 {
/**
 * Overview of register layout:
 *
 * A 8x8x8 cell of Rhs is stored in 8bit in q12-q13
 * A 8x8x4 cell of Lhs is stored in 8bit in q0-q3
 * A 4x8 block of accumulators is stored in 32bit in q4-q11
 * zero_point_A is stored in 8bit in q14
 * zero_point_B is stored in 8bit in q15.
 *
 *                     +--------+--------+
 *                     |v12[0-8]|v13[0-8]|
 *                Rhs  +--------+--------+
 *    Lhs              |        |        |
 *
 *  +--------+ - - - - +-----------------+
 *  |v0[0-8]|          | v4[0-4]| v5[0-4]|
 *  |v1[0-8]|          | v6[0-4]| v7[0-4]|
 *  |v2[0-8]|          | v8[0-4]| v9[0-4]|
 *  |v3[0-8]|          |v10[0-4]|v11[0-4]|
 *  +--------+ - - - - +-----------------+
 *
 *                            Accumulator
 */
static void kern_4x8(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, size_t m_remain,
                     uint8_t za, uint8_t zb) {
    K /= 8;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
    size_t x0 = 0;

// clang-format off
#define LOAD_LINE(reg_index1, reg_index2, reg_index3, reg_index4, n)    \
    "cmp %[x0], #0 \n"                                                  \
    "beq 100f\n"                                                        \
    "vld1.32 {d" reg_index1 ", d" reg_index2 ", d" reg_index3 ", d"     \
    reg_index4 "}, [r" n "]!\n"                                         \
    "subs %[x0], %[x0], #1\n"

#define LOAD_C                                  \
    "mov %[x0], %[m_remain]\n"                  \
    LOAD_LINE("8", "9", "10", "11", "0")        \
    LOAD_LINE("12", "13", "14", "15", "1")      \
    LOAD_LINE("16", "17", "18", "19", "2")      \
    LOAD_LINE("20", "21", "22", "23", "3")      \
    "100:\n"

#define STORE_LINE(reg_index1, reg_index2, reg_index3, reg_index4, n)   \
    "cmp %[x0], #0 \n"                                                  \
    "beq 101f\n"                                                        \
    "vst1.32 {d" reg_index1 ", d" reg_index2 ", d" reg_index3 ", d"     \
    reg_index4 "}, [r" n "]!\n"                                         \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                                 \
    "mov %[x0], %[m_remain]\n"                  \
    STORE_LINE("8", "9", "10", "11", "0")       \
    STORE_LINE("12", "13", "14", "15", "1")     \
    STORE_LINE("16", "17", "18", "19", "2")     \
    STORE_LINE("20", "21", "22", "23", "3")     \
    "101:\n"

    // clang-format on

    register int32_t* outptr asm("r0") = output;
    asm volatile(
            // load accumulator C
            "add r1, r0, %[LDC]\n"
            "add r2, r1, %[LDC]\n"
            "add r3, r2, %[LDC]\n"
            "vdup.8 d28, %[za]\n"
            "vdup.8 d30, %[zb]\n"
            "cmp %[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"
            "veor.s32 q8, q8, q8\n"
            "veor.s32 q9, q9, q9\n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q11, q11, q11\n"

            "2: \n"
            "vld1.8 {d24}, [%[b_ptr]]!\n"
            "vld1.8 {d0}, [%[a_ptr]]!\n"
            "vld1.8 {d2}, [%[a_ptr]]!\n"
            "vld1.8 {d4}, [%[a_ptr]]!\n"
            "vld1.8 {d6}, [%[a_ptr]]!\n"
            "vsubl.u8 q12, d24, d30\n"
            "vsubl.u8 q0, d0, d28\n"
            "vsubl.u8 q1, d2, d28\n"
            "vsubl.u8 q2, d4, d28\n"
            "vsubl.u8 q3, d6, d28\n"

            "vld1.8 {d26}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d24, d0[0]\n"
            "vmlal.s16 q6, d24, d2[0]\n"
            "vmlal.s16 q8, d24, d4[0]\n"
            "vmlal.s16 q10, d24, d6[0]\n"
            "vsubl.u8 q13, d26, d30\n"
            "vmlal.s16 q5, d25, d0[0]\n"
            "vmlal.s16 q7, d25, d2[0]\n"
            "vmlal.s16 q9, d25, d4[0]\n"
            "vmlal.s16 q11, d25, d6[0]\n"

            "vld1.8 {d24}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d26, d0[1]\n"
            "vmlal.s16 q6, d26, d2[1]\n"
            "vmlal.s16 q8, d26, d4[1]\n"
            "vmlal.s16 q10, d26, d6[1]\n"
            "vsubl.u8 q12, d24, d30\n"
            "vmlal.s16 q5, d27, d0[1]\n"
            "vmlal.s16 q7, d27, d2[1]\n"
            "vmlal.s16 q9, d27, d4[1]\n"
            "vmlal.s16 q11, d27, d6[1]\n"

            "vld1.8 {d26}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d24, d0[2]\n"
            "vmlal.s16 q6, d24, d2[2]\n"
            "vmlal.s16 q8, d24, d4[2]\n"
            "vmlal.s16 q10, d24, d6[2]\n"
            "vsubl.u8 q13, d26, d30\n"
            "vmlal.s16 q5, d25, d0[2]\n"
            "vmlal.s16 q7, d25, d2[2]\n"
            "vmlal.s16 q9, d25, d4[2]\n"
            "vmlal.s16 q11, d25, d6[2]\n"

            "vld1.8 {d24}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d26, d0[3]\n"
            "vmlal.s16 q6, d26, d2[3]\n"
            "vmlal.s16 q8, d26, d4[3]\n"
            "vmlal.s16 q10, d26, d6[3]\n"
            "vsubl.u8 q12, d24, d30\n"
            "vmlal.s16 q5, d27, d0[3]\n"
            "vmlal.s16 q7, d27, d2[3]\n"
            "vmlal.s16 q9, d27, d4[3]\n"
            "vmlal.s16 q11, d27, d6[3]\n"

            "vld1.8 {d26}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d24, d1[0]\n"
            "vmlal.s16 q6, d24, d3[0]\n"
            "vmlal.s16 q8, d24, d5[0]\n"
            "vmlal.s16 q10, d24, d7[0]\n"
            "vsubl.u8 q13, d26, d30\n"
            "vmlal.s16 q5, d25, d1[0]\n"
            "vmlal.s16 q7, d25, d3[0]\n"
            "vmlal.s16 q9, d25, d5[0]\n"
            "vmlal.s16 q11, d25, d7[0]\n"

            "vld1.8 {d24}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d26, d1[1]\n"
            "vmlal.s16 q6, d26, d3[1]\n"
            "vmlal.s16 q8, d26, d5[1]\n"
            "vmlal.s16 q10, d26, d7[1]\n"
            "vsubl.u8 q12, d24, d30\n"
            "vmlal.s16 q5, d27, d1[1]\n"
            "vmlal.s16 q7, d27, d3[1]\n"
            "vmlal.s16 q9, d27, d5[1]\n"
            "vmlal.s16 q11, d27, d7[1]\n"

            "vld1.8 {d26}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d24, d1[2]\n"
            "vmlal.s16 q6, d24, d3[2]\n"
            "vmlal.s16 q8, d24, d5[2]\n"
            "vmlal.s16 q10, d24, d7[2]\n"
            "vsubl.u8 q13, d26, d30\n"
            "vmlal.s16 q5, d25, d1[2]\n"
            "vmlal.s16 q7, d25, d3[2]\n"
            "vmlal.s16 q9, d25, d5[2]\n"
            "vmlal.s16 q11, d25, d7[2]\n"

            "vmlal.s16 q4, d26, d1[3]\n"
            "vmlal.s16 q6, d26, d3[3]\n"
            "vmlal.s16 q8, d26, d5[3]\n"
            "vmlal.s16 q10, d26, d7[3]\n"
            "vmlal.s16 q5, d27, d1[3]\n"
            "vmlal.s16 q7, d27, d3[3]\n"
            "vmlal.s16 q9, d27, d5[3]\n"
            "vmlal.s16 q11, d27, d7[3]\n"

            "subs %[K], %[K], #1\n"
            "bne 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [x0] "+r"(x0), [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [m_remain] "+r"(m_remain), [za] "+r"(za), [zb] "+r"(zb),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31", "r1", "r2", "r3", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/**
 * Overview of register layout:
 *
 * A 8x4x8 cell of Rhs is stored in 8bit in q8-q9
 * A 8x8x4 cell of Lhs is stored in 8bit in q0-q3
 * A 4x4 block of accumulators is stored in 32bit in q4-q7
 * zero_point_A is stored in 8bit in q10
 * zero_point_B is stored in 8bit in q11.
 *
 *                     +--------+
 *                     | v8[0-4]|
 *                Rhs  +--------+
 *                     | v9[0-4]|
 *     Lhs             +--------+
 *
 *  +--------+ - - - - +--------+
 *  |v0[0-8]|          | v4[0-4]|
 *  |v1[0-8]|          | v5[0-4]|
 *  |v2[0-8]|          | v6[0-4]|
 *  |v3[0-8]|          | v7[0-4]|
 *  +--------+ - - - - +--------+
 *
 *                            Accumulator
 */
static void kern_4x4(const uint8_t* packA, const uint8_t* packB, int K,
                     int32_t* output, int LDC, bool is_first_k, size_t m_remain,
                     size_t n_remain, uint8_t za, uint8_t zb) {
    K /= 8;
    const uint8_t* a_ptr = packA;
    const uint8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
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
    "mov r1, r0\n"                  \
    LOAD_LINE("8", "9", "1")        \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    LOAD_LINE("10", "11", "1")      \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    LOAD_LINE("12", "13", "1")      \
    "add r1, r0, %[LDC]\n"          \
    LOAD_LINE("14", "15", "1")      \
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
    "mov r1, r0\n"                  \
    STORE_LINE("8", "9", "1")       \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    STORE_LINE("10", "11", "1")     \
    "add r1, r0, %[LDC]\n"          \
    "add r0, r0, %[LDC]\n"          \
    STORE_LINE("12", "13", "1")     \
    "add r1, r0, %[LDC]\n"          \
    STORE_LINE("14", "15", "1")     \
    "105:\n"

    // clang-format on

    register int32_t* outptr asm("r0") = output;
    asm volatile(
            // load accumulator C
            "vdup.8 d20, %[za]\n"
            "vdup.8 d22, %[zb]\n"
            "cmp %[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "veor.s32 q4, q4, q4\n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"

            "2: \n"
            "vld1.32 {d16[0]}, [%[b_ptr]]!\n"
            "vld1.8 {d0}, [%[a_ptr]]!\n"
            "vld1.8 {d2}, [%[a_ptr]]!\n"
            "vld1.8 {d4}, [%[a_ptr]]!\n"
            "vld1.8 {d6}, [%[a_ptr]]!\n"
            "vsubl.u8 q8, d16, d22\n"
            "vsubl.u8 q0, d0, d20\n"
            "vsubl.u8 q1, d2, d20\n"
            "vsubl.u8 q2, d4, d20\n"
            "vsubl.u8 q3, d6, d20\n"

            "vld1.32 {d18[0]}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d16, d0[0]\n"
            "vmlal.s16 q5, d16, d2[0]\n"
            "vsubl.u8 q9, d18, d22\n"
            "vmlal.s16 q6, d16, d4[0]\n"
            "vmlal.s16 q7, d16, d6[0]\n"

            "vld1.32 {d16[0]}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d18, d0[1]\n"
            "vmlal.s16 q5, d18, d2[1]\n"
            "vsubl.u8 q8, d16, d22\n"
            "vmlal.s16 q6, d18, d4[1]\n"
            "vmlal.s16 q7, d18, d6[1]\n"

            "vld1.32 {d18[0]}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d16, d0[2]\n"
            "vmlal.s16 q5, d16, d2[2]\n"
            "vsubl.u8 q9, d18, d22\n"
            "vmlal.s16 q6, d16, d4[2]\n"
            "vmlal.s16 q7, d16, d6[2]\n"

            "vld1.32 {d16[0]}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d18, d0[3]\n"
            "vmlal.s16 q5, d18, d2[3]\n"
            "vsubl.u8 q8, d16, d22\n"
            "vmlal.s16 q6, d18, d4[3]\n"
            "vmlal.s16 q7, d18, d6[3]\n"

            "vld1.32 {d18[0]}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d16, d1[0]\n"
            "vmlal.s16 q5, d16, d3[0]\n"
            "vsubl.u8 q9, d18, d22\n"
            "vmlal.s16 q6, d16, d5[0]\n"
            "vmlal.s16 q7, d16, d7[0]\n"

            "vld1.32 {d16[0]}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d18, d1[1]\n"
            "vmlal.s16 q5, d18, d3[1]\n"
            "vsubl.u8 q8, d16, d22\n"
            "vmlal.s16 q6, d18, d5[1]\n"
            "vmlal.s16 q7, d18, d7[1]\n"

            "vld1.32 {d18[0]}, [%[b_ptr]]!\n"
            "vmlal.s16 q4, d16, d1[2]\n"
            "vmlal.s16 q5, d16, d3[2]\n"
            "vsubl.u8 q9, d18, d22\n"
            "vmlal.s16 q6, d16, d5[2]\n"
            "vmlal.s16 q7, d16, d7[2]\n"

            "vmlal.s16 q4, d18, d1[3]\n"
            "vmlal.s16 q5, d18, d3[3]\n"
            "vmlal.s16 q6, d18, d5[3]\n"
            "vmlal.s16 q7, d18, d7[3]\n"

            "subs %[K], %[K], #1\n"
            "bne 2b\n"

            "3:\n" STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [outptr] "+r"(outptr),
              [K] "+r"(K), [is_first_k] "+r"(is_first_k), [LDC] "+r"(LDC),
              [x0] "+r"(x0), [m_remain] "+r"(m_remain),
              [n_remain] "+r"(n_remain), [za] "+r"(za), [zb] "+r"(zb)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31", "r1", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_u8_4x8_pack_A_n(dt_uint8* outptr, const dt_uint8* inptr,
                                 int ldin, int y0, int ymax, int k0, int kmax,
                                 uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);

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

            interleave_4x8_2_b(inptr0, inptr1, inptr2, inptr3, outptr);
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 8, K,
                         zero_point);
        }
    }
}

static void gemm_u8_4x8_transpose_pack_A_n(dt_uint8* out, const dt_uint8* in,
                                           int ldin, int x0, int xmax, int k0,
                                           int kmax, uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 8) * 4;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;

    int k = k0;
    for (; k < kmax; k += 8) {
        const uint8_t* inptr0 = in + k * ldin + x0;
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

        int x = x0;
        outptr = outptr_base;

        for (; x + 3 < xmax; x += 4) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
                    case 6:
                        inptr1 = zerobuff;
                    case 5:
                        inptr2 = zerobuff;
                    case 4:
                        inptr3 = zerobuff;
                    case 3:
                        inptr4 = zerobuff;
                    case 2:
                        inptr5 = zerobuff;
                    case 1:
                        inptr6 = zerobuff;
                    case 0:
                        inptr7 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                              inptr6, inptr7, outptr);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
                    case 6:
                        inptr1 = zerobuff;
                    case 5:
                        inptr2 = zerobuff;
                    case 4:
                        inptr3 = zerobuff;
                    case 3:
                        inptr4 = zerobuff;
                    case 2:
                        inptr5 = zerobuff;
                    case 1:
                        inptr6 = zerobuff;
                    case 0:
                        inptr7 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                        inptr7, outptr, 4, xmax - x, zero_point);
        }

        outptr_base += 4 * 8;
    }
}

static void gemm_u8_4x8_pack_B_n(dt_uint8* out, const dt_uint8* in, int ldin,
                                 int x0, int xmax, int k0, int kmax,
                                 uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 8) * 4;
    const int ksize8 = ksize4 * 2;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;
    uint8_t* outptr_interleave = nullptr;
    uint8_t* outptr_base4 = out + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 8) {
        const uint8_t* inptr0 = in + k * ldin + x0;
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

        int x = x0;
        outptr = outptr_base;

        for (; x + 7 < xmax; x += 8) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
                    case 6:
                        inptr1 = zerobuff;
                    case 5:
                        inptr2 = zerobuff;
                    case 4:
                        inptr3 = zerobuff;
                    case 3:
                        inptr4 = zerobuff;
                    case 2:
                        inptr5 = zerobuff;
                    case 1:
                        inptr6 = zerobuff;
                    case 0:
                        inptr7 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            outptr_interleave = outptr;
            interleave_8x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr_interleave);
            outptr += ksize8;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
                    case 6:
                        inptr1 = zerobuff;
                    case 5:
                        inptr2 = zerobuff;
                    case 4:
                        inptr3 = zerobuff;
                    case 3:
                        inptr4 = zerobuff;
                    case 2:
                        inptr5 = zerobuff;
                    case 1:
                        inptr6 = zerobuff;
                    case 0:
                        inptr7 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            outptr_interleave = outptr;
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr_interleave, 4, 4, zero_point);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 7 >= kmax) {
                switch (k + 7 - kmax) {
                    case 6:
                        inptr1 = zerobuff;
                    case 5:
                        inptr2 = zerobuff;
                    case 4:
                        inptr3 = zerobuff;
                    case 3:
                        inptr4 = zerobuff;
                    case 2:
                        inptr5 = zerobuff;
                    case 1:
                        inptr6 = zerobuff;
                    case 0:
                        inptr7 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            outptr_interleave = outptr;
            interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr_interleave, 4, xmax - x, zero_point);
        }

        outptr_base += 8 * 8;
        outptr_base4 += 4 * 8;
    }
}

static void gemm_u8_4x8_transpose_pack_B_n(dt_uint8* outptr,
                                           const dt_uint8* inptr, int ldin,
                                           int y0, int ymax, int k0, int kmax,
                                           uint8_t zero_point) {
    uint8_t zerobuff[16];
    std::fill(zerobuff, zerobuff + 16, zero_point);
    constexpr int interleave4 = 32;
    constexpr int interleave8 = 64;

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
        for (; K > 7; K -= 8) {
            transpose_8x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                              inptr6, inptr7, outptr);
            outptr += interleave8;
        }

        if (K > 0) {
            transpose_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                        inptr7, outptr, 8, K, zero_point);
            outptr += interleave8;
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
        for (; K > 7; K -= 8) {
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

            transpose_8x4_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += interleave4;
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
            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 8, K,
                        zero_point);
            outptr += interleave4;
        }
    }
}

}  // namespace matmul_4x8x8
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
