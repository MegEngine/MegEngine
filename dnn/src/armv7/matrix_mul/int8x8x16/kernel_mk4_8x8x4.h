/**
 * \file dnn/src/armv7/matrix_mul/int8x8x16/kernel_mk4_8x8x4.h
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

namespace megdnn {
namespace armv7 {
namespace matmul_mk4_8x8x4 {

//! optimize for A7

/**
 * Overview of register layout:
 *
 * A 8x8x8 cell of Lhs is stored in 16bit in q0, q1
 * A 8x8x8 cell of Rhs is stored in 8bit in q2, q3
 * A 8x8 block of accumulators is stored in 16bit in q8-q15
 *
 *                     +--------+
 *                     | q4[0-8]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +--------+ - - - - +---------
 *  |q0[0]|            | q8 [0-8]|
 *  |q0[1]|            | q9 [0-8]|
 *  |q0[2]|            | q10[0-8]|
 *  |q0[3]|            | q11[0-8]|
 *  |q0[4]|            | q12[0-8]|
 *  |q0[5]|            | q13[0-8]|
 *  |q0[6]|            | q14[0-8]|
 *  |q0[7]|            | q15[0-8]|
 *  +--------+ - - - - +---------
 *
 *                            Accumulator
 */
static void kern_8x8(const int16_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int remain_n) {
    K /= 4;
    const int16_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);
    int x0 = 0;

// clang-format off
#define STORE_LINE(reg_index1, reg_index2)     \
    "cmp %[x0], #0 \n"                         \
    "beq 101f\n"                               \
    "vst1.16 {d" reg_index1 "}, [r0]!\n"       \
    "vst1.16 {d" reg_index2 "}, [r1]!\n"       \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                 \
    "mov %[x0], %[remain_n]\n"  \
    STORE_LINE("16", "17")      \
    STORE_LINE("18", "19")      \
    STORE_LINE("20", "21")      \
    STORE_LINE("22", "23")      \
    STORE_LINE("24", "25")      \
    STORE_LINE("26", "27")      \
    STORE_LINE("28", "29")      \
    STORE_LINE("30", "31")      \
    "101:\n"

    // clang-format on

    register int16_t* outptr asm("r0") = output;
    asm volatile(
            // load accumulator C
            "add r1, r0, %[LDC]\n"
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"

            "b 2f\n"

            "1:\n"
            "veor.s32 q8 , q8 , q8 \n"
            "veor.s32 q9 , q9 , q9 \n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q11, q11, q11\n"
            "veor.s32 q12, q12, q12\n"
            "veor.s32 q13, q13, q13\n"
            "veor.s32 q14, q14, q14\n"
            "veor.s32 q15, q15, q15\n"

            "2: \n"
            "vld1.8 {d4}, [%[b_ptr]]!\n"
            "vld1.16 {d0, d1}, [%[a_ptr]]!\n"
            "vmovl.s8 q2, d4\n"
            "vld1.16 {d2, d3}, [%[a_ptr]]!\n"
            "vld1.8 {d6}, [%[b_ptr]]!\n"
            //! k0
            "vmla.s16 q8,  q0, d4[0]\n"
            "vmla.s16 q9,  q0, d4[1]\n"
            "vmla.s16 q10, q0, d4[2]\n"
            "vmla.s16 q11, q0, d4[3]\n"
            "vmovl.s8 q3, d6\n"
            "vmla.s16 q12, q0, d5[0]\n"
            "vmla.s16 q13, q0, d5[1]\n"
            "vmla.s16 q14, q0, d5[2]\n"
            "vmla.s16 q15, q0, d5[3]\n"
            //! k1
            "vld1.16 {d0, d1}, [%[a_ptr]]!\n"
            "vld1.8 {d4}, [%[b_ptr]]!\n"
            "vmla.s16 q8,  q1, d6[0]\n"
            "vmla.s16 q9,  q1, d6[1]\n"
            "vmla.s16 q10, q1, d6[2]\n"
            "vmla.s16 q11, q1, d6[3]\n"
            "vmovl.s8 q2, d4\n"
            "vmla.s16 q12, q1, d7[0]\n"
            "vmla.s16 q13, q1, d7[1]\n"
            "vmla.s16 q14, q1, d7[2]\n"
            "vmla.s16 q15, q1, d7[3]\n"
            //! k2
            "vld1.16 {d2, d3}, [%[a_ptr]]!\n"
            "vld1.8 {d6}, [%[b_ptr]]!\n"
            "vmla.s16 q8,  q0, d4[0]\n"
            "vmla.s16 q9,  q0, d4[1]\n"
            "vmla.s16 q10, q0, d4[2]\n"
            "vmla.s16 q11, q0, d4[3]\n"
            "vmovl.s8 q3, d6\n"
            "vmla.s16 q12, q0, d5[0]\n"
            "vmla.s16 q13, q0, d5[1]\n"
            "vmla.s16 q14, q0, d5[2]\n"
            "vmla.s16 q15, q0, d5[3]\n"
            //! k3
            "vmla.s16 q8,  q1, d6[0]\n"
            "vmla.s16 q9,  q1, d6[1]\n"
            "vmla.s16 q10, q1, d6[2]\n"
            "vmla.s16 q11, q1, d6[3]\n"
            "vmla.s16 q12, q1, d7[0]\n"
            "vmla.s16 q13, q1, d7[1]\n"
            "vmla.s16 q14, q1, d7[2]\n"
            "vmla.s16 q15, q1, d7[3]\n"

            "subs %[K], %[K], #1\n"
            "bne 2b\n"

            "3:\n"
            "cmp %[remain_n], #8\n"
            "bne 4f\n"
            "vstr d16, [r0]\n"
            "vstr d18, [r0, #8]\n"
            "vstr d20, [r0, #16]\n"
            "vstr d22, [r0, #24]\n"
            "vstr d24, [r0, #32]\n"
            "vstr d26, [r0, #40]\n"
            "vstr d28, [r0, #48]\n"
            "vstr d30, [r0, #56]\n"

            "vstr d17, [r1]\n"
            "vstr d19, [r1, #8]\n"
            "vstr d21, [r1, #16]\n"
            "vstr d23, [r1, #24]\n"
            "vstr d25, [r1, #32]\n"
            "vstr d27, [r1, #40]\n"
            "vstr d29, [r1, #48]\n"
            "vstr d31, [r1, #56]\n"

            "b 101f\n"

            "4:\n " STORE_C
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [x0] "+r"(x0), [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [outptr] "+r"(outptr), [remain_n] "+r"(remain_n)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31", "r1", "r2", "r3", "cc", "memory");
#undef STORE_C
#undef STORE_LINE
}

/**
 * Overview of register layout:
 *
 * A 8x8x8 cell of Lhs is stored in 16bit in d0, d2
 * A 8x8x8 cell of Rhs is stored in 8bit in q2, q3
 * A 8x8 block of accumulators is stored in 16bit in q8-11
 *
 *                     +--------+
 *                     | q4[0-8]|
 *                Rhs  +--------+
 *    Lhs              |        |
 *
 *  +--------+ - - - - +---------
 *  |d0[0]|            | q8 [0-8]|
 *  |d0[1]|            | q9 [0-8]|
 *  |d0[2]|            | q10[0-8]|
 *  |d0[3]|            | q11[0-8]|
 *  +--------+ - - - - +---------
 *
 *                            Accumulator
 */
static void kern_4x8(const int16_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int remain_n) {
    K /= 4;
    const int16_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);
    int x0 = 0;

// clang-format off
#define STORE_LINE(reg_index1)                 \
    "cmp %[x0], #0 \n"                         \
    "beq 101f\n"                               \
    "vst1.16 {d" reg_index1 "}, [r0]!\n"       \
    "subs %[x0], %[x0], #1\n"

#define STORE_C                 \
    "mov %[x0], %[remain_n]\n"  \
    STORE_LINE("16")            \
    STORE_LINE("18")            \
    STORE_LINE("20")            \
    STORE_LINE("22")            \
    STORE_LINE("24")            \
    STORE_LINE("26")            \
    STORE_LINE("28")            \
    STORE_LINE("30")            \
    "101:\n"

    // clang-format on

    register int16_t* outptr asm("r0") = output;
    asm volatile(
            //! load accumulator C
            "add r1, r0, %[LDC]\n"
            "cmp %[is_first_k], #1\n"
            "beq 1f\n"

            "b 2f\n"

            "1:\n"
            "veor.s32 q8 , q8 , q8 \n"
            "veor.s32 q9 , q9 , q9 \n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q11, q11, q11\n"
            "veor.s32 q12, q12, q12\n"
            "veor.s32 q13, q13, q13\n"
            "veor.s32 q14, q14, q14\n"
            "veor.s32 q15, q15, q15\n"

            "2: \n"
            "vld1.8 {d4}, [%[b_ptr]]!\n"
            "vld1.16 {d0}, [%[a_ptr]]!\n"
            "vmovl.s8 q2, d4\n"
            "vld1.16 {d2}, [%[a_ptr]]!\n"
            "vld1.8 {d6}, [%[b_ptr]]!\n"
            //! k0
            "vmla.s16 d16,  d0, d4[0]\n"
            "vmla.s16 d18,  d0, d4[1]\n"
            "vmla.s16 d20, d0, d4[2]\n"
            "vmla.s16 d22, d0, d4[3]\n"
            "vmovl.s8 q3, d6\n"
            "vmla.s16 d24, d0, d5[0]\n"
            "vmla.s16 d26, d0, d5[1]\n"
            "vmla.s16 d28, d0, d5[2]\n"
            "vmla.s16 d30, d0, d5[3]\n"
            //! k1
            "vld1.16 {d0}, [%[a_ptr]]!\n"
            "vld1.8 {d4}, [%[b_ptr]]!\n"
            "vmla.s16 d16,  d2, d6[0]\n"
            "vmla.s16 d18,  d2, d6[1]\n"
            "vmla.s16 d20, d2, d6[2]\n"
            "vmla.s16 d22, d2, d6[3]\n"
            "vmovl.s8 q2, d4\n"
            "vmla.s16 d24, d2, d7[0]\n"
            "vmla.s16 d26, d2, d7[1]\n"
            "vmla.s16 d28, d2, d7[2]\n"
            "vmla.s16 d30, d2, d7[3]\n"
            //! k2
            "vld1.16 {d2}, [%[a_ptr]]!\n"
            "vld1.8 {d6}, [%[b_ptr]]!\n"
            "vmla.s16 d16,  d0, d4[0]\n"
            "vmla.s16 d18,  d0, d4[1]\n"
            "vmla.s16 d20, d0, d4[2]\n"
            "vmla.s16 d22, d0, d4[3]\n"
            "vmovl.s8 q3, d6\n"
            "vmla.s16 d24, d0, d5[0]\n"
            "vmla.s16 d26, d0, d5[1]\n"
            "vmla.s16 d28, d0, d5[2]\n"
            "vmla.s16 d30, d0, d5[3]\n"
            //! k3
            "vmla.s16 d16,  d2, d6[0]\n"
            "vmla.s16 d18,  d2, d6[1]\n"
            "vmla.s16 d20, d2, d6[2]\n"
            "vmla.s16 d22, d2, d6[3]\n"
            "vmla.s16 d24, d2, d7[0]\n"
            "vmla.s16 d26, d2, d7[1]\n"
            "vmla.s16 d28, d2, d7[2]\n"
            "vmla.s16 d30, d2, d7[3]\n"

            "subs %[K], %[K], #1\n"
            "bne 2b\n"

            "3:\n"
            "cmp %[remain_n], #8\n"
            "bne 4f\n"
            "vstr d16, [r0]\n"
            "vstr d18, [r0, #8]\n"
            "vstr d20, [r0, #16]\n"
            "vstr d22, [r0, #24]\n"
            "vstr d24, [r0, #32]\n"
            "vstr d26, [r0, #40]\n"
            "vstr d28, [r0, #48]\n"
            "vstr d30, [r0, #56]\n"
            "b 101f\n"

            "4:\n " STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [x0] "+r"(x0), [LDC] "+r"(LDC), [is_first_k] "+r"(is_first_k),
              [outptr] "+r"(outptr), [remain_n] "+r"(remain_n)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31", "r1", "r2", "r3", "cc", "memory");
#undef STORE_C
#undef STORE_LINE
}

static void gemm_s8x8x16_mk4_8x8_pack_A_n(dt_int16* outptr,
                                          const dt_int8* inptr, int ldin,
                                          int m0, int mmax, int k0, int kmax) {
    megdnn_assert(m0 % 4 == 0 && mmax % 4 == 0, "M must be time of 4");
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    constexpr int pack_m = 8;
    constexpr int pack_k = 4;
    constexpr int pack_size = 4;
    const int m_size = mmax - m0;
    const int m_end = m_size / pack_m * pack_m + m0;
    const int remain_m = mmax - m_end;

    for (int m_idx = m0; m_idx < m_end; m_idx += pack_m) {
        const int8_t* inptr0 = inptr + m_idx / pack_size * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);

        for (int k_idx = k0; k_idx < kmax; k_idx += pack_size) {
            interleave_4x4_8x4_s8_s16(inptr0, inptr1, outptr);
            inptr0 += pack_size * pack_size;
            inptr1 += pack_size * pack_size;
            outptr += pack_m * pack_k;
        }
    }
    if (remain_m > 0) {
        const int8_t* inptr0 = inptr + m_end / pack_size * ldin + k0;
        const int k_size = kmax - k0;
        memcpy_s8_s16(inptr0, outptr, k_size * pack_size);
    }
}

static void gemm_s8x8x16_mk4_8x8_pack_B_n(dt_int8* out, const dt_int8* in,
                                          int ldin, int n0, int nmax, int k0,
                                          int kmax) {
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    int8_t tmpbuff[32] = {0};

    constexpr int pack_n = 8;
    constexpr int pack_size = 4;
    const int ksize = kmax - k0;
    const int nsize = nmax - n0;
    const int n_end = nsize / pack_n * pack_n + n0;
    const int remain_n = nsize % pack_n;
    int output_stride = ksize * pack_n;
    int8_t* outptr_base = out;

    for (int k_idx = k0; k_idx < kmax; k_idx += pack_size) {
        const int8_t* inptr = in + k_idx / pack_size * ldin + n0 * pack_size;
        prefetch_3x(inptr);

        auto outptr = outptr_base;
        for (int n_idx = n0; n_idx < n_end; n_idx += pack_n) {
            transpos_8x4_int8(inptr, outptr);
            inptr += pack_n * pack_size;
            outptr += output_stride;
        }
        if (remain_n > 0) {
            memcpy(tmpbuff, inptr, sizeof(int8_t) * remain_n * pack_size);
            transpos_8x4_int8(tmpbuff, outptr);
            outptr += output_stride;
        }
        outptr_base += pack_n * pack_size;
    }
}

}  // namespace matmul_mk4_8x8x4
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
