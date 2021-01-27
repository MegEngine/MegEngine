/**
 * \file dnn/src/armv7/matrix_mul/int8x8x16/kernel_8x8x4.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
namespace matmul_8x8x4 {
/*                             +--------+---------------------------------+
 *                             |   q4   | b00 b01 b02 b03 b04 b05 b06 b07 |
 *                             +--------+---------------------------------+
 *                             |   q5   | b10 b11 b12 b13 b14 b15 b16 b17 |
 *                             +--------+---------------------------------+
 *                             |   q6   | b20 b21 b22 b23 b24 b25 b26 b27 |
 *                             +--------+---------------------------------+
 *                             |   q7   | b30 b31 b32 b33 b34 b35 b36 b37 |
 *                             +--------+---------------------------------+
 * +----+-----------------+    +--------+---------------------------------+
 * | d0 | a00 a01 a02 a03 |    |   q8   | c00 c01 c02 c03 c04 c05 c06 c07 |
 * | d1 | a10 a11 a12 a13 |    |   q9   | c10 c11 c12 c13 c14 c15 c16 c17 |
 * | d2 | a20 a21 a22 a23 |    |   q10  | c20 c21 c22 c23 c24 c25 c26 c27 |
 * | d3 | a30 a31 a32 a33 |    |   q11  | c30 c31 c32 c33 c34 c35 c36 c37 |
 * | d4 | a40 a41 a42 a43 |    |   q12  | c40 c41 c42 c43 c44 c45 c46 c47 |
 * | d5 | a50 a51 a52 a53 |    |   q13  | c50 c51 c52 c53 c54 c55 c56 c57 |
 * | d6 | a60 a61 a62 a63 |    |   q14  | c60 c61 c62 c63 c64 c65 c66 c67 |
 * | d7 | a70 a71 a72 a73 |    |   q15  | c70 c71 c72 c73 c74 c75 c76 c77 |
 * +----+-----------------+    +--------+---------------------------------+
 *
 */

static void kern_8x8(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k,
                     size_t n_remain) {
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    LDC = LDC * sizeof(int16_t);
    size_t nr = n_remain;

    // clang-format off

#define LOAD_C                               \
    "mov r1, r0\n"                           \
    "vld1.16 {d16, d17}, [r1], %[LDC]\n"     \
    "vld1.16 {d18, d19}, [r1], %[LDC]\n"     \
    "vld1.16 {d20, d21}, [r1], %[LDC]\n"     \
    "vld1.16 {d22, d23}, [r1], %[LDC]\n"     \
    "vld1.16 {d24, d25}, [r1], %[LDC]\n"     \
    "vld1.16 {d26, d27}, [r1], %[LDC]\n"     \
    "vld1.16 {d28, d29}, [r1], %[LDC]\n"     \
    "vld1.16 {d30, d31}, [r1], %[LDC]\n"                 


#define STORE_LINE(id1, id2)                 \
    "mov r2, r1\n"                           \
    "cmp %[nr], #8\n"                        \
    "bne 100f\n"                             \
    "vst1.16 {d" id1 ", d" id2 "}, [r2]!\n"  \
    "b 101f\n"                               \
    "100:\n"                                 \
    "cmp %[nr], #0\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[0]}, [r2]!\n"         \
    "cmp %[nr], #1\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[1]}, [r2]!\n"         \
    "cmp %[nr], #2\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[2]}, [r2]!\n"         \
    "cmp %[nr], #3\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[3]}, [r2]!\n"         \
    "cmp %[nr], #4\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id2 "[0]}, [r2]!\n"         \
    "cmp %[nr], #5\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id2 "[1]}, [r2]!\n"         \
    "cmp %[nr], #6\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id2 "[2]}, [r2]!\n"         \
    "101:\n"                                                              
#define STORE_C                              \
    "mov r1, r0\n"                           \
    STORE_LINE("16", "17")                   \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("18", "19")                   \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("20", "21")                   \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("22", "23")                   \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("24", "25")                   \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("26", "27")                   \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("28", "29")                   \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("30", "31")
    // clang-format on

    register int16_t* outptr asm("r0") = output;
    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "veor.s32 q8, q8, q8\n"
            "veor.s32 q9, q9, q9\n"
            "veor.s32 q10, q10, q10\n"
            "veor.s32 q11, q11, q11\n"
            "veor.s32 q12, q12, q12\n"
            "veor.s32 q13, q13, q13\n"
            "veor.s32 q14, q14, q14\n"
            "veor.s32 q15, q15, q15\n"

            "2:\n"
            "vld1.8 {d0}, [%[a_ptr]]!\n"
            "vld1.8 {d2}, [%[a_ptr]]!\n"
            "vld1.8 {d4}, [%[a_ptr]]!\n"
            "vld1.8 {d6}, [%[a_ptr]]!\n"
            "vmovl.s8 q0, d0\n"
            "vmovl.s8 q1, d2\n"
            "vmovl.s8 q2, d4\n"
            "vmovl.s8 q3, d6\n"

            "vld1.8 {d8}, [%[b_ptr]]!\n"
            "vld1.8 {d10}, [%[b_ptr]]!\n"
            "vld1.8 {d12}, [%[b_ptr]]!\n"
            "vld1.8 {d14}, [%[b_ptr]]!\n"
            "vmovl.s8 q4, d8\n"
            "vmovl.s8 q5, d10\n"
            "vmovl.s8 q6, d12\n"
            "vmovl.s8 q7, d14\n"

            "vmla.s16 q8, q4, d0[0]\n"
            "vmla.s16 q9, q4, d1[0]\n"
            "vmla.s16 q10, q4, d2[0]\n"
            "vmla.s16 q11, q4, d3[0]\n"
            "vmla.s16 q12, q4, d4[0]\n"
            "vmla.s16 q13, q4, d5[0]\n"
            "vmla.s16 q14, q4, d6[0]\n"
            "vmla.s16 q15, q4, d7[0]\n"

            "vmla.s16 q8, q5, d0[1]\n"
            "vmla.s16 q9, q5, d1[1]\n"
            "vmla.s16 q10, q5, d2[1]\n"
            "vmla.s16 q11, q5, d3[1]\n"
            "vmla.s16 q12, q5, d4[1]\n"
            "vmla.s16 q13, q5, d5[1]\n"
            "vmla.s16 q14, q5, d6[1]\n"
            "vmla.s16 q15, q5, d7[1]\n"

            "vmla.s16 q8, q6, d0[2]\n"
            "vmla.s16 q9, q6, d1[2]\n"
            "vmla.s16 q10, q6, d2[2]\n"
            "vmla.s16 q11, q6, d3[2]\n"
            "vmla.s16 q12, q6, d4[2]\n"
            "vmla.s16 q13, q6, d5[2]\n"
            "vmla.s16 q14, q6, d6[2]\n"
            "vmla.s16 q15, q6, d7[2]\n"

            "vmla.s16 q8, q7, d0[3]\n"
            "vmla.s16 q9, q7, d1[3]\n"
            "vmla.s16 q10, q7, d2[3]\n"
            "vmla.s16 q11, q7, d3[3]\n"
            "vmla.s16 q12, q7, d4[3]\n"
            "vmla.s16 q13, q7, d5[3]\n"
            "vmla.s16 q14, q7, d6[3]\n"
            "vmla.s16 q15, q7, d7[3]\n"

            "subs %[K], %[K], #1\n"
            "bne 2b\n"

            "3:\n" STORE_C
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ LDC ] "+r"(LDC), [ is_first_k ] "+r"(is_first_k),
              [ outptr ] "+r"(outptr), [ nr ] "+r"(nr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
              "q11", "q12", "q13", "q14", "q15", "r1", "r2", "cc", "memory");
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

/*                             +--------+---------------------------------+
 *                             |   q2   | b00 b01 b02 b03 b04 b05 b06 b07 |
 *                             +--------+---------------------------------+
 *                             |   q3   | b10 b11 b12 b13 b14 b15 b16 b17 |
 *                             +--------+---------------------------------+
 *                             |   q4   | b20 b21 b22 b23 b24 b25 b26 b27 |
 *                             +--------+---------------------------------+
 *                             |   q5   | b30 b31 b32 b33 b34 b35 b36 b37 |
 *                             +--------+---------------------------------+
 * +----+-----------------+    +--------+---------------------------------+
 * | d0 | a00 a01 a02 a03 |    |   q6   | c00 c01 c02 c03 c04 c05 c06 c07 |
 * | d1 | a10 a11 a12 a13 |    |   q7   | c10 c11 c12 c13 c14 c15 c16 c17 |
 * | d2 | a20 a21 a22 a23 |    |   q8   | c20 c21 c22 c23 c24 c25 c26 c27 |
 * | d3 | a30 a31 a32 a33 |    |   q9   | c30 c31 c32 c33 c34 c35 c36 c37 |
 * +----+-----------------+    +--------+---------------------------------+
 *
 */
static void kern_4x8(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, size_t m_remain,
                     size_t n_remain) {
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    LDC = LDC * sizeof(int16_t);
    size_t mr = m_remain;
    size_t nr = n_remain;

    // clang-format off

#define LOAD_C                               \
    "cmp %[mr], #0\n"                        \
    "beq 100f\n"                             \
    "mov r1, r0\n"                           \
    "vld1.16 {d12, d13}, [r1], %[LDC]\n"     \
    "cmp %[mr], #1\n"                        \
    "beq 100f\n"                             \
    "vld1.16 {d14, d15}, [r1], %[LDC]\n"     \
    "cmp %[mr], #2\n"                        \
    "beq 100f\n"                             \
    "vld1.16 {d16, d17}, [r1], %[LDC]\n"     \
    "cmp %[mr], #3\n"                        \
    "beq 100f\n"                             \
    "vld1.16 {d18, d19}, [r1], %[LDC]\n"     \
    "100:\n"                                 \

#define STORE_LINE(id1, id2)                 \
    "mov r2, r1\n"                           \
    "cmp %[nr], #0\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[0]}, [r2]!\n"         \
    "cmp %[nr], #1\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[1]}, [r2]!\n"         \
    "cmp %[nr], #2\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[2]}, [r2]!\n"         \
    "cmp %[nr], #3\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id1 "[3]}, [r2]!\n"         \
    "cmp %[nr], #4\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id2 "[0]}, [r2]!\n"         \
    "cmp %[nr], #5\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id2 "[1]}, [r2]!\n"         \
    "cmp %[nr], #6\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id2 "[2]}, [r2]!\n"         \
    "cmp %[nr], #7\n"                        \
    "beq 101f\n"                             \
    "vst1.16 {d" id2 "[3]}, [r2]!\n"         \
    "101:\n"                                                              
#define STORE_C                              \
    "cmp %[mr], #0\n"                        \
    "beq 102f\n"                             \
    "mov r1, r0\n"                           \
    STORE_LINE("12", "13")                   \
    "cmp %[mr], #1\n"                        \
    "beq 102f\n"                             \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("14", "15")                   \
    "cmp %[mr], #2\n"                        \
    "beq 102f\n"                             \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("16", "17")                   \
    "cmp %[mr], #3\n"                        \
    "beq 102f\n"                             \
    "add r1, r1, %[LDC]\n"                   \
    STORE_LINE("18", "19")                   \
    "102:\n"

    // clang-format on

    register int16_t* outptr asm("r0") = output;
    asm volatile(
            "cmp %[is_first_k], #1\n"
            "beq 1f\n" LOAD_C

            "b 2f\n"

            "1:\n"
            "veor.s32 q6, q6, q6\n"
            "veor.s32 q7, q7, q7\n"
            "veor.s32 q8, q8, q8\n"
            "veor.s32 q9, q9, q9\n"

            "2:\n"
            "vld1.8 {d0}, [%[a_ptr]]!\n"
            "vld1.8 {d2}, [%[a_ptr]]!\n"
            "vmovl.s8 q0, d0\n"
            "vmovl.s8 q1, d2\n"

            "vld1.8 {d4}, [%[b_ptr]]!\n"
            "vld1.8 {d6}, [%[b_ptr]]!\n"
            "vld1.8 {d8}, [%[b_ptr]]!\n"
            "vld1.8 {d10}, [%[b_ptr]]!\n"
            "vmovl.s8 q2, d4\n"
            "vmovl.s8 q3, d6\n"
            "vmovl.s8 q4, d8\n"
            "vmovl.s8 q5, d10\n"

            "vmla.s16 q6, q2, d0[0]\n"
            "vmla.s16 q7, q2, d1[0]\n"
            "vmla.s16 q8, q2, d2[0]\n"
            "vmla.s16 q9, q2, d3[0]\n"

            "vmla.s16 q6, q3, d0[1]\n"
            "vmla.s16 q7, q3, d1[1]\n"
            "vmla.s16 q8, q3, d2[1]\n"
            "vmla.s16 q9, q3, d3[1]\n"

            "vmla.s16 q6, q4, d0[2]\n"
            "vmla.s16 q7, q4, d1[2]\n"
            "vmla.s16 q8, q4, d2[2]\n"
            "vmla.s16 q9, q4, d3[2]\n"

            "vmla.s16 q6, q5, d0[3]\n"
            "vmla.s16 q7, q5, d1[3]\n"
            "vmla.s16 q8, q5, d2[3]\n"
            "vmla.s16 q9, q5, d3[3]\n"

            "subs %[K], %[K], #1\n"
            "bne 2b\n"

            "3:\n" STORE_C
            : [ a_ptr ] "+r"(a_ptr), [ b_ptr ] "+r"(b_ptr), [ K ] "+r"(K),
              [ LDC ] "+r"(LDC), [ is_first_k ] "+r"(is_first_k),
              [ outptr ] "+r"(outptr), [ mr ] "+r"(mr), [ nr ] "+r"(nr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "r1",
              "r2", "cc", "memory");
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_s8x8x16_8x8_pack_A_n(dt_int8* out, const dt_int8* inptr,
                                      int ldin, int y0, int ymax, int k0,
                                      int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    int8_t* outptr = out;
    int y = y0;
    for (; y + 7 < ymax; y += 8) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        const int8_t* inptr6 = inptr5 + ldin;
        const int8_t* inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            interleave_8x4_4_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr);
        }

        if (K > 0) {
            for (; K > 0; K -= 4)
                interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                             inptr6, inptr7, outptr, 4, std::min(K, 4));
        }
    }
    for (; y < ymax; y += 4) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        for (; K > 0; K -= 4) {
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
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 4,
                         std::min(K, 4));
        }
    }
}

static void gemm_s8x8x16_8x8_pack_A_t(dt_int8* out, const dt_int8* in, int ldin,
                                      int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    int8_t* outbase = out;
    size_t K = round_up(kmax - k0, 4);

    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = in + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = xmax - x0;
        int8_t* outptr = outbase;
        int8_t* out_tmp = outptr;
        for (; x > 7; x -= 8) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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
            out_tmp = outptr;
            transpose_8x4_1_b(inptr0, inptr1, inptr2, inptr3, out_tmp);
            outptr += (K - k) * 8 + (x > 15 ? 8 : 4) * k;
        }

        if (x > 0) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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
            out_tmp = outptr;
            if (x > 4) {
                transpose_4(inptr0, inptr1, inptr2, inptr3, out_tmp, 4, 4);
                x -= 4;
                out_tmp = outptr + K * 4;
            }
            transpose_4(inptr0, inptr1, inptr2, inptr3, out_tmp, 4, x);
        }
        outbase += 4 * ((xmax - x0) > 7 ? 8 : 4);
    }
}

static void gemm_s8x8x16_8x8_pack_B_n(dt_int8* out, const dt_int8* in, int ldin,
                                      int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    int8_t* outbase = out;
    int8_t* out_interleave = out;
    const size_t K8 = round_up(kmax - k0, 4) * 8;
    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = in + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = xmax - x0;

        int8_t* outptr = outbase;
        for (; x > 7; x -= 8) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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
            out_interleave = outptr;
            asm volatile(
                    "vld1.32 {d0}, [%[inptr0]]!\n"
                    "vld1.32 {d1}, [%[inptr1]]!\n"
                    "vld1.32 {d2}, [%[inptr2]]!\n"
                    "vld1.32 {d3}, [%[inptr3]]!\n"
                    "vst1.32 {d0}, [%[out_interleave]]!\n"
                    "vst1.32 {d1}, [%[out_interleave]]!\n"
                    "vst1.32 {d2}, [%[out_interleave]]!\n"
                    "vst1.32 {d3}, [%[out_interleave]]!\n"
                    : [ inptr0 ] "+r"(inptr0), [ inptr1 ] "+r"(inptr1),
                      [ inptr2 ] "+r"(inptr2), [ inptr3 ] "+r"(inptr3),
                      [ out_interleave ] "+r"(out_interleave)
                    :
                    : "q0", "q1", "cc", "memory");
            outptr += K8;
        }

        if (x > 0) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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
            out_interleave = outptr;
            interleave_4(inptr0, inptr1, inptr2, inptr3, out_interleave, 8, x);
        }
        outbase += 4 * 8;
    }
}

static void gemm_s8x8x16_8x8_pack_B_t(dt_int8* out, const dt_int8* inptr,
                                      int ldin, int y0, int ymax, int k0,
                                      int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    int8_t* outptr = out;

    int y = y0;
    for (; y < ymax; y += 8) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        const int8_t* inptr6 = inptr5 + ldin;
        const int8_t* inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int k = k0;

        for (; k + 3 < kmax; k += 4) {
            if (y + 7 >= ymax) {
                switch (y + 7 - ymax) {
                    case 6:
                        inptr1 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 5:
                        inptr2 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 4:
                        inptr3 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 3:
                        inptr4 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 2:
                        inptr5 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 1:
                        inptr6 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 0:
                        inptr7 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            transpose_4x8_1_b(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                              inptr6, inptr7, outptr);
            outptr += 4 * 8;
        }

        if (k < kmax) {
            if (y + 7 >= ymax) {
                switch (y + 7 - ymax) {
                    case 6:
                        inptr1 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 5:
                        inptr2 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 4:
                        inptr3 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 3:
                        inptr4 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 2:
                        inptr5 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 1:
                        inptr6 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 0:
                        inptr7 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            transpose_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                        inptr7, outptr, 4, kmax - k);
            outptr += 4 * 8;
        }
    }
}
}  // namespace matmul_8x8x4
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
