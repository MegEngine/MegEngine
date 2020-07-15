/**
 * \file dnn/src/armv7/matrix_mul/int8/kernel_mk4_4x2x16.h
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
namespace matmul_mk4_4x2x16 {

/**
 * Overview of register layout.
 *
 * A 2x16 block of Rhs is stored in 8 bit in d0--d3.
 * A 4x16 block of Lhs is stored in 8 bit in d4--d7. That is only
 * half of the register space required, so we loop over these registers
 * twice. Only half of it, a 2x16 block, is stored in d4--d7 at
 * any given time.
 *
 * A 4x2 block of accumulators is stored in q8--q15 (as 4x32 bit
 * components which need to be horizontally-added at the end)
 *
 * Only then, having processed 16 levels of depth, do we need to
 * horizontally add these int16x8 accumulators into the final
 * int32x4 accumulators.
 *
 * As we do not have enough registers to store all 16 int16x8
 * temporary-16bit-accumulators, we have them cycle through q4--q7.
 *
 * \warning Fast kernel operating on int8 operands.
 * It is assumed that one of the two int8 operands only takes values
 * in [-127, 127], while the other may freely range in [-128, 127].
 * The issue with both operands taking the value -128 is that:
 * -128*-128 + -128*-128 == -32768 overflows int16.
 * Every other expression a*b + c*d, for any int8 a,b,c,d, fits in int16
 * range. That is the basic idea of this kernel.
 *
 *
 *                              +--------+--------+
 *                              |d0[0-8] |d2[0-8] |
 *                         Rhs  +--------+--------+
 *                              |d1[0-8] |d3[0-8] |
 *                              +--------+--------+
 *                              |        |        |
 *
 *    Lhs                       |        |        |
 *
 *  +--------+--------+ - - - - +------------------
 *  |d4[0-8] |d5[0-8] |         |q8[0-4] |q9[0-4] |
 *  |d6[0-8] |d7[0-8] |         |q10[0-4]|q11[0-4]|
 *  |d4[0-8] |d5[0-8] |         |q12[0-4]|q13[0-4]|
 *  |d6[0-8] |d7[0-8] |         |q14[0-4]|q15[0-4]|
 *  +--------+--------+ - - - - +------------------
 *
 *                               Accumulator
 */

static void kern_4x2(const int8_t* packA, const int8_t* packB, int K,
                     int32_t* output, bool is_first_k, int n_remain) {
    MEGDNN_MARK_USED_VAR(n_remain);
    K /= 16;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    asm volatile(
            "vldr d0, [%[b_ptr], #0]\n"
            "vmov.i32 q8, #0\n"
            "vldr d4, [%[a_ptr], #0]\n"
            "vmov.i32 q9, #0\n"
            "vldr d1, [%[b_ptr], #8]\n"
            "vmov.i32 q10, q8\n"
            "vldr d5, [%[a_ptr], #8]\n"
            "vmov.i32 q11, q8\n"
            "vldr d2, [%[b_ptr], #16]\n"
            "vmov.i32 q12, q8\n"
            "vldr d6, [%[a_ptr], #16]\n"
            "vmov.i32 q13, q8\n"
            "vldr d3, [%[b_ptr], #24]\n"
            "vmov.i32 q14, q8\n"
            "vldr d7, [%[a_ptr], #24]\n"
            "vmov.i32 q15, q8\n"

            // General loop.
            "1:\n"
            "vmull.s8    q4,  d0,  d4\n"
            "add %[b_ptr], %[b_ptr], #32\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vldr d4, [%[a_ptr], #32]\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vmull.s8    q7,  d2,  d6\n"
            "vldr d6, [%[a_ptr], #48]\n"

            "vmlal.s8    q4,  d1,  d5\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vldr d5, [%[a_ptr], #40]\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vmlal.s8    q7,  d3,  d7\n"
            "vldr d7, [%[a_ptr], #56]\n"

            "vpadal.s16   q8,  q4\n"
            "add %[a_ptr], %[a_ptr], #64\n"
            "vpadal.s16   q9,  q5\n"
            "subs %[K], %[K], #1\n"
            "vpadal.s16   q10, q6\n"
            "vpadal.s16   q11, q7\n"

            "beq 2f\n"

            "vmull.s8    q4,  d0,  d4\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vldr d4, [%[a_ptr], #0]\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vldr d0, [%[b_ptr], #0]\n"
            "vmull.s8    q7,  d2,  d6\n"
            "vldr d2, [%[b_ptr], #16]\n"

            "vmlal.s8    q4,  d1,  d5\n"
            "vldr d6, [%[a_ptr], #16]\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vldr d5, [%[a_ptr], #8]\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vldr d1, [%[b_ptr], #8]\n"
            "vmlal.s8    q7,  d3,  d7\n"
            "vldr d3, [%[b_ptr], #24]\n"

            // Add pairwise, accumulate into 32-bit accumulators.
            "vpadal.s16   q12, q4\n"
            "vldr d7, [%[a_ptr], #24]\n"
            "vpadal.s16   q13, q5\n"
            "vpadal.s16   q14, q6\n"
            "vpadal.s16   q15, q7\n"

            "b 1b\n"

            "2:\n"
            // Multiply first half.
            "vmull.s8    q4,  d0,  d4\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vmull.s8    q7,  d2,  d6\n"

            "vmlal.s8    q4,  d1,  d5\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vmlal.s8    q7,  d3,  d7\n"

            "vpadal.s16   q12, q4\n"
            "vpadal.s16   q13, q5\n"
            "vpadal.s16   q14, q6\n"
            "vpadal.s16   q15, q7\n"
            "cmp %[is_first_k], #1\n"

            // Reduce 32bit accumulators horizontally.
            "vpadd.s32 d0, d16, d17\n"
            "vpadd.s32 d1, d18, d19\n"
            "vpadd.s32 d2, d20, d21\n"
            "vpadd.s32 d3, d22, d23\n"
            "vpadd.s32 d4, d24, d25\n"
            "vpadd.s32 d5, d26, d27\n"
            "vpadd.s32 d6, d28, d29\n"
            "vpadd.s32 d7, d30, d31\n"

            "bne 3f\n"
            "vpadd.s32 d8, d0, d2\n"
            "vpadd.s32 d9, d4, d6\n"
            "vpadd.s32 d10, d1, d3\n"
            "vpadd.s32 d11, d5, d7\n"
            "b 5f\n"

            "3:\n"
            "cmp %[n_remain], #1\n"
            "beq 4f\n"
            "vldr d18, [%[outptr], #16]\n"
            "vldr d19, [%[outptr], #24]\n"
            "4:\n"
            "vldr d16, [%[outptr]]\n"
            "vldr d17, [%[outptr], #8]\n"

            "vpadd.s32 d8, d0, d2\n"
            "vpadd.s32 d9, d4, d6\n"
            "vpadd.s32 d10, d1, d3\n"
            "vpadd.s32 d11, d5, d7\n"

            "vadd.s32 q4, q8, q4\n"
            "vadd.s32 q5, q9, q5\n"

            "5:\n"
            "cmp %[n_remain], #1\n"
            "beq 6f\n"
            "vstr d10, [%[outptr], #16]\n"
            "vstr d11, [%[outptr], #24]\n"
            "6:\n"
            "vstr d8, [%[outptr]]\n"
            "vstr d9, [%[outptr], #8]\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [outptr] "+r"(output),
              [n_remain] "+r"(n_remain)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

static void gemm_mk4_s8_4x2_pack_A(dt_int8* outptr, const dt_int8* inptr,
                                   int ldin, int y0, int ymax, int k0,
                                   int kmax) {
    //! pack form {oc/4, ic/4, 4(ic), 4(oc)} to {oc/4, ic/16, 4(oc), 16(ic)}
    int8_t zerobuff[4][64];
    std::memset(zerobuff, 0, sizeof(int8_t) * 64 * 4);
    megdnn_assert(ymax % 4 == 0 && y0 % 4 == 0 && (ymax - y0) % 4 == 0,
                  "mk4 matmul with m is not times of 4");
    megdnn_assert(kmax % 4 == 0 && k0 % 4 == 0 && (kmax - k0) % 4 == 0,
                  "mk4 matmul with k is not times of 4");
    size_t roundk = round_up(kmax - k0, 16);
    size_t out_offset = roundk * 4;
    int y = y0;
    int start_y = y0 / 4;
    for (; y + 15 < ymax; y += 16, start_y += 4) {
        const int8_t* inptr0 = inptr + start_y * ldin + k0 * 4;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        int8_t* output = outptr + (y - y0) / 4 * out_offset;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            transpose_interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, output,
                                         out_offset);
            output += 64;
        }
        if (K > 0) {
            std::memcpy(zerobuff[0], inptr0, sizeof(int8_t) * K * 4);
            std::memcpy(zerobuff[1], inptr1, sizeof(int8_t) * K * 4);
            std::memcpy(zerobuff[2], inptr2, sizeof(int8_t) * K * 4);
            std::memcpy(zerobuff[3], inptr3, sizeof(int8_t) * K * 4);
            inptr0 = zerobuff[0];
            inptr1 = zerobuff[1];
            inptr2 = zerobuff[2];
            inptr3 = zerobuff[3];
            transpose_interleave_4x4_4_b(inptr0, inptr1, inptr2, inptr3, output,
                                         out_offset);
            output += 64;
        }
    }
    for (; y + 3 < ymax; y += 4, start_y++) {
        const int8_t* inptr0 = inptr + start_y * ldin + k0 * 4;
        int8_t* output = outptr + (y - y0) / 4 * out_offset;
        prefetch_2x(inptr0);
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            transpose_interleave_1x4_4_b(inptr0, output);
            output += 64;
        }
        if (K > 0) {
            std::memcpy(zerobuff[0], inptr0, sizeof(int8_t) * K * 4);
            inptr0 = zerobuff[0];
            transpose_interleave_1x4_4_b(inptr0, output);
            output += 64;
        }
    }
}

static void gemm_mk4_s8_4x2_pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                   int x0, int xmax, int k0, int kmax) {
    int32_t zerobuff[4];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ICB = (ksize) / 4;
    const int ksize2 = round_up<int>(ICB, 4) * 2;
    int32_t* outptr = reinterpret_cast<int32_t*>(out);
    megdnn_assert(kmax % 4 == 0 && k0 % 4 == 0 && ksize % 4 == 0,
                  "mk4 matmul with k is not times of 4");

    int k = k0 / 4;
    for (; k + 3 < ICB; k += 4) {
        const int32_t* inptr0 =
                reinterpret_cast<const int32_t*>(in + k * ldin + x0);
        const int32_t* inptr1 =
                reinterpret_cast<const int32_t*>(in + (k + 1) * ldin + x0);
        const int32_t* inptr2 =
                reinterpret_cast<const int32_t*>(in + (k + 2) * ldin + x0);
        const int32_t* inptr3 =
                reinterpret_cast<const int32_t*>(in + (k + 3) * ldin + x0);
        int32_t* outptr_inner = outptr;

        int x = x0;
        for (; x + 1 < xmax; x += 2) {
            transpose_4x2_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner);
            outptr_inner += ksize2;
        }
        if (x < xmax) {
            *outptr_inner++ = *inptr0++;
            *outptr_inner++ = *inptr1++;
            *outptr_inner++ = *inptr2++;
            *outptr_inner++ = *inptr3++;
        }
        outptr += 4 * 2;
    }
    if (k < ICB) {
        const int32_t* inptr0 =
                reinterpret_cast<const int32_t*>(in + k * ldin + x0);
        const int32_t* inptr1 =
                reinterpret_cast<const int32_t*>(in + (k + 1) * ldin + x0);
        const int32_t* inptr2 =
                reinterpret_cast<const int32_t*>(in + (k + 2) * ldin + x0);
        const int32_t* inptr3 =
                reinterpret_cast<const int32_t*>(in + (k + 3) * ldin + x0);
        int32_t* outptr_inner = outptr;

        int x = x0;
        for (; x + 1 < xmax; x += 2) {
            if (k + 3 >= ICB) {
                switch (k + 3 - ICB) {
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
            transpose_4x2_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner);
            outptr_inner += ksize2;
        }
        if (x < xmax) {
            if (k + 3 >= ICB) {
                switch (k + 3 - ICB) {
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
            *outptr_inner++ = *inptr0;
            *outptr_inner++ = *inptr1;
            *outptr_inner++ = *inptr2;
            *outptr_inner++ = *inptr3;
        }
        outptr += 4 * 2;
    }
}

}  // namespace matmul_mk4_4x2x16
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
