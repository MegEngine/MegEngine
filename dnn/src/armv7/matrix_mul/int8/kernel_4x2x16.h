/**
 * \file dnn/src/armv7/matrix_mul/int8/kernel_4x2x16.h
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
namespace matmul_4x2x16 {

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
                     int32_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    MEGDNN_MARK_USED_VAR(m_remain);
    MEGDNN_MARK_USED_VAR(n_remain);
    K /= 16;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int32_t);
// clang-format off
#define LOAD_LINE(reg_index, n)               \
    "cmp r5, #0 \n"                           \
    "beq 102f\n"                              \
    "cmp %[n_remain], #2\n"                   \
    "blt 100" n "f\n"                         \
    "vld1.32 {d" reg_index "}, [r" n "]\n"    \
    "b 101" n "f\n"                           \
    "100" n ":\n"                             \
    "cmp %[n_remain], #0\n"                   \
    "beq 101" n "f\n"                         \
    "vld1.32 {d" reg_index "[0]}, [r" n "]\n" \
    "101" n ":\n" \
    "subs r5, r5, #1\n"

#define LOAD_C                \
    "mov r5, %[m_remain]\n"   \
    LOAD_LINE("16", "0")      \
    LOAD_LINE("17", "1")      \
    LOAD_LINE("18", "2")      \
    LOAD_LINE("19", "3")      \
    "102:\n"

#define STORE_LINE(reg_index, n)               \
    "cmp r5, #0 \n"                            \
    "beq 105f\n"                               \
    "cmp %[n_remain], #2\n"                    \
    "blt 103" n "f\n"                          \
    "vst1.32 {d" reg_index "}, [r" n "]\n"     \
    "b 104" n "f\n"                            \
    "103" n ":\n"                              \
    "cmp %[n_remain], #0\n"                    \
    "beq 104" n "f\n"                          \
    "vst1.32 {d" reg_index "[0]}, [r" n " ]\n" \
    "104" n ":\n" \
    "subs r5, r5, #1\n"

#define STORE_C              \
    "mov r5, %[m_remain]\n"  \
    STORE_LINE("8", "0")     \
    STORE_LINE("9", "1")     \
    STORE_LINE("10", "2")    \
    STORE_LINE("11", "3")    \
    "105:\n"

    register int32_t* outptr asm("r0") = output;
    asm volatile(
            "add r1, r0, %[LDC]\n"
            "add r2, r1, %[LDC]\n"
            "add r3, r2, %[LDC]\n"
            "vldr d0, [%[b_ptr], #0]\n"
            "vmov.i32 q8, #0\n"
            "vldr d4, [%[a_ptr], #0]\n"
            "vmov.i32 q9, #0\n"
            "vldr d2, [%[b_ptr], #16]\n"
            "vmov.i32 q10, q8\n"
            "vldr d6, [%[a_ptr], #16]\n"
            "vmov.i32 q11, q8\n"
            "vldr d1, [%[b_ptr], #8]\n"
            "vmov.i32 q12, q8\n"
            "vldr d5, [%[a_ptr], #8]\n"
            "vmov.i32 q13, q8\n"
            "vldr d3, [%[b_ptr], #24]\n"
            "vmov.i32 q14, q8\n"
            "vldr d7, [%[a_ptr], #24]\n"
            "vmov.i32 q15, q8\n"

            // General loop.
            "1:\n"

            // Multiply 8 first levels of depth.
            "vmull.s8    q4,  d0,  d4\n"
            "add %[b_ptr], %[b_ptr], #32\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vldr d4, [%[a_ptr], #32]\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vmull.s8    q7,  d2,  d6\n"
            "vldr d6, [%[a_ptr], #48]\n"

            // Multiply-accumulate second-half, again into the same
            // 16bit local accumulator registers. This is where we
            // take advantage of having int8 instead of uint8 and therefore
            // being able to accumulate two products into int16.
            "vmlal.s8    q4,  d1,  d5\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vldr d5, [%[a_ptr], #40]\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vmlal.s8    q7,  d3,  d7\n"
            "vldr d7, [%[a_ptr], #56]\n"

            // Add pairwise, accumulate into 32-bit accumulators.
            "vpadal.s16   q8,  q4\n"
            "add %[a_ptr], %[a_ptr], #64\n"
            "vpadal.s16   q9,  q5\n"
            "subs %[K], %[K], #1\n"
            "vpadal.s16   q10, q6\n"
            "vpadal.s16   q11, q7\n"

            "beq 2f\n"

            // Multiply first half.
            "vmull.s8    q4,  d0,  d4\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vldr d4, [%[a_ptr], #0]\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vldr d0, [%[b_ptr], #0]\n"
            "vmull.s8    q7,  d2,  d6\n"
            "vldr d2, [%[b_ptr], #16]\n"

            // Multiply-accumulate second-half, again into the same
            // 16bit local accumulator registers. This is where we
            // take advantage of having int8 instead of uint8 and therefore
            // being able to accumulate two products into int16.
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

            // Multiply-accumulate second-half, again into the same
            // 16bit local accumulator registers. This is where we
            // take advantage of having int8 instead of uint8 and therefore
            // being able to accumulate two products into int16.
            "vmlal.s8    q4,  d1,  d5\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vmlal.s8    q7,  d3,  d7\n"

            // Add pairwise, accumulate into 32-bit accumulators.
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

            // Reduce 32bit accumulators horizontally, second pass
            // (each pass adds pairwise. we need to add 4-wise).
            "vpadd.s32 d8, d0, d1\n"
            "vpadd.s32 d9, d2, d3\n"
            "vpadd.s32 d10, d4, d5\n"
            "vpadd.s32 d11, d6, d7\n"

            "b 4f\n"

            "3:\n"

            // Reduce 32bit accumulators horizontally, second pass
            // (each pass adds pairwise. we need to add 4-wise),
            // and load destination values from memory.
            LOAD_C //
            "vpadd.s32 d8, d0, d1\n"
            "vpadd.s32 d9, d2, d3\n"
            "vpadd.s32 d10, d4, d5\n"
            "vpadd.s32 d11, d6, d7\n"

            // Add horizontally-reduced accumulators into
            // the values loaded from memory
            "vadd.s32 q4, q8, q4\n"
            "vadd.s32 q5, q9, q5\n"

            "4:\n"
            // Store back into memory
            STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr] "+r"(outptr), [m_remain] "+r" (m_remain),
              [n_remain] "+r" (n_remain)
            :
            : "cc", "memory", "r1", "r2", "r3", "r4", "r5",
              "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
              "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18",
              "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
              "d28", "d29", "d30", "d31");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_s8_4x2_pack_A_n(dt_int8* outptr, const dt_int8* inptr,
                                 int ldin, int y0, int ymax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 3 < ymax; y += 4) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int K = kmax - k0;
        //! read 4 * 16 in each row
        for (; K > 15; K -= 16) {
            interleave_4x16_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 16, K);
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
        //! read 4 * 16 in each row
        for (; K > 15; K -= 16) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff;MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff;MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4x16_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        inptr1 = zerobuff;MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff;MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 16, K);
        }
    }
}

static void gemm_s8_4x2_pack_A_t(dt_int8* out, const dt_int8* in, int ldin,
                                 int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize4 = round_up(ksize, 16) * 4;
    int8_t* outptr = out;

    int k = k0;
    for (; k < kmax; k += 16) {
        int ki = k;
        for (int cnt = 0; cnt < 2; ki += 8, cnt++) {
            const int8_t* inptr0 = in + ki * ldin + x0;
            const int8_t* inptr1 = inptr0 + ldin;
            const int8_t* inptr2 = inptr1 + ldin;
            const int8_t* inptr3 = inptr2 + ldin;
            const int8_t* inptr4 = inptr3 + ldin;
            const int8_t* inptr5 = inptr4 + ldin;
            const int8_t* inptr6 = inptr5 + ldin;
            const int8_t* inptr7 = inptr6 + ldin;
            int8_t* outptr_inner = outptr + ki - k;

            int remain = std::min(ki + 7 - kmax, 7);
            int x = x0;
            for (; x + 3 < xmax; x += 4) {
                if (remain >= 0) {
                    switch (remain) {
                        case 7:
                            inptr0 = zerobuff;MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff;MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff;MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff;MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff;MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff;MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff;MEGDNN_FALLTHRU
                        case 0:
                            inptr7 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                transpose_4x16_1_b_helper(inptr0, inptr1, inptr2, inptr3,
                                          inptr4, inptr5, inptr6, inptr7,
                                          outptr_inner);
                outptr_inner += ksize4;
            }

            if (x < xmax) {
                if (remain >= 0) {
                    switch (remain) {
                        case 7:
                            inptr0 = zerobuff;MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff;MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff;MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff;MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff;MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff;MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff;MEGDNN_FALLTHRU
                        case 0:
                            inptr7 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                for (; x < xmax; x++) {
                    *outptr_inner++ = *inptr0++;
                    *outptr_inner++ = *inptr1++;
                    *outptr_inner++ = *inptr2++;
                    *outptr_inner++ = *inptr3++;
                    *outptr_inner++ = *inptr4++;
                    *outptr_inner++ = *inptr5++;
                    *outptr_inner++ = *inptr6++;
                    *outptr_inner++ = *inptr7++;
                    outptr_inner += 8;
                }
            }
        }

        outptr += 16 * 4;
    }
}

static void gemm_s8_4x2_pack_B_n(dt_int8* out, const dt_int8* in, int ldin,
                                 int x0, int xmax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize2 = round_up(ksize, 16) * 2;
    int8_t* outptr = out;

    int k = k0;
    for (; k < kmax; k += 16) {
        int ki = k;
        for (int cnt = 0; cnt < 2; ki += 8, cnt++) {
            const int8_t* inptr0 = in + ki * ldin + x0;
            const int8_t* inptr1 = inptr0 + ldin;
            const int8_t* inptr2 = inptr1 + ldin;
            const int8_t* inptr3 = inptr2 + ldin;
            const int8_t* inptr4 = inptr3 + ldin;
            const int8_t* inptr5 = inptr4 + ldin;
            const int8_t* inptr6 = inptr5 + ldin;
            const int8_t* inptr7 = inptr6 + ldin;
            int8_t* outptr_inner = outptr + ki - k;

            int remain = std::min(ki + 7 - kmax, 7);
            int x = x0;
            for (; x + 1 < xmax; x += 2) {
                if (remain >= 0) {
                    switch (remain) {
                        case 7:
                            inptr0 = zerobuff;MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff;MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff;MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff;MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff;MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff;MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff;MEGDNN_FALLTHRU
                        case 0:
                            inptr7 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                transpose_2x16_1_b_helper(inptr0, inptr1, inptr2, inptr3,
                                          inptr4, inptr5, inptr6, inptr7,
                                          outptr_inner);
                outptr_inner += ksize2;
            }

            if (x < xmax) {
                if (remain >= 0) {
                    switch (remain) {
                        case 7:
                            inptr0 = zerobuff;MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff;MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff;MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff;MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff;MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff;MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff;MEGDNN_FALLTHRU
                        case 0:
                            inptr7 = zerobuff;
                            break;
                        default:
                            megdnn_assert(0);
                    }
                }

                for (; x < xmax; x++) {
                    *outptr_inner++ = *inptr0++;
                    *outptr_inner++ = *inptr1++;
                    *outptr_inner++ = *inptr2++;
                    *outptr_inner++ = *inptr3++;
                    *outptr_inner++ = *inptr4++;
                    *outptr_inner++ = *inptr5++;
                    *outptr_inner++ = *inptr6++;
                    *outptr_inner++ = *inptr7++;
                    outptr_inner += 8;
                }
            }
        }

        outptr += 16 * 2;
    }
}

static void gemm_s8_4x2_pack_B_t(dt_int8* outptr, const dt_int8* inptr,
                                 int ldin, int y0, int ymax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 1 < ymax; y += 2) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);

        int K = kmax - k0;
        //! read 16 * 2 in each row
        for (; K > 15; K -= 16) {
            interleave_2x16_1_b(inptr0, inptr1, outptr);
        }

        if (K > 0) {
            interleave_2(inptr0, inptr1, outptr, 16, K);
        }
    }
    for (; y < ymax; y += 2) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);

        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            if (y + 1 >= ymax) {
                switch (y + 1 - ymax) {
                    case 0:
                        inptr1 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            interleave_2x16_1_b(inptr0, inptr1, outptr);
        }

        if (K > 0) {
            if (y + 1 >= ymax) {
                switch (y + 1 - ymax) {
                    case 0:
                        inptr1 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_2(inptr0, inptr1, outptr, 16, K);
        }
    }
}

}  // matmul_4x2x16
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
