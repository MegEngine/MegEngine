/**
 * \file dnn/src/aarch64/matrix_mul/fp16/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/matrix_mul/fp16/strategy.h"
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

namespace {

void interleave_8x1(__fp16* out, const __fp16* in, int ldin, int y0, int ymax,
                    int k0, int kmax) {
    __fp16* outptr = out;
    const __fp16* inptr = in;
    __fp16 zerobuff[24];
    std::memset(zerobuff, 0, sizeof(__fp16) * 24);

    int y = y0;
    for (; y + 8 <= ymax; y += 8) {
        const __fp16* inptr0 = inptr + y * ldin + k0;
        const __fp16* inptr1 = inptr0 + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;
        const __fp16* inptr4 = inptr3 + ldin;
        const __fp16* inptr5 = inptr4 + ldin;
        const __fp16* inptr6 = inptr5 + ldin;
        const __fp16* inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int x = (kmax - k0);
        for (; x > 7; x -= 8) {
            int skippf = (x & 31);
            interleave_8x1_8_h(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr, skippf);
        }

        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
            *outptr++ = *inptr6++;
            *outptr++ = *inptr7++;
        }
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

            interleave_4x1_4_h(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
        }
    }
}

void interleave_24x1(__fp16* out, const __fp16* in, const int ldin, const int y0,
                    const int ymax, const int k0, const int kmax) {
    __fp16* outptr = out;
    const __fp16* inptr = in;
    __fp16 zerobuff[24];
    std::memset(zerobuff, 0, sizeof(__fp16) * 24);
    int K16 = 16 * (kmax - k0);
    int K24 = 24 * (kmax - k0);

    int y = y0;
    for (; y + 24 <= ymax; y += 24) {
        int yi = y;
        for (; yi < y + 24; yi += 8) {
            const __fp16* inptr0 = inptr + yi * ldin + k0;
            const __fp16* inptr1 = inptr0 + ldin;
            const __fp16* inptr2 = inptr1 + ldin;
            const __fp16* inptr3 = inptr2 + ldin;
            const __fp16* inptr4 = inptr3 + ldin;
            const __fp16* inptr5 = inptr4 + ldin;
            const __fp16* inptr6 = inptr5 + ldin;
            const __fp16* inptr7 = inptr6 + ldin;
            __fp16* outptr_inner = outptr + yi - y;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);
            prefetch_2x(inptr4);
            prefetch_2x(inptr5);
            prefetch_2x(inptr6);
            prefetch_2x(inptr7);

            int x = (kmax - k0);
            for (; x > 7; x -= 8) {
                int skippf = (x & 31);
                interleave_24x1_8_h_helper(inptr0, inptr1, inptr2, inptr3,
                                           inptr4, inptr5, inptr6, inptr7,
                                           outptr_inner, skippf);
            }
            for (; x > 0; x--) {
                *outptr_inner++ = *inptr0++;
                *outptr_inner++ = *inptr1++;
                *outptr_inner++ = *inptr2++;
                *outptr_inner++ = *inptr3++;
                *outptr_inner++ = *inptr4++;
                *outptr_inner++ = *inptr5++;
                *outptr_inner++ = *inptr6++;
                *outptr_inner++ = *inptr7++;
                outptr_inner += 16;
            }
        }
        outptr += K24;
    }

    for (; y + 16 <= ymax; y += 16) {
        int yi = y;
        for (; yi < y + 16; yi += 8) {
            const __fp16* inptr0 = inptr + yi * ldin + k0;
            const __fp16* inptr1 = inptr0 + ldin;
            const __fp16* inptr2 = inptr1 + ldin;
            const __fp16* inptr3 = inptr2 + ldin;
            const __fp16* inptr4 = inptr3 + ldin;
            const __fp16* inptr5 = inptr4 + ldin;
            const __fp16* inptr6 = inptr5 + ldin;
            const __fp16* inptr7 = inptr6 + ldin;
            __fp16* outptr_inner = outptr + yi - y;

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);
            prefetch_2x(inptr4);
            prefetch_2x(inptr5);
            prefetch_2x(inptr6);
            prefetch_2x(inptr7);

            int x = (kmax - k0);
            for (; x > 7; x -= 8) {
                int skippf = (x & 31);
                interleave_16x1_8_h_helper(inptr0, inptr1, inptr2, inptr3,
                                           inptr4, inptr5, inptr6, inptr7,
                                           outptr_inner, skippf);
            }
            for (; x > 0; x--) {
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
        outptr += K16;
    }

    for (; y + 8 <= ymax; y += 8) {
        const __fp16* inptr0 = inptr + y * ldin + k0;
        const __fp16* inptr1 = inptr0 + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;
        const __fp16* inptr4 = inptr3 + ldin;
        const __fp16* inptr5 = inptr4 + ldin;
        const __fp16* inptr6 = inptr5 + ldin;
        const __fp16* inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int x = (kmax - k0);
        for (; x > 7; x -= 8) {
            int skippf = (x & 31);
            interleave_8x1_8_h(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                               inptr6, inptr7, outptr, skippf);
        }

        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
            *outptr++ = *inptr6++;
            *outptr++ = *inptr7++;
        }
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

            interleave_4x1_4_h(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        for (; x > 0; x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
        }
    }
}

void transpose_1x8(__fp16* out, const __fp16* in, int ldin, int x0, int xmax,
                    int k0, int kmax) {
    int ksize = kmax - k0;
    int ksize8 = (ksize << 3);
    int ksize4 = (ksize << 2);
    int k = ksize;
    __fp16* outptr_base8 = out;
    __fp16* outptr_base4 = out;

    const __fp16* inptr_base = in + x0 + k0 * ldin;

    for (; k > 3; k -= 4) {
        __fp16* outptr = outptr_base8;

        const __fp16* inptr = inptr_base;
        const __fp16* inptr1 = inptr + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);
        prefetch_3x(inptr3);

        int x = x0;
        for (; x + 8 <= xmax; x += 8) {
            transpose_8x4_1_h(inptr, inptr1, inptr2, inptr3, outptr);
            outptr += ksize8;
        }
        outptr += outptr_base4 - outptr_base8;
        for (; x < xmax; x += 4) {
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                *outptr++ = val;
            }
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr1++) : (__fp16)(0);
                *outptr++ = val;
            }
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr2++) : (__fp16)(0);
                *outptr++ = val;
            }
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr3++) : (__fp16)(0);
                *outptr++ = val;
            }
            outptr -= 16;
            outptr += ksize4;
        }

        inptr_base += ldin * 4;
        outptr_base8 += 8 * 4;
        outptr_base4 += 4 * 4;
    }

    if (k) {
        __fp16* outptr = outptr_base8;
        const __fp16* inptr = inptr_base;
        const __fp16* inptr1 = inptr + ldin;
        const __fp16* inptr2 = inptr1 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);

        int x = x0;
        for (; x + 8 <= xmax; x += 8) {
            switch (k) {
                case 3:
                    transpose_8x2_1_h(inptr, inptr1, outptr);
                    transpose_8x1_1_h(inptr2, outptr + 8 * 2);
                    break;

                case 2:
                    transpose_8x2_1_h(inptr, inptr1, outptr);
                    break;

                case 1:
                    transpose_8x1_1_h(inptr, outptr);
                    break;

                default:
                    megdnn_assert(0);
            }
            outptr += ksize8;
        }

        outptr += outptr_base4 - outptr_base8;
        for (; x < xmax; x += 4) {
            switch (k) {
                case 3:
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr1++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr2++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    outptr -= 12;
                    break;
                case 2:
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr1++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    outptr -= 8;
                    break;

                case 1:
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    outptr -= 4;
                    break;

                default:
                    megdnn_assert(0);
            }
            outptr += ksize4;
        }
    }
}

void transpose_1x24(__fp16* out, const __fp16* in, const int ldin, const int x0,
                    const int xmax, const int k0, const int kmax) {
    int ksize = kmax - k0;
    int ksize24 = ksize * 24;
    int ksize16 = (ksize << 4);
    int ksize8 = (ksize << 3);
    int ksize4 = (ksize << 2);
    int k = ksize;
    __fp16* outptr_base = out;
    __fp16* outptr_base16 = out;
    __fp16* outptr_base8 = out;
    __fp16* outptr_base4 = out;

    const __fp16* inptr_base = in + x0 + k0 * ldin;

    for (; k > 3; k -= 4) {
        __fp16* outptr = outptr_base;

        const __fp16* inptr = inptr_base;
        const __fp16* inptr1 = inptr + ldin;
        const __fp16* inptr2 = inptr1 + ldin;
        const __fp16* inptr3 = inptr2 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);
        prefetch_3x(inptr3);

        int x = x0;
        for (; x + 24 <= xmax; x += 24) {
            transpose_24x4_1_h(inptr, inptr1, inptr2, inptr3, outptr);
            outptr += ksize24;
        }
        outptr += outptr_base16 - outptr_base;
        for (; x + 16 <= xmax; x += 16) {
            transpose_16x4_1_h(inptr, inptr1, inptr2, inptr3, outptr);
            outptr += ksize16;
        }
        outptr += outptr_base8 - outptr_base16;
        for (; x + 8 <= xmax; x += 8) {
            transpose_8x4_1_h(inptr, inptr1, inptr2, inptr3, outptr);
            outptr += ksize8;
        }
        outptr += outptr_base4 - outptr_base8;
        for (; x < xmax; x += 4) {
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                *outptr++ = val;
            }
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr1++) : (__fp16)(0);
                *outptr++ = val;
            }
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr2++) : (__fp16)(0);
                *outptr++ = val;
            }
            for (int i = 0; i < 4; i++) {
                __fp16 val = (x + i < xmax) ? (*inptr3++) : (__fp16)(0);
                *outptr++ = val;
            }
            outptr -= 16;
            outptr += ksize4;
        }

        inptr_base += ldin * 4;
        outptr_base += 24 * 4;
        outptr_base16 += 16 * 4;
        outptr_base8 += 8 * 4;
        outptr_base4 += 4 * 4;
    }

    if (k) {
        __fp16* outptr = outptr_base;
        const __fp16* inptr = inptr_base;
        const __fp16* inptr1 = inptr + ldin;
        const __fp16* inptr2 = inptr1 + ldin;

        prefetch_3x(inptr);
        prefetch_3x(inptr1);
        prefetch_3x(inptr2);

        int x = x0;
        for (; x + 24 <= xmax; x += 24) {
            switch (k) {
                case 3:
                    transpose_24x2_1_h(inptr, inptr1, outptr);
                    transpose_24x1_1_h(inptr2, outptr + 24 * 2);
                    break;

                case 2:
                    transpose_24x2_1_h(inptr, inptr1, outptr);
                    break;

                case 1:
                    transpose_24x1_1_h(inptr, outptr);
                    break;

                default:
                    megdnn_assert(0);
            }
            outptr += ksize24;
        }

        outptr += outptr_base16 - outptr_base;
        for (; x + 16 <= xmax; x += 16) {
            switch (k) {
                case 3:
                    transpose_16x2_1_h(inptr, inptr1, outptr);
                    transpose_16x1_1_h(inptr2, outptr + 16 * 2);
                    break;

                case 2:
                    transpose_16x2_1_h(inptr, inptr1, outptr);
                    break;

                case 1:
                    transpose_16x1_1_h(inptr, outptr);
                    break;

                default:
                    megdnn_assert(0);
            }
            outptr += ksize16;
        }

        outptr += outptr_base8 - outptr_base16;
        for (; x + 8 <= xmax; x += 8) {
            switch (k) {
                case 3:
                    transpose_8x2_1_h(inptr, inptr1, outptr);
                    transpose_8x1_1_h(inptr2, outptr + 8 * 2);
                    break;

                case 2:
                    transpose_8x2_1_h(inptr, inptr1, outptr);
                    break;

                case 1:
                    transpose_8x1_1_h(inptr, outptr);
                    break;

                default:
                    megdnn_assert(0);
            }
            outptr += ksize8;
        }

        outptr += outptr_base4 - outptr_base8;
        for (; x < xmax; x += 4) {
            switch (k) {
                case 3:
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr1++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr2++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    outptr -= 12;
                    break;
                case 2:
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr1++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    outptr -= 8;
                    break;

                case 1:
                    for (int i = 0; i < 4; i++) {
                        __fp16 val = (x + i < xmax) ? (*inptr++) : (__fp16)(0);
                        *outptr++ = val;
                    }
                    outptr -= 4;
                    break;

                default:
                    megdnn_assert(0);
            }
            outptr += ksize4;
        }
    }
}

// Overview of register layout:
//
// A 2x24 cell of Rhs is stored in 16bit in q2-q7.
// A 8x2 cell of Lhs is stored in 16bit in q0-q1
// A 8x24 block of accumulators is stored in 16bit in q8--q31.
//
//                   +--------+--------+--------+
//                   | v2[0-7]| v3[0-7]| v4[0-7]|
//              Rhs  +--------+--------+--------+
//                   | v5[0-7]| v6[0-7]| v7[0-7]|
//                   +--------+--------+--------+
//
//                   |        |        |        |
//
//    Lhs            |        |        |        |
//
//  +--+--+ - - - -  +--------+--------+--------+
//  |v0|v1|          | v8[0-7]|v16[0-7]|v24[0-7]|
//  |v0|v1|          | v9[0-7]|v17[0-7]|v25[0-7]|
//  |v0|v1|          |v10[0-7]|v18[0-7]|v26[0-7]|
//  |v0|v1|          |v11[0-7]|v19[0-7]|v27[0-7]|
//  |v0|v1|          |v12[0-7]|v20[0-7]|v28[0-7]|
//  |v0|v1|          |v13[0-7]|v21[0-7]|v29[0-7]|
//  |v0|v1|          |v14[0-7]|v22[0-7]|v30[0-7]|
//  |v0|v1|          |v15[0-7]|v23[0-7]|v31[0-7]|
//  +--+--+ - - - -  +--------+--------+--------+
//
//                            Accumulator

void aarch64_hgemm_assembly_kernel_24x8(const __fp16* a_ptr,
                                        const __fp16*& b_ptr, int K,
                                        __fp16* outptr0, int ldout, int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b1 asm("v3");
    register float16x8_t b2 asm("v4");
    register float16x8_t b0a asm("v5");
    register float16x8_t b1a asm("v6");
    register float16x8_t b2a asm("v7");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;
    __fp16* outptr4 = outptr3 + ldout;
    __fp16* outptr5 = outptr4 + ldout;
    __fp16* outptr6 = outptr5 + ldout;
    __fp16* outptr7 = outptr6 + ldout;

    asm volatile(
            ".arch armv8.2-a+fp16\n"

            // load accumulator C
            "cmp %w[type], #0\n"
            "beq 5f\n"
            "ldp q8, q16, [%[outptr0]]\n"
            "ldr q24, [%[outptr0], #32]\n"
            "ldp q9, q17, [%[outptr1]]\n"
            "ldr q25, [%[outptr1], #32]\n"
            "ldp q10, q18, [%[outptr2]]\n"
            "ldr q26, [%[outptr2], #32]\n"
            "ldp q11, q19, [%[outptr3]]\n"
            "ldr q27, [%[outptr3], #32]\n"
            "ldp q12, q20, [%[outptr4]]\n"
            "ldr q28, [%[outptr4], #32]\n"
            "ldp q13, q21, [%[outptr5]]\n"
            "ldr q29, [%[outptr5], #32]\n"
            "ldp q14, q22, [%[outptr6]]\n"
            "ldr q30, [%[outptr6], #32]\n"
            "ldp q15, q23, [%[outptr7]]\n"
            "ldr q31, [%[outptr7], #32]\n"
            "b 6f\n"

            "5:\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
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
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"
            "eor v28.16b, v28.16b, v28.16b\n"
            "eor v29.16b, v29.16b, v29.16b\n"
            "eor v30.16b, v30.16b, v30.16b\n"
            "eor v31.16b, v31.16b, v31.16b\n"

            "6:\n"
            "ldr %q[a0], [%[a_ptr]]\n"
            "ldr %q[b0], [%[b_ptr]]\n"
            "ldr %q[b1], [%[b_ptr], #16]\n"
            "ldr %q[b2], [%[b_ptr], #32]\n"
            "ldr %q[b0a], [%[b_ptr], #48]\n"
            "ldr %q[b1a], [%[b_ptr], #64]\n"

            ASM_PREFETCH("[%[b_ptr], #64]")
            ASM_PREFETCH("[%[b_ptr], #128]")
            ASM_PREFETCH("[%[b_ptr], #192]")
            ASM_PREFETCH("[%[b_ptr], #256]")
            ASM_PREFETCH("[%[b_ptr], #320]")

            "cbz %w[k], 4f\n"

            "1:\n"
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "ldr %q[b2a], [%[b_ptr], #80]\n"
            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"
            "ldr %q[b0], [%[b_ptr], #96]\n"

            "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
            "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
            ASM_PREFETCH("[%[a_ptr], #128]")
            "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
            "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
            "add %[b_ptr], %[b_ptr], #96\n"
            "fmla v20.8h, %[b1].8h, %[a0].h[4]\n"
            "fmla v21.8h, %[b1].8h, %[a0].h[5]\n"
            "fmla v22.8h, %[b1].8h, %[a0].h[6]\n"
            "fmla v23.8h, %[b1].8h, %[a0].h[7]\n"
            "ldr %q[b1], [%[b_ptr], #16]\n"

            "fmla v24.8h, %[b2].8h, %[a0].h[0]\n"
            "fmla v25.8h, %[b2].8h, %[a0].h[1]\n"
            ASM_PREFETCH("[%[b_ptr], #288]")
            "fmla v26.8h, %[b2].8h, %[a0].h[2]\n"
            "fmla v27.8h, %[b2].8h, %[a0].h[3]\n"
            "fmla v28.8h, %[b2].8h, %[a0].h[4]\n"
            "fmla v29.8h, %[b2].8h, %[a0].h[5]\n"
            "fmla v30.8h, %[b2].8h, %[a0].h[6]\n"
            "fmla v31.8h, %[b2].8h, %[a0].h[7]\n"
            "ldr %q[a0], [%[a_ptr], #32]\n"

            "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
            "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
            "ldr %q[b2], [%[b_ptr], #32]\n"

            "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
            "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
            "fmla v12.8h, %[b0a].8h, %[a0a].h[4]\n"
            "fmla v13.8h, %[b0a].8h, %[a0a].h[5]\n"
            "fmla v14.8h, %[b0a].8h, %[a0a].h[6]\n"
            "fmla v15.8h, %[b0a].8h, %[a0a].h[7]\n"
            "ldr %q[b0a], [%[b_ptr], #48]\n"

            "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
            "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
            ASM_PREFETCH("[%[b_ptr], #352]")
            "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
            "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"
            "fmla v20.8h, %[b1a].8h, %[a0a].h[4]\n"
            "fmla v21.8h, %[b1a].8h, %[a0a].h[5]\n"
            "fmla v22.8h, %[b1a].8h, %[a0a].h[6]\n"
            "fmla v23.8h, %[b1a].8h, %[a0a].h[7]\n"
            "ldr %q[b1a], [%[b_ptr], #64]\n"

            "fmla v24.8h, %[b2a].8h, %[a0a].h[0]\n"
            "fmla v25.8h, %[b2a].8h, %[a0a].h[1]\n"
            "add %[a_ptr], %[a_ptr], #32\n"
            "fmla v26.8h, %[b2a].8h, %[a0a].h[2]\n"
            "fmla v27.8h, %[b2a].8h, %[a0a].h[3]\n"
            "fmla v28.8h, %[b2a].8h, %[a0a].h[4]\n"
            "fmla v29.8h, %[b2a].8h, %[a0a].h[5]\n"
            "subs %w[k], %w[k], #1\n"
            "fmla v30.8h, %[b2a].8h, %[a0a].h[6]\n"
            "fmla v31.8h, %[b2a].8h, %[a0a].h[7]\n"

            "bne 1b\n"
            "4:\n"
            // Jump to odd tail if necessary.
            "cbnz %w[oddk], 2f\n"

            // Even tail
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "ldr %q[b2a], [%[b_ptr], #80]\n"
            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"

            "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
            "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
            "add %[b_ptr], %[b_ptr], #96\n"
            "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
            "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
            "fmla v20.8h, %[b1].8h, %[a0].h[4]\n"
            "fmla v21.8h, %[b1].8h, %[a0].h[5]\n"
            "add %[a_ptr], %[a_ptr], #32\n"
            "fmla v22.8h, %[b1].8h, %[a0].h[6]\n"
            "fmla v23.8h, %[b1].8h, %[a0].h[7]\n"

            "fmla v24.8h, %[b2].8h, %[a0].h[0]\n"
            "fmla v25.8h, %[b2].8h, %[a0].h[1]\n"
            "fmla v26.8h, %[b2].8h, %[a0].h[2]\n"
            "fmla v27.8h, %[b2].8h, %[a0].h[3]\n"
            "fmla v28.8h, %[b2].8h, %[a0].h[4]\n"
            "fmla v29.8h, %[b2].8h, %[a0].h[5]\n"
            "fmla v30.8h, %[b2].8h, %[a0].h[6]\n"
            "fmla v31.8h, %[b2].8h, %[a0].h[7]\n"

            "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
            "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
            "str q8, [%[outptr0]]\n"
            "fmla v24.8h, %[b2a].8h, %[a0a].h[0]\n"
            "str q16, [%[outptr0], #16]\n"

            "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
            "str q24, [%[outptr0], #32]\n"
            "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
            "str q9, [%[outptr1]]\n"
            "fmla v25.8h, %[b2a].8h, %[a0a].h[1]\n"
            "str q17, [%[outptr1], #16]\n"

            "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
            "str q25, [%[outptr1], #32]\n"
            "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
            "str q10, [%[outptr2]]\n"
            "fmla v26.8h, %[b2a].8h, %[a0a].h[2]\n"
            "str q18, [%[outptr2], #16]\n"

            "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
            "str q26, [%[outptr2], #32]\n"
            "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"
            "str q11, [%[outptr3]]\n"
            "fmla v27.8h, %[b2a].8h, %[a0a].h[3]\n"
            "str q19, [%[outptr3], #16]\n"

            "fmla v12.8h, %[b0a].8h, %[a0a].h[4]\n"
            "str q27, [%[outptr3], #32]\n"
            "fmla v20.8h, %[b1a].8h, %[a0a].h[4]\n"
            "str q12, [%[outptr4]]\n"
            "fmla v28.8h, %[b2a].8h, %[a0a].h[4]\n"
            "str q20, [%[outptr4], #16]\n"

            "fmla v13.8h, %[b0a].8h, %[a0a].h[5]\n"
            "str q28, [%[outptr4], #32]\n"
            "fmla v21.8h, %[b1a].8h, %[a0a].h[5]\n"
            "str q13, [%[outptr5]]\n"
            "fmla v29.8h, %[b2a].8h, %[a0a].h[5]\n"
            "str q21, [%[outptr5], #16]\n"

            "fmla v14.8h, %[b0a].8h, %[a0a].h[6]\n"
            "str q29, [%[outptr5], #32]\n"
            "fmla v22.8h, %[b1a].8h, %[a0a].h[6]\n"
            "str q14, [%[outptr6]]\n"
            "fmla v30.8h, %[b2a].8h, %[a0a].h[6]\n"
            "str q22, [%[outptr6], #16]\n"

            "fmla v15.8h, %[b0a].8h, %[a0a].h[7]\n"
            "str q30, [%[outptr6], #32]\n"
            "fmla v23.8h, %[b1a].8h, %[a0a].h[7]\n"
            "str q15, [%[outptr7]]\n"
            "fmla v31.8h, %[b2a].8h, %[a0a].h[7]\n"
            "b 3f\n"

            // Odd tail
            "2:\n"
            "add %[a_ptr], %[a_ptr], #16\n"
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "add %[b_ptr], %[b_ptr], #48\n"
            "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
            "str q8, [%[outptr0]]\n"
            "fmla v24.8h, %[b2].8h, %[a0].h[0]\n"
            "str q16, [%[outptr0], #16]\n"

            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "str q24, [%[outptr0], #32]\n"
            "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
            "str q9, [%[outptr1]]\n"
            "fmla v25.8h, %[b2].8h, %[a0].h[1]\n"
            "str q17, [%[outptr1], #16]\n"

            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "str q25, [%[outptr1], #32]\n"
            "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
            "str q10, [%[outptr2]]\n"
            "fmla v26.8h, %[b2].8h, %[a0].h[2]\n"
            "str q18, [%[outptr2], #16]\n"

            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "str q26, [%[outptr2], #32]\n"
            "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
            "str q11, [%[outptr3]]\n"
            "fmla v27.8h, %[b2].8h, %[a0].h[3]\n"
            "str q19, [%[outptr3], #16]\n"

            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "str q27, [%[outptr3], #32]\n"
            "fmla v20.8h, %[b1].8h, %[a0].h[4]\n"
            "str q12, [%[outptr4]]\n"
            "fmla v28.8h, %[b2].8h, %[a0].h[4]\n"
            "str q20, [%[outptr4], #16]\n"

            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "str q28, [%[outptr4], #32]\n"
            "fmla v21.8h, %[b1].8h, %[a0].h[5]\n"
            "str q13, [%[outptr5]]\n"
            "fmla v29.8h, %[b2].8h, %[a0].h[5]\n"
            "str q21, [%[outptr5], #16]\n"

            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "str q29, [%[outptr5], #32]\n"
            "fmla v22.8h, %[b1].8h, %[a0].h[6]\n"
            "str q14, [%[outptr6]]\n"
            "fmla v30.8h, %[b2].8h, %[a0].h[6]\n"
            "str q22, [%[outptr6], #16]\n"

            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"
            "str q30, [%[outptr6], #32]\n"
            "fmla v23.8h, %[b1].8h, %[a0].h[7]\n"
            "str q15, [%[outptr7]]\n"
            "fmla v31.8h, %[b2].8h, %[a0].h[7]\n"

            "3:\n"
            "str q23, [%[outptr7], #16]\n"
            "str q31, [%[outptr7], #32]\n"
            : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0), [b1] "+w"(b1),
              [b2] "+w"(b2), [k] "+r"(k), [b0a] "+w"(b0a), [b1a] "+w"(b1a),
              [b2a] "+w"(b2a), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
              [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3),
              [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5),
              [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7)
            : [oddk] "r"(oddk), [type] "r"(type)
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
              "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory");
}

// Overview of register layout:
//
// A 2x16 cell of Rhs is stored in 16bit in q2,q3,q5,q6.
// A 8x2 cell of Lhs is stored in 16bit in q0-q1
// A 8x16 block of accumulators is stored in 16bit in q8-q15, q16-q23.
//
//                   +--------+--------+
//                   | v2[0-7]| v3[0-7]|
//              Rhs  +--------+--------+
//                   | v5[0-7]| v6[0-7]|
//                   +--------+--------+
//
//                   |        |        |
//
//    Lhs            |        |        |
//
//  +--+--+ - - - -  +--------+--------+
//  |v0|v1|          | v8[0-7]|v16[0-7]|
//  |v0|v1|          | v9[0-7]|v17[0-7]|
//  |v0|v1|          |v10[0-7]|v18[0-7]|
//  |v0|v1|          |v11[0-7]|v19[0-7]|
//  |v0|v1|          |v12[0-7]|v20[0-7]|
//  |v0|v1|          |v13[0-7]|v21[0-7]|
//  |v0|v1|          |v14[0-7]|v22[0-7]|
//  |v0|v1|          |v15[0-7]|v23[0-7]|
//  +--+--+ - - - -  +--------+--------+
//
//                        Accumulator
void aarch64_hgemm_assembly_kernel_16x8(const __fp16* a_ptr,
                                        const __fp16*& b_ptr, int K,
                                        __fp16* outptr0, int ldout, int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b1 asm("v3");
    register float16x8_t b0a asm("v5");
    register float16x8_t b1a asm("v6");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;
    __fp16* outptr4 = outptr3 + ldout;
    __fp16* outptr5 = outptr4 + ldout;
    __fp16* outptr6 = outptr5 + ldout;
    __fp16* outptr7 = outptr6 + ldout;

    asm volatile(
            ".arch armv8.2-a+fp16\n"

            // load accumulator C
            "cmp %w[type], #0\n"
            "beq 5f\n"
            "ldp q8, q16, [%[outptr0]]\n"
            "ldp q9, q17, [%[outptr1]]\n"
            "ldp q10, q18, [%[outptr2]]\n"
            "ldp q11, q19, [%[outptr3]]\n"
            "ldp q12, q20, [%[outptr4]]\n"
            "ldp q13, q21, [%[outptr5]]\n"
            "ldp q14, q22, [%[outptr6]]\n"
            "ldp q15, q23, [%[outptr7]]\n"
            "b 6f\n"

            "5:\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
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

            "6:\n"
            "ldr %q[a0], [%[a_ptr]]\n"
            "ldr %q[b0], [%[b_ptr]]\n"
            "ldr %q[b1], [%[b_ptr], #16]\n"
            "ldr %q[b0a], [%[b_ptr], #32]\n"
            "ldr %q[b1a], [%[b_ptr], #48]\n"

            "cbz %w[k], 4f\n"

            "1:\n"
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"
            "ldr %q[b0], [%[b_ptr], #64]\n"

            "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
            "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
            "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
            "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
            "add %[b_ptr], %[b_ptr], #64\n"
            "fmla v20.8h, %[b1].8h, %[a0].h[4]\n"
            "fmla v21.8h, %[b1].8h, %[a0].h[5]\n"
            "fmla v22.8h, %[b1].8h, %[a0].h[6]\n"
            "fmla v23.8h, %[b1].8h, %[a0].h[7]\n"
            "ldr %q[b1], [%[b_ptr], #16]\n"

            "ldr %q[a0], [%[a_ptr], #32]\n"

            "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
            "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
            "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
            "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
            "fmla v12.8h, %[b0a].8h, %[a0a].h[4]\n"
            "fmla v13.8h, %[b0a].8h, %[a0a].h[5]\n"
            "fmla v14.8h, %[b0a].8h, %[a0a].h[6]\n"
            "fmla v15.8h, %[b0a].8h, %[a0a].h[7]\n"
            "ldr %q[b0a], [%[b_ptr], #32]\n"

            "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
            "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
            "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
            "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"
            "fmla v20.8h, %[b1a].8h, %[a0a].h[4]\n"
            "fmla v21.8h, %[b1a].8h, %[a0a].h[5]\n"
            "fmla v22.8h, %[b1a].8h, %[a0a].h[6]\n"
            "fmla v23.8h, %[b1a].8h, %[a0a].h[7]\n"
            "ldr %q[b1a], [%[b_ptr], #48]\n"

            "add %[a_ptr], %[a_ptr], #32\n"
            "subs %w[k], %w[k], #1\n"

            "bne 1b\n"
            "4:\n"
            // Jump to odd tail if necessary.
            "cbnz %w[oddk], 2f\n"

            // Even tail
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"

            "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
            "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
            "add %[b_ptr], %[b_ptr], #64\n"
            "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
            "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
            "fmla v20.8h, %[b1].8h, %[a0].h[4]\n"
            "fmla v21.8h, %[b1].8h, %[a0].h[5]\n"
            "add %[a_ptr], %[a_ptr], #32\n"
            "fmla v22.8h, %[b1].8h, %[a0].h[6]\n"
            "fmla v23.8h, %[b1].8h, %[a0].h[7]\n"

            "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
            "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
            "str q8, [%[outptr0]]\n"
            "str q16, [%[outptr0], #16]\n"

            "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
            "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
            "str q9, [%[outptr1]]\n"
            "str q17, [%[outptr1], #16]\n"

            "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
            "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
            "str q10, [%[outptr2]]\n"
            "str q18, [%[outptr2], #16]\n"

            "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
            "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"
            "str q11, [%[outptr3]]\n"
            "str q19, [%[outptr3], #16]\n"

            "fmla v12.8h, %[b0a].8h, %[a0a].h[4]\n"
            "fmla v20.8h, %[b1a].8h, %[a0a].h[4]\n"
            "str q12, [%[outptr4]]\n"
            "str q20, [%[outptr4], #16]\n"

            "fmla v13.8h, %[b0a].8h, %[a0a].h[5]\n"
            "fmla v21.8h, %[b1a].8h, %[a0a].h[5]\n"
            "str q13, [%[outptr5]]\n"
            "str q21, [%[outptr5], #16]\n"

            "fmla v14.8h, %[b0a].8h, %[a0a].h[6]\n"
            "fmla v22.8h, %[b1a].8h, %[a0a].h[6]\n"
            "str q14, [%[outptr6]]\n"
            "str q22, [%[outptr6], #16]\n"

            "fmla v15.8h, %[b0a].8h, %[a0a].h[7]\n"
            "fmla v23.8h, %[b1a].8h, %[a0a].h[7]\n"
            "str q15, [%[outptr7]]\n"
            "b 3f\n"

            // Odd tail
            "2:\n"
            "add %[a_ptr], %[a_ptr], #16\n"
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "add %[b_ptr], %[b_ptr], #32\n"
            "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
            "str q8, [%[outptr0]]\n"
            "str q16, [%[outptr0], #16]\n"

            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
            "str q9, [%[outptr1]]\n"
            "str q17, [%[outptr1], #16]\n"

            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
            "str q10, [%[outptr2]]\n"
            "str q18, [%[outptr2], #16]\n"

            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
            "str q11, [%[outptr3]]\n"
            "str q19, [%[outptr3], #16]\n"

            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "fmla v20.8h, %[b1].8h, %[a0].h[4]\n"
            "str q12, [%[outptr4]]\n"
            "str q20, [%[outptr4], #16]\n"

            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "fmla v21.8h, %[b1].8h, %[a0].h[5]\n"
            "str q13, [%[outptr5]]\n"
            "str q21, [%[outptr5], #16]\n"

            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "fmla v22.8h, %[b1].8h, %[a0].h[6]\n"
            "str q14, [%[outptr6]]\n"
            "str q22, [%[outptr6], #16]\n"

            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"
            "fmla v23.8h, %[b1].8h, %[a0].h[7]\n"
            "str q15, [%[outptr7]]\n"

            "3:\n"
            "str q23, [%[outptr7], #16]\n"
            : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0), [b1] "+w"(b1),
              [k] "+r"(k), [b0a] "+w"(b0a), [b1a] "+w"(b1a),
              [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [outptr0] "+r"(outptr0),
              [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2),
              [outptr3] "+r"(outptr3), [outptr4] "+r"(outptr4),
              [outptr5] "+r"(outptr5), [outptr6] "+r"(outptr6),
              [outptr7] "+r"(outptr7)
            : [oddk] "r"(oddk), [type] "r"(type)
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "cc", "memory");
}

// Overview of register layout:
//
// A 2x8 cell of Rhs is stored in 16bit in q2,q5
// A 8x2 cell of Lhs is stored in 16bit in q0-q1
// A 8x8 block of accumulators is stored in 16bit in q8-q15.
//
//                   +--------+
//                   | v2[0-7]|
//              Rhs  +--------+
//                   | v5[0-7]|
//                   +--------+
//
//                   |        |
//
//    Lhs            |        |
//
//  +--+--+ - - - -  +--------+
//  |v0|v1|          | v8[0-7]|
//  |v0|v1|          | v9[0-7]|
//  |v0|v1|          |v10[0-7]|
//  |v0|v1|          |v11[0-7]|
//  |v0|v1|          |v12[0-7]|
//  |v0|v1|          |v13[0-7]|
//  |v0|v1|          |v14[0-7]|
//  |v0|v1|          |v15[0-7]|
//  +--+--+ - - - -  +--------+
//
//                  Accumulator
void aarch64_hgemm_assembly_kernel_8x8(const __fp16* a_ptr,
                                       const __fp16*& b_ptr, int K,
                                       __fp16* outptr0, int ldout, int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b0a asm("v5");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;
    __fp16* outptr4 = outptr3 + ldout;
    __fp16* outptr5 = outptr4 + ldout;
    __fp16* outptr6 = outptr5 + ldout;
    __fp16* outptr7 = outptr6 + ldout;

    asm volatile(
            ".arch armv8.2-a+fp16\n"

            // load accumulator C
            "cmp %w[type], #0\n"
            "beq 5f\n"
            "ldr q8, [%[outptr0]]\n"
            "ldr q9, [%[outptr1]]\n"
            "ldr q10, [%[outptr2]]\n"
            "ldr q11, [%[outptr3]]\n"
            "ldr q12, [%[outptr4]]\n"
            "ldr q13, [%[outptr5]]\n"
            "ldr q14, [%[outptr6]]\n"
            "ldr q15, [%[outptr7]]\n"
            "b 6f\n"

            "5:\n"
            "eor v8.16b,  v8.16b,  v8.16b\n"
            "eor v9.16b,  v9.16b,  v9.16b\n"
            "eor v10.16b, v10.16b, v10.16b\n"
            "eor v11.16b, v11.16b, v11.16b\n"
            "eor v12.16b, v12.16b, v12.16b\n"
            "eor v13.16b, v13.16b, v13.16b\n"
            "eor v14.16b, v14.16b, v14.16b\n"
            "eor v15.16b, v15.16b, v15.16b\n"

            "6:\n"
            "ldr %q[a0], [%[a_ptr]]\n"
            "ldr %q[b0], [%[b_ptr]]\n"
            "ldr %q[b0a], [%[b_ptr], #16]\n"

            "cbz %w[k], 4f\n"

            "1:\n"
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"
            "ldr %q[b0], [%[b_ptr], #32]\n"

            "add %[b_ptr], %[b_ptr], #32\n"
            "ldr %q[a0], [%[a_ptr], #32]\n"

            "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
            "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"

            "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
            "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
            "fmla v12.8h, %[b0a].8h, %[a0a].h[4]\n"
            "fmla v13.8h, %[b0a].8h, %[a0a].h[5]\n"
            "fmla v14.8h, %[b0a].8h, %[a0a].h[6]\n"
            "fmla v15.8h, %[b0a].8h, %[a0a].h[7]\n"
            "ldr %q[b0a], [%[b_ptr], #16]\n"

            "add %[a_ptr], %[a_ptr], #32\n"
            "subs %w[k], %w[k], #1\n"

            "bne 1b\n"
            "4:\n"
            // Jump to odd tail if necessary.
            "cbnz %w[oddk], 2f\n"

            // Even tail
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"

            "add %[b_ptr], %[b_ptr], #32\n"
            "add %[a_ptr], %[a_ptr], #32\n"

            "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
            "str q8, [%[outptr0]]\n"

            "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
            "str q9, [%[outptr1]]\n"

            "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
            "str q10, [%[outptr2]]\n"

            "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
            "str q11, [%[outptr3]]\n"

            "fmla v12.8h, %[b0a].8h, %[a0a].h[4]\n"
            "str q12, [%[outptr4]]\n"

            "fmla v13.8h, %[b0a].8h, %[a0a].h[5]\n"
            "str q13, [%[outptr5]]\n"

            "fmla v14.8h, %[b0a].8h, %[a0a].h[6]\n"
            "str q14, [%[outptr6]]\n"

            "fmla v15.8h, %[b0a].8h, %[a0a].h[7]\n"
            "str q15, [%[outptr7]]\n"
            "b 3f\n"

            // Odd tail
            "2:\n"
            "add %[a_ptr], %[a_ptr], #16\n"
            "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
            "add %[b_ptr], %[b_ptr], #16\n"
            "str q8, [%[outptr0]]\n"

            "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
            "str q9, [%[outptr1]]\n"

            "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
            "str q10, [%[outptr2]]\n"

            "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
            "str q11, [%[outptr3]]\n"

            "fmla v12.8h, %[b0].8h, %[a0].h[4]\n"
            "str q12, [%[outptr4]]\n"

            "fmla v13.8h, %[b0].8h, %[a0].h[5]\n"
            "str q13, [%[outptr5]]\n"

            "fmla v14.8h, %[b0].8h, %[a0].h[6]\n"
            "str q14, [%[outptr6]]\n"

            "fmla v15.8h, %[b0].8h, %[a0].h[7]\n"
            "str q15, [%[outptr7]]\n"

            "3:\n"
            : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0), [k] "+r"(k),
              [b0a] "+w"(b0a), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
              [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3),
              [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5),
              [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7)
            : [oddk] "r"(oddk), [type] "r"(type)
            : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "cc",
              "memory");
}

// Overview of register layout:
//
// A 2x8 cell of Rhs is stored in 16bit in d2, d5
// A 8x2 cell of Lhs is stored in 16bit in q0 - q1
// A 8x8 block of accumulators is stored in 16bit in d8 - d15.
//
//                   +--------+
//                   | d2[0-3]|
//              Rhs  +--------+
//                   | d5[0-3]|
//                   +--------+
//
//                   |        |
//
//    Lhs            |        |
//
//  +--+--+ - - - -  +--------+
//  |v0|v1|          | d8[0-3]|
//  |v0|v1|          | d9[0-3]|
//  |v0|v1|          |d10[0-3]|
//  |v0|v1|          |d11[0-3]|
//  |v0|v1|          |d12[0-3]|
//  |v0|v1|          |d13[0-3]|
//  |v0|v1|          |d14[0-3]|
//  |v0|v1|          |d15[0-3]|
//  +--+--+ - - - -  +--------+
//
//                  Accumulator
void aarch64_hgemm_assembly_kernel_4x8(const __fp16* a_ptr,
                                       const __fp16*& b_ptr, int K,
                                       __fp16* outptr0, int ldout, int x_remain,
                                       int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b0a asm("v5");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;
    __fp16* outptr4 = outptr3 + ldout;
    __fp16* outptr5 = outptr4 + ldout;
    __fp16* outptr6 = outptr5 + ldout;
    __fp16* outptr7 = outptr6 + ldout;

#define LOAD_LINE(reg_index, n)            \
    "mov x0, %[outptr" n                   \
    "]\n"                                  \
    "cmp %w[x_remain], #4\n"               \
    "b.lt REMAIN_LOAD_LINE_LESS_THAN_4_" n \
    "\n"                                   \
    "ldr d" reg_index                      \
    ", [x0]\n"                             \
    "b LOAD_LINE_END_" n                   \
    "\n"                                   \
                                           \
    "REMAIN_LOAD_LINE_LESS_THAN_4_" n      \
    ":\n"                                  \
    "cmp %w[x_remain], #0\n"               \
    "beq LOAD_LINE_END_" n                 \
    "\n"                                   \
    "ld1 {v" reg_index                     \
    ".h}[0], [x0], #2\n"                   \
    "cmp %w[x_remain], #1\n"               \
    "beq LOAD_LINE_END_" n                 \
    "\n"                                   \
    "ld1 {v" reg_index                     \
    ".h}[1], [x0], #2\n"                   \
    "cmp %w[x_remain], #2\n"               \
    "beq LOAD_LINE_END_" n                 \
    "\n"                                   \
    "ld1 {v" reg_index                     \
    ".h}[2], [x0], #2\n"                   \
    "LOAD_LINE_END_" n ":\n"

#define LOAD_C           \
    LOAD_LINE("8", "0")  \
    LOAD_LINE("9", "1")  \
    LOAD_LINE("10", "2") \
    LOAD_LINE("11", "3") \
    LOAD_LINE("12", "4") \
    LOAD_LINE("13", "5") \
    LOAD_LINE("14", "6") \
    LOAD_LINE("15", "7")

#define STORE_LINE(reg_index, n)            \
    "mov x0, %[outptr" n                    \
    "]\n"                                   \
    "cmp %w[x_remain], #4\n"                \
    "b.lt REMAIN_STORE_LINE_LESS_THAN_4_" n \
    "\n"                                    \
    "str d" reg_index                       \
    ", [x0]\n"                              \
    "b STORE_LINE_END_" n                   \
    "\n"                                    \
                                            \
    "REMAIN_STORE_LINE_LESS_THAN_4_" n      \
    ":\n"                                   \
    "cmp %w[x_remain], #0\n"                \
    "beq STORE_LINE_END_" n                 \
    "\n"                                    \
    "st1 {v" reg_index                      \
    ".h}[0], [x0], #2\n"                    \
    "cmp %w[x_remain], #1\n"                \
    "beq STORE_LINE_END_" n                 \
    "\n"                                    \
    "st1 {v" reg_index                      \
    ".h}[1], [x0], #2\n"                    \
    "cmp %w[x_remain], #2\n"                \
    "beq STORE_LINE_END_" n                 \
    "\n"                                    \
    "st1 {v" reg_index                      \
    ".h}[2], [x0], #2\n"                    \
    "STORE_LINE_END_" n ":\n"

#define STORE_C           \
    STORE_LINE("8", "0")  \
    STORE_LINE("9", "1")  \
    STORE_LINE("10", "2") \
    STORE_LINE("11", "3") \
    STORE_LINE("12", "4") \
    STORE_LINE("13", "5") \
    STORE_LINE("14", "6") \
    STORE_LINE("15", "7")

    asm volatile(
            ".arch armv8.2-a+fp16\n"

            // load accumulator C
            "cmp %w[type], #0\n"
            "beq 5f\n" LOAD_C
            "b 6f\n"

            "5:\n"
            "eor v8.8b,  v8.8b,  v8.8b\n"
            "eor v9.8b,  v9.8b,  v9.8b\n"
            "eor v10.8b, v10.8b, v10.8b\n"
            "eor v11.8b, v11.8b, v11.8b\n"
            "eor v12.8b, v12.8b, v12.8b\n"
            "eor v13.8b, v13.8b, v13.8b\n"
            "eor v14.8b, v14.8b, v14.8b\n"
            "eor v15.8b, v15.8b, v15.8b\n"

            "6:\n"
            "ldr %q[a0], [%[a_ptr]]\n"

            "cbz %w[k], 4f\n"

            "1:\n"
            "ldp %d[b0], %d[b0a], [%[b_ptr]]\n"
            "fmla v8.4h , %[b0].4h, %[a0].h[0]\n"
            "fmla v9.4h , %[b0].4h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.4h, %[b0].4h, %[a0].h[2]\n"
            "fmla v11.4h, %[b0].4h, %[a0].h[3]\n"
            "fmla v12.4h, %[b0].4h, %[a0].h[4]\n"
            "fmla v13.4h, %[b0].4h, %[a0].h[5]\n"
            "fmla v14.4h, %[b0].4h, %[a0].h[6]\n"
            "fmla v15.4h, %[b0].4h, %[a0].h[7]\n"

            "add %[b_ptr], %[b_ptr], #16\n"
            "ldr %q[a0], [%[a_ptr], #32]\n"

            "fmla v8.4h , %[b0a].4h, %[a0a].h[0]\n"
            "fmla v9.4h , %[b0a].4h, %[a0a].h[1]\n"
            "fmla v10.4h, %[b0a].4h, %[a0a].h[2]\n"
            "fmla v11.4h, %[b0a].4h, %[a0a].h[3]\n"
            "fmla v12.4h, %[b0a].4h, %[a0a].h[4]\n"
            "fmla v13.4h, %[b0a].4h, %[a0a].h[5]\n"
            "fmla v14.4h, %[b0a].4h, %[a0a].h[6]\n"
            "fmla v15.4h, %[b0a].4h, %[a0a].h[7]\n"

            "add %[a_ptr], %[a_ptr], #32\n"
            "subs %w[k], %w[k], #1\n"

            "bne 1b\n"
            "4:\n"
            // Jump to odd tail if necessary.
            "cbnz %w[oddk], 2f\n"

            // Even tail
            "ldp %d[b0], %d[b0a], [%[b_ptr]]\n"
            "fmla v8.4h , %[b0].4h, %[a0].h[0]\n"
            "fmla v9.4h , %[b0].4h, %[a0].h[1]\n"
            "ldr %q[a0a], [%[a_ptr], #16]\n"
            "fmla v10.4h, %[b0].4h, %[a0].h[2]\n"
            "fmla v11.4h, %[b0].4h, %[a0].h[3]\n"
            "fmla v12.4h, %[b0].4h, %[a0].h[4]\n"
            "fmla v13.4h, %[b0].4h, %[a0].h[5]\n"
            "fmla v14.4h, %[b0].4h, %[a0].h[6]\n"
            "fmla v15.4h, %[b0].4h, %[a0].h[7]\n"

            "add %[b_ptr], %[b_ptr], #16\n"
            "add %[a_ptr], %[a_ptr], #32\n"

            "fmla v8.4h , %[b0a].4h, %[a0a].h[0]\n"
            "fmla v9.4h , %[b0a].4h, %[a0a].h[1]\n"
            "fmla v10.4h, %[b0a].4h, %[a0a].h[2]\n"
            "fmla v11.4h, %[b0a].4h, %[a0a].h[3]\n"
            "fmla v12.4h, %[b0a].4h, %[a0a].h[4]\n"
            "fmla v13.4h, %[b0a].4h, %[a0a].h[5]\n"
            "fmla v14.4h, %[b0a].4h, %[a0a].h[6]\n"
            "fmla v15.4h, %[b0a].4h, %[a0a].h[7]\n"
            "b 3f\n"

            // Odd tail
            "2:\n"
            "ldr %d[b0], [%[b_ptr]]\n"
            "add %[a_ptr], %[a_ptr], #16\n"
            "fmla v8.4h , %[b0].4h, %[a0].h[0]\n"
            "add %[b_ptr], %[b_ptr], #8\n"
            "fmla v9.4h , %[b0].4h, %[a0].h[1]\n"
            "fmla v10.4h, %[b0].4h, %[a0].h[2]\n"
            "fmla v11.4h, %[b0].4h, %[a0].h[3]\n"
            "fmla v12.4h, %[b0].4h, %[a0].h[4]\n"
            "fmla v13.4h, %[b0].4h, %[a0].h[5]\n"
            "fmla v14.4h, %[b0].4h, %[a0].h[6]\n"
            "fmla v15.4h, %[b0].4h, %[a0].h[7]\n"

            "3:\n" STORE_C
            : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0), [k] "+r"(k),
              [b0a] "+w"(b0a), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
              [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3),
              [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5),
              [outptr6] "+r"(outptr6), [outptr7] "+r"(outptr7)
            : [oddk] "r"(oddk), [x_remain] "r"(x_remain), [type] "r"(type)
            : "x0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "cc",
              "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A 2x24 cell of Rhs is stored in 16bit in q2 - q7
// A 4x2 cell of Lhs is stored in 16bit in d0, d1
// A 4x24 block of accumulators is stored in 16bit in q8-q11, q16-q19, q24-q27.
//
//                   +--------+--------+--------+
//                   | v2[0-7]| v3[0-7]| v4[0-7]|
//              Rhs  +--------+--------+--------+
//                   | v5[0-7]| v6[0-7]| v7[0-7]|
//                   +--------+--------+--------+
//
//                   |        |        |        |
//
//    Lhs            |        |        |        |
//
//  +--+--+ - - - -  +--------+--------+--------+
//  |v0|v1|          | v8[0-7]|v16[0-7]|v24[0-7]|
//  |v0|v1|          | v9[0-7]|v17[0-7]|v25[0-7]|
//  |v0|v1|          |v10[0-7]|v18[0-7]|v26[0-7]|
//  |v0|v1|          |v11[0-7]|v19[0-7]|v27[0-7]|
//  +--+--+ - - - -  +--------+--------+--------+
//
//                            Accumulator
//! cannot load %[a0] and %[a0a] at same time!
void aarch64_hgemm_assembly_kernel_24x4(const __fp16* a_ptr,
                                        const __fp16*& b_ptr, int K,
                                        __fp16* outptr0, int ldout,
                                        int y_remain, int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b1 asm("v3");
    register float16x8_t b2 asm("v4");
    register float16x8_t b0a asm("v5");
    register float16x8_t b1a asm("v6");
    register float16x8_t b2a asm("v7");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;

// clang-format off
#define LOAD_LINE(v1, v2, v3, n)        \
    "cbz w0, LOAD_24x4_C_END\n"         \
    "ldp q" v1 ", q" v2 ", [%[outptr" n \
    "]]\n"                              \
    "ldr q" v3 ", [%[outptr" n          \
    "], #32]\n"                         \
    "subs w0, w0, #1\n"

#define LOAD_C                       \
    "mov w0, %w[y_remain]\n"         \
    LOAD_LINE("8", "16", "24", "0")  \
    LOAD_LINE("9", "17", "25", "1")  \
    LOAD_LINE("10", "18", "26", "2") \
    LOAD_LINE("11", "19", "27", "3") \
    "LOAD_24x4_C_END:\n"

#define STORE_LINE(v1, v2, v3, n)       \
    "cbz w0, STORE_24x4_C_END\n"        \
    "stp q" v1 ", q" v2 ", [%[outptr" n \
    "]]\n"                              \
    "str q" v3 ", [%[outptr" n          \
    "], #32]\n"                         \
    "subs w0, w0, #1\n"

#define STORE_C "mov w0, %w[y_remain]\n"      \
            STORE_LINE("8", "16", "24", "0")  \
            STORE_LINE("9", "17", "25", "1")  \
            STORE_LINE("10", "18", "26", "2") \
            STORE_LINE("11", "19", "27", "3") \
            "STORE_24x4_C_END:\n"
// clang-format on

            asm volatile(
                    ".arch armv8.2-a+fp16\n"

                    // load accumulator C
                    "cmp %w[type], #0\n"
                    "beq 5f\n"
                    LOAD_C
                    "b 6f\n"
                    "5:\n"
                    "eor v8.16b,  v8.16b,  v8.16b\n"
                    "eor v9.16b,  v9.16b,  v9.16b\n"
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "eor v11.16b, v11.16b, v11.16b\n"

                    "eor v16.16b, v16.16b, v16.16b\n"
                    "eor v17.16b, v17.16b, v17.16b\n"
                    "eor v18.16b, v18.16b, v18.16b\n"
                    "eor v19.16b, v19.16b, v19.16b\n"

                    "eor v24.16b, v24.16b, v24.16b\n"
                    "eor v25.16b, v25.16b, v25.16b\n"
                    "eor v26.16b, v26.16b, v26.16b\n"
                    "eor v27.16b, v27.16b, v27.16b\n"

                    "6:\n"
                    "ldr %d[a0], [%[a_ptr]]\n"
                    "ldr %q[b0], [%[b_ptr]]\n"
                    "ldr %q[b1], [%[b_ptr], #16]\n"
                    "ldr %q[b2], [%[b_ptr], #32]\n"
                    "ldr %q[b0a], [%[b_ptr], #48]\n"
                    "ldr %q[b1a], [%[b_ptr], #64]\n"

                    "cbz %w[k], 4f\n"

                    "1:\n"
                    "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
                    "ldr %q[b2a], [%[b_ptr], #80]\n"
                    "ldr %q[b0], [%[b_ptr], #96]\n"

                    "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
                    "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
                    "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
                    "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
                    "add %[b_ptr], %[b_ptr], #96\n"
                    "ldr %q[b1], [%[b_ptr], #16]\n"

                    "fmla v24.8h, %[b2].8h, %[a0].h[0]\n"
                    "fmla v25.8h, %[b2].8h, %[a0].h[1]\n"
                    "fmla v26.8h, %[b2].8h, %[a0].h[2]\n"
                    "fmla v27.8h, %[b2].8h, %[a0].h[3]\n"
                    "ldr %d[a0], [%[a_ptr], #16]\n"

                    "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
                    "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
                    "ldr %q[b2], [%[b_ptr], #32]\n"

                    "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                    "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
                    "ldr %q[b0a], [%[b_ptr], #48]\n"

                    "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
                    "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
                    "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
                    "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"
                    "ldr %q[b1a], [%[b_ptr], #64]\n"

                    "fmla v24.8h, %[b2a].8h, %[a0a].h[0]\n"
                    "fmla v25.8h, %[b2a].8h, %[a0a].h[1]\n"
                    "add %[a_ptr], %[a_ptr], #16\n"
                    "fmla v26.8h, %[b2a].8h, %[a0a].h[2]\n"
                    "fmla v27.8h, %[b2a].8h, %[a0a].h[3]\n"
                    "subs %w[k], %w[k], #1\n"

                    "bne 1b\n"
                    "4:\n"
                    // Jump to odd tail if necessary.
                    "cbnz %w[oddk], 2f\n"

                    // Even tail
                    "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
                    "ldr %q[b2a], [%[b_ptr], #80]\n"

                    "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
                    "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
                    "add %[b_ptr], %[b_ptr], #96\n"
                    "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
                    "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
                    "add %[a_ptr], %[a_ptr], #16\n"

                    "fmla v24.8h, %[b2].8h, %[a0].h[0]\n"
                    "fmla v25.8h, %[b2].8h, %[a0].h[1]\n"
                    "fmla v26.8h, %[b2].8h, %[a0].h[2]\n"
                    "fmla v27.8h, %[b2].8h, %[a0].h[3]\n"

                    "fmla v8.8h, %[b0a].8h, %[a0a].h[0]\n"
                    "fmla v9.8h, %[b0a].8h, %[a0a].h[1]\n"
                    "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                    "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"

                    "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
                    "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
                    "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
                    "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"

                    "fmla v24.8h, %[b2a].8h, %[a0a].h[0]\n"
                    "fmla v25.8h, %[b2a].8h, %[a0a].h[1]\n"
                    "fmla v26.8h, %[b2a].8h, %[a0a].h[2]\n"
                    "fmla v27.8h, %[b2a].8h, %[a0a].h[3]\n"
                    "b 3f\n"

                    // Odd tail
                    "2:\n"
                    "add %[a_ptr], %[a_ptr], #8\n"
                    "add %[b_ptr], %[b_ptr], #48\n"

                    "fmla v8.8h, %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h, %[b0].8h, %[a0].h[1]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"

                    "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
                    "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
                    "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
                    "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"

                    "fmla v24.8h, %[b2].8h, %[a0].h[0]\n"
                    "fmla v25.8h, %[b2].8h, %[a0].h[1]\n"
                    "fmla v26.8h, %[b2].8h, %[a0].h[2]\n"
                    "fmla v27.8h, %[b2].8h, %[a0].h[3]\n"

                    "3:\n" STORE_C
                    : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0),
                      [b1] "+w"(b1), [b2] "+w"(b2), [k] "+r"(k),
                      [b0a] "+w"(b0a), [b1a] "+w"(b1a), [b2a] "+w"(b2a),
                      [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                      [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
                      [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3)
                    :
                    [oddk] "r"(oddk), [y_remain] "r"(y_remain), [type] "r"(type)
                    : "w0", "v8", "v9", "v10", "v11", "v16", "v17", "v18",
                      "v19", "v24", "v25", "v26", "v27", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A 2x16 cell of Rhs is stored in 16bit in q2, q3, q5, q6
// A 4x2 cell of Lhs is stored in 16bit in d0, d1
// A 4x16 block of accumulators is stored in 16bit in q8-q11, q16-q19.
//
//                   +--------+--------+
//                   | v2[0-7]| v3[0-7]|
//              Rhs  +--------+--------+
//                   | v5[0-7]| v6[0-7]|
//                   +--------+--------+
//
//                   |        |        |
//
//    Lhs            |        |        |
//
//  +--+--+ - - - -  +--------+--------+
//  |v0|v1|          | v8[0-7]|v16[0-7]|
//  |v0|v1|          | v9[0-7]|v17[0-7]|
//  |v0|v1|          |v10[0-7]|v18[0-7]|
//  |v0|v1|          |v11[0-7]|v19[0-7]|
//  +--+--+ - - - -  +--------+--------+
//
//                       Accumulator
void aarch64_hgemm_assembly_kernel_16x4(const __fp16* a_ptr,
                                        const __fp16*& b_ptr, int K,
                                        __fp16* outptr0, int ldout,
                                        int y_remain, int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b1 asm("v3");
    register float16x8_t b0a asm("v5");
    register float16x8_t b1a asm("v6");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;

// clang-format off

#define LOAD_LINE(v1, v2, n)            \
    "cbz w0, LOAD_16x4_C_END\n"         \
    "ldp q" v1 ", q" v2 ", [%[outptr" n \
    "]]\n"                              \
    "subs w0, w0, #1\n"

#define LOAD_C "mov w0, %w[y_remain]\n" \
    LOAD_LINE("8", "16", "0")           \
    LOAD_LINE("9", "17", "1")           \
    LOAD_LINE("10", "18", "2")          \
    LOAD_LINE("11", "19", "3")          \
    "LOAD_16x4_C_END:\n"

#define STORE_LINE(v1, v2, n)           \
    "cbz w0, STORE_16x4_C_END\n"        \
    "stp q" v1 ", q" v2 ", [%[outptr" n \
    "]]\n"                              \
    "subs w0, w0, #1\n"

#define STORE_C "mov w0, %w[y_remain]\n" \
            STORE_LINE("8", "16", "0")   \
            STORE_LINE("9", "17", "1")   \
            STORE_LINE("10", "18", "2")  \
            STORE_LINE("11", "19", "3")  \
            "STORE_16x4_C_END:\n"

// clang-format on

            asm volatile(
                    ".arch armv8.2-a+fp16\n"

                    // load accumulator C
                    "cmp %w[type], #0\n"
                    "beq 5f\n" LOAD_C
                    "b 6f\n"

                    "5:\n"
                    "eor v8.16b,  v8.16b,  v8.16b\n"
                    "eor v9.16b,  v9.16b,  v9.16b\n"
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "eor v11.16b, v11.16b, v11.16b\n"

                    "eor v16.16b, v16.16b, v16.16b\n"
                    "eor v17.16b, v17.16b, v17.16b\n"
                    "eor v18.16b, v18.16b, v18.16b\n"
                    "eor v19.16b, v19.16b, v19.16b\n"

                    "6:\n"
                    "ldr %d[a0], [%[a_ptr]]\n"
                    "ldr %q[b0], [%[b_ptr]]\n"
                    "ldr %q[b1], [%[b_ptr], #16]\n"
                    "ldr %q[b0a], [%[b_ptr], #32]\n"
                    "ldr %q[b1a], [%[b_ptr], #48]\n"

                    "cbz %w[k], 4f\n"

                    "1:\n"
                    "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
                    "ldr %q[b0], [%[b_ptr], #64]\n"

                    "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
                    "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
                    "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
                    "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
                    "add %[b_ptr], %[b_ptr], #64\n"
                    "ldr %q[b1], [%[b_ptr], #16]\n"

                    "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
                    "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
                    "ldr %d[a0], [%[a_ptr], #16]\n"
                    "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                    "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
                    "ldr %q[b0a], [%[b_ptr], #32]\n"

                    "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
                    "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
                    "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
                    "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"
                    "ldr %q[b1a], [%[b_ptr], #48]\n"

                    "add %[a_ptr], %[a_ptr], #16\n"
                    "subs %w[k], %w[k], #1\n"

                    "bne 1b\n"
                    "4:\n"
                    // Jump to odd tail if necessary.
                    "cbnz %w[oddk], 2f\n"

                    // Even tail
                    "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"

                    "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
                    "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
                    "add %[b_ptr], %[b_ptr], #64\n"
                    "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
                    "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"
                    "add %[a_ptr], %[a_ptr], #16\n"

                    "fmla v8.8h, %[b0a].8h, %[a0a].h[0]\n"
                    "fmla v9.8h, %[b0a].8h, %[a0a].h[1]\n"
                    "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                    "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"

                    "fmla v16.8h, %[b1a].8h, %[a0a].h[0]\n"
                    "fmla v17.8h, %[b1a].8h, %[a0a].h[1]\n"
                    "fmla v18.8h, %[b1a].8h, %[a0a].h[2]\n"
                    "fmla v19.8h, %[b1a].8h, %[a0a].h[3]\n"

                    "b 3f\n"

                    // Odd tail
                    "2:\n"
                    "add %[a_ptr], %[a_ptr], #8\n"
                    "add %[b_ptr], %[b_ptr], #32\n"

                    "fmla v8.8h, %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h, %[b0].8h, %[a0].h[1]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"

                    "fmla v16.8h, %[b1].8h, %[a0].h[0]\n"
                    "fmla v17.8h, %[b1].8h, %[a0].h[1]\n"
                    "fmla v18.8h, %[b1].8h, %[a0].h[2]\n"
                    "fmla v19.8h, %[b1].8h, %[a0].h[3]\n"

                    "3:\n" STORE_C
                    : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0),
                      [b1] "+w"(b1), [k] "+r"(k), [b0a] "+w"(b0a),
                      [b1a] "+w"(b1a), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
                      [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1),
                      [outptr2] "+r"(outptr2), [outptr3] "+r"(outptr3)
                    :
                    [oddk] "r"(oddk), [y_remain] "r"(y_remain), [type] "r"(type)
                    : "w0", "v8", "v9", "v10", "v11", "v16", "v17", "v18",
                      "v19", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A 2x8 cell of Rhs is stored in 16bit in q2, q5
// A 4x2 cell of Lhs is stored in 16bit in d0, d1
// A 4x8 block of accumulators is stored in 16bit in q8-q11.
//
//                   +--------+
//                   | v2[0-7]|
//              Rhs  +--------+
//                   | v5[0-7]|
//                   +--------+
//
//                   |        |
//
//    Lhs            |        |
//
//  +--+--+ - - - -  +--------+
//  |v0|v1|          | v8[0-7]|
//  |v0|v1|          | v9[0-7]|
//  |v0|v1|          |v10[0-7]|
//  |v0|v1|          |v11[0-7]|
//  +--+--+ - - - -  +--------+
//
//                  Accumulator
void aarch64_hgemm_assembly_kernel_8x4(const __fp16* a_ptr,
                                       const __fp16*& b_ptr, int K,
                                       __fp16* outptr0, int ldout, int y_remain,
                                       int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b0a asm("v5");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;

// clang-format off
#define LOAD_LINE(v1, n)       \
    "cbz w0, LOAD_8x4_C_END\n" \
    "ldr q" v1 ", [%[outptr" n \
    "]]\n"                     \
    "subs w0, w0, #1\n"

#define LOAD_C               \
    "mov w0, %w[y_remain]\n" \
    LOAD_LINE("8", "0")      \
    LOAD_LINE("9", "1")      \
    LOAD_LINE("10", "2")     \
    LOAD_LINE("11", "3")     \
    "LOAD_8x4_C_END:\n"

#define STORE_LINE(v1, n)       \
    "cbz w0, STORE_8x4_C_END\n" \
    "str q" v1 ", [%[outptr" n  \
    "]]\n"                      \
    "subs w0, w0, #1\n"

#define STORE_C              \
    "mov w0, %w[y_remain]\n" \
    STORE_LINE("8", "0")     \
    STORE_LINE("9", "1")     \
    STORE_LINE("10", "2")    \
    STORE_LINE("11", "3")    \
    "STORE_8x4_C_END:\n"
// clang-format on

            asm volatile(
                    ".arch armv8.2-a+fp16\n"

                    // load accumulator C
                    "cmp %w[type], #0\n"
                    "beq 5f\n" LOAD_C
                    "b 6f\n"
                    "5:\n"
                    "eor v8.16b,  v8.16b,  v8.16b\n"
                    "eor v9.16b,  v9.16b,  v9.16b\n"
                    "eor v10.16b, v10.16b, v10.16b\n"
                    "eor v11.16b, v11.16b, v11.16b\n"

                    "6:\n"
                    "ldr %d[a0], [%[a_ptr]]\n"
                    "ldr %q[b0], [%[b_ptr]]\n"
                    "ldr %q[b0a], [%[b_ptr], #16]\n"

                    "cbz %w[k], 4f\n"

                    "1:\n"
                    "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"
                    "ldr %q[b0], [%[b_ptr], #32]\n"

                    "add %[b_ptr], %[b_ptr], #32\n"
                    "ldr %d[a0], [%[a_ptr], #16]\n"

                    "fmla v8.8h , %[b0a].8h, %[a0a].h[0]\n"
                    "fmla v9.8h , %[b0a].8h, %[a0a].h[1]\n"
                    "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                    "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"
                    "ldr %q[b0a], [%[b_ptr], #16]\n"

                    "add %[a_ptr], %[a_ptr], #16\n"
                    "subs %w[k], %w[k], #1\n"

                    "bne 1b\n"
                    "4:\n"
                    // Jump to odd tail if necessary.
                    "cbnz %w[oddk], 2f\n"

                    // Even tail
                    "fmla v8.8h , %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h , %[b0].8h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"

                    "add %[b_ptr], %[b_ptr], #32\n"
                    "add %[a_ptr], %[a_ptr], #16\n"

                    "fmla v8.8h, %[b0a].8h, %[a0a].h[0]\n"
                    "fmla v9.8h, %[b0a].8h, %[a0a].h[1]\n"
                    "fmla v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                    "fmla v11.8h, %[b0a].8h, %[a0a].h[3]\n"

                    "b 3f\n"

                    // Odd tail
                    "2:\n"
                    "add %[a_ptr], %[a_ptr], #8\n"
                    "add %[b_ptr], %[b_ptr], #16\n"

                    "fmla v8.8h, %[b0].8h, %[a0].h[0]\n"
                    "fmla v9.8h, %[b0].8h, %[a0].h[1]\n"
                    "fmla v10.8h, %[b0].8h, %[a0].h[2]\n"
                    "fmla v11.8h, %[b0].8h, %[a0].h[3]\n"

                    "3:\n" STORE_C
                    : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0),
                      [k] "+r"(k), [b0a] "+w"(b0a), [a_ptr] "+r"(a_ptr),
                      [b_ptr] "+r"(b_ptr), [outptr0] "+r"(outptr0),
                      [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2),
                      [outptr3] "+r"(outptr3)
                    :
                    [oddk] "r"(oddk), [y_remain] "r"(y_remain), [type] "r"(type)
                    : "w0", "v8", "v9", "v10", "v11", "cc", "memory");
#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

// Overview of register layout:
//
// A 2x8 cell of Rhs is stored in 16bit in d2, d5
// A 4x2 cell of Lhs is stored in 16bit in d0, d1
// A 4x8 block of accumulators is stored in 16bit in d8-d11.
//
//                   +--------+
//                   | d2[0-3]|
//              Rhs  +--------+
//                   | d5[0-3]|
//                   +--------+
//
//                   |        |
//
//    Lhs            |        |
//
//  +--+--+ - - - -  +--------+
//  |d0|d1|          | d8[0-3]|
//  |d0|d1|          | d9[0-3]|
//  |d0|d1|          |d10[0-3]|
//  |d0|d1|          |d11[0-3]|
//  +--+--+ - - - -  +--------+
//
//                  Accumulator
void aarch64_hgemm_assembly_kernel_4x4(const __fp16* a_ptr,
                                       const __fp16*& b_ptr, int K,
                                       __fp16* outptr0, int ldout, int x_remain,
                                       int y_remain, int type) {
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register float16x8_t a0 asm("v0");
    register float16x8_t a0a asm("v1");
    register float16x8_t b0 asm("v2");
    register float16x8_t b0a asm("v5");

    __fp16* outptr1 = outptr0 + ldout;
    __fp16* outptr2 = outptr1 + ldout;
    __fp16* outptr3 = outptr2 + ldout;

#define LOAD_LINE(reg_index, n)                \
    "cbz w1, LOAD_4x4_C_END\n"                 \
    "mov x0, %[outptr" n                       \
    "]\n"                                      \
    "cmp %w[x_remain], #4\n"                   \
    "b.lt REMAIN_LOAD_4x4_LINE_LESS_THAN_4_" n \
    "\n"                                       \
    "ldr d" reg_index                          \
    ", [x0]\n"                                 \
    "b LOAD_4x4_LINE_END_" n                   \
    "\n"                                       \
                                               \
    "REMAIN_LOAD_4x4_LINE_LESS_THAN_4_" n      \
    ":\n"                                      \
    "cmp %w[x_remain], #0\n"                   \
    "beq LOAD_4x4_LINE_END_" n                 \
    "\n"                                       \
    "ld1 {v" reg_index                         \
    ".h}[0], [x0], #2\n"                       \
    "cmp %w[x_remain], #1\n"                   \
    "beq LOAD_4x4_LINE_END_" n                 \
    "\n"                                       \
    "ld1 {v" reg_index                         \
    ".h}[1], [x0], #2\n"                       \
    "cmp %w[x_remain], #2\n"                   \
    "beq LOAD_4x4_LINE_END_" n                 \
    "\n"                                       \
    "ld1 {v" reg_index                         \
    ".h}[2], [x0], #2\n"                       \
    "LOAD_4x4_LINE_END_" n                     \
    ":\n"                                      \
    "subs w1, w1, #1\n"

#define LOAD_C               \
    "mov w1, %w[y_remain]\n" \
    LOAD_LINE("8", "0")      \
    LOAD_LINE("9", "1")      \
    LOAD_LINE("10", "2")     \
    LOAD_LINE("11", "3")     \
    "LOAD_4x4_C_END:\n"

#define STORE_LINE(reg_index, n)                \
    "cbz w1, STORE_4x4_C_END\n"                 \
    "mov x0, %[outptr" n                        \
    "]\n"                                       \
    "cmp %w[x_remain], #4\n"                    \
    "b.lt REMAIN_STORE_4x4_LINE_LESS_THAN_4_" n \
    "\n"                                        \
    "str d" reg_index                           \
    ", [x0]\n"                                  \
    "b STORE_4x4_LINE_END_" n                   \
    "\n"                                        \
                                                \
    "REMAIN_STORE_4x4_LINE_LESS_THAN_4_" n      \
    ":\n"                                       \
    "cmp %w[x_remain], #0\n"                    \
    "beq STORE_4x4_LINE_END_" n                 \
    "\n"                                        \
    "st1 {v" reg_index                          \
    ".h}[0], [x0], #2\n"                        \
    "cmp %w[x_remain], #1\n"                    \
    "beq STORE_4x4_LINE_END_" n                 \
    "\n"                                        \
    "st1 {v" reg_index                          \
    ".h}[1], [x0], #2\n"                        \
    "cmp %w[x_remain], #2\n"                    \
    "beq STORE_4x4_LINE_END_" n                 \
    "\n"                                        \
    "st1 {v" reg_index                          \
    ".h}[2], [x0], #2\n"                        \
    "STORE_4x4_LINE_END_" n                     \
    ":\n"                                       \
    "subs w1, w1, #1\n"

#define STORE_C "mov w1, %w[y_remain]\n" \
            STORE_LINE("8", "0")         \
            STORE_LINE("9", "1")         \
            STORE_LINE("10", "2")        \
            STORE_LINE("11", "3")        \
            "STORE_4x4_C_END:\n"

            asm volatile(
                    ".arch armv8.2-a+fp16\n"

                    // load accumulator C
                    "cmp %w[type], #0\n"
                    "beq 5f\n" LOAD_C
                    "b 6f\n"

                    "5:\n"
                    "eor v8.8b,  v8.8b,  v8.8b\n"
                    "eor v9.8b,  v9.8b,  v9.8b\n"
                    "eor v10.8b, v10.8b, v10.8b\n"
                    "eor v11.8b, v11.8b, v11.8b\n"

                    "6:\n"
                    "ldr %d[a0], [%[a_ptr]]\n"

                    "cbz %w[k], 4f\n"

                    "1:\n"
                    "ldp %d[b0], %d[b0a], [%[b_ptr]]\n"
                    "fmla v8.4h , %[b0].4h, %[a0].h[0]\n"
                    "fmla v9.4h , %[b0].4h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.4h, %[b0].4h, %[a0].h[2]\n"
                    "fmla v11.4h, %[b0].4h, %[a0].h[3]\n"

                    "add %[b_ptr], %[b_ptr], #16\n"
                    "ldr %d[a0], [%[a_ptr], #16]\n"

                    "fmla v8.4h , %[b0a].4h, %[a0a].h[0]\n"
                    "fmla v9.4h , %[b0a].4h, %[a0a].h[1]\n"
                    "fmla v10.4h, %[b0a].4h, %[a0a].h[2]\n"
                    "fmla v11.4h, %[b0a].4h, %[a0a].h[3]\n"

                    "add %[a_ptr], %[a_ptr], #16\n"
                    "subs %w[k], %w[k], #1\n"

                    "bne 1b\n"
                    "4:\n"
                    // Jump to odd tail if necessary.
                    "cbnz %w[oddk], 2f\n"

                    // Even tail
                    "ldp %d[b0], %d[b0a], [%[b_ptr]]\n"
                    "fmla v8.4h , %[b0].4h, %[a0].h[0]\n"
                    "fmla v9.4h , %[b0].4h, %[a0].h[1]\n"
                    "ldr %d[a0a], [%[a_ptr], #8]\n"
                    "fmla v10.4h, %[b0].4h, %[a0].h[2]\n"
                    "fmla v11.4h, %[b0].4h, %[a0].h[3]\n"

                    "add %[b_ptr], %[b_ptr], #16\n"
                    "add %[a_ptr], %[a_ptr], #16\n"

                    "fmla v8.4h, %[b0a].4h, %[a0a].h[0]\n"
                    "fmla v9.4h, %[b0a].4h, %[a0a].h[1]\n"
                    "fmla v10.4h, %[b0a].4h, %[a0a].h[2]\n"
                    "fmla v11.4h, %[b0a].4h, %[a0a].h[3]\n"
                    "b 3f\n"

                    // Odd tail
                    "2:\n"
                    "ldr %d[b0], [%[b_ptr]]\n"
                    "add %[a_ptr], %[a_ptr], #8\n"
                    "add %[b_ptr], %[b_ptr], #8\n"

                    "fmla v8.4h, %[b0].4h, %[a0].h[0]\n"
                    "fmla v9.4h, %[b0].4h, %[a0].h[1]\n"
                    "fmla v10.4h, %[b0].4h, %[a0].h[2]\n"
                    "fmla v11.4h, %[b0].4h, %[a0].h[3]\n"

                    "3:\n" STORE_C
                    : [a0] "+w"(a0), [a0a] "+w"(a0a), [b0] "+w"(b0),
                      [k] "+r"(k), [b0a] "+w"(b0a), [a_ptr] "+r"(a_ptr),
                      [b_ptr] "+r"(b_ptr), [outptr0] "+r"(outptr0),
                      [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2),
                      [outptr3] "+r"(outptr3)
                    : [oddk] "r"(oddk), [x_remain] "r"(x_remain),
                      [y_remain] "r"(y_remain), [type] "r"(type)
                    : "x0", "w1", "v8", "v9", "v10", "v11", "cc", "memory");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

void aarch64_hgemm_asimd_8x24(const __fp16* Apanel, const __fp16* Bpanel,
                              __fp16* out, int ldout, int x0, int xmax, int y0,
                              int ymax, int K, bool is_first_k) {
    const __fp16* a_ptr = Apanel;
    const int A_interleave = 8;
    const int B_transpose1xW = 24;
    const int K8 = (K << 3);
    const int K4 = (K << 2);
    int type = is_first_k ? 0 : 1;

    int y = y0;
    for (; y + A_interleave <= ymax; y += A_interleave) {
        const __fp16* a_ptr0 = a_ptr;
        const __fp16* b_ptr = Bpanel;

        __fp16* outptr0 = out + (y * ldout) + x0;

        int x = x0;

        for (; x + B_transpose1xW <= xmax; x += B_transpose1xW) {
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_24x8(a_ptr, b_ptr, K, outptr0, ldout,
                                               type);
            outptr0 += B_transpose1xW;
        }

        for (; x + 16 <= xmax; x += 16) {
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_16x8(a_ptr, b_ptr, K, outptr0, ldout,
                                               type);
            outptr0 += 16;
        }
        for (; x + 8 <= xmax; x += 8) {
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_8x8(a_ptr, b_ptr, K, outptr0, ldout,
                                              type);
            outptr0 += 8;
        }
        for (; x < xmax; x += 4) {
            int x_remain = xmax - x;
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_4x8(a_ptr, b_ptr, K, outptr0, ldout,
                                              x_remain, type);
            outptr0 += 4;
        }
        a_ptr = a_ptr0 + K8;
    }

    for (; y < ymax; y += 4) {
        const __fp16* a_ptr0 = a_ptr;
        const __fp16* b_ptr = Bpanel;

        __fp16* outptr0 = out + (y * ldout) + x0;

        int x = x0;
        for (; x + B_transpose1xW <= xmax; x += B_transpose1xW) {
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_24x4(a_ptr, b_ptr, K, outptr0, ldout,
                                               ymax - y, type);
            outptr0 += B_transpose1xW;
        }

        for (; x + 16 <= xmax; x += 16) {
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_16x4(a_ptr, b_ptr, K, outptr0, ldout,
                                               ymax - y, type);
            outptr0 += 16;
        }
        for (; x + 8 <= xmax; x += 8) {
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_8x4(a_ptr, b_ptr, K, outptr0, ldout,
                                              ymax - y, type);
            outptr0 += 8;
        }
        for (; x < xmax; x += 4) {
            a_ptr = a_ptr0;
            aarch64_hgemm_assembly_kernel_4x4(a_ptr, b_ptr, K, outptr0, ldout,
                                              xmax - x, ymax - y, type);
            outptr0 += 4;
        }
        a_ptr = a_ptr0 + K4;
    }
}
}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL(hgemm_8x24);

void hgemm_8x24::pack_A(dt_float16* out, const dt_float16* in, int ldin, int y0,
                        int ymax, int k0, int kmax, bool transpose_A) const {
    if (transpose_A) {
        transpose_1x8(reinterpret_cast<__fp16*>(out),
                      reinterpret_cast<const __fp16*>(in), ldin, y0, ymax, k0,
                      kmax);
    } else {
        interleave_8x1(reinterpret_cast<__fp16*>(out),
                       reinterpret_cast<const __fp16*>(in), ldin, y0, ymax, k0,
                       kmax);
    }
}

void hgemm_8x24::pack_B(dt_float16* out, const dt_float16* in, int ldin, int x0,
                        int xmax, int k0, int kmax, bool transpose_B) const {
    if (transpose_B) {
        interleave_24x1(reinterpret_cast<__fp16*>(out),
                        reinterpret_cast<const __fp16*>(in), ldin, x0, xmax, k0,
                        kmax);
    } else {
        transpose_1x24(reinterpret_cast<__fp16*>(out),
                       reinterpret_cast<const __fp16*>(in), ldin, x0, xmax, k0,
                       kmax);
    }
}

void hgemm_8x24::kern(const dt_float16* packA, const dt_float16* packB,
                      size_t M, size_t N, size_t K, dt_float16* C, size_t LDC,
                      bool is_first_k, const dt_float16*, dt_float16*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == C_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Float16);
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);
    aarch64_hgemm_asimd_8x24(reinterpret_cast<const __fp16*>(packA),
                             reinterpret_cast<const __fp16*>(packB),
                             reinterpret_cast<__fp16*>(C), LDC, 0, N, 0, M, K,
                             is_first_k);
}
#endif
// vim: syntax=cpp.doxygen
