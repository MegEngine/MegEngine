/**
 * \file dnn/src/aarch64/matrix_mul/int8x8x16/kernel_4x4x16.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <inttypes.h>
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_4x4x16 {

/**
 * Overview of register layout:
 *
 *                     +---------+---------+---------+---------+
 *                     |v20[0-15]|v21[0-15]|v22[0-15]|v23[0-15]|
 *                Rhs  +---------+---------+---------+---------+
 *    Lhs              |         |         |
 *
 *  +--------+ - - - - +---------+---------+---------+---------+
 *  |v0[0-15]|         | v4[0-8] |  v8[0-8]| v12[0-8]| v16[0-8]|
 *  |v1[0-15]|         | v5[0-8] |  v9[0-8]| v13[0-8]| v17[0-8]|
 *  |v2[0-15]|         | v6[0-8] | v10[0-8]| v14[0-8]| v18[0-8]|
 *  |v3[0-15]|         | v7[0-8] | v11[0-8]| v15[0-8]| v19[0-8]|
 *  +--------+ - - - - +---------+---------+---------+---------+
 *
 *                            Accumulator
 */
static void kern_4x4(const int8_t* packA, const int8_t* packB, int K,
                     int16_t* output, int LDC, bool is_first_k, int m_remain,
                     int n_remain) {
    K /= 16;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int16_t);
// clang-format off
#define LOAD_LINE(reg_index, n)                    \
    "cmp x5, #0 \n"                                \
    "beq 105f\n"                                   \
    "cmp %w[n_remain], #4\n"                       \
    "blt 100" n "f\n"                              \
    "ld1 {v" reg_index ".4h}, [x" n "], #8\n"      \
    "b 101" n "f\n"                                \
    "100" n ":\n"                                  \
    "cmp %w[n_remain], #0\n"                       \
    "blt 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[0], [x" n "], #2\n"    \
    "cmp %w[n_remain], #1\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[1], [x" n "], #2\n"    \
    "cmp %w[n_remain], #2\n"                       \
    "beq 101" n "f\n"                              \
    "ld1 {v" reg_index ".h}[2], [x" n "], #2\n"    \
    "101" n ":\n"                                  \
    "sub x5, x5, #1\n"

#define LOAD_C                     \
    "mov x5, %x[m_remain]\n"       \
    LOAD_LINE("24", "0")           \
    LOAD_LINE("25", "1")           \
    LOAD_LINE("26", "2")           \
    LOAD_LINE("27", "3")           \
    "105:\n"


#define STORE_LINE(reg_index, n)                \
    "cmp x5, #0 \n"                             \
    "beq 105f\n"                                \
    "cmp %w[n_remain], #4\n"                    \
    "blt 102" n "f\n"                           \
    "st1 {v" reg_index ".4h}, [x" n "], #8\n"   \
    "b 103" n "f\n"                             \
    "102" n ":\n"                               \
    "cmp %w[n_remain], #0\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[0], [x" n "], #2\n" \
    "cmp %w[n_remain], #1\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[1], [x" n "], #2\n" \
    "cmp %w[n_remain], #2\n"                    \
    "beq 103" n "f\n"                           \
    "st1 {v" reg_index ".h}[2], [x" n "], #2\n" \
    "103" n ":\n"                               \
    "sub x5, x5, #1\n"

#define STORE_C                     \
    "mov x5, %x[m_remain]\n"        \
    STORE_LINE("24", "0")           \
    STORE_LINE("25", "1")           \
    STORE_LINE("26", "2")           \
    STORE_LINE("27", "3")           \
    "105:\n"
    // clang-format on

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            "add x1, x0, %x[LDC]\n"
            "add x2, x1, %x[LDC]\n"
            "add x3, x2, %x[LDC]\n"

            // Clear accumulators
            "eor  v4.16b,  v4.16b,  v4.16b\n"
            "eor  v5.16b,  v5.16b,  v5.16b\n"
            "eor  v6.16b,  v6.16b,  v6.16b\n"
            "eor  v7.16b,  v7.16b,  v7.16b\n"
            "eor  v8.16b,  v8.16b,  v8.16b\n"
            "eor  v9.16b,  v9.16b,  v9.16b\n"
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

            // General loop.
            "1:\n"
            "ld1 {v20.16b}, [%[b_ptr]], 16\n"
            "ld1 {v0.16b}, [%[a_ptr]], 16\n"
            "ld1 {v1.16b}, [%[a_ptr]], 16\n"
            "ld1 {v2.16b}, [%[a_ptr]], 16\n"
            "ld1 {v3.16b}, [%[a_ptr]], 16\n"

            "ld1 {v21.16b}, [%[b_ptr]], 16\n"
            "smlal     v4.8h,  v0.8b,  v20.8b\n"
            "smlal     v5.8h,  v1.8b,  v20.8b\n"
            "smlal     v6.8h,  v2.8b,  v20.8b\n"
            "smlal     v7.8h,  v3.8b,  v20.8b\n"
            "smlal2    v4.8h,  v0.16b,  v20.16b\n"
            "smlal2    v5.8h,  v1.16b,  v20.16b\n"
            "smlal2    v6.8h,  v2.16b,  v20.16b\n"
            "smlal2    v7.8h,  v3.16b,  v20.16b\n"

            "ld1 {v22.16b}, [%[b_ptr]], 16\n"
            "smlal     v8.8h,  v0.8b,  v21.8b\n"
            "smlal     v9.8h,  v1.8b,  v21.8b\n"
            "smlal    v10.8h,  v2.8b,  v21.8b\n"
            "smlal    v11.8h,  v3.8b,  v21.8b\n"
            "smlal2    v8.8h,  v0.16b,  v21.16b\n"
            "smlal2    v9.8h,  v1.16b,  v21.16b\n"
            "smlal2   v10.8h,  v2.16b,  v21.16b\n"
            "smlal2   v11.8h,  v3.16b,  v21.16b\n"

            "ld1 {v23.16b}, [%[b_ptr]], 16\n"
            "smlal     v12.8h,  v0.8b,  v22.8b\n"
            "smlal     v13.8h,  v1.8b,  v22.8b\n"
            "smlal     v14.8h,  v2.8b,  v22.8b\n"
            "smlal     v15.8h,  v3.8b,  v22.8b\n"
            "smlal2    v12.8h,  v0.16b,  v22.16b\n"
            "smlal2    v13.8h,  v1.16b,  v22.16b\n"
            "smlal2    v14.8h,  v2.16b,  v22.16b\n"
            "smlal2    v15.8h,  v3.16b,  v22.16b\n"

            "smlal    v16.8h,  v0.8b,  v23.8b\n"
            "smlal    v17.8h,  v1.8b,  v23.8b\n"
            "smlal    v18.8h,  v2.8b,  v23.8b\n"
            "smlal    v19.8h,  v3.8b,  v23.8b\n"
            "smlal2   v16.8h,  v0.16b,  v23.16b\n"
            "smlal2   v17.8h,  v1.16b,  v23.16b\n"
            "smlal2   v18.8h,  v2.16b,  v23.16b\n"
            "smlal2   v19.8h,  v3.16b,  v23.16b\n"

            "subs %w[K], %w[K], #1\n"
            "cbnz %w[K], 1b\n"

            "cmp %w[is_first_k], #1\n"
            "beq 2f\n" LOAD_C
            "b 3f\n"

            "2:\n"  // Clear the C regs.
            "eor v24.16b, v24.16b, v24.16b\n"
            "eor v25.16b, v25.16b, v25.16b\n"
            "eor v26.16b, v26.16b, v26.16b\n"
            "eor v27.16b, v27.16b, v27.16b\n"

            "3:\n"
            // Reduce v4-v19 to v0-v3
            "addv h20, v4.8h\n"
            "addv h21, v8.8h\n"
            "addv h22, v12.8h\n"
            "addv h23, v16.8h\n"
            "ins v0.h[0], v20.h[0]\n"
            "ins v0.h[1], v21.h[0]\n"
            "ins v0.h[2], v22.h[0]\n"
            "ins v0.h[3], v23.h[0]\n"
            "add v24.4h, v24.4h, v0.4h\n"

            "addv h28, v5.8h\n"
            "addv h29, v9.8h\n"
            "addv h30, v13.8h\n"
            "addv h31, v17.8h\n"
            "ins v1.h[0], v28.h[0]\n"
            "ins v1.h[1], v29.h[0]\n"
            "ins v1.h[2], v30.h[0]\n"
            "ins v1.h[3], v31.h[0]\n"
            "add v25.4h, v25.4h, v1.4h\n"

            "addv h20, v6.8h\n"
            "addv h21, v10.8h\n"
            "addv h22, v14.8h\n"
            "addv h23, v18.8h\n"
            "ins v2.h[0], v20.h[0]\n"
            "ins v2.h[1], v21.h[0]\n"
            "ins v2.h[2], v22.h[0]\n"
            "ins v2.h[3], v23.h[0]\n"
            "add v26.4h, v26.4h, v2.4h\n"

            "addv h28, v7.8h\n"
            "addv h29, v11.8h\n"
            "addv h30, v15.8h\n"
            "addv h31, v19.8h\n"
            "ins v3.h[0], v28.h[0]\n"
            "ins v3.h[1], v29.h[0]\n"
            "ins v3.h[2], v30.h[0]\n"
            "ins v3.h[3], v31.h[0]\n"
            "add v27.4h, v27.4h, v3.4h\n"

            // Store back into memory
            STORE_C

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr),
              [is_first_k] "+r"(is_first_k), [K] "+r"(K), [LDC] "+r"(LDC),
              [outptr] "+r"(outptr), [m_remain] "+r"(m_remain),
              [n_remain] "+r"(n_remain)
            :
            : "cc", "memory", "x1", "x2", "x3", "x4", "x5", "v0", "v1", "v2",
              "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
              "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
              "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
              "v31");

#undef LOAD_LINE
#undef LOAD_C
#undef STORE_LINE
#undef STORE_C
}

static void gemm_s8x8x16_4x4_pack_A_n(dt_int8* outptr, const dt_int8* inptr,
                                      int ldin, int y0, int ymax, int k0,
                                      int kmax) {
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
                        inptr1 = zerobuff; MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff; MEGDNN_FALLTHRU
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
                        inptr1 = zerobuff; MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff; MEGDNN_FALLTHRU
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

static void gemm_s8x8x16_4x4_pack_B_n(dt_int8* out, const dt_int8* in, int ldin,
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

            prefetch_2x(inptr0);
            prefetch_2x(inptr1);
            prefetch_2x(inptr2);
            prefetch_2x(inptr3);
            prefetch_2x(inptr4);
            prefetch_2x(inptr5);
            prefetch_2x(inptr6);
            prefetch_2x(inptr7);

            int8_t* outptr_inner = outptr + ki - k;

            int remain = std::min(ki + 7 - kmax, 7);
            int x = x0;
            for (; x + 3 < xmax; x += 4) {
                if (remain >= 0) {
                    switch (remain) {
                        case 7:
                            inptr0 = zerobuff; MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff; MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff; MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff; MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff; MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff; MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff; MEGDNN_FALLTHRU
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
                            inptr0 = zerobuff; MEGDNN_FALLTHRU
                        case 6:
                            inptr1 = zerobuff; MEGDNN_FALLTHRU
                        case 5:
                            inptr2 = zerobuff; MEGDNN_FALLTHRU
                        case 4:
                            inptr3 = zerobuff; MEGDNN_FALLTHRU
                        case 3:
                            inptr4 = zerobuff; MEGDNN_FALLTHRU
                        case 2:
                            inptr5 = zerobuff; MEGDNN_FALLTHRU
                        case 1:
                            inptr6 = zerobuff; MEGDNN_FALLTHRU
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

}  // namespace matmul_4x4x16
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
