/**
 * \file dnn/src/aarch64/matrix_mul/int8x8x16/kernel_mk4_4x4x8_a72.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <inttypes.h>
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace aarch64 {
namespace matmul_mk4_4x4x8_a72 {

//! optimize for A72

// clang-format off
/**
 * Overview of register layout:
 *
 * A 4x4x8 cell of Lhs is stored in 8bit in q0-q3, q4-q7
 * A 4x4x8 cell of Rhs is stored in 8bit in q8-q11, q12-q15
 * A 4x4 block of accumulators is stored in 16bit in q16-q31
 *
 *                     +------------------------+
 *                     |  q8  |  q9 | q10 | q11 |
 *                Rhs  +------------------------+
 *    Lhs              |      |     |     |     |
 *  +--------+ - - - - +------------------------+
 *  | q0 |             |  q16 | q20 | q24 | q28 |
 *  | q1 |             |  q17 | q21 | q25 | q29 |
 *  | q2 |             |  q18 | q22 | q26 | q30 |
 *  | q3 |             |  q19 | q23 | q27 | q31 |
 *  +--------+ - - - - +------------------------+
 *
 *                            Accumulator
 */

// clang-format on
static inline void kern_4x4(const int8_t* packA, const int8_t* packB, int K,
                            int16_t* output, int LDC, bool, int remain_n) {
    K = div_ceil(K, 8);
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;

    LDC = LDC * sizeof(int8_t);
// clang-format off
    #define STORE_LINE(reg0)                 \
    "cmp w10, #0 \n"                         \
    "beq 101f\n"                             \
    "st1 {v" reg0 ".4h}, [x0], #8\n"       \
    "subs w10, w10, #1\n"

    #define STORE_C                \
        "mov w10, %w[remain_n]\n"  \
        STORE_LINE("16")           \
        STORE_LINE("20")           \
        STORE_LINE("24")           \
        STORE_LINE("28")

    // clang-format on

    register int16_t* outptr asm("x0") = output;
    asm volatile(
            // load accumulator C

            "1:\n"

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

            "2: \n"

            "ld1 {v0.8b, v1.8b}, [%[a_ptr]], #16\n"
            "ld1 {v2.8b, v3.8b}, [%[a_ptr]], #16\n"
            "ld1 {v8.8b, v9.8b}, [%[b_ptr]], #16\n"
            "ld1 {v10.8b, v11.8b}, [%[b_ptr]], #16\n"

            "cmp %w[K], #0\n"
            "beq 4f\n"

            "3: \n"
            //! k = 0
            "smlal v16.8h, v0.8b, v8.8b\n"
            "ld1 {v4.8b}, [%[a_ptr]], #8\n"
            "smlal v17.8h, v1.8b, v8.8b\n"
            "smlal v18.8h, v2.8b, v8.8b\n"
            "ld1 {v5.8b}, [%[a_ptr]], #8\n"
            "smlal v19.8h, v3.8b, v8.8b\n"
            "smlal v20.8h, v0.8b, v9.8b\n"
            "ld1 {v6.8b}, [%[a_ptr]], #8\n"
            "smlal v21.8h, v1.8b, v9.8b\n"
            "smlal v22.8h, v2.8b, v9.8b\n"
            "ld1 {v7.8b}, [%[a_ptr]], #8\n"
            "smlal v23.8h, v3.8b, v9.8b\n"
            "smlal v24.8h, v0.8b, v10.8b\n"
            "ld1 {v12.8b}, [%[b_ptr]], #8\n"
            "smlal v25.8h, v1.8b, v10.8b\n"
            "smlal v26.8h, v2.8b, v10.8b\n"
            "ld1 {v13.8b}, [%[b_ptr]], #8\n"
            "smlal v27.8h, v3.8b, v10.8b\n"
            "smlal v28.8h, v0.8b, v11.8b\n"
            "ld1 {v14.8b}, [%[b_ptr]], #8\n"
            "smlal v29.8h, v1.8b, v11.8b\n"
            "smlal v30.8h, v2.8b, v11.8b\n"
            "ld1 {v15.8b}, [%[b_ptr]], #8\n"
            "smlal v31.8h, v3.8b, v11.8b\n"
            //! k = 8
            "smlal v16.8h, v4.8b, v12.8b\n"
            "ld1 {v0.8b}, [%[a_ptr]], #8\n"
            "smlal v17.8h, v5.8b, v12.8b\n"
            "smlal v18.8h, v6.8b, v12.8b\n"
            "ld1 {v1.8b}, [%[a_ptr]], #8\n"
            "smlal v19.8h, v7.8b, v12.8b\n"
            "smlal v20.8h, v4.8b, v13.8b\n"
            "ld1 {v2.8b}, [%[a_ptr]], #8\n"
            "smlal v21.8h, v5.8b, v13.8b\n"
            "smlal v22.8h, v6.8b, v13.8b\n"
            "ld1 {v3.8b}, [%[a_ptr]], #8\n"
            "smlal v23.8h, v7.8b, v13.8b\n"
            "smlal v24.8h, v4.8b, v14.8b\n"
            "ld1 {v8.8b}, [%[b_ptr]], #8\n"
            "smlal v25.8h, v5.8b, v14.8b\n"
            "smlal v26.8h, v6.8b, v14.8b\n"
            "ld1 {v9.8b}, [%[b_ptr]], #8\n"
            "smlal v27.8h, v7.8b, v14.8b\n"
            "smlal v28.8h, v4.8b, v15.8b\n"
            "ld1 {v10.8b}, [%[b_ptr]], #8\n"
            "smlal v29.8h, v5.8b, v15.8b\n"
            "smlal v30.8h, v6.8b, v15.8b\n"
            "ld1 {v11.8b}, [%[b_ptr]], #8\n"
            "smlal v31.8h, v7.8b, v15.8b\n"

            "subs %w[K], %w[K], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"
            //! even tail
            //! k = 0
            "smlal v16.8h, v0.8b, v8.8b\n"
            "ld1 {v4.8b}, [%[a_ptr]], #8\n"
            "smlal v17.8h, v1.8b, v8.8b\n"
            "smlal v18.8h, v2.8b, v8.8b\n"
            "ld1 {v5.8b}, [%[a_ptr]], #8\n"
            "smlal v19.8h, v3.8b, v8.8b\n"
            "smlal v20.8h, v0.8b, v9.8b\n"
            "ld1 {v6.8b}, [%[a_ptr]], #8\n"
            "smlal v21.8h, v1.8b, v9.8b\n"
            "smlal v22.8h, v2.8b, v9.8b\n"
            "ld1 {v7.8b}, [%[a_ptr]], #8\n"
            "smlal v23.8h, v3.8b, v9.8b\n"
            "smlal v24.8h, v0.8b, v10.8b\n"
            "ld1 {v12.8b}, [%[b_ptr]], #8\n"
            "smlal v25.8h, v1.8b, v10.8b\n"
            "smlal v26.8h, v2.8b, v10.8b\n"
            "ld1 {v13.8b}, [%[b_ptr]], #8\n"
            "smlal v27.8h, v3.8b, v10.8b\n"
            "smlal v28.8h, v0.8b, v11.8b\n"
            "ld1 {v14.8b}, [%[b_ptr]], #8\n"
            "smlal v29.8h, v1.8b, v11.8b\n"
            "smlal v30.8h, v2.8b, v11.8b\n"
            "ld1 {v15.8b}, [%[b_ptr]], #8\n"
            "smlal v31.8h, v3.8b, v11.8b\n"
            //! k = 8
            "smlal v16.8h, v4.8b, v12.8b\n"
            "smlal v17.8h, v5.8b, v12.8b\n"
            "smlal v18.8h, v6.8b, v12.8b\n"
            "smlal v19.8h, v7.8b, v12.8b\n"
            "smlal v20.8h, v4.8b, v13.8b\n"
            "smlal v21.8h, v5.8b, v13.8b\n"
            "smlal v22.8h, v6.8b, v13.8b\n"
            "smlal v23.8h, v7.8b, v13.8b\n"
            "smlal v24.8h, v4.8b, v14.8b\n"
            "smlal v25.8h, v5.8b, v14.8b\n"
            "smlal v26.8h, v6.8b, v14.8b\n"
            "smlal v27.8h, v7.8b, v14.8b\n"
            "smlal v28.8h, v4.8b, v15.8b\n"
            "smlal v29.8h, v5.8b, v15.8b\n"
            "smlal v30.8h, v6.8b, v15.8b\n"
            "smlal v31.8h, v7.8b, v15.8b\n"
            "b 6f\n"

            "5:\n"
            //! odd tail
            "smlal v16.8h, v0.8b, v8.8b\n"
            "smlal v17.8h, v1.8b, v8.8b\n"
            "smlal v18.8h, v2.8b, v8.8b\n"
            "smlal v19.8h, v3.8b, v8.8b\n"
            "smlal v20.8h, v0.8b, v9.8b\n"
            "smlal v21.8h, v1.8b, v9.8b\n"
            "smlal v22.8h, v2.8b, v9.8b\n"
            "smlal v23.8h, v3.8b, v9.8b\n"
            "smlal v24.8h, v0.8b, v10.8b\n"
            "smlal v25.8h, v1.8b, v10.8b\n"
            "smlal v26.8h, v2.8b, v10.8b\n"
            "smlal v27.8h, v3.8b, v10.8b\n"
            "smlal v28.8h, v0.8b, v11.8b\n"
            "smlal v29.8h, v1.8b, v11.8b\n"
            "smlal v30.8h, v2.8b, v11.8b\n"
            "smlal v31.8h, v3.8b, v11.8b\n"

            "6:\n"
            //! reduece
            "addp v16.8h, v16.8h, v17.8h\n"
            "addp v18.8h, v18.8h, v19.8h\n"
            "addp v20.8h, v20.8h, v21.8h\n"
            "addp v22.8h, v22.8h, v23.8h\n"
            "addp v24.8h, v24.8h, v25.8h\n"
            "addp v26.8h, v26.8h, v27.8h\n"

            "addp v16.8h, v16.8h, v18.8h\n"
            "addp v28.8h, v28.8h, v29.8h\n"
            "addp v30.8h, v30.8h, v31.8h\n"
            "addp v20.8h, v20.8h, v22.8h\n"

            "addp v16.8h, v16.8h, v16.8h\n"
            "addp v20.8h, v20.8h, v20.8h\n"

            "addp v24.8h, v24.8h, v26.8h\n"
            "addp v24.8h, v24.8h, v24.8h\n"

            "addp v28.8h, v28.8h, v30.8h\n"
            "addp v28.8h, v28.8h, v28.8h\n"

            "cmp %w[remain_n], #4\n"
            "bne 7f\n"

            "st1 {v16.4h}, [x0], #8\n"
            "st1 {v20.4h}, [x0], #8\n"
            "st1 {v24.4h}, [x0], #8\n"
            "st1 {v28.4h}, [x0], #8\n"
            "b 101f\n"

            "7:\n" STORE_C

            "101:\n"
            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [oddk] "+r"(oddk), [LDC] "+r"(LDC), [outptr] "+r"(outptr),
              [remain_n] "+r"(remain_n)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
              "x8", "x9", "x10", "cc", "memory");

#undef STORE_C
#undef STORE_LINE
}
static inline void transpose_8x4_b(const dt_int8* inptr, dt_int8* outptr) {
    int8x8x4_t in0 = vld4_s8(inptr);
    vst1_s8(outptr + 0 * 8, in0.val[0]);
    vst1_s8(outptr + 1 * 8, in0.val[1]);
    vst1_s8(outptr + 2 * 8, in0.val[2]);
    vst1_s8(outptr + 3 * 8, in0.val[3]);
}

static inline void interleve_8x4_b(const dt_int8* inptr, const dt_int8* inptr2,
                                   dt_int8* outptr) {
    int8x16_t in0 = vld1q_s8(inptr);
    int8x16_t in1 = vld1q_s8(inptr2);
    int32x4x2_t in_x2 = {
            {vreinterpretq_s32_s8(in0), vreinterpretq_s32_s8(in1)}};
    vst2q_s32(reinterpret_cast<int32_t*>(outptr), in_x2);
}

static inline void interleve_8x4_b_pad(const dt_int8* inptr, dt_int8* outptr) {
    int8x16_t in0 = vld1q_s8(inptr);
    int8x16_t in1 = vdupq_n_s8(0);
    int32x4x2_t in_x2 = {
            {vreinterpretq_s32_s8(in0), vreinterpretq_s32_s8(in1)}};
    vst2q_s32(reinterpret_cast<int32_t*>(outptr), in_x2);
}

static void gemm_s8x8x16_mk4_4x4x8_pack_A(dt_int8* out, const dt_int8* in,
                                          int ldin, int m0, int mmax, int k0,
                                          int kmax) {
    megdnn_assert(m0 % 4 == 0 && mmax % 4 == 0, "M must be time of 4");
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    constexpr int pack_m = 4;
    constexpr int pack_k = 8;
    constexpr int pack_size = 4;
    const int ksize = kmax - k0;
    const int remain_k = ksize % pack_k;
    const int kend = kmax - remain_k;
    int8_t tmpbuff[pack_m * pack_k]{0};

    for (int m_idx = m0; m_idx < mmax; m_idx += pack_m) {
        const int8_t* inptr0 = in + m_idx / pack_size * ldin + k0;

        for (int k_idx = k0; k_idx < kend; k_idx += pack_k) {
            transpose_8x4_b(inptr0, out);
            inptr0 += pack_m * pack_k;
            out += pack_m * pack_k;
        }
        if (remain_k > 0) {
            int8x16_t tmp = vld1q_s8(inptr0);
            vst1q_s8(&tmpbuff[0], tmp);
            transpose_8x4_b(&tmpbuff[0], out);
            inptr0 += pack_m * pack_size;
            out += pack_m * pack_k;
        }
    }
}

static void gemm_s8x8x16_mk4_4x4x8_pack_B(dt_int8* out, const dt_int8* in,
                                          int ldin, int n0, int nmax, int k0,
                                          int kmax) {
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");

    constexpr int pack_n = 4;
    constexpr int pack_k = 8;
    constexpr int pack_size = 4;
    const int ksize = kmax - k0;
    const int packed_ksize = round_up(ksize, pack_k);
    const int remain_k = ksize % pack_k;
    const int kend = kmax - remain_k;
    const int nsize = nmax - n0;
    const int remain_n = nsize % pack_n;
    const int nend = nmax - remain_n;
    const int stride_input = pack_size * nsize;
    int8_t tmpbuff[pack_n * pack_k]{0};
    int8_t tmpbuff2[pack_n * pack_k]{0};

    for (int k_idx = k0; k_idx < kend; k_idx += pack_k) {
        const int8_t* inptr = in + k_idx / pack_size * ldin + n0 * pack_size;
        const int8_t* inptr2 = inptr + stride_input;
        int8_t* outptr = out + k_idx * pack_n;
        for (int n_idx = n0; n_idx < nend; n_idx += pack_n) {
            interleve_8x4_b(inptr, inptr2, outptr);
            inptr += pack_n * pack_size;
            inptr2 += pack_n * pack_size;
            outptr += pack_n * packed_ksize;
        }
        if (remain_n > 0) {
            memcpy(&tmpbuff[0], inptr, remain_n * pack_size * sizeof(int8_t));
            memcpy(&tmpbuff2[0], inptr2, remain_n * pack_size * sizeof(int8_t));
            interleve_8x4_b(&tmpbuff[0], &tmpbuff2[0], outptr);
            outptr += pack_n * packed_ksize;
        }
    }
    if (remain_k > 0) {
        const int8_t* inptr = in + kend / pack_size * ldin + n0 * pack_size;
        int8_t* outptr = out + kend * pack_n;
        for (int n_idx = n0; n_idx < nend; n_idx += pack_n) {
            interleve_8x4_b_pad(inptr, outptr);
            inptr += pack_n * pack_size;
            outptr += pack_n * packed_ksize;
        }
        if (remain_n > 0) {
            memcpy(&tmpbuff[0], inptr, remain_n * pack_size * sizeof(int8_t));
            interleve_8x4_b_pad(&tmpbuff[0], outptr);
            outptr += pack_n * packed_ksize;
        }
    }
}

}  // namespace matmul_mk4_4x4x8_a72
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
