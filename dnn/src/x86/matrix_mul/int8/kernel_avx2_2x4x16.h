/**
 * \file dnn/src/x86/matrix_mul/int8/kernel_avx2_2x4x16.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "src/common/utils.h"
#include "src/x86/matrix_mul/common/common.h"

namespace megdnn {
namespace x86 {

namespace matmul_avx2_2x4x16 {

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_2x4x16_remain(
        const int8_t* pack_a_ptr, const int8_t* pack_b_ptr, int32_t* c_ptr,
        const int32_t ldc, const int32_t k, const int32_t remain_m,
        const int32_t remain_n) {
    int32_t* c0_ptr = c_ptr + 0 * ldc;
    int32_t* c1_ptr = c_ptr + 1 * ldc;

    constexpr int32_t k_step = 16;

    int32_t nk = (k + k_step - 1) / k_step;

    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i temp_vec[4];
    __m256i c_vec[8];

    c_vec[0] = _mm256_setzero_si256();
    c_vec[1] = _mm256_setzero_si256();
    c_vec[2] = _mm256_setzero_si256();
    c_vec[3] = _mm256_setzero_si256();
    c_vec[4] = _mm256_setzero_si256();
    c_vec[5] = _mm256_setzero_si256();
    c_vec[6] = _mm256_setzero_si256();
    c_vec[7] = _mm256_setzero_si256();

    for (int32_t k_iter = 0; k_iter < nk; ++k_iter) {
        a_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_a_ptr);
        a_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_a_ptr + 16);

        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
        temp_vec[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[0] = _mm256_add_epi32(temp_vec[0], c_vec[0]);
        temp_vec[1] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[1] = _mm256_add_epi32(temp_vec[1], c_vec[1]);

        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);
        temp_vec[2] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[2] = _mm256_add_epi32(temp_vec[2], c_vec[2]);
        temp_vec[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[3] = _mm256_add_epi32(temp_vec[3], c_vec[3]);

        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 32);
        temp_vec[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[4] = _mm256_add_epi32(temp_vec[0], c_vec[4]);
        temp_vec[1] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[5] = _mm256_add_epi32(temp_vec[1], c_vec[5]);

        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 48);
        temp_vec[2] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[6] = _mm256_add_epi32(temp_vec[2], c_vec[6]);
        temp_vec[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[7] = _mm256_add_epi32(temp_vec[3], c_vec[7]);

        pack_a_ptr += 32;
        pack_b_ptr += 64;
    }

    a_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_hadd_epi32(c_vec[0], c_vec[2]);
    c_vec[0] = _mm256_hadd_epi32(c_vec[0], a_vec[0]);
    c_vec[2] = _mm256_permute2x128_si256(c_vec[0], a_vec[0], 0x31);
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_vec[2]);
    c0_ptr[0] = _mm256_extract_epi32(c_vec[0], 0);
    if (remain_n > 1) {
        c0_ptr[1] = _mm256_extract_epi32(c_vec[0], 1);
    }
    c_vec[4] = _mm256_hadd_epi32(c_vec[4], c_vec[6]);
    c_vec[4] = _mm256_hadd_epi32(c_vec[4], a_vec[0]);
    c_vec[6] = _mm256_permute2x128_si256(c_vec[4], a_vec[0], 0x31);
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_vec[6]);
    if (remain_n > 2) {
        c0_ptr[2] = _mm256_extract_epi32(c_vec[4], 0);
    }
    if (remain_n > 3) {
        c0_ptr[3] = _mm256_extract_epi32(c_vec[4], 1);
    }
    if (remain_m > 1) {
        c_vec[1] = _mm256_hadd_epi32(c_vec[1], c_vec[3]);
        c_vec[1] = _mm256_hadd_epi32(c_vec[1], a_vec[0]);
        c_vec[3] = _mm256_permute2x128_si256(c_vec[1], a_vec[0], 0x31);
        c_vec[1] = _mm256_add_epi32(c_vec[1], c_vec[3]);
        c1_ptr[0] = _mm256_extract_epi32(c_vec[1], 0);
        if (remain_n > 1) {
            c1_ptr[1] = _mm256_extract_epi32(c_vec[1], 1);
        }

        c_vec[5] = _mm256_hadd_epi32(c_vec[5], c_vec[7]);
        c_vec[5] = _mm256_hadd_epi32(c_vec[5], a_vec[0]);
        c_vec[7] = _mm256_permute2x128_si256(c_vec[5], a_vec[0], 0x31);
        c_vec[5] = _mm256_add_epi32(c_vec[5], c_vec[7]);
        if (remain_n > 2) {
            c1_ptr[2] = _mm256_extract_epi32(c_vec[5], 0);
        }
        if (remain_n > 3) {
            c1_ptr[3] = _mm256_extract_epi32(c_vec[5], 1);
        }
    }
}
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_2x4x16(const int8_t* pack_a_ptr,
                                            const int8_t* pack_b_ptr,
                                            int32_t* c_ptr, const int32_t ldc,
                                            const int32_t k) {
    int32_t* c0_ptr = c_ptr + 0 * ldc;
    int32_t* c1_ptr = c_ptr + 1 * ldc;

    constexpr int32_t k_step = 16;

    // TODO try define c_temp
    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i temp_vec[4];
    __m256i c_vec[8];

    a_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_a_ptr);
    a_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_a_ptr + 16);

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
    temp_vec[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_add_epi32(temp_vec[0], c_vec[0]);
    temp_vec[1] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_vec[1] = _mm256_setzero_si256();
    c_vec[1] = _mm256_add_epi32(temp_vec[1], c_vec[1]);

    b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);
    temp_vec[2] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[2] = _mm256_setzero_si256();
    c_vec[2] = _mm256_add_epi32(temp_vec[2], c_vec[2]);
    temp_vec[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[3] = _mm256_setzero_si256();
    c_vec[3] = _mm256_add_epi32(temp_vec[3], c_vec[3]);

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 32);
    temp_vec[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_vec[4] = _mm256_setzero_si256();
    c_vec[4] = _mm256_add_epi32(temp_vec[0], c_vec[4]);
    temp_vec[1] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_vec[5] = _mm256_setzero_si256();
    c_vec[5] = _mm256_add_epi32(temp_vec[1], c_vec[5]);

    b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 48);
    temp_vec[2] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[6] = _mm256_setzero_si256();
    c_vec[6] = _mm256_add_epi32(temp_vec[2], c_vec[6]);
    temp_vec[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[7] = _mm256_setzero_si256();
    c_vec[7] = _mm256_add_epi32(temp_vec[3], c_vec[7]);

    pack_a_ptr += 32;
    pack_b_ptr += 64;

    for (int32_t k_iter = 16; k_iter < k; k_iter += k_step) {
        a_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_a_ptr);
        a_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_a_ptr + 16);

        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
        temp_vec[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[0] = _mm256_add_epi32(temp_vec[0], c_vec[0]);
        temp_vec[1] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[1] = _mm256_add_epi32(temp_vec[1], c_vec[1]);

        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);
        temp_vec[2] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[2] = _mm256_add_epi32(temp_vec[2], c_vec[2]);
        temp_vec[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[3] = _mm256_add_epi32(temp_vec[3], c_vec[3]);

        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 32);
        temp_vec[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[4] = _mm256_add_epi32(temp_vec[0], c_vec[4]);
        temp_vec[1] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[5] = _mm256_add_epi32(temp_vec[1], c_vec[5]);

        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 48);
        temp_vec[2] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[6] = _mm256_add_epi32(temp_vec[2], c_vec[6]);
        temp_vec[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[7] = _mm256_add_epi32(temp_vec[3], c_vec[7]);

        pack_a_ptr += 32;
        pack_b_ptr += 64;
    }

    a_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_hadd_epi32(c_vec[0], c_vec[2]);
    c_vec[0] = _mm256_hadd_epi32(c_vec[0], a_vec[0]);
    c_vec[2] = _mm256_permute2x128_si256(c_vec[0], a_vec[0], 0x31);
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_vec[2]);
    c0_ptr[0] = _mm256_extract_epi32(c_vec[0], 0);
    c0_ptr[1] = _mm256_extract_epi32(c_vec[0], 1);

    c_vec[4] = _mm256_hadd_epi32(c_vec[4], c_vec[6]);
    c_vec[4] = _mm256_hadd_epi32(c_vec[4], a_vec[0]);
    c_vec[6] = _mm256_permute2x128_si256(c_vec[4], a_vec[0], 0x31);
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_vec[6]);
    c0_ptr[2] = _mm256_extract_epi32(c_vec[4], 0);
    c0_ptr[3] = _mm256_extract_epi32(c_vec[4], 1);

    c_vec[1] = _mm256_hadd_epi32(c_vec[1], c_vec[3]);
    c_vec[1] = _mm256_hadd_epi32(c_vec[1], a_vec[0]);
    c_vec[3] = _mm256_permute2x128_si256(c_vec[1], a_vec[0], 0x31);
    c_vec[1] = _mm256_add_epi32(c_vec[1], c_vec[3]);
    c1_ptr[0] = _mm256_extract_epi32(c_vec[1], 0);
    c1_ptr[1] = _mm256_extract_epi32(c_vec[1], 1);

    c_vec[5] = _mm256_hadd_epi32(c_vec[5], c_vec[7]);
    c_vec[5] = _mm256_hadd_epi32(c_vec[5], a_vec[0]);
    c_vec[7] = _mm256_permute2x128_si256(c_vec[5], a_vec[0], 0x31);
    c_vec[5] = _mm256_add_epi32(c_vec[5], c_vec[7]);
    c1_ptr[2] = _mm256_extract_epi32(c_vec[5], 0);
    c1_ptr[3] = _mm256_extract_epi32(c_vec[5], 1);
}

static inline void gemm_avx2_s8s8s32_2x4x16_pack_bn(dt_int8* out,
                                                    const dt_int8* in, int ldin,
                                                    int n_start, int n_max,
                                                    int k_start, int k_max) {
    constexpr int tile_n = 4;
    constexpr int tile_k = 16;
    constexpr int tile_len = tile_n * tile_k;
    const int k_size = k_max - k_start;
    const int k_end = k_size / tile_k * tile_k + k_start;
    const int k_remain = k_max - k_end;
    const int n_size = n_max - n_start;
    const int n_end = n_size / tile_n * tile_n + n_start;
    const int n_remain = n_max - n_end;
    const int pack_line_len = round_up(k_size, tile_k) * tile_n;
    int k = k_start;
    for (; k < k_end; k += tile_k) {
        int8_t* outptr = out;
        for (int n = n_start; n < n_end; n += tile_n) {
            naive_transpose_kn(outptr, in + k * ldin + n, ldin, tile_k, tile_n);
            outptr += pack_line_len;
        }
        if (n_end < n_max) {
            naive_transpose_kn_pad(outptr, in + k * ldin + n_end, ldin, tile_k,
                                   n_remain, tile_k, tile_n);
        }
        out += tile_len;
    }
    if (k_max > k_end) {
        int8_t* outptr = out;
        for (int n = n_start; n < n_end; n += tile_n) {
            naive_transpose_kn_pad(outptr, in + k_end * ldin + n, ldin,
                                   k_remain, tile_n, tile_k, tile_n);
            outptr += pack_line_len;
        }
        if (n_end < n_max) {
            naive_transpose_kn_pad(outptr, in + k * ldin + n_end, ldin,
                                   k_remain, n_remain, tile_k, tile_n);
        }
    }
}

static inline void gemm_avx2_s8s8s32_2x4x16_pack_bt(dt_int8* out,
                                                    const dt_int8* in, int ldin,
                                                    int n_start, int n_max,
                                                    int k_start, int k_max) {
    constexpr int tile_n = 4;
    constexpr int tile_k = 16;
    constexpr int tile_len = tile_n * tile_k;
    const int k_size = k_max - k_start;
    const int n_end = (n_max - n_start) / tile_n * tile_n + n_start;

    for (int n = n_start; n < n_end; n += tile_n) {
        const dt_int8* in0 = in + n * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        const dt_int8* in2 = in1 + ldin;
        const dt_int8* in3 = in2 + ldin;
        int remain_k = k_size;
        for (; remain_k >= tile_k; remain_k -= tile_k) {
            interleave_4x16(out, in0, in1, in2, in3);
            out += tile_len;
            in0 += tile_k;
            in1 += tile_k;
            in2 += tile_k;
            in3 += tile_k;
        }
        if (remain_k > 0) {
            interleave_4x16_pad(out, in0, in1, in2, in3, remain_k);
            out += tile_len;
        }
    }
    if (n_end < n_max) {
        dt_int8 zerobuff[16];
        std::memset(zerobuff, 0, sizeof(int8_t) * 16);
        const int remain_n = n_max - n_end;
        const dt_int8* in0 = in + n_end * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        const dt_int8* in2 = in1 + ldin;
        const dt_int8* in3 = in2 + ldin;
        int remain_k = k_size;
        if (remain_n == 1) {
            in1 = &zerobuff[0];
            in2 = &zerobuff[0];
            in3 = &zerobuff[0];
            for (; remain_k >= tile_k; remain_k -= tile_k) {
                interleave_4x16(out, in0, in1, in2, in3);
                out += tile_len;
                in0 += tile_k;
            }
        } else if (remain_n == 2) {
            in2 = &zerobuff[0];
            in3 = &zerobuff[0];
            for (; remain_k >= tile_k; remain_k -= tile_k) {
                interleave_4x16(out, in0, in1, in2, in3);
                out += tile_len;
                in0 += tile_k;
                in1 += tile_k;
            }
        } else if (remain_n == 3) {
            in3 = &zerobuff[0];
            for (; remain_k >= tile_k; remain_k -= tile_k) {
                interleave_4x16(out, in0, in1, in2, in3);
                out += tile_len;
                in0 += tile_k;
                in1 += tile_k;
                in2 += tile_k;
            }
        } else if (remain_n == 4) {
            for (; remain_k >= tile_k; remain_k -= tile_k) {
                interleave_4x16(out, in0, in1, in2, in3);
                out += tile_len;
                in0 += tile_k;
                in1 += tile_k;
                in2 += tile_k;
                in3 += tile_k;
            }
        }
        if (remain_k > 0) {
            interleave_4x16_pad(out, in0, in1, in2, in3, remain_k);
            out += tile_len;
        }
    }
}

static inline void gemm_avx2_s8s8s32_2x4x16_pack_an(dt_int8* out,
                                                    const dt_int8* in, int ldin,
                                                    int m_start, int m_max,
                                                    int k_start, int k_max) {
    constexpr int tile_m = 2;
    constexpr int tile_k = 16;
    constexpr int tile_len = tile_m * tile_k;
    const int k_size = k_max - k_start;
    const int m_end = (m_max - m_start) / tile_m * tile_m + m_start;

    for (int m = m_start; m < m_end; m += tile_m) {
        const dt_int8* in0 = in + m * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        int remain_k = k_size;
        for (; remain_k >= tile_k; remain_k -= tile_k) {
            interleave_2x16(out, in0, in1);
            out += tile_len;
            in0 += tile_k;
            in1 += tile_k;
        }

        if (remain_k > 0) {
            interleave_2x16_pad(out, in0, in1, remain_k);
            out += tile_len;
        }
    }
    if (m_end < m_max) {
        dt_int8 zerobuff[16];
        std::memset(zerobuff, 0, sizeof(int8_t) * 16);
        const dt_int8* in0 = in + m_end * ldin + k_start;
        const dt_int8* in1 = &zerobuff[0];
        int remain_k = k_size;
        for (; remain_k >= tile_k; remain_k -= tile_k) {
            interleave_2x16(out, in0, in1);
            out += tile_len;
            in0 += tile_k;
        }

        if (remain_k > 0) {
            interleave_2x16_pad(out, in0, in1, remain_k);
            out += tile_len;
        }
    }
}
static inline void gemm_avx2_s8s8s32_2x4x16_pack_at(dt_int8* out,
                                                    const dt_int8* in, int ldin,
                                                    int m_start, int m_max,
                                                    int k_start, int k_max) {
    constexpr int tile_m = 2;
    constexpr int tile_k = 16;
    constexpr int tile_len = tile_m * tile_k;
    const int k_size = k_max - k_start;
    const int k_end = k_size / tile_k * tile_k + k_start;
    const int k_remain = k_max - k_end;
    const int m_size = m_max - m_start;
    const int m_end = m_size / tile_m * tile_m + m_start;
    const int m_remain = m_max - m_end;
    const int pack_line_len = round_up(k_size, tile_k) * tile_m;
    int k = k_start;
    for (; k < k_end; k += tile_k) {
        int8_t* outptr = out;
        for (int m = m_start; m < m_end; m += tile_m) {
            naive_transpose_kn(outptr, in + k * ldin + m, ldin, tile_k, tile_m);
            outptr += pack_line_len;
        }
        if (m_end < m_max) {
            naive_transpose_kn_pad(outptr, in + k * ldin + m_end, ldin, tile_k,
                                   m_remain, tile_k, tile_m);
        }
        out += tile_len;
    }
    if (k_max > k_end) {
        int8_t* outptr = out;
        for (int m = m_start; m < m_end; m += tile_m) {
            naive_transpose_kn_pad(outptr, in + k_end * ldin + m, ldin,
                                   k_remain, tile_m, tile_k, tile_m);
            outptr += pack_line_len;
        }
        if (m_end < m_max) {
            naive_transpose_kn_pad(outptr, in + k * ldin + m_end, ldin,
                                   k_remain, m_remain, tile_k, tile_m);
        }
    }
}

}  // namespace matmul_avx2_2x4x16

}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen