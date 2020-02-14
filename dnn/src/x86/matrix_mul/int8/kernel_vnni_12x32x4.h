/**
 * \file dnn/src/x86/matrix_mul/int8/kernel_vnni_12x32x4.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if MEGDNN_X86_WITH_VNNI
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "src/common/utils.h"
#include "src/x86/matrix_mul/common/common.h"

namespace megdnn {
namespace x86 {
namespace matmul_vnni_12x32x4 {

MEGDNN_ATTRIBUTE_TARGET("avx512vl,avx512vnni")
static void kern_12x32x4(const uint8_t* packA, const int8_t* packB, int K,
                         int32_t* output, int LDC, bool is_first_k) {
    constexpr size_t unroll_k = 4;
    __m512i v[24];
    __m512i sub_t0 = _mm512_setzero_epi32();
    __m512i sub_t1 = _mm512_setzero_epi32();
    // init register
    if (is_first_k) {
        v[0] = _mm512_setzero_epi32();
        v[1] = _mm512_setzero_epi32();
        v[2] = _mm512_setzero_epi32();
        v[3] = _mm512_setzero_epi32();
        v[4] = _mm512_setzero_epi32();
        v[5] = _mm512_setzero_epi32();
        v[6] = _mm512_setzero_epi32();
        v[7] = _mm512_setzero_epi32();
        v[8] = _mm512_setzero_epi32();
        v[9] = _mm512_setzero_epi32();
        v[10] = _mm512_setzero_epi32();
        v[11] = _mm512_setzero_epi32();
        v[12] = _mm512_setzero_epi32();
        v[13] = _mm512_setzero_epi32();
        v[14] = _mm512_setzero_epi32();
        v[15] = _mm512_setzero_epi32();
        v[16] = _mm512_setzero_epi32();
        v[17] = _mm512_setzero_epi32();
        v[18] = _mm512_setzero_epi32();
        v[19] = _mm512_setzero_epi32();
        v[20] = _mm512_setzero_epi32();
        v[21] = _mm512_setzero_epi32();
        v[22] = _mm512_setzero_epi32();
        v[23] = _mm512_setzero_epi32();
    } else {
        for (size_t i = 0; i < 12; i++) {
            int32_t* out_temp = output + i * LDC;
            v[2 * i] = _mm512_load_epi32(out_temp);
            v[2 * i + 1] = _mm512_load_epi32(out_temp + 64);
        }
    }
    // loop k block
    size_t kblocks = (K + unroll_k - 1) / unroll_k;
    // int8 trick: add 128 means add b1000 0000, it is same to -128
    const __m512i const_m512 = _mm512_set1_epi8(-128);
    for (size_t bk = 0; bk < kblocks; bk++) {
        __m512i b0 = _mm512_load_si512(packB + bk * 128);
        __m512i b1 = _mm512_load_si512(packB + bk * 128 + 64);
        sub_t0 = _mm512_dpbusds_epi32(sub_t0, const_m512, b0);
        sub_t1 = _mm512_dpbusds_epi32(sub_t1, const_m512, b1);

        __m512i a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48));
        __m512i a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 4));
        __m512i a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 8));
        __m512i a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 12));

        v[0] = _mm512_dpbusds_epi32(v[0], a0, b0);
        v[1] = _mm512_dpbusds_epi32(v[1], a0, b1);
        v[2] = _mm512_dpbusds_epi32(v[2], a1, b0);
        v[3] = _mm512_dpbusds_epi32(v[3], a1, b1);
        v[4] = _mm512_dpbusds_epi32(v[4], a2, b0);
        v[5] = _mm512_dpbusds_epi32(v[5], a2, b1);
        v[6] = _mm512_dpbusds_epi32(v[6], a3, b0);
        v[7] = _mm512_dpbusds_epi32(v[7], a3, b1);

        a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 16));
        a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 20));
        a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 24));
        a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 28));

        v[8] = _mm512_dpbusds_epi32(v[8], a0, b0);
        v[9] = _mm512_dpbusds_epi32(v[9], a0, b1);
        v[10] = _mm512_dpbusds_epi32(v[10], a1, b0);
        v[11] = _mm512_dpbusds_epi32(v[11], a1, b1);
        v[12] = _mm512_dpbusds_epi32(v[12], a2, b0);
        v[13] = _mm512_dpbusds_epi32(v[13], a2, b1);
        v[14] = _mm512_dpbusds_epi32(v[14], a3, b0);
        v[15] = _mm512_dpbusds_epi32(v[15], a3, b1);

        a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 32));
        a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 36));
        a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 40));
        a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 44));

        v[16] = _mm512_dpbusds_epi32(v[16], a0, b0);
        v[17] = _mm512_dpbusds_epi32(v[17], a0, b1);
        v[18] = _mm512_dpbusds_epi32(v[18], a1, b0);
        v[19] = _mm512_dpbusds_epi32(v[19], a1, b1);
        v[20] = _mm512_dpbusds_epi32(v[20], a2, b0);
        v[21] = _mm512_dpbusds_epi32(v[21], a2, b1);
        v[22] = _mm512_dpbusds_epi32(v[22], a3, b0);
        v[23] = _mm512_dpbusds_epi32(v[23], a3, b1);
    }

    // store value
    v[0] = _mm512_sub_epi32(v[0], sub_t0);
    v[2] = _mm512_sub_epi32(v[2], sub_t0);
    v[4] = _mm512_sub_epi32(v[4], sub_t0);
    v[6] = _mm512_sub_epi32(v[6], sub_t0);
    v[8] = _mm512_sub_epi32(v[8], sub_t0);
    v[10] = _mm512_sub_epi32(v[10], sub_t0);
    v[12] = _mm512_sub_epi32(v[12], sub_t0);
    v[14] = _mm512_sub_epi32(v[14], sub_t0);
    v[16] = _mm512_sub_epi32(v[16], sub_t0);
    v[18] = _mm512_sub_epi32(v[18], sub_t0);
    v[20] = _mm512_sub_epi32(v[20], sub_t0);
    v[22] = _mm512_sub_epi32(v[22], sub_t0);

    v[1] = _mm512_sub_epi32(v[1], sub_t1);
    v[3] = _mm512_sub_epi32(v[3], sub_t1);
    v[5] = _mm512_sub_epi32(v[5], sub_t1);
    v[7] = _mm512_sub_epi32(v[7], sub_t1);
    v[9] = _mm512_sub_epi32(v[9], sub_t1);
    v[11] = _mm512_sub_epi32(v[11], sub_t1);
    v[13] = _mm512_sub_epi32(v[13], sub_t1);
    v[15] = _mm512_sub_epi32(v[15], sub_t1);
    v[17] = _mm512_sub_epi32(v[17], sub_t1);
    v[19] = _mm512_sub_epi32(v[19], sub_t1);
    v[21] = _mm512_sub_epi32(v[21], sub_t1);
    v[23] = _mm512_sub_epi32(v[23], sub_t1);

    _mm512_storeu_si512(output, v[0]);
    _mm512_storeu_si512(output + 16, v[1]);
    _mm512_storeu_si512(output + LDC, v[2]);
    _mm512_storeu_si512(output + LDC + 16, v[3]);
    _mm512_storeu_si512(output + 2 * LDC, v[4]);
    _mm512_storeu_si512(output + 2 * LDC + 16, v[5]);
    _mm512_storeu_si512(output + 3 * LDC, v[6]);
    _mm512_storeu_si512(output + 3 * LDC + 16, v[7]);
    _mm512_storeu_si512(output + 4 * LDC, v[8]);
    _mm512_storeu_si512(output + 4 * LDC + 16, v[9]);
    _mm512_storeu_si512(output + 5 * LDC, v[10]);
    _mm512_storeu_si512(output + 5 * LDC + 16, v[11]);
    _mm512_storeu_si512(output + 6 * LDC, v[12]);
    _mm512_storeu_si512(output + 6 * LDC + 16, v[13]);
    _mm512_storeu_si512(output + 7 * LDC, v[14]);
    _mm512_storeu_si512(output + 7 * LDC + 16, v[15]);
    _mm512_storeu_si512(output + 8 * LDC, v[16]);
    _mm512_storeu_si512(output + 8 * LDC + 16, v[17]);
    _mm512_storeu_si512(output + 9 * LDC, v[18]);
    _mm512_storeu_si512(output + 9 * LDC + 16, v[19]);
    _mm512_storeu_si512(output + 10 * LDC, v[20]);
    _mm512_storeu_si512(output + 10 * LDC + 16, v[21]);
    _mm512_storeu_si512(output + 11 * LDC, v[22]);
    _mm512_storeu_si512(output + 11 * LDC + 16, v[23]);
}

MEGDNN_ATTRIBUTE_TARGET("avx512vl,avx512vnni")
static void kern_12x16x4(const uint8_t* packA, const int8_t* packB, int K,
                         int32_t* output, int LDC, bool is_first_k,
                         size_t n_remain = 16) {
    megdnn_assert(n_remain <= 16,
                  "kernel vnni kern_12x32x4 n_remain is not allow big than 16");
    constexpr size_t unroll_k = 4;
    __m512i v[12];
    __m512i sub_t0 = _mm512_setzero_epi32();
    // init register
    if (is_first_k) {
        v[0] = _mm512_setzero_epi32();
        v[1] = _mm512_setzero_epi32();
        v[2] = _mm512_setzero_epi32();
        v[3] = _mm512_setzero_epi32();
        v[4] = _mm512_setzero_epi32();
        v[5] = _mm512_setzero_epi32();
        v[6] = _mm512_setzero_epi32();
        v[7] = _mm512_setzero_epi32();
        v[8] = _mm512_setzero_epi32();
        v[9] = _mm512_setzero_epi32();
        v[10] = _mm512_setzero_epi32();
        v[11] = _mm512_setzero_epi32();
    } else {
        int32_t temp_out[16] = {0};
        for (size_t i = 0; i < 12; i++) {
            int32_t* out_src = output + i * LDC;
            for (size_t j = 0; j < n_remain; j++) {
                temp_out[j] = out_src[j];
            }
            v[i] = _mm512_load_si512(temp_out);
        }
    }
    // loop k block
    size_t kblocks = (K + unroll_k - 1) / unroll_k;
    // int8 trick: add 128 means add b1000 0000, it is same to -128
    const __m512i const_m512 = _mm512_set1_epi8(-128);
    for (size_t bk = 0; bk < kblocks; bk++) {
        __m512i b0 = _mm512_load_si512(packB + bk * 64);
        sub_t0 = _mm512_dpbusds_epi32(sub_t0, const_m512, b0);

        __m512i a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48));
        __m512i a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 4));
        __m512i a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 8));
        __m512i a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 12));

        v[0] = _mm512_dpbusds_epi32(v[0], a0, b0);
        v[1] = _mm512_dpbusds_epi32(v[1], a1, b0);
        v[2] = _mm512_dpbusds_epi32(v[2], a2, b0);
        v[3] = _mm512_dpbusds_epi32(v[3], a3, b0);

        a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 16));
        a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 20));
        a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 24));
        a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 28));

        v[4] = _mm512_dpbusds_epi32(v[4], a0, b0);
        v[5] = _mm512_dpbusds_epi32(v[5], a1, b0);
        v[6] = _mm512_dpbusds_epi32(v[6], a2, b0);
        v[7] = _mm512_dpbusds_epi32(v[7], a3, b0);

        a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 32));
        a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 36));
        a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 40));
        a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 48 + 44));

        v[8] = _mm512_dpbusds_epi32(v[8], a0, b0);
        v[9] = _mm512_dpbusds_epi32(v[9], a1, b0);
        v[10] = _mm512_dpbusds_epi32(v[10], a2, b0);
        v[11] = _mm512_dpbusds_epi32(v[11], a3, b0);
    }

    // store value
    v[0] = _mm512_sub_epi32(v[0], sub_t0);
    v[1] = _mm512_sub_epi32(v[1], sub_t0);
    v[2] = _mm512_sub_epi32(v[2], sub_t0);
    v[3] = _mm512_sub_epi32(v[3], sub_t0);
    v[4] = _mm512_sub_epi32(v[4], sub_t0);
    v[5] = _mm512_sub_epi32(v[5], sub_t0);
    v[6] = _mm512_sub_epi32(v[6], sub_t0);
    v[7] = _mm512_sub_epi32(v[7], sub_t0);
    v[8] = _mm512_sub_epi32(v[8], sub_t0);
    v[9] = _mm512_sub_epi32(v[9], sub_t0);
    v[10] = _mm512_sub_epi32(v[10], sub_t0);
    v[11] = _mm512_sub_epi32(v[11], sub_t0);

    if (n_remain == 16) {
        _mm512_storeu_si512(output, v[0]);
        _mm512_storeu_si512(output + LDC, v[1]);
        _mm512_storeu_si512(output + 2 * LDC, v[2]);
        _mm512_storeu_si512(output + 3 * LDC, v[3]);
        _mm512_storeu_si512(output + 4 * LDC, v[4]);
        _mm512_storeu_si512(output + 5 * LDC, v[5]);
        _mm512_storeu_si512(output + 6 * LDC, v[6]);
        _mm512_storeu_si512(output + 7 * LDC, v[7]);
        _mm512_storeu_si512(output + 8 * LDC, v[8]);
        _mm512_storeu_si512(output + 9 * LDC, v[9]);
        _mm512_storeu_si512(output + 10 * LDC, v[10]);
        _mm512_storeu_si512(output + 11 * LDC, v[11]);
    } else {
        for (size_t m = 0; m < 12; m++) {
            int32_t* out_dst = output + m * LDC;
            int32_t* out = reinterpret_cast<int32_t*>(&(v[m]));
            for (size_t n = 0; n < n_remain; n++) {
                out_dst[n] = out[n];
            }
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("avx512vl,avx512vnni")
static void kern_4x32x4(const uint8_t* packA, const int8_t* packB, int K,
                        int32_t* output, int LDC, bool is_first_k,
                        size_t m_remain = 4) {
    megdnn_assert(m_remain <= 4,
                  "kernel vnni kern_4x32x4 m_remain is not allow big than 4");
    constexpr size_t unroll_k = 4;
    __m512i v[8];
    __m512i sub_t0 = _mm512_setzero_epi32();
    __m512i sub_t1 = _mm512_setzero_epi32();
    // init register
    if (is_first_k) {
        v[0] = _mm512_setzero_epi32();
        v[1] = _mm512_setzero_epi32();
        v[2] = _mm512_setzero_epi32();
        v[3] = _mm512_setzero_epi32();
        v[4] = _mm512_setzero_epi32();
        v[5] = _mm512_setzero_epi32();
        v[6] = _mm512_setzero_epi32();
        v[7] = _mm512_setzero_epi32();
    } else {
        for (size_t i = 0; i < m_remain; i++) {
            int32_t* out_current = output + i * LDC;
            v[2 * i] = _mm512_load_epi32(out_current);
            v[2 * i + 1] = _mm512_load_epi32(out_current + 64);
        }
    }
    // loop k block
    size_t kblocks = (K + unroll_k - 1) / unroll_k;
    // int8 trick: add 128 means add b1000 0000, it is same to -128
    const __m512i const_m512 = _mm512_set1_epi8(-128);
    for (size_t bk = 0; bk < kblocks; bk++) {
        __m512i b0 = _mm512_load_si512(packB + bk * 128);
        __m512i b1 = _mm512_load_si512(packB + bk * 128 + 64);
        sub_t0 = _mm512_dpbusds_epi32(sub_t0, const_m512, b0);
        sub_t1 = _mm512_dpbusds_epi32(sub_t1, const_m512, b1);

        __m512i a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16));
        __m512i a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16 + 4));
        __m512i a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16 + 8));
        __m512i a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16 + 12));
        v[0] = _mm512_dpbusds_epi32(v[0], a0, b0);
        v[1] = _mm512_dpbusds_epi32(v[1], a0, b1);
        v[2] = _mm512_dpbusds_epi32(v[2], a1, b0);
        v[3] = _mm512_dpbusds_epi32(v[3], a1, b1);
        v[4] = _mm512_dpbusds_epi32(v[4], a2, b0);
        v[5] = _mm512_dpbusds_epi32(v[5], a2, b1);
        v[6] = _mm512_dpbusds_epi32(v[6], a3, b0);
        v[7] = _mm512_dpbusds_epi32(v[7], a3, b1);
    }
    // store value
    for (size_t m = 0; m < m_remain; m++) {
        v[2 * m] = _mm512_sub_epi32(v[2 * m], sub_t0);
        v[2 * m + 1] = _mm512_sub_epi32(v[2 * m + 1], sub_t1);
        _mm512_storeu_si512(output + m * LDC, v[2 * m]);
        _mm512_storeu_si512(output + m * LDC + 16, v[2 * m + 1]);
    }
}

MEGDNN_ATTRIBUTE_TARGET("avx512vl,avx512vnni")
static void kern_4x16x4(const uint8_t* packA, const int8_t* packB, int K,
                        int32_t* output, int LDC, bool is_first_k,
                        size_t m_remain = 4, size_t n_remain = 16) {
    megdnn_assert(m_remain <= 4,
                  "kernel vnni kern_4x32x4 m_remain is not allow big than 4");
    megdnn_assert(n_remain <= 16,
                  "kernel vnni kern_4x32x4 n_remain is not allow big than 16");

    constexpr size_t unroll_k = 4;
    __m512i v[4];
    __m512i sub_t0 = _mm512_setzero_epi32();
    // int8 trick: add 128 means add b1000 0000, it is same to -128
    const __m512i const_m512 = _mm512_set1_epi8(-128);
    // init register
    if (is_first_k) {
        v[0] = _mm512_setzero_epi32();
        v[1] = _mm512_setzero_epi32();
        v[2] = _mm512_setzero_epi32();
        v[3] = _mm512_setzero_epi32();
    } else {
        int32_t temp_out[16] = {0};
        size_t i = 0;
        for (; i < m_remain; i++) {
            int32_t* out_src = output + i * LDC;
            for (size_t j = 0; j < n_remain; j++) {
                temp_out[j] = out_src[j];
            }
            v[i] = _mm512_load_si512(temp_out);
        }
        for (; i < 4; i++) {
            v[i] = _mm512_setzero_epi32();
        }
    }
    // loop k block
    size_t kblocks = (K + unroll_k - 1) / unroll_k;
    for (size_t bk = 0; bk < kblocks; bk++) {
        __m512i b0 = _mm512_load_si512(packB + bk * 64);
        sub_t0 = _mm512_dpbusds_epi32(sub_t0, const_m512, b0);

        __m512i a0 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16));
        __m512i a1 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16 + 4));
        __m512i a2 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16 + 8));
        __m512i a3 = _mm512_set1_epi32(*(int32_t*)(packA + bk * 16 + 12));
        v[0] = _mm512_dpbusds_epi32(v[0], a0, b0);
        v[1] = _mm512_dpbusds_epi32(v[1], a1, b0);
        v[2] = _mm512_dpbusds_epi32(v[2], a2, b0);
        v[3] = _mm512_dpbusds_epi32(v[3], a3, b0);
    }
    // store value
    for (size_t m = 0; m < m_remain; m++) {
        v[m] = _mm512_sub_epi32(v[m], sub_t0);
        int32_t* out_dst = output + m * LDC;
        for (size_t n = 0; n < n_remain; n++) {
            out_dst[n] = (reinterpret_cast<int32_t*>(&v[m]))[n];
        }
    }
}
static void gemm_pack_A_n(dt_uint8* outptr, const dt_int8* inptr, int ldin,
                          int y0, int ymax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 11 < ymax; y += 12) {
        const int8_t* input[12];
        input[0] = inptr + y * ldin + k0;
        input[1] = input[0] + ldin;
        input[2] = input[1] + ldin;
        input[3] = input[2] + ldin;
        input[4] = input[3] + ldin;
        input[5] = input[4] + ldin;
        input[6] = input[5] + ldin;
        input[7] = input[6] + ldin;
        input[8] = input[7] + ldin;
        input[9] = input[8] + ldin;
        input[10] = input[9] + ldin;
        input[11] = input[10] + ldin;
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            interleave_12x4_4_b_add_128(input, outptr);
        }
        if (K > 0) {
            interleave_12_add_128(input, outptr, 4, K);
        }
    }
    for (; y < ymax; y += 4) {
        const int8_t* input[4];
        input[0] = inptr + y * ldin + k0;
        input[1] = input[0] + ldin;
        input[2] = input[1] + ldin;
        input[3] = input[2] + ldin;
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        input[1] = static_cast<int8_t*>(zerobuff);
                        input[2] = static_cast<int8_t*>(zerobuff);
                        input[3] = static_cast<int8_t*>(zerobuff);
                        break;
                    case 1:
                        input[2] = static_cast<int8_t*>(zerobuff);
                        input[3] = static_cast<int8_t*>(zerobuff);
                        break;
                    case 0:
                        input[3] = static_cast<int8_t*>(zerobuff);
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4x4_4_b_add_128(input[0], input[1], input[2], input[3],
                                       outptr);
        }
        if (K > 0) {
            if (y + 3 >= ymax) {
                switch (y + 3 - ymax) {
                    case 2:
                        input[1] = static_cast<int8_t*>(zerobuff);
                        input[2] = static_cast<int8_t*>(zerobuff);
                        input[3] = static_cast<int8_t*>(zerobuff);
                        break;
                    case 1:
                        input[2] = static_cast<int8_t*>(zerobuff);
                        input[3] = static_cast<int8_t*>(zerobuff);
                        break;
                    case 0:
                        input[3] = static_cast<int8_t*>(zerobuff);
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4_add_128(input[0], input[1], input[2], input[3], outptr,
                                 4, K);
        }
    }
}

static void gemm_pack_A_t(dt_uint8* out, const dt_int8* in, int ldin, int x0,
                          int xmax, int k0, int kmax) {
    int8_t zerobuff[12];
    std::memset(zerobuff, 0, sizeof(int8_t) * 12);
    const int ksize = kmax - k0;
    const int ksize12 = round_up<int>(ksize, 4) * 12;
    const int ksize4 = round_up<int>(ksize, 4) * 4;
    uint8_t* outptr = out;
    uint8_t* outptr_base = out;
    uint8_t* outptr_base4 = out + ((xmax - x0) / 12) * ksize12;

    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = in + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        int x = x0;
        outptr = outptr_base;
        for (; x + 11 < xmax; x += 12) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = static_cast<int8_t*>(zerobuff);
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 1:
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 0:
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            transpose_4x12_1_b_add_128(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += ksize12;
        }
        outptr = outptr_base4;
        for (; x < xmax; x += 4) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = static_cast<int8_t*>(zerobuff);
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 1:
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 0:
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            transpose_4_add_128(inptr0, inptr1, inptr2, inptr3, outptr, 4,
                                std::min<size_t>(4, (xmax - x)));
            outptr += ksize4;
        }
        outptr_base += 12 * 4;
        outptr_base4 += 4 * 4;
    }
}

static void gemm_pack_B_n(dt_int8* out, const dt_int8* in, int ldin, int x0,
                          int xmax, int k0, int kmax) {
    int8_t zerobuff[32];
    std::memset(zerobuff, 0, sizeof(int8_t) * 32);
    const int ksize = kmax - k0;
    const int ksize32 = round_up<int>(ksize, 4) * 32;
    const int ksize16 = round_up(ksize, 4) * 16;
    int8_t* outptr = out;
    int8_t* outptr_base = out;
    //! 4x4 block output start pos
    int8_t* outptr_base16 = out + ((xmax - x0) / 32) * ksize32;

    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = in + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;

        int x = x0;
        outptr = outptr_base;
        for (; x + 31 < xmax; x += 32) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = static_cast<int8_t*>(zerobuff);
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 1:
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 0:
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4x32_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += ksize32;
        }

        outptr = outptr_base16;
        for (; x + 15 < xmax; x += 16) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = static_cast<int8_t*>(zerobuff);
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 1:
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 0:
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4x16_1_b(inptr0, inptr1, inptr2, inptr3, outptr);
            outptr += ksize16;
        }
        if (x < xmax) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
                    case 2:
                        inptr1 = static_cast<int8_t*>(zerobuff);
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 1:
                        inptr2 = static_cast<int8_t*>(zerobuff);
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    case 0:
                        inptr3 = static_cast<int8_t*>(zerobuff);
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            transpose_4(inptr0, inptr1, inptr2, inptr3, outptr, 16, xmax - x);
        }
        outptr_base += 32 * 4;
        outptr_base16 += 16 * 4;
    }
}

static void gemm_pack_B_t(dt_int8* outptr, const dt_int8* inptr, int ldin,
                          int y0, int ymax, int k0, int kmax) {
    int8_t zerobuff[16];
    std::memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y + 31 < ymax; y += 32) {
        const int8_t* input[32];
        input[0] = inptr + y * ldin + k0;
        for (int i = 1; i < 32; i++)
            input[i] = input[i - 1] + ldin;

        int K = kmax - k0;
        //! read 12 * 4 in each row
        for (; K > 15; K -= 16) {
            interleave_32x4_4_b(input, outptr);
        }
        if (K > 0) {
            interleave_32(input, outptr, 4, K);
        }
    }
    for (; y < ymax; y += 16) {
        const int8_t* input[16];
        input[0] = inptr + y * ldin + k0;
        for (int i = 1; i < 16; i++)
            input[i] = input[i - 1] + ldin;

        int K = kmax - k0;
        //! read 4 * 4 in each row
        for (; K > 15; K -= 16) {
            for (int i = 0; i < 16; i++) {
                if (i >= ymax - y) {
                    input[i] = zerobuff;
                }
            }
            interleave_16x4_4_b(input, outptr);
        }

        if (K > 0) {
            for (int i = 0; i < 16; i++) {
                if (i >= ymax - y) {
                    input[i] = zerobuff;
                }
            }
            interleave_16(input, outptr, 4, K);
        }
    }
}

}  // namespace matmul_vnni_12x32x4
}  // namespace x86
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
