/**
 * \file dnn/src/x86/matrix_mul/int8/kernel_sse_4x8x2.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <immintrin.h>
#ifdef WIN32
#include <avx2intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <smmintrin.h>
#endif
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "src/common/utils.h"
#include "src/x86/matrix_mul/common/common.h"

namespace megdnn {
namespace x86 {

namespace matmul_sse_4x8x2 {

template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
void store_overflow(void* ptr, __m128i a);

template <>
void store_overflow<int16_t>(void* ptr, __m128i a) {
    a = _mm_shufflelo_epi16(a, 0x08);
    a = _mm_shufflehi_epi16(a, 0x08);
    a = _mm_shuffle_epi32(a, 0x08);
    _mm_storel_epi64((__m128i*)ptr, a);
}
template <>
void store_overflow<int32_t>(void* ptr, __m128i a) {
    _mm_storeu_si128((__m128i*)(ptr), a);
}
template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
void store_overflow(void* ptr, __m128i a, int remain);

template <>
void store_overflow<int16_t>(void* ptr, __m128i a, int remain) {
    __m128i mask = _mm_continue_mask(remain * sizeof(int16_t));
    a = _mm_shufflelo_epi16(a, 0x08);
    a = _mm_shufflehi_epi16(a, 0x08);
    a = _mm_shuffle_epi32(a, 0x08);
    _mm_maskmoveu_si128(a, mask, reinterpret_cast<char*>(ptr));
}
template <>
void store_overflow<int32_t>(void* ptr, __m128i a, int remain) {
    __m128i mask = _mm_continue_mask(remain * sizeof(int32_t));
    _mm_maskmoveu_si128(a, mask, reinterpret_cast<char*>(ptr));
}

template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void kern_gemm_s8s8s32_sse_4x8x2(const int16_t* pack_a_ptr,
                                               const int8_t* pack_b_ptr,
                                               CType* c_ptr, const int ldc,
                                               const int k) {
    constexpr int k_step = 2;

    __m128i a_vec[2];
    __m128i b_vec[2];
    __m128i c_vec[4 * 2];
    __m128i c_temp[4];

    b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm_setzero_si128();
    c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm_setzero_si128();
    c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm_setzero_si128();
    c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm_setzero_si128();
    c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm_setzero_si128();
    c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm_setzero_si128();
    c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm_setzero_si128();
    c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm_setzero_si128();
    c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 16;

    for (int iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 16;
    }
    store_overflow<CType>(c_ptr, c_vec[0]);
    store_overflow<CType>(c_ptr + 4, c_vec[1]);
    store_overflow<CType>(c_ptr + ldc, c_vec[2]);
    store_overflow<CType>(c_ptr + ldc + 4, c_vec[3]);
    store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
    store_overflow<CType>(c_ptr + 2 * ldc + 4, c_vec[5]);
    store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6]);
    store_overflow<CType>(c_ptr + 3 * ldc + 4, c_vec[7]);
}

template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void kern_gemm_s8s8s32_sse_4x8x2_remain_m(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const int ldc, const int k, const int remain_m) {
    constexpr int k_step = 2;

    __m128i a_vec[2];
    __m128i b_vec[2];
    __m128i c_vec[4 * 2];
    __m128i c_temp[4];

    b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm_setzero_si128();
    c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm_setzero_si128();
    c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm_setzero_si128();
    c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm_setzero_si128();
    c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm_setzero_si128();
    c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm_setzero_si128();
    c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm_setzero_si128();
    c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm_setzero_si128();
    c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 16;

    for (int iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 16;
    }

    store_overflow<CType>(c_ptr, c_vec[0]);
    store_overflow<CType>(c_ptr + 4, c_vec[1]);
    switch (remain_m) {
        case 2:
            store_overflow<CType>(c_ptr + ldc, c_vec[2]);
            store_overflow<CType>(c_ptr + ldc + 4, c_vec[3]);
            break;
        case 3:
            store_overflow<CType>(c_ptr + ldc, c_vec[2]);
            store_overflow<CType>(c_ptr + ldc + 4, c_vec[3]);
            store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
            store_overflow<CType>(c_ptr + 2 * ldc + 4, c_vec[5]);
            break;
        case 4:
            store_overflow<CType>(c_ptr + ldc, c_vec[2]);
            store_overflow<CType>(c_ptr + ldc + 4, c_vec[3]);
            store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
            store_overflow<CType>(c_ptr + 2 * ldc + 4, c_vec[5]);
            store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6]);
            store_overflow<CType>(c_ptr + 3 * ldc + 4, c_vec[7]);
        default:
            break;
    }
}

template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void kern_gemm_s8s8s32_sse_4x8x2_remain_n(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const int ldc, const int k, int remain_n) {
    constexpr int k_step = 2;

    __m128i a_vec[2];
    __m128i b_vec[2];
    __m128i c_vec[4 * 2];
    __m128i c_temp[4];

    b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm_setzero_si128();
    c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm_setzero_si128();
    c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm_setzero_si128();
    c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm_setzero_si128();
    c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm_setzero_si128();
    c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm_setzero_si128();
    c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm_setzero_si128();
    c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm_setzero_si128();
    c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 16;

    for (int iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 16;
    }

    if (remain_n >= 4) {
        store_overflow<CType>(c_ptr, c_vec[0]);
        store_overflow<CType>(c_ptr + ldc, c_vec[2]);
        store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
        store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6]);
        c_ptr += 4;
        remain_n -= 4;
        c_vec[0] = c_vec[1];
        c_vec[2] = c_vec[3];
        c_vec[4] = c_vec[5];
        c_vec[6] = c_vec[7];
    }
    store_overflow<CType>(c_ptr, c_vec[0], remain_n);
    store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
    store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4], remain_n);
    store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6], remain_n);
}

template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void kern_gemm_s8s8s32_sse_4x8x2_remain_m_n(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const int ldc, const int k, int remain_m, int remain_n) {
    constexpr int k_step = 2;

    __m128i a_vec[2];
    __m128i b_vec[2];
    __m128i c_vec[4 * 2];
    __m128i c_temp[4];

    b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm_setzero_si128();
    c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm_setzero_si128();
    c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm_setzero_si128();
    c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm_setzero_si128();
    c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm_setzero_si128();
    c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm_setzero_si128();
    c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm_setzero_si128();
    c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm_setzero_si128();
    c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 16;

    for (int iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm_cvtepi8_epi16_from_ptr(pack_b_ptr + 8);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 16;
    }
    int index_array[4]{0, 2, 4, 6};
    if (remain_n >= 4) {
        for (int m = 0; m < remain_m; ++m) {
            store_overflow<CType>(c_ptr + m * ldc, c_vec[index_array[m]]);
        }
        c_ptr += 4;
        remain_n -= 4;
        c_vec[0] = c_vec[1];
        c_vec[2] = c_vec[3];
        c_vec[4] = c_vec[5];
        c_vec[6] = c_vec[7];
    }
    for (int m = 0; m < remain_m; ++m) {
        store_overflow<CType>(c_ptr + m * ldc, c_vec[index_array[m]], remain_n);
    }
}

static inline void gemm_s8s8s32_sse_4x8x2_pack_an(dt_int16* out,
                                                  const dt_int8* in, int ldin,
                                                  int m_start, int m_max,
                                                  int k_start, int k_max) {
    constexpr int tile_m = 4;
    constexpr int tile_k_step = 8;
    constexpr int tile_k = 2;
    constexpr int tile_len = tile_m * tile_k_step;
    const int k_size = k_max - k_start;
    const int m_end = (m_max - m_start) / tile_m * tile_m + m_start;
    const int m_remain = m_max - m_end;
    for (int m = m_start; m < m_end; m += tile_m) {
        const dt_int8* in0 = in + m * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        const dt_int8* in2 = in1 + ldin;
        const dt_int8* in3 = in2 + ldin;
        int remain_k = k_size;
        for (; remain_k >= tile_k_step; remain_k -= tile_k_step) {
            transpose_4x8_k2_int8_to_int16(in0, in1, in2, in3, out);
            out += tile_len;
            in0 += tile_k_step;
            in1 += tile_k_step;
            in2 += tile_k_step;
            in3 += tile_k_step;
        }

        if (remain_k > 0) {
            transpose_4xk_int8_to_int16_pad(in0, in1, in2, in3, out, remain_k);
            out += tile_m * round_up(remain_k, tile_k);
        }
    }
    if (m_remain > 0) {
        dt_int8 zerobuff[tile_k_step];
        std::memset(zerobuff, 0, sizeof(int8_t) * tile_k_step);
        const dt_int8* in0 = in + m_end * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        const dt_int8* in2 = in1 + ldin;
        const dt_int8* in3 = &zerobuff[0];
        int in1_step = tile_k_step;
        int in2_step = tile_k_step;
        if (m_remain < 3) {
            in2 = &zerobuff[0];
            in2_step = 0;
        }
        if (m_remain < 2) {
            in1 = &zerobuff[0];
            in1_step = 0;
        }
        int remain_k = k_size;
        for (; remain_k >= tile_k_step; remain_k -= tile_k_step) {
            transpose_4x8_k2_int8_to_int16(in0, in1, in2, in3, out);
            out += tile_len;
            in0 += tile_k_step;
            in1 += in1_step;
            in2 += in2_step;
        }
        if (remain_k > 0) {
            transpose_4xk_int8_to_int16_pad(in0, in1, in2, in3, out, remain_k);
            out += tile_m * round_up(remain_k, tile_k);
            in0 += tile_k_step;
            in1 += in1_step;
            in2 += in2_step;
        }
    }
}

static inline void gemm_s8s8s32_sse_4x8x2_pack_bn(dt_int8* out,
                                                  const dt_int8* in, int ldin,
                                                  int n_start, int n_max,
                                                  int k_start, int k_max) {
    constexpr int tile_n = 8;
    constexpr int tile_k = 2;
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
            const dt_int8* inptr_0 = in + k * ldin + n;
            const dt_int8* inptr_1 = inptr_0 + ldin;
            transpose_2x8_no_inc(inptr_0, inptr_1, outptr);
            outptr += pack_line_len;
        }
        if (n_end < n_max) {
            naive_transpose_kn_pad(outptr, in + k * ldin + n_end, ldin, tile_k,
                                   n_remain, tile_k, tile_n);
        }
        out += tile_len;
    }
    if (k_remain > 0) {
        int8_t* outptr = out;
        dt_int8 zerobuff[tile_n];
        std::memset(zerobuff, 0, sizeof(int8_t) * tile_n);
        for (int n = n_start; n < n_end; n += tile_n) {
            const dt_int8* inptr_0 = in + k * ldin + n;
            const dt_int8* inptr_1 = &zerobuff[0];
            transpose_2x8_no_inc(inptr_0, inptr_1, outptr);
            outptr += pack_line_len;
        }
        if (n_end < n_max) {
            naive_transpose_kn_pad(outptr, in + k * ldin + n_end, ldin,
                                   k_remain, n_remain, tile_k, tile_n);
        }
    }
}

static inline void gemm_s8s8s32_sse_4x8x2_pack_bt(dt_int8* out,
                                                  const dt_int8* in, int ldin,
                                                  int n_start, int n_max,
                                                  int k_start, int k_max) {
    constexpr int tile_n = 8;
    constexpr int tile_k = 2;
    constexpr int tile_k_step = 16;
    const int k_size = k_max - k_start;
    const int k_end = k_size / tile_k_step * tile_k_step + k_start;
    const int k_remain = k_max - k_end;
    const int n_size = n_max - n_start;
    const int n_end = n_size / tile_n * tile_n + n_start;
    const int n_remain = n_max - n_end;
    for (int n = n_start; n < n_end; n += tile_n) {
        const dt_int8* in0 = in + n * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        const dt_int8* in2 = in1 + ldin;
        const dt_int8* in3 = in2 + ldin;
        const dt_int8* in4 = in3 + ldin;
        const dt_int8* in5 = in4 + ldin;
        const dt_int8* in6 = in5 + ldin;
        const dt_int8* in7 = in6 + ldin;
        for (int k = k_start; k < k_end; k += tile_k_step) {
            transpose_8x16_k2(out, in0, in1, in2, in3, in4, in5, in6, in7);
            in0 += tile_k_step;
            in1 += tile_k_step;
            in2 += tile_k_step;
            in3 += tile_k_step;
            in4 += tile_k_step;
            in5 += tile_k_step;
            in6 += tile_k_step;
            in7 += tile_k_step;
            out += tile_n * tile_k_step;
        }
        naive_transpose_8xk_k2(out, in0, in1, in2, in3, in4, in5, in6, in7,
                               k_remain);
        out += tile_n * round_up(k_remain, tile_k);
    }
    if (n_remain > 0) {
        const dt_int8* in0 = in + n_end * ldin + k_start;
        naive_transpose_nk_k2(out, in0, ldin, n_remain, k_size, tile_n);
    }
}

static inline void gemm_s8s8s32_sse_4x8x2_pack_at(dt_int16* out,
                                                  const dt_int8* in, int ldin,
                                                  int m_start, int m_max,
                                                  int k_start, int k_max) {
    constexpr int tile_m = 8;
    constexpr int tile_m_step = 4;
    constexpr int tile_k = 2;

    const int k_size = k_max - k_start;
    const int k_end = k_size / tile_k * tile_k + k_start;
    const int k_remain = k_max - k_end;
    const int m_size = m_max - m_start;
    const int m_end = m_size / tile_m * tile_m + m_start;

    const int pack_line_len = round_up(k_size, tile_k) * tile_m_step;
    int k = k_start;
    for (; k < k_end; k += tile_k) {
        dt_int16* outptr = out;
        for (int m = m_start; m < m_end; m += tile_m) {
            const dt_int8* inptr_0 = in + k * ldin + m;
            const dt_int8* inptr_1 = inptr_0 + ldin;
            transpose_km_2x8_k2_tile4_int8_to_int16(inptr_0, inptr_1, outptr,
                                                    pack_line_len);
            outptr += (tile_m / tile_m_step) * pack_line_len;
        }
        if (m_end < m_max) {
            for (int m = m_end; m < m_max; m += tile_m_step) {
                const int m_remain =
                        m_max - m >= tile_m_step ? tile_m_step : m_max - m;
                naive_transpose_kn_pad(outptr, in + k * ldin + m, ldin, tile_k,
                                       m_remain, tile_k, tile_m_step);
                outptr += pack_line_len;
            }
        }
        out += tile_m_step * tile_k;
    }
    if (k_remain > 0) {
        dt_int16* outptr = out;
        dt_int8 zerobuff[tile_m];
        std::memset(zerobuff, 0, sizeof(int8_t) * tile_m);
        for (int n = m_start; n < m_end; n += tile_m) {
            const dt_int8* inptr_0 = in + k * ldin + n;
            const dt_int8* inptr_1 = &zerobuff[0];
            transpose_km_2x8_k2_tile4_int8_to_int16(inptr_0, inptr_1, outptr,
                                                    pack_line_len);
            outptr += (tile_m / tile_m_step) * pack_line_len;
        }
        if (m_end < m_max) {
            for (int m = m_end; m < m_max; m += tile_m_step) {
                const int m_remain =
                        m_max - m >= tile_m_step ? tile_m_step : m_max - m;
                naive_transpose_kn_pad(outptr, in + k * ldin + m, ldin,
                                       k_remain, m_remain, tile_k, tile_m_step);
                outptr += pack_line_len;
            }
        }
    }
}

}  // namespace matmul_sse_4x8x2
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen