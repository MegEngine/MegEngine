/**
 * \file dnn/src/x86/matrix_mul/int8/kernel_avx2_4x16x2.h
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
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "src/common/utils.h"
#include "src/x86/matrix_mul/common/common.h"

#define DNN_AVX2_TARGET
#if !defined(__clang__)
//! bypass gcc bug https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1642109
#pragma GCC target("avx2")
#else
#undef DNN_AVX2_TARGET
#define DNN_AVX2_TARGET MEGDNN_ATTRIBUTE_TARGET("avx2")
#endif
namespace megdnn {
namespace x86 {

namespace matmul_avx2_4x16x2 {

template <typename CType>
DNN_AVX2_TARGET void store_overflow(void* ptr, __m256i a);

template <>
void store_overflow<int16_t>(void* ptr, __m256i a) {
    static __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    a = _mm256_shufflelo_epi16(a, 0x08);
    a = _mm256_shufflehi_epi16(a, 0x08);
    a = _mm256_permutevar8x32_epi32(a, idx);
    _mm_storeu_si128((__m128i*)ptr, _mm256_extractf128_si256(a, 0));
}

template <>
void store_overflow<int32_t>(void* ptr, __m256i a) {
    _mm256_storeu_si256((__m256i*)(ptr), a);
}

template <typename CType>
DNN_AVX2_TARGET void store_overflow(void* ptr, __m256i a, int remain);

template <>
void store_overflow<int16_t>(void* ptr, __m256i a, int remain) {
    __m128i mask = _mm_continue_mask(remain * sizeof(int16_t));
    static __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    a = _mm256_shufflelo_epi16(a, 0x08);
    a = _mm256_shufflehi_epi16(a, 0x08);
    a = _mm256_permutevar8x32_epi32(a, idx);
    _mm_maskmoveu_si128(_mm256_extractf128_si256(a, 0), mask,
                        reinterpret_cast<char*>(ptr));
}

template <>
void store_overflow<int32_t>(void* ptr, __m256i a, int remain) {
    __m256i mask = _m256_continue_mask(remain);
    _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(ptr), mask, a);
}

template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_avx2_4x16x2(const int16_t* pack_a_ptr,
                                                 const int8_t* pack_b_ptr,
                                                 CType* c_ptr,
                                                 const uint32_t ldc,
                                                 const uint32_t k) {
    constexpr uint32_t k_step = 2;

    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i c_vec[4 * 2];
    __m256i c_temp[4];

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm256_setzero_si256();
    c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm256_setzero_si256();
    c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm256_setzero_si256();
    c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm256_setzero_si256();
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm256_setzero_si256();
    c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm256_setzero_si256();
    c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm256_setzero_si256();
    c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 32;

    for (uint32_t iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 32;
    }

    store_overflow<CType>(c_ptr, c_vec[0]);
    store_overflow<CType>(c_ptr + 8, c_vec[1]);
    store_overflow<CType>(c_ptr + ldc, c_vec[2]);
    store_overflow<CType>(c_ptr + ldc + 8, c_vec[3]);
    store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
    store_overflow<CType>(c_ptr + 2 * ldc + 8, c_vec[5]);
    store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6]);
    store_overflow<CType>(c_ptr + 3 * ldc + 8, c_vec[7]);
}
template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_avx2_4x16x2_n8_remain_n(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const uint32_t ldc, const uint32_t k, const uint32_t remain_n) {
    constexpr uint32_t k_step = 2;

    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i c_vec[4 * 2];
    __m256i c_temp[3];

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_vec[2] = _mm256_setzero_si256();
    c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_vec[4] = _mm256_setzero_si256();
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_vec[6] = _mm256_setzero_si256();
    c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);

    pack_a_ptr += 8;
    pack_b_ptr += 32;

    for (uint32_t iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);

        pack_a_ptr += 8;
        pack_b_ptr += 32;
    }

    store_overflow<CType>(c_ptr, c_vec[0], remain_n);
    store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
    store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4], remain_n);
    store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6], remain_n);
}
template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_avx2_4x16x2_n8_remain_m_n(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const uint32_t ldc, const uint32_t k, const uint32_t remain_m,
        uint32_t remain_n) {
    constexpr uint32_t k_step = 2;

    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i c_vec[4 * 2];
    __m256i c_temp[3];

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_vec[2] = _mm256_setzero_si256();
    c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_vec[4] = _mm256_setzero_si256();
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_vec[6] = _mm256_setzero_si256();
    c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);

    pack_a_ptr += 8;
    pack_b_ptr += 32;

    for (uint32_t iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);

        pack_a_ptr += 8;
        pack_b_ptr += 32;
    }

    store_overflow<CType>(c_ptr, c_vec[0], remain_n);
    switch (remain_m) {
        case 2:
            store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);

            break;
        case 3:
            store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
            store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4], remain_n);
            break;
        case 4:
            store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
            store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4], remain_n);
            store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6], remain_n);
            break;
        default:
            break;
    }
}
template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_avx2_4x16x2_remain_m(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const uint32_t ldc, const uint32_t k, const uint32_t remain_m) {
    constexpr uint32_t k_step = 2;

    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i c_vec[4 * 2];
    __m256i c_temp[4];

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm256_setzero_si256();
    c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm256_setzero_si256();
    c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm256_setzero_si256();
    c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm256_setzero_si256();
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm256_setzero_si256();
    c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm256_setzero_si256();
    c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm256_setzero_si256();
    c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 32;

    for (uint32_t iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 32;
    }

    store_overflow<CType>(c_ptr, c_vec[0]);
    store_overflow<CType>(c_ptr + 8, c_vec[1]);

    switch (remain_m) {
        case 2:
            store_overflow<CType>(c_ptr + ldc, c_vec[2]);
            store_overflow<CType>(c_ptr + ldc + 8, c_vec[3]);
            break;
        case 3:
            store_overflow<CType>(c_ptr + ldc, c_vec[2]);
            store_overflow<CType>(c_ptr + ldc + 8, c_vec[3]);
            store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
            store_overflow<CType>(c_ptr + 2 * ldc + 8, c_vec[5]);
            break;
        case 4:
            store_overflow<CType>(c_ptr + ldc, c_vec[2]);
            store_overflow<CType>(c_ptr + ldc + 8, c_vec[3]);
            store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
            store_overflow<CType>(c_ptr + 2 * ldc + 8, c_vec[5]);
            store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6]);
            store_overflow<CType>(c_ptr + 3 * ldc + 8, c_vec[7]);
        default:
            break;
    }
}
template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_avx2_4x16x2_remain_n(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const uint32_t ldc, const uint32_t k, uint32_t remain_n) {
    constexpr uint32_t k_step = 2;

    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i c_vec[4 * 2];
    __m256i c_temp[4];

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm256_setzero_si256();
    c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm256_setzero_si256();
    c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm256_setzero_si256();
    c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm256_setzero_si256();
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm256_setzero_si256();
    c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm256_setzero_si256();
    c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm256_setzero_si256();
    c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 32;

    for (uint32_t iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 32;
    }

    if (remain_n >= 8) {
        store_overflow<CType>(c_ptr, c_vec[0]);
        store_overflow<CType>(c_ptr + ldc, c_vec[2]);
        store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
        store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6]);
        remain_n -= 8;
        if (remain_n > 0) {
            store_overflow<CType>(c_ptr + 8, c_vec[1], remain_n);
            store_overflow<CType>(c_ptr + ldc + 8, c_vec[3], remain_n);
            store_overflow<CType>(c_ptr + 2 * ldc + 8, c_vec[5], remain_n);
            store_overflow<CType>(c_ptr + 3 * ldc + 8, c_vec[7], remain_n);
        }
    } else {
        store_overflow<CType>(c_ptr, c_vec[0], remain_n);
        store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
        store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4], remain_n);
        store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6], remain_n);
    }
}
template <typename CType>
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_gemm_s8s8s32_avx2_4x16x2_remain_m_n(
        const int16_t* pack_a_ptr, const int8_t* pack_b_ptr, CType* c_ptr,
        const uint32_t ldc, const uint32_t k, const uint32_t remain_m,
        uint32_t remain_n) {
    constexpr uint32_t k_step = 2;

    __m256i a_vec[2];
    __m256i b_vec[2];
    __m256i c_vec[4 * 2];
    __m256i c_temp[4];

    b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
    b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[0] = _mm256_setzero_si256();
    c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
    c_vec[1] = _mm256_setzero_si256();
    c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[2] = _mm256_setzero_si256();
    c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
    c_vec[3] = _mm256_setzero_si256();
    c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

    a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
    a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
    c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
    c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
    c_vec[4] = _mm256_setzero_si256();
    c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
    c_vec[5] = _mm256_setzero_si256();
    c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

    c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
    c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
    c_vec[6] = _mm256_setzero_si256();
    c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
    c_vec[7] = _mm256_setzero_si256();
    c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

    pack_a_ptr += 8;
    pack_b_ptr += 32;

    for (uint32_t iter_k = 2; iter_k < k; iter_k += k_step) {
        b_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr);
        b_vec[1] = _mm256_cvtepi8_epi16_from_ptr(pack_b_ptr + 16);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 2));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[0] = _mm256_add_epi32(c_vec[0], c_temp[0]);
        c_vec[1] = _mm256_add_epi32(c_vec[1], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[2] = _mm256_add_epi32(c_vec[2], c_temp[2]);
        c_vec[3] = _mm256_add_epi32(c_vec[3], c_temp[3]);

        a_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 4));
        a_vec[1] = _mm256_set1_epi32(*(int32_t*)(pack_a_ptr + 6));
        c_temp[0] = _mm256_madd_epi16(a_vec[0], b_vec[0]);
        c_temp[1] = _mm256_madd_epi16(a_vec[0], b_vec[1]);
        c_vec[4] = _mm256_add_epi32(c_vec[4], c_temp[0]);
        c_vec[5] = _mm256_add_epi32(c_vec[5], c_temp[1]);

        c_temp[2] = _mm256_madd_epi16(a_vec[1], b_vec[0]);
        c_temp[3] = _mm256_madd_epi16(a_vec[1], b_vec[1]);
        c_vec[6] = _mm256_add_epi32(c_vec[6], c_temp[2]);
        c_vec[7] = _mm256_add_epi32(c_vec[7], c_temp[3]);

        pack_a_ptr += 8;
        pack_b_ptr += 32;
    }

    if (remain_n >= 8) {
        store_overflow<CType>(c_ptr, c_vec[0]);
        switch (remain_m) {
            case 2:
                store_overflow<CType>(c_ptr + ldc, c_vec[2]);
                break;
            case 3:
                store_overflow<CType>(c_ptr + ldc, c_vec[2]);
                store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
                break;
            case 4:
                store_overflow<CType>(c_ptr + ldc, c_vec[2]);
                store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4]);
                store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6]);
                break;
            default:
                break;
        }

        remain_n -= 8;
        if (remain_n > 0) {
            store_overflow<CType>(c_ptr + 8, c_vec[1], remain_n);
            switch (remain_m) {
                case 2:
                    store_overflow<CType>(c_ptr + ldc + 8, c_vec[3], remain_n);
                    break;
                case 3:
                    store_overflow<CType>(c_ptr + ldc + 8, c_vec[3], remain_n);
                    store_overflow<CType>(c_ptr + 2 * ldc + 8, c_vec[5],
                                          remain_n);
                    break;
                case 4:
                    store_overflow<CType>(c_ptr + ldc + 8, c_vec[3], remain_n);
                    store_overflow<CType>(c_ptr + 2 * ldc + 8, c_vec[5],
                                          remain_n);
                    store_overflow<CType>(c_ptr + 3 * ldc + 8, c_vec[7],
                                          remain_n);
                    break;
                default:
                    break;
            }
        }
    } else {
        store_overflow<CType>(c_ptr, c_vec[0], remain_n);
        switch (remain_m) {
            case 2:
                store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
                break;
            case 3:
                store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
                store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4], remain_n);
                break;
            case 4:
                store_overflow<CType>(c_ptr + ldc, c_vec[2], remain_n);
                store_overflow<CType>(c_ptr + 2 * ldc, c_vec[4], remain_n);
                store_overflow<CType>(c_ptr + 3 * ldc, c_vec[6], remain_n);
                break;
            default:
                break;
        }
    }
}

static inline void gemm_s8s8s32_avx2_4x16x2_pack_an(dt_int16* out,
                                                    const dt_int8* in, int ldin,
                                                    int m_start, int m_max,
                                                    int k_start, int k_max) {
    constexpr int tile_m = 4;

    constexpr int tile_k = 16;
    constexpr int tile_k_step = 2;
    constexpr int tile_len = tile_m * tile_k;
    const int k_size = k_max - k_start;
    const int m_end = (m_max - m_start) / tile_m * tile_m + m_start;
    const int m_remain = m_max - m_end;
    for (int m = m_start; m < m_end; m += tile_m) {
        const dt_int8* in0 = in + m * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        const dt_int8* in2 = in1 + ldin;
        const dt_int8* in3 = in2 + ldin;
        int remain_k = k_size;
        for (; remain_k >= tile_k; remain_k -= tile_k) {
            transpose_4x16_k2_int8_to_int16(in0, in1, in2, in3, out);
            out += tile_len;
            in0 += tile_k;
            in1 += tile_k;
            in2 += tile_k;
            in3 += tile_k;
        }

        if (remain_k > 0) {
            transpose_4xk_int8_to_int16_pad(in0, in1, in2, in3, out, remain_k);
            out += tile_m * round_up(remain_k, tile_k_step);
            in0 += tile_k;
            in1 += tile_k;
            in2 += tile_k;
            in3 += tile_k;
        }
    }
    if (m_remain > 0) {
        dt_int8 zerobuff[16];
        std::memset(zerobuff, 0, sizeof(int8_t) * 16);
        const dt_int8* in0 = in + m_end * ldin + k_start;
        const dt_int8* in1 = in0 + ldin;
        const dt_int8* in2 = in1 + ldin;
        const dt_int8* in3 = &zerobuff[0];
        int in1_step = tile_k;
        int in2_step = tile_k;
        if (m_remain < 3) {
            in2 = &zerobuff[0];
            in2_step = 0;
        }
        if (m_remain < 2) {
            in1 = &zerobuff[0];
            in1_step = 0;
        }
        int remain_k = k_size;
        for (; remain_k >= tile_k; remain_k -= tile_k) {
            transpose_4x16_k2_int8_to_int16(in0, in1, in2, in3, out);
            out += tile_len;
            in0 += tile_k;
            in1 += in1_step;
            in2 += in2_step;
        }
        if (remain_k > 0) {
            transpose_4xk_int8_to_int16_pad(in0, in1, in2, in3, out, remain_k);
            out += tile_m * round_up(remain_k, tile_k_step);
            in0 += tile_k;
            in1 += in1_step;
            in2 += in2_step;
        }
    }
}

static inline void gemm_s8s8s32_avx2_4x16x2_pack_bn(dt_int8* out,
                                                    const dt_int8* in, int ldin,
                                                    int n_start, int n_max,
                                                    int k_start, int k_max) {
    constexpr int tile_n = 16;
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
            transpose_2x16_no_inc(inptr_0, inptr_1, outptr);
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
        dt_int8 zerobuff[16];
        std::memset(zerobuff, 0, sizeof(int8_t) * 16);
        for (int n = n_start; n < n_end; n += tile_n) {
            const dt_int8* inptr_0 = in + k * ldin + n;
            const dt_int8* inptr_1 = &zerobuff[0];
            transpose_2x16_no_inc(inptr_0, inptr_1, outptr);
            outptr += pack_line_len;
        }
        if (n_end < n_max) {
            naive_transpose_kn_pad(outptr, in + k * ldin + n_end, ldin,
                                   k_remain, n_remain, tile_k, tile_n);
        }
    }
}

static inline void gemm_s8s8s32_avx2_4x16x2_pack_bt(dt_int8* out,
                                                    const dt_int8* in, int ldin,
                                                    int n_start, int n_max,
                                                    int k_start, int k_max) {
    constexpr int tile_n = 16;
    constexpr int tile_k = 2;
    const int k_size = k_max - k_start;
    const int roundup_k_size = round_up(k_size, tile_k);
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
        const dt_int8* in8 = in7 + ldin;
        const dt_int8* in9 = in8 + ldin;
        const dt_int8* in10 = in9 + ldin;
        const dt_int8* in11 = in10 + ldin;
        const dt_int8* in12 = in11 + ldin;
        const dt_int8* in13 = in12 + ldin;
        const dt_int8* in14 = in13 + ldin;
        const dt_int8* in15 = in14 + ldin;
        naive_transpose_16xk_k2(out, in0, in1, in2, in3, in4, in5, in6, in7,
                                in8, in9, in10, in11, in12, in13, in14, in15,
                                k_size);
        out += tile_n * roundup_k_size;
    }
    if (n_remain > 0) {
        const dt_int8* in0 = in + n_end * ldin + k_start;
        naive_transpose_nk_k2(out, in0, ldin, n_remain, k_size, tile_n);
    }
}

static inline void gemm_s8s8s32_avx2_4x16x2_pack_at(dt_int16* out,
                                                    const dt_int8* in, int ldin,
                                                    int m_start, int m_max,
                                                    int k_start, int k_max) {
    constexpr int tile_m = 16;
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
            transpose_km_2x16_k2_tile4_int8_to_int16(inptr_0, inptr_1, outptr,
                                                     pack_line_len);
            outptr += 4 * pack_line_len;
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
        dt_int8 zerobuff[16];
        std::memset(zerobuff, 0, sizeof(int8_t) * 16);
        for (int n = m_start; n < m_end; n += tile_m) {
            const dt_int8* inptr_0 = in + k * ldin + n;
            const dt_int8* inptr_1 = &zerobuff[0];
            transpose_km_2x16_k2_tile4_int8_to_int16(inptr_0, inptr_1, outptr,
                                                     pack_line_len);
            outptr += 4 * pack_line_len;
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

}  // namespace matmul_avx2_4x16x2

}  // namespace x86
}  // namespace megdnn

#if !defined(__clang__)
#pragma GCC reset_options
#endif
#undef DNN_AVX2_TARGET

// vim: syntax=cpp.doxygen