/**
 * \file dnn/src/x86/conv_bias/int8/common_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include <immintrin.h>
#include "megdnn/arch.h"
#include "src/common/unroll_macro.h"
#include "src/x86/conv_bias/int8/chanwise_helper.h"
#ifdef WIN32
#include <smmintrin.h>
#endif

namespace megdnn {
namespace x86 {

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline __v8si _m256_continue_mask_v8si(const int& x) {
    static __v8si map[8] = {
            {0, 0, 0, 0, 0, 0, 0, 0},       {-1, 0, 0, 0, 0, 0, 0, 0},
            {-1, -1, 0, 0, 0, 0, 0, 0},     {-1, -1, -1, 0, 0, 0, 0, 0},
            {-1, -1, -1, -1, 0, 0, 0, 0},   {-1, -1, -1, -1, -1, 0, 0, 0},
            {-1, -1, -1, -1, -1, -1, 0, 0}, {-1, -1, -1, -1, -1, -1, -1, 0}};
    return map[x];
}

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline __m256i _m256_continue_mask(const int& x) {
    return (__m256i)_m256_continue_mask_v8si(x);
}

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline __m256i _mm256_cvtepi8_epi16_from_ptr(const int8_t* ptr) {
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)ptr));
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline __m128i _mm_cvtepi8_epi16_from_ptr(const int8_t* ptr) {
    return _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)ptr));
}

static inline void transpose_4x2xn_int8_int16(
        const int8_t* inptr0_0, const int8_t* inptr0_1, const int8_t* inptr1_0,
        const int8_t* inptr1_1, const int8_t* inptr2_0, const int8_t* inptr2_1,
        const int8_t* inptr3_0, const int8_t* inptr3_1, int16_t* out_ptr,
        int length) {
    for (int i = 0; i < length; i++) {
        *out_ptr++ = (int16_t)(*inptr0_0++);
        *out_ptr++ = (int16_t)(*inptr0_1++);
        *out_ptr++ = (int16_t)(*inptr1_0++);
        *out_ptr++ = (int16_t)(*inptr1_1++);
        *out_ptr++ = (int16_t)(*inptr2_0++);
        *out_ptr++ = (int16_t)(*inptr2_1++);
        *out_ptr++ = (int16_t)(*inptr3_0++);
        *out_ptr++ = (int16_t)(*inptr3_1++);
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void transpose_4x2x8_int8_int16(
        const int8_t* inptr0_0, const int8_t* inptr0_1, const int8_t* inptr1_0,
        const int8_t* inptr1_1, const int8_t* inptr2_0, const int8_t* inptr2_1,
        const int8_t* inptr3_0, const int8_t* inptr3_1, int16_t* out_ptr) {
#define cb(iter, a...)                                                  \
    __m128i r##iter##_0 = _mm_cvtepi8_epi16_from_ptr(inptr##iter##_0);  \
    __m128i r##iter##_1 = _mm_cvtepi8_epi16_from_ptr(inptr##iter##_1);  \
    __m128i r##iter##_l = _mm_unpacklo_epi16(r##iter##_0, r##iter##_1); \
    __m128i r##iter##_h = _mm_unpackhi_epi16(r##iter##_0, r##iter##_1);
    UNROLL_CALL_NOWRAPPER(4, cb)
#undef cb
    __m128i ab_01 = _mm_unpacklo_epi32(r0_l, r1_l);
    __m128i ab_23 = _mm_unpackhi_epi32(r0_l, r1_l);
    __m128i ab_45 = _mm_unpacklo_epi32(r0_h, r1_h);
    __m128i ab_67 = _mm_unpackhi_epi32(r0_h, r1_h);
    __m128i cd_01 = _mm_unpacklo_epi32(r2_l, r3_l);
    __m128i cd_23 = _mm_unpackhi_epi32(r2_l, r3_l);
    __m128i cd_45 = _mm_unpacklo_epi32(r2_h, r3_h);
    __m128i cd_67 = _mm_unpackhi_epi32(r2_h, r3_h);

    __m128i abcd_0 = _mm_unpacklo_epi64(ab_01, cd_01);
    __m128i abcd_1 = _mm_unpackhi_epi64(ab_01, cd_01);
    __m128i abcd_2 = _mm_unpacklo_epi64(ab_23, cd_23);
    __m128i abcd_3 = _mm_unpackhi_epi64(ab_23, cd_23);
    __m128i abcd_4 = _mm_unpacklo_epi64(ab_45, cd_45);
    __m128i abcd_5 = _mm_unpackhi_epi64(ab_45, cd_45);
    __m128i abcd_6 = _mm_unpacklo_epi64(ab_67, cd_67);
    __m128i abcd_7 = _mm_unpackhi_epi64(ab_67, cd_67);

    _mm_storeu_si128((__m128i*)(out_ptr + 0 * 8), abcd_0);
    _mm_storeu_si128((__m128i*)(out_ptr + 1 * 8), abcd_1);
    _mm_storeu_si128((__m128i*)(out_ptr + 2 * 8), abcd_2);
    _mm_storeu_si128((__m128i*)(out_ptr + 3 * 8), abcd_3);
    _mm_storeu_si128((__m128i*)(out_ptr + 4 * 8), abcd_4);
    _mm_storeu_si128((__m128i*)(out_ptr + 5 * 8), abcd_5);
    _mm_storeu_si128((__m128i*)(out_ptr + 6 * 8), abcd_6);
    _mm_storeu_si128((__m128i*)(out_ptr + 7 * 8), abcd_7);
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void transpose_2x16_int8(const int8_t* inptr0,
                                       const int8_t* inptr1, int8_t* out_ptr) {
    __m128i r0 = _mm_loadu_si128((__m128i*)inptr0);
    __m128i r1 = _mm_loadu_si128((__m128i*)inptr1);
    __m128i r01l = _mm_unpacklo_epi8(r0, r1);
    __m128i r01h = _mm_unpackhi_epi8(r0, r1);
    _mm_storeu_si128((__m128i*)out_ptr, r01l);
    _mm_storeu_si128((__m128i*)(out_ptr + 16), r01h);
}

static inline void transpose_2xn_int8(const int8_t* inptr0,
                                      const int8_t* inptr1, int8_t* out_ptr,
                                      int length) {
    for (int i = 0; i < length; i++) {
        *out_ptr++ = *inptr0++;
        *out_ptr++ = *inptr1++;
    }
}

static inline void append_zero_and_inc(int8_t*& out_ptr, int length) {
    memset(out_ptr, 0, sizeof(int8_t) * length);
    out_ptr += length;
}
static inline void append_zero_and_inc(int8_t*& even_out_ptr,
                                       int8_t*& odd_out_ptr, const int length,
                                       const int c_step) {
    int even_length = div_ceil(length, c_step) * c_step;
    int odd_length = length - even_length;
    memset(even_out_ptr, 0, sizeof(int8_t) * even_length);
    memset(odd_out_ptr, 0, sizeof(int8_t) * odd_length);
    even_out_ptr += even_length;
    odd_out_ptr += odd_length;
}
static inline void append_zero(int8_t* out_ptr, int length) {
    memset(out_ptr, 0, sizeof(int8_t) * length);
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void transpose_2x16_int8_odd_even(const int8_t* inptr0,
                                                const int8_t* inptr1,
                                                int8_t* odd_out_ptr,
                                                int8_t* even_out_ptr) {
    const static __m128i shuffle0 =
            _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    __m128i r0 = _mm_loadu_si128((__m128i*)inptr0);
    __m128i r1 = _mm_loadu_si128((__m128i*)inptr1);
    __m128i r01l = _mm_unpacklo_epi8(r0, r1);
    __m128i r01h = _mm_unpackhi_epi8(r0, r1);
    __m128i odd_even_low = _mm_shuffle_epi8(r01l, shuffle0);
    __m128i odd_even_high = _mm_shuffle_epi8(r01h, shuffle0);
    __m128i odd = _mm_unpacklo_epi64(odd_even_low, odd_even_high);
    __m128i even = _mm_unpackhi_epi64(odd_even_low, odd_even_high);
    _mm_storeu_si128((__m128i*)odd_out_ptr, odd);
    _mm_storeu_si128((__m128i*)even_out_ptr, even);
}

static inline void transpose_2xn_int8_odd_even(const int8_t* inptr0,
                                               const int8_t* inptr1,
                                               int8_t* odd_out_ptr,
                                               int8_t* even_out_ptr,
                                               int length) {
    for (int i = 0; i < length; i++) {
        if (i % 2 == 0) {
            *odd_out_ptr++ = *inptr0++;
            *odd_out_ptr++ = *inptr1++;
        } else {
            *even_out_ptr++ = *inptr0++;
            *even_out_ptr++ = *inptr1++;
        }
    }
}

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
