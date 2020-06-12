/**
 * \file dnn/src/x86/matrix_mul/common/common.h
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
#include <x86intrin.h>

#ifdef WIN32
#include <avx2intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <smmintrin.h>
#endif
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
namespace megdnn {
namespace x86 {

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void _mm256_reduce_two_epi32_to_ptr(__m256i& a, __m256i& b,
                                                  int32_t* output_ptr) {
    __m256i vec_zero = _mm256_setzero_si256();
    a = _mm256_hadd_epi32(a, b);
    a = _mm256_hadd_epi32(a, vec_zero);
    a = _mm256_add_epi32(a, _mm256_permute2x128_si256(a, vec_zero, 0x31));
    output_ptr[0] = _mm256_extract_epi32(a, 0);
    output_ptr[1] = _mm256_extract_epi32(a, 1);
}

template <typename T>
static inline void interleave_helper(const T*& inptr, T*& outptr, int unroll_k,
                                     int ksize, T val = 0) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}

static inline void interleave_helper_add_128(const int8_t*& inptr,
                                             uint8_t*& outptr, int unroll_k,
                                             int ksize, uint8_t val = 0) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = static_cast<uint8_t>((*inptr++) + 128u);
    }
    for (; k < unroll_k; k++) {
        *outptr++ = static_cast<uint8_t>(val + 128u);
    }
}
template <typename T>
static inline void interleave_helper_no_inc(T* outptr, const T* inptr,
                                            int unroll_k, int ksize,
                                            T val = 0) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}
static inline void interleave_2x16_pad(dt_int8* out, const dt_int8* in0,
                                       const dt_int8* in1, int k) {
    interleave_helper_no_inc(out, in0, 16, k);
    interleave_helper_no_inc(out + 16, in1, 16, k);
}
static inline void interleave_4x16_pad(dt_int8* out, const dt_int8* in0,
                                       const dt_int8* in1, const dt_int8* in2,
                                       const dt_int8* in3, int k) {
    interleave_helper_no_inc(out, in0, 16, k);
    interleave_helper_no_inc(out + 16, in1, 16, k);
    interleave_helper_no_inc(out + 32, in2, 16, k);
    interleave_helper_no_inc(out + 48, in3, 16, k);
}
MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void interleave_2x16(dt_int8* out, const dt_int8* in0,
                                   const dt_int8* in1) {
    _mm_storeu_si128((__m128i*)out, _mm_loadu_si128((const __m128i*)in0));
    _mm_storeu_si128((__m128i*)(out + 16),
                     _mm_loadu_si128((const __m128i*)in1));
}
MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void interleave_4x16(dt_int8* out, const dt_int8* in0,
                                   const dt_int8* in1, const dt_int8* in2,
                                   const dt_int8* in3) {
    _mm_storeu_si128((__m128i*)out, _mm_loadu_si128((const __m128i*)in0));
    _mm_storeu_si128((__m128i*)(out + 16),
                     _mm_loadu_si128((const __m128i*)in1));
    _mm_storeu_si128((__m128i*)(out + 32),
                     _mm_loadu_si128((const __m128i*)in2));
    _mm_storeu_si128((__m128i*)(out + 48),
                     _mm_loadu_si128((const __m128i*)in3));
}
template <typename T>
static inline void interleave_4(const T*& inptr0, const T*& inptr1,
                                const T*& inptr2, const T*& inptr3, T*& outptr,
                                int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        interleave_helper(inptr3, outptr, unroll_k, size, val);
    }
}

static inline void interleave_4_add_128(const int8_t*& inptr0,
                                        const int8_t*& inptr1,
                                        const int8_t*& inptr2,
                                        const int8_t*& inptr3, uint8_t*& outptr,
                                        int unroll_k, int ksize,
                                        uint8_t val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper_add_128(inptr0, outptr, unroll_k, size, val);
        interleave_helper_add_128(inptr1, outptr, unroll_k, size, val);
        interleave_helper_add_128(inptr2, outptr, unroll_k, size, val);
        interleave_helper_add_128(inptr3, outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_12(const T* (&input)[12], T*& outptr,
                                 int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        for (int i = 0; i < 12; i++)
            interleave_helper(input[i], outptr, unroll_k, size, val);
    }
}

static inline void interleave_12_add_128(const int8_t* (&input)[12],
                                         uint8_t*& outptr, int unroll_k,
                                         int ksize, uint8_t val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        for (int i = 0; i < 12; i++)
            interleave_helper_add_128(input[i], outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_16(const T* (&input)[16], T*& outptr,
                                 int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        for (int i = 0; i < 16; i++)
            interleave_helper(input[i], outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_32(const T* (&input)[32], T*& outptr,
                                 int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        for (int i = 0; i < 32; i++)
            interleave_helper(input[i], outptr, unroll_k, size, val);
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void interleave_4x4_4_b_add_128(const int8_t*& input0,
                                              const int8_t*& input1,
                                              const int8_t*& input2,
                                              const int8_t*& input3,
                                              uint8_t*& outptr) {
    // int8 trick: add 128 means add b1000 0000, it is same to -128
    __m128i const_128 = _mm_set1_epi8(-128);
    __m128i R0 = _mm_loadu_si128((__m128i*)input0);  //    A3 A2 A1 A0
    __m128i R1 = _mm_loadu_si128((__m128i*)input1);  //    B3 B2 B1 B0
    __m128i R2 = _mm_loadu_si128((__m128i*)input2);  //    C3 C2 C1 C0
    __m128i R3 = _mm_loadu_si128((__m128i*)input3);  //    D3 D2 D1 D0

    R0 = _mm_add_epi8(R0, const_128);
    R1 = _mm_add_epi8(R1, const_128);
    R2 = _mm_add_epi8(R2, const_128);
    R3 = _mm_add_epi8(R3, const_128);

    __m128i R01L = _mm_unpacklo_epi32(R0, R1);  //    B1 A1 B0 A0
    __m128i R01H = _mm_unpackhi_epi32(R0, R1);  //    B3 A3 B2 A2
    __m128i R23L = _mm_unpacklo_epi32(R2, R3);  //    D1 C1 D0 C0
    __m128i R23H = _mm_unpackhi_epi32(R2, R3);  //    D3 C3 D2 C2

    _mm_storeu_si128((__m128i*)(outptr), _mm_unpacklo_epi64(R01L, R23L));
    _mm_storeu_si128((__m128i*)(outptr + 16), _mm_unpackhi_epi64(R01L, R23L));
    _mm_storeu_si128((__m128i*)(outptr + 32), _mm_unpacklo_epi64(R01H, R23H));
    _mm_storeu_si128((__m128i*)(outptr + 48), _mm_unpackhi_epi64(R01H, R23H));
    input0 += 16;
    input1 += 16;
    input2 += 16;
    input3 += 16;
    outptr += 64;
}

MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void interleave_12x4_4_b_add_128(const int8_t* (&input)[12],
                                               uint8_t*& outptr) {
    __m128i O0[3], O1[3], O2[3], O3[3];
    // int8 trick: add 128 means add b1000 0000, it is same to -128
    __m128i const_128 = _mm_set1_epi8(-128);
    for (int i = 0; i < 3; i++) {
        __m128i R0 = _mm_loadu_si128(
                ((__m128i*)input[i * 4 + 0]));  //    A3 A2 A1 A0
        __m128i R1 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 1]));  //    B3 B2 B1 B0
        __m128i R2 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 2]));  //    C3 C2 C1 C0
        __m128i R3 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 3]));  //    D3 D2 D1 D0

        R0 = _mm_add_epi8(R0, const_128);
        R1 = _mm_add_epi8(R1, const_128);
        R2 = _mm_add_epi8(R2, const_128);
        R3 = _mm_add_epi8(R3, const_128);

        __m128i R01L = _mm_unpacklo_epi32(R0, R1);  //    B1 A1 B0 A0
        __m128i R01H = _mm_unpackhi_epi32(R0, R1);  //    B3 A3 B2 A2
        __m128i R23L = _mm_unpacklo_epi32(R2, R3);  //    D1 C1 D0 C0
        __m128i R23H = _mm_unpackhi_epi32(R2, R3);  //    D3 C3 D2 C2

        O0[i] = _mm_unpacklo_epi64(R01L, R23L);
        O1[i] = _mm_unpackhi_epi64(R01L, R23L);
        O2[i] = _mm_unpacklo_epi64(R01H, R23H);
        O3[i] = _mm_unpackhi_epi64(R01H, R23H);
    }
    for (int i = 0; i < 3; i++) {
        _mm_storeu_si128((__m128i*)outptr, O0[i]);
        outptr += 16;
    }
    for (int i = 0; i < 3; i++) {
        _mm_storeu_si128((__m128i*)outptr, O1[i]);
        outptr += 16;
    }
    for (int i = 0; i < 3; i++) {
        _mm_storeu_si128((__m128i*)outptr, O2[i]);
        outptr += 16;
    }
    for (int i = 0; i < 3; i++) {
        _mm_storeu_si128((__m128i*)outptr, O3[i]);
        outptr += 16;
    }
    for (auto& ptr : input) {
        ptr += 16;
    }
}
template <typename T>
MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void interleave_16x4_4_b(const T* (&input)[16], T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_16x4_4_b only support uint8_t and int8_t");

    __m128i O0[4], O1[4], O2[4], O3[4];
    for (int i = 0; i < 4; i++) {
        __m128i R0 = _mm_loadu_si128(
                ((__m128i*)input[i * 4 + 0]));  //    A3 A2 A1 A0
        __m128i R1 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 1]));      //    B3 B2 B1 B0
        __m128i R01L = _mm_unpacklo_epi32(R0, R1);  //    B1 A1 B0 A0
        __m128i R01H = _mm_unpackhi_epi32(R0, R1);  //    B3 A3 B2 A2

        __m128i R2 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 2]));  //    C3 C2 C1 C0
        __m128i R3 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 3]));      //    D3 D2 D1 D0
        __m128i R23L = _mm_unpacklo_epi32(R2, R3);  //    D1 C1 D0 C0
        __m128i R23H = _mm_unpackhi_epi32(R2, R3);  //    D3 C3 D2 C2

        O0[i] = _mm_unpacklo_epi64(R01L, R23L);
        O1[i] = _mm_unpackhi_epi64(R01L, R23L);
        O2[i] = _mm_unpacklo_epi64(R01H, R23H);
        O3[i] = _mm_unpackhi_epi64(R01H, R23H);
    }
    for (int i = 0; i < 4; i++) {
        _mm_storeu_si128((__m128i*)outptr, O0[i]);
        outptr += 16;
    }
    for (int i = 0; i < 4; i++) {
        _mm_storeu_si128((__m128i*)outptr, O1[i]);
        outptr += 16;
    }
    for (int i = 0; i < 4; i++) {
        _mm_storeu_si128((__m128i*)outptr, O2[i]);
        outptr += 16;
    }
    for (int i = 0; i < 4; i++) {
        _mm_storeu_si128((__m128i*)outptr, O3[i]);
        outptr += 16;
    }
    for (auto& ptr : input) {
        ptr += 16;
    }
}
template <typename T>
MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void interleave_32x4_4_b(const T* (&input)[32], T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_32x4_4_b only support uint8_t and int8_t");

    __m128i O0[8], O1[8], O2[8], O3[8];
    for (int i = 0; i < 8; i++) {
        __m128i R0 = _mm_loadu_si128(
                ((__m128i*)input[i * 4 + 0]));  //    A3 A2 A1 A0
        __m128i R1 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 1]));      //    B3 B2 B1 B0
        __m128i R01L = _mm_unpacklo_epi32(R0, R1);  //    B1 A1 B0 A0
        __m128i R01H = _mm_unpackhi_epi32(R0, R1);  //    B3 A3 B2 A2

        __m128i R2 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 2]));  //    C3 C2 C1 C0
        __m128i R3 = _mm_loadu_si128(
                (__m128i*)(input[i * 4 + 3]));      //    D3 D2 D1 D0
        __m128i R23L = _mm_unpacklo_epi32(R2, R3);  //    D1 C1 D0 C0
        __m128i R23H = _mm_unpackhi_epi32(R2, R3);  //    D3 C3 D2 C2

        O0[i] = _mm_unpacklo_epi64(R01L, R23L);
        O1[i] = _mm_unpackhi_epi64(R01L, R23L);
        O2[i] = _mm_unpacklo_epi64(R01H, R23H);
        O3[i] = _mm_unpackhi_epi64(R01H, R23H);
    }
    for (int i = 0; i < 8; i++) {
        _mm_storeu_si128((__m128i*)outptr, O0[i]);
        outptr += 16;
    }
    for (int i = 0; i < 8; i++) {
        _mm_storeu_si128((__m128i*)outptr, O1[i]);
        outptr += 16;
    }
    for (int i = 0; i < 8; i++) {
        _mm_storeu_si128((__m128i*)outptr, O2[i]);
        outptr += 16;
    }
    for (int i = 0; i < 8; i++) {
        _mm_storeu_si128((__m128i*)outptr, O3[i]);
        outptr += 16;
    }

    for (auto& ptr : input) {
        ptr += 16;
    }
}
static inline void naive_transpose_16xn(
        dt_int8* out, const dt_int8* in0, const dt_int8* in1,
        const dt_int8* in2, const dt_int8* in3, const dt_int8* in4,
        const dt_int8* in5, const dt_int8* in6, const dt_int8* in7,
        const dt_int8* in8, const dt_int8* in9, const dt_int8* in10,
        const dt_int8* in11, const dt_int8* in12, const dt_int8* in13,
        const dt_int8* in14, const dt_int8* in15, int n) {
    for (int i = 0; i < n; ++i) {
#define cb(iter, a...) *out++ = *in##iter++;

        UNROLL_CALL(16, cb);
#undef cb
    }
}
static inline void naive_transpose_nk_k2(dt_int8* out, const dt_int8* in,
                                         int ldin, int n, int k, int n_unroll) {
    constexpr int k_step = 2;
    for (int k_iter = 0; k_iter < k; k_iter += k_step) {
        for (int n_iter = 0; n_iter < n; ++n_iter) {
            *out++ = *(in + n_iter * ldin + k_iter);
            if (k_iter + 1 < k) {
                *out++ = *(in + n_iter * ldin + k_iter + 1);
            } else {
                *out++ = 0;
            }
        }
        for (int n_iter = n; n_iter < n_unroll; ++n_iter) {
            *out++ = 0;
            *out++ = 0;
        }
    }
}
static inline void naive_transpose_16xk_k2(
        dt_int8* out, const dt_int8* in0, const dt_int8* in1,
        const dt_int8* in2, const dt_int8* in3, const dt_int8* in4,
        const dt_int8* in5, const dt_int8* in6, const dt_int8* in7,
        const dt_int8* in8, const dt_int8* in9, const dt_int8* in10,
        const dt_int8* in11, const dt_int8* in12, const dt_int8* in13,
        const dt_int8* in14, const dt_int8* in15, int k_max) {
    constexpr int k_step = 2;
    const int k_end = k_max / k_step * k_step;
    const int k_remain = k_max - k_end;
    for (int k = 0; k < k_end; k += k_step) {
#define cb(iter, a...)    \
    *out++ = *in##iter++; \
    *out++ = *in##iter++;

        UNROLL_CALL(16, cb);
#undef cb
    }
    if (k_remain > 0) {
#define cb(iter, a...)    \
    *out++ = *in##iter++; \
    *out++ = 0;
        UNROLL_CALL(16, cb);
#undef cb
    }
}

static inline void naive_transpose_8xk_k2(
        dt_int8* out, const dt_int8* in0, const dt_int8* in1,
        const dt_int8* in2, const dt_int8* in3, const dt_int8* in4,
        const dt_int8* in5, const dt_int8* in6, const dt_int8* in7, int k_max) {
    constexpr int k_step = 2;
    const int k_end = k_max / k_step * k_step;
    const int k_remain = k_max - k_end;
    for (int k = 0; k < k_end; k += k_step) {
#define cb(iter, a...)    \
    *out++ = *in##iter++; \
    *out++ = *in##iter++;

        UNROLL_CALL(8, cb);
#undef cb
    }
    if (k_remain > 0) {
#define cb(iter, a...)    \
    *out++ = *in##iter++; \
    *out++ = 0;
        UNROLL_CALL(8, cb);
#undef cb
    }
}
static inline void naive_transpose_kn(dt_int8* out, const dt_int8* in, int ldin,
                                      int k, int n) {
    for (int n_iter = 0; n_iter < n; ++n_iter) {
        for (int k_iter = 0; k_iter < k; ++k_iter) {
            *out++ = *(in + k_iter * ldin + n_iter);
        }
    }
}
template <typename OutType>
static inline void naive_transpose_kn_pad(OutType* out, const dt_int8* in,
                                          int ldin, int k, int n, int k_unroll,
                                          int n_unroll, OutType pad = 0) {
    for (int n_iter = 0; n_iter < n_unroll; ++n_iter) {
        for (int k_iter = 0; k_iter < k_unroll; ++k_iter) {
            if (k_iter < k && n_iter < n) {
                *out++ = *(in + k_iter * ldin + n_iter);
            } else {
                *out++ = pad;
            }
        }
    }
}

template <typename T>
static inline void transpose_4(const T*& inptr0, const T*& inptr1,
                               const T*& inptr2, const T*& inptr3, T* outptr,
                               int interleave, int size, T val = 0) {
    megdnn_assert(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
    }
    for (; i < interleave; i++) {
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
    }
}

template <typename T>
static inline void transpose_2_no_inc(const T* inptr0, const T* inptr1,
                                      T* outptr, int interleave, int size,
                                      T val = 0) {
    megdnn_assert(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
    }
    for (; i < interleave; i++) {
        *outptr++ = val;
        *outptr++ = val;
    }
}

static inline void transpose_4_add_128(const int8_t*& inptr0,
                                       const int8_t*& inptr1,
                                       const int8_t*& inptr2,
                                       const int8_t*& inptr3, uint8_t* outptr,
                                       int interleave, int size,
                                       uint8_t val = 0) {
    megdnn_assert(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = static_cast<uint8_t>((*inptr0++) + 128u);
        *outptr++ = static_cast<uint8_t>((*inptr1++) + 128u);
        *outptr++ = static_cast<uint8_t>((*inptr2++) + 128u);
        *outptr++ = static_cast<uint8_t>((*inptr3++) + 128u);
    }
    for (; i < interleave; i++) {
        *outptr++ = static_cast<uint8_t>(val + 128u);
        *outptr++ = static_cast<uint8_t>(val + 128u);
        *outptr++ = static_cast<uint8_t>(val + 128u);
        *outptr++ = static_cast<uint8_t>(val + 128u);
    }
}
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void transpose_2x32_no_inc(const int8_t* inptr0,
                                         const int8_t* inptr1, int8_t* outptr) {
    //    A32 ... A14 A13 A12 A11 A10 A9 A8 A7 A6 A5 A4 A3 A2 A1 A0
    __m256i r0 = _mm256_loadu_si256((__m256i*)(inptr0));
    //    B32 ... B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0
    __m256i r1 = _mm256_loadu_si256((__m256i*)(inptr1));
    //  B23 A23 B22 A22 B21 A21 B20 A20 B19 A19 B18 A18 B17 A17 B16 A16
    //  B7 A7 B6 A6 B5 A5 B4 A4 B3 A3 B2 A2 B1 A1 B0 A0
    __m256i r01l = _mm256_unpacklo_epi8(r0, r1);
    //  B31 A31 B30 A30 B29 A29 B28 A28 B27 A27 B26 A26 B25 A25 B24 A24
    //  B15 A15 B14 A14 B13 A13 B12 A12 B11 A11 B10 A10 B9 A9 B8 A8
    __m256i r01h = _mm256_unpackhi_epi8(r0, r1);

    _mm_storeu_si128((__m128i*)outptr, _mm256_extracti128_si256(r01l, 0));
    _mm_storeu_si128((__m128i*)(outptr + 16),
                     _mm256_extracti128_si256(r01h, 0));
    _mm_storeu_si128((__m128i*)(outptr + 32),
                     _mm256_extracti128_si256(r01l, 1));
    _mm_storeu_si128((__m128i*)(outptr + 48),
                     _mm256_extracti128_si256(r01h, 1));
}

MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void transpose_2x16_no_inc(const int8_t* inptr0,
                                         const int8_t* inptr1, int8_t* outptr) {
    //    A15 A14 A13 A12 A11 A10 A9 A8 A7 A6 A5 A4 A3 A2 A1 A0
    __m128i r0 = _mm_loadu_si128((__m128i*)inptr0);
    //    B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0
    __m128i r1 = _mm_loadu_si128((__m128i*)inptr1);
    //    B3 A3 B2 A2 B1 A1 B0 A0
    __m128i r01l = _mm_unpacklo_epi8(r0, r1);
    //    B15 A15 B14 A14 B13 A13 B12 A12 B11 A11 B10 A10 B9 A9 B8 A8
    __m128i r01h = _mm_unpackhi_epi8(r0, r1);

    _mm_storeu_si128((__m128i*)outptr, r01l);
    _mm_storeu_si128((__m128i*)(outptr + 16), r01h);
}

MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void transpose_2x8_no_inc(const int8_t* inptr0,
                                        const int8_t* inptr1, int8_t* outptr) {
    //    A7 A6 A5 A4 A3 A2 A1 A0
    __m128i r0 = _mm_loadl_epi64((__m128i*)inptr0);
    //    B7 B6 B5 B4 B3 B2 B1 B0
    __m128i r1 = _mm_loadl_epi64((__m128i*)inptr1);
    //    B7 A7 B6 A6 B5 A5 B4 A4 B3 A3 B2 A2 B1 A1 B0 A0
    __m128i r01l = _mm_unpacklo_epi8(r0, r1);

    _mm_storeu_si128((__m128i*)outptr, r01l);
}

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline __m256i _mm256_cvtepi8_epi16_from_ptr(const int8_t* ptr) {
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)ptr));
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline __m128i _mm_cvtepi8_epi16_from_ptr(const int8_t* ptr) {
    return _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*)ptr));
}

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void transpose_2x16_k2_int8_to_int16(const int8_t* inptr0,
                                                   const int8_t* inptr1,
                                                   int16_t* outptr) {
    //    A7 A6 A5 A4 A3 A2 A1 A0
    __m256i r0 = _mm256_cvtepi8_epi16_from_ptr(inptr0);
    //    B7 B6 B5 B4 B3 B2 B1 B0
    __m256i r1 = _mm256_cvtepi8_epi16_from_ptr(inptr1);
    //    B5 A5 B4 A4 B1 A1 B0 A0
    __m256i r01l = _mm256_unpacklo_epi32(r0, r1);
    //    B7 A7 B6 A6 B3 A3 B2 A2
    __m256i r01h = _mm256_unpackhi_epi32(r0, r1);

    _mm_storeu_si128((__m128i*)(outptr + 0), _mm256_extracti128_si256(r01l, 0));
    _mm_storeu_si128((__m128i*)(outptr + 8), _mm256_extracti128_si256(r01h, 0));
    _mm_storeu_si128((__m128i*)(outptr + 16),
                     _mm256_extracti128_si256(r01l, 1));
    _mm_storeu_si128((__m128i*)(outptr + 24),
                     _mm256_extracti128_si256(r01h, 1));
}
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void transpose_km_2x16_k2_tile4_int8_to_int16(
        const int8_t* inptr0, const int8_t* inptr1, int16_t* outptr,
        int tile_step) {
    //    A15 A14 A13 A12 A11 A10 A9 A8 A7 A6 A5 A4 A3 A2 A1 A0
    __m256i r0 = _mm256_cvtepi8_epi16_from_ptr(inptr0);
    //    B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0
    __m256i r1 = _mm256_cvtepi8_epi16_from_ptr(inptr1);
    //    B11 A11 B10 A10 B9 A9 B8 A8 B3 A3 B2 A2 B1 A1 B0 A0
    __m256i r01l = _mm256_unpacklo_epi16(r0, r1);
    //    B15 A15 B14 A14 B13 A13 B12 A12 B7 A7 B6 A6 B5 A5 B4 A4
    __m256i r01h = _mm256_unpackhi_epi16(r0, r1);

    _mm_storeu_si128((__m128i*)(outptr + 0 * tile_step),
                     _mm256_extracti128_si256(r01l, 0));
    _mm_storeu_si128((__m128i*)(outptr + 1 * tile_step),
                     _mm256_extracti128_si256(r01h, 0));
    _mm_storeu_si128((__m128i*)(outptr + 2 * tile_step),
                     _mm256_extracti128_si256(r01l, 1));
    _mm_storeu_si128((__m128i*)(outptr + 3 * tile_step),
                     _mm256_extracti128_si256(r01h, 1));
}
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void transpose_8x16_k2(dt_int8* out, const dt_int8* in0,
                                     const dt_int8* in1, const dt_int8* in2,
                                     const dt_int8* in3, const dt_int8* in4,
                                     const dt_int8* in5, const dt_int8* in6,
                                     const dt_int8* in7) {
    //    A7 A6 A5 A4 A3 A2 A1 A0
    __m128i r0 = _mm_loadu_si128((__m128i*)in0);
    //    B7 B6 B5 B4 B3 B2 B1 B0
    __m128i r1 = _mm_loadu_si128((__m128i*)in1);
    //    C7 C6 C5 C4 C3 C2 C1 C0
    __m128i r2 = _mm_loadu_si128((__m128i*)in2);
    //    D7 D6 D5 D4 D3 D2 D1 D0
    __m128i r3 = _mm_loadu_si128((__m128i*)in3);
    //    E7 E6 E5 E4 E3 E2 E1 E0
    __m128i r4 = _mm_loadu_si128((__m128i*)in4);
    //    F7 F6 F5 F4 F3 F2 F1 F0
    __m128i r5 = _mm_loadu_si128((__m128i*)in5);
    //    G7 G6 G5 G4 G3 G2 G1 G0
    __m128i r6 = _mm_loadu_si128((__m128i*)in6);
    //    H7 H6 H5 H4 H3 H2 H1 H0
    __m128i r7 = _mm_loadu_si128((__m128i*)in7);

    // do 8x8 epi16 transpose
    //    B3 A3 B2 A2 B1 A1 B0 A0
    __m128i rab0123 = _mm_unpacklo_epi16(r0, r1);
    __m128i rab4567 = _mm_unpackhi_epi16(r0, r1);
    __m128i rcd0123 = _mm_unpacklo_epi16(r2, r3);
    __m128i rcd4567 = _mm_unpackhi_epi16(r2, r3);
    __m128i ref0123 = _mm_unpacklo_epi16(r4, r5);
    __m128i ref4567 = _mm_unpackhi_epi16(r4, r5);
    __m128i rgh0123 = _mm_unpacklo_epi16(r6, r7);
    __m128i rgh4567 = _mm_unpackhi_epi16(r6, r7);

    //    D1 C1 B1 A1 D0 C0 B0 A0
    __m128i rabcd01 = _mm_unpacklo_epi32(rab0123, rcd0123);
    __m128i rabcd23 = _mm_unpackhi_epi32(rab0123, rcd0123);
    __m128i rabcd45 = _mm_unpacklo_epi32(rab4567, rcd4567);
    __m128i rabcd67 = _mm_unpackhi_epi32(rab4567, rcd4567);
    __m128i refgh01 = _mm_unpacklo_epi32(ref0123, rgh0123);
    __m128i refgh23 = _mm_unpackhi_epi32(ref0123, rgh0123);
    __m128i refgh45 = _mm_unpacklo_epi32(ref4567, rgh4567);
    __m128i refgh67 = _mm_unpackhi_epi32(ref4567, rgh4567);

    //    H0 G0 F0 E0 D0 C0 B0 A0
    __m128i rabcdefgh0 = _mm_unpacklo_epi64(rabcd01, refgh01);
    __m128i rabcdefgh1 = _mm_unpackhi_epi64(rabcd01, refgh01);
    __m128i rabcdefgh2 = _mm_unpacklo_epi64(rabcd23, refgh23);
    __m128i rabcdefgh3 = _mm_unpackhi_epi64(rabcd23, refgh23);
    __m128i rabcdefgh4 = _mm_unpacklo_epi64(rabcd45, refgh45);
    __m128i rabcdefgh5 = _mm_unpackhi_epi64(rabcd45, refgh45);
    __m128i rabcdefgh6 = _mm_unpacklo_epi64(rabcd67, refgh67);
    __m128i rabcdefgh7 = _mm_unpackhi_epi64(rabcd67, refgh67);

    _mm_storeu_si128((__m128i*)(out + 0 * 16), rabcdefgh0);
    _mm_storeu_si128((__m128i*)(out + 1 * 16), rabcdefgh1);
    _mm_storeu_si128((__m128i*)(out + 2 * 16), rabcdefgh2);
    _mm_storeu_si128((__m128i*)(out + 3 * 16), rabcdefgh3);
    _mm_storeu_si128((__m128i*)(out + 4 * 16), rabcdefgh4);
    _mm_storeu_si128((__m128i*)(out + 5 * 16), rabcdefgh5);
    _mm_storeu_si128((__m128i*)(out + 6 * 16), rabcdefgh6);
    _mm_storeu_si128((__m128i*)(out + 7 * 16), rabcdefgh7);
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void transpose_km_2x8_k2_tile4_int8_to_int16(const int8_t* inptr0,
                                                           const int8_t* inptr1,
                                                           int16_t* outptr,
                                                           int tile_step) {
    //    A7 A6 A5 A4 A3 A2 A1 A0
    __m128i r0 = _mm_cvtepi8_epi16_from_ptr(inptr0);
    //    B7 B6 B5 B4 B3 B2 B1 B0
    __m128i r1 = _mm_cvtepi8_epi16_from_ptr(inptr1);
    //    B3 A3 B2 A2 B1 A1 B0 A0
    __m128i r01l = _mm_unpacklo_epi16(r0, r1);
    //    B7 A7 B6 A6 B5 A5 B4 A4
    __m128i r01h = _mm_unpackhi_epi16(r0, r1);

    _mm_storeu_si128((__m128i*)(outptr + 0 * tile_step), r01l);
    _mm_storeu_si128((__m128i*)(outptr + 1 * tile_step), r01h);
}

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void transpose_4x16_k2_int8_to_int16(const int8_t* inptr0,
                                                   const int8_t* inptr1,
                                                   const int8_t* inptr2,
                                                   const int8_t* inptr3,
                                                   int16_t* outptr) {
    //    A7 A6 A5 A4 A3 A2 A1 A0
    __m256i r0 = _mm256_cvtepi8_epi16_from_ptr(inptr0);
    //    B7 B6 B5 B4 B3 B2 B1 B0
    __m256i r1 = _mm256_cvtepi8_epi16_from_ptr(inptr1);
    //    C7 C6 C5 C4 C3 C2 C1 C0
    __m256i r2 = _mm256_cvtepi8_epi16_from_ptr(inptr2);
    //    D7 D6 D5 D4 D3 D2 D1 D0
    __m256i r3 = _mm256_cvtepi8_epi16_from_ptr(inptr3);

    //    B5 A5 B4 A4 B1 A1 B0 A0
    __m256i r01l = _mm256_unpacklo_epi32(r0, r1);
    //    B7 A7 B6 A6 B3 A3 B2 A2
    __m256i r01h = _mm256_unpackhi_epi32(r0, r1);
    //    D5 C5 D4 C4 D1 C1 D0 C0
    __m256i r23l = _mm256_unpacklo_epi32(r2, r3);
    //    D7 C7 D6 C6 D3 C3 D2 C2
    __m256i r23h = _mm256_unpackhi_epi32(r2, r3);

    //   D4 C4 B4 A4 D0 C0 B0 A0
    __m256i out_0_4 = _mm256_unpacklo_epi64(r01l, r23l);
    __m256i out_1_5 = _mm256_unpackhi_epi64(r01l, r23l);
    __m256i out_2_6 = _mm256_unpacklo_epi64(r01h, r23h);
    __m256i out_3_7 = _mm256_unpackhi_epi64(r01h, r23h);

    _mm_storeu_si128((__m128i*)(outptr + 0),
                     _mm256_extracti128_si256(out_0_4, 0));
    _mm_storeu_si128((__m128i*)(outptr + 8),
                     _mm256_extracti128_si256(out_1_5, 0));
    _mm_storeu_si128((__m128i*)(outptr + 16),
                     _mm256_extracti128_si256(out_2_6, 0));
    _mm_storeu_si128((__m128i*)(outptr + 24),
                     _mm256_extracti128_si256(out_3_7, 0));
    _mm_storeu_si128((__m128i*)(outptr + 32),
                     _mm256_extracti128_si256(out_0_4, 1));
    _mm_storeu_si128((__m128i*)(outptr + 40),
                     _mm256_extracti128_si256(out_1_5, 1));
    _mm_storeu_si128((__m128i*)(outptr + 48),
                     _mm256_extracti128_si256(out_2_6, 1));
    _mm_storeu_si128((__m128i*)(outptr + 56),
                     _mm256_extracti128_si256(out_3_7, 1));
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void transpose_4x8_k2_int8_to_int16(const int8_t* inptr0,
                                                  const int8_t* inptr1,
                                                  const int8_t* inptr2,
                                                  const int8_t* inptr3,
                                                  int16_t* outptr) {
    //    A3 A2 A1 A0
    __m128i r0 = _mm_cvtepi8_epi16_from_ptr(inptr0);
    //    B3 B2 B1 B0
    __m128i r1 = _mm_cvtepi8_epi16_from_ptr(inptr1);
    //    C3 C2 C1 C0
    __m128i r2 = _mm_cvtepi8_epi16_from_ptr(inptr2);
    //    D3 D2 D1 D0
    __m128i r3 = _mm_cvtepi8_epi16_from_ptr(inptr3);

    //    B1 A1 B0 A0
    __m128i r01l = _mm_unpacklo_epi32(r0, r1);
    //    B3 A3 B2 A2
    __m128i r01h = _mm_unpackhi_epi32(r0, r1);
    //    D1 C1 D0 C0
    __m128i r23l = _mm_unpacklo_epi32(r2, r3);
    //    D3 C3 D2 C2
    __m128i r23h = _mm_unpackhi_epi32(r2, r3);

    //   D0 C0 B0 A0
    __m128i out_0_4 = _mm_unpacklo_epi64(r01l, r23l);
    __m128i out_1_5 = _mm_unpackhi_epi64(r01l, r23l);
    __m128i out_2_6 = _mm_unpacklo_epi64(r01h, r23h);
    __m128i out_3_7 = _mm_unpackhi_epi64(r01h, r23h);

    _mm_storeu_si128((__m128i*)(outptr + 0), out_0_4);
    _mm_storeu_si128((__m128i*)(outptr + 8), out_1_5);
    _mm_storeu_si128((__m128i*)(outptr + 16), out_2_6);
    _mm_storeu_si128((__m128i*)(outptr + 24), out_3_7);
}

MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline __v8si _m256_continue_mask_v8si(const int& x) {
    // clang-format off
    static __v8si map[9] = {
            {00, 00, 00, 00, 00, 00, 00, 00}, 
            {-1, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, 00, 00, 00, 00}, 
            {-1, -1, -1, -1, -1, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, 00, 00}, 
            {-1, -1, -1, -1, -1, -1, -1, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1}};
    return map[x];
    // clang-format on
}
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline __m256i _m256_continue_mask(const int& x) {
    return (__m256i)_m256_continue_mask_v8si(x);
}

MEGDNN_ATTRIBUTE_TARGET("sse2")
static inline __m128i _mm_continue_mask(const int& x) {
    static __v16qi map[17] = {
            {00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, 00, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, 00, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, 00, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 00, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 00, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 00, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 00, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 00, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 00},
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    };
    return (__m128i)map[x];
}

MEGDNN_ATTRIBUTE_TARGET("sse2")
static inline void transpose_4xk_int8_to_int16_pad(const int8_t* inptr0,
                                                   const int8_t* inptr1,
                                                   const int8_t* inptr2,
                                                   const int8_t* inptr3,
                                                   int16_t* outptr, int k) {
    int i = 0;
    constexpr int k_step = 2;
    const int k_end = k / k_step * k_step;
    const int k_remain = k - k_end;
    for (; i < k_end; i += k_step) {
        *outptr++ = (int16_t)(*inptr0++);
        *outptr++ = (int16_t)(*inptr0++);
        *outptr++ = (int16_t)(*inptr1++);
        *outptr++ = (int16_t)(*inptr1++);
        *outptr++ = (int16_t)(*inptr2++);
        *outptr++ = (int16_t)(*inptr2++);
        *outptr++ = (int16_t)(*inptr3++);
        *outptr++ = (int16_t)(*inptr3++);
    }
    if (k_remain > 0) {
        *outptr++ = (int16_t)(*inptr0++);
        *outptr++ = 0;
        *outptr++ = (int16_t)(*inptr1++);
        *outptr++ = 0;
        *outptr++ = (int16_t)(*inptr2++);
        *outptr++ = 0;
        *outptr++ = (int16_t)(*inptr3++);
        *outptr++ = 0;
        i += k_step;
    }
}
MEGDNN_ATTRIBUTE_TARGET("sse2")
static inline void transpose_2xk_k2_pad(const int8_t* inptr0,
                                        const int8_t* inptr1, int16_t* outptr,
                                        int k) {
    int i = 0;
    constexpr int k_step = 2;
    const int k_end = k / k_step * k_step;
    const int k_remain = k - k_end;
    for (; i < k_end; i += k_step) {
        *outptr++ = (int16_t)(*inptr0++);
        *outptr++ = (int16_t)(*inptr0++);
        *outptr++ = (int16_t)(*inptr1++);
        *outptr++ = (int16_t)(*inptr1++);
    }
    if (k_remain > 0) {
        *outptr++ = (int16_t)(*inptr0++);
        *outptr++ = 0;
        *outptr++ = (int16_t)(*inptr1++);
        *outptr++ = 0;
        i += k_step;
    }
}
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void transpose_4x32_1_b(const int8_t*& inptr0,
                                      const int8_t*& inptr1,
                                      const int8_t*& inptr2,
                                      const int8_t*& inptr3, int8_t* outptr) {
    //    A32 ... A14 A13 A12 A11 A10 A9 A8 A7 A6 A5 A4 A3 A2 A1 A0
    __m256i R0 = _mm256_loadu_si256((__m256i*)(inptr0));
    //    B32 ... B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0
    __m256i R1 = _mm256_loadu_si256((__m256i*)(inptr1));
    //    C32 ... C15 C14 C13 C12 C11 C10 C9 C8 C7 C6 C5 C4 C3 C2 C1 C0
    __m256i R2 = _mm256_loadu_si256((__m256i*)(inptr2));
    //    D32 ... D15 D14 D13 D12 D11 D10 D9 D8 D7 D6 D5 D4 D3 D2 D1 D0
    __m256i R3 = _mm256_loadu_si256((__m256i*)(inptr3));

    //  B23 A23 B22 A22 B21 A21 B20 A20 B19 A19 B18 A18 B17 A17 B16 A16
    //  B7 A7 B6 A6 B5 A5 B4 A4 B3 A3 B2 A2 B1 A1 B0 A0
    __m256i R01L = _mm256_unpacklo_epi8(R0, R1);
    //  B31 A31 B30 A30 B29 A29 B28 A28 B27 A27 B26 A26 B25 A25 B24 A24
    //  B15 A15 B14 A14 B13 A13 B12 A12 B11 A11 B10 A10 B9 A9 B8 A8
    __m256i R01H = _mm256_unpackhi_epi8(R0, R1);
    //  D23 C23 D22 C22 D21 C21 D20 C20 D19 C19 D18 C18 D17 C17 D16 C16
    //  D7 C7 D6 C6 D5 C5 D4 C4 D3 C3 D2 C2 D1 C1 D0 C0
    __m256i R23L = _mm256_unpacklo_epi8(R2, R3);
    //  D31 C31 D30 C30 D29 C29 D28 C28 D27 C27 D26 C26 D25 C25 D24 C24
    //  D15 C15 D14 C14 D13 C13 D12 C12 D11 C11 D10 C10 D9 C9 D8 C8
    __m256i R23H = _mm256_unpackhi_epi8(R2, R3);

    // D19 C19 B19 A19 ... D16 C16 B16 A16
    // D3 C3 B3 A3 ... D0 C0 B0 A0
    __m256i Out0_3 = _mm256_unpacklo_epi16(R01L, R23L);
    // D23 C23 B23 A23 ... D20 C20 B20 A20
    // D7 C7 B7 A7 ... D4 C4 B4 A4
    __m256i Out4_7 = _mm256_unpackhi_epi16(R01L, R23L);
    // D27 C27 B27 A27 ... D24 C24 B24 A24
    // D11 C11 B11 A11 ... D8 C8 B8 A8
    __m256i Out8_11 = _mm256_unpacklo_epi16(R01H, R23H);
    // D31 C31 B31 A31 ... D28 C28 B28 A28
    // D15 C15 B15 A15 ... D12 C12 B12 A12
    __m256i Out12_15 = _mm256_unpackhi_epi16(R01H, R23H);

    _mm_storeu_si128((__m128i*)outptr, _mm256_extracti128_si256(Out0_3, 0));
    _mm_storeu_si128((__m128i*)(outptr + 16),
                     _mm256_extracti128_si256(Out4_7, 0));
    _mm_storeu_si128((__m128i*)(outptr + 32),
                     _mm256_extracti128_si256(Out8_11, 0));
    _mm_storeu_si128((__m128i*)(outptr + 48),
                     _mm256_extracti128_si256(Out12_15, 0));
    _mm_storeu_si128((__m128i*)(outptr + 64),
                     _mm256_extracti128_si256(Out0_3, 1));
    _mm_storeu_si128((__m128i*)(outptr + 80),
                     _mm256_extracti128_si256(Out4_7, 1));
    _mm_storeu_si128((__m128i*)(outptr + 96),
                     _mm256_extracti128_si256(Out8_11, 1));
    _mm_storeu_si128((__m128i*)(outptr + 112),
                     _mm256_extracti128_si256(Out12_15, 1));
    inptr0 += 32;
    inptr1 += 32;
    inptr2 += 32;
    inptr3 += 32;
}

template <typename T>
MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void transpose_4x16_1_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_4x16_1_h only support uint8_t and int8_t");
    //    A15 A14 A13 A12 A11 A10 A9 A8 A7 A6 A5 A4 A3 A2 A1 A0
    __m128i R0 = _mm_loadu_si128((__m128i*)inptr0);
    //    B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0
    __m128i R1 = _mm_loadu_si128((__m128i*)inptr1);
    //    C15 C14 C13 C12 C11 C10 C9 C8 C7 C6 C5 C4 C3 C2 C1 C0
    __m128i R2 = _mm_loadu_si128((__m128i*)inptr2);
    //    D15 D14 D13 D12 D11 D10 D9 D8 D7 D6 D5 D4 D3 D2 D1 D0
    __m128i R3 = _mm_loadu_si128((__m128i*)inptr3);

    // B7 A7 B6 A6 B5 A5 B4 A4 B3 A3 B2 A2 B1 A1 B0 A0
    __m128i R01L = _mm_unpacklo_epi8(R0, R1);
    // B15 A15 B14 A14 B13 A13 B12 A12 B11 A11 B10 A10 B9 A9 B8 A8
    __m128i R01H = _mm_unpackhi_epi8(R0, R1);

    // D7 C7 D6 C6 D5 C5 D4 C4 D3 C3 D2 C2 D1 C1 D0 C0
    __m128i R23L = _mm_unpacklo_epi8(R2, R3);
    // D15 C15 D14 C14 D13 C13 D12 C12 D11 C11 D10 C10 D9 C9 D8 C8
    __m128i R23H = _mm_unpackhi_epi8(R2, R3);

    // D3 C3 B3 A3 D2 C2 B2 A2 D1 C1 B1 A1 D0 C0 B0 A0
    __m128i Out0_3 = _mm_unpacklo_epi16(R01L, R23L);
    // D7 C7 B7 A7 D6 C6 B6 A6 D5 C5 B5 A5 D4 C4 B4 A4
    __m128i Out4_7 = _mm_unpackhi_epi16(R01L, R23L);
    // D11 C11 B11 A11 D10 C10 B10 A10 D9 C9 B9 A9 D8 C8 B8 A8
    __m128i Out8_11 = _mm_unpacklo_epi16(R01H, R23H);
    // D11 C11 B11 A11 D10 C10 B10 A10 D9 C9 B9 A9 D8 C8 B8 A8
    __m128i Out12_15 = _mm_unpackhi_epi16(R01H, R23H);

    _mm_storeu_si128((__m128i*)outptr, Out0_3);
    _mm_storeu_si128((__m128i*)(outptr + 16), Out4_7);
    _mm_storeu_si128((__m128i*)(outptr + 32), Out8_11);
    _mm_storeu_si128((__m128i*)(outptr + 48), Out12_15);
    inptr0 += 16;
    inptr1 += 16;
    inptr2 += 16;
    inptr3 += 16;
}

MEGDNN_ATTRIBUTE_TARGET("sse3")
static inline void transpose_4x12_1_b_add_128(const int8_t*& inptr0,
                                              const int8_t*& inptr1,
                                              const int8_t*& inptr2,
                                              const int8_t*& inptr3,
                                              uint8_t*& outptr) {
    // int8 trick, we want to add 128, means adding b1000 0000, it is same to
    // -128
    __m128i const_128 = _mm_set1_epi8(-128);
    //    A15 A14 A13 A12 A11 A10 A9 A8 A7 A6 A5 A4 A3 A2 A1 A0
    __m128i R0 = _mm_loadu_si128((__m128i*)inptr0);
    //    B15 B14 B13 B12 B11 B10 B9 B8 B7 B6 B5 B4 B3 B2 B1 B0
    __m128i R1 = _mm_loadu_si128((__m128i*)inptr1);
    //    C15 C14 C13 C12 C11 C10 C9 C8 C7 C6 C5 C4 C3 C2 C1 C0
    __m128i R2 = _mm_loadu_si128((__m128i*)inptr2);
    //    D15 D14 D13 D12 D11 D10 D9 D8 D7 D6 D5 D4 D3 D2 D1 D0
    __m128i R3 = _mm_loadu_si128((__m128i*)inptr3);

    R0 = _mm_add_epi8(R0, const_128);
    R1 = _mm_add_epi8(R1, const_128);
    R2 = _mm_add_epi8(R2, const_128);
    R3 = _mm_add_epi8(R3, const_128);

    // B7 A7 B6 A6 B5 A5 B4 A4 B3 A3 B2 A2 B1 A1 B0 A0
    __m128i R01L = _mm_unpacklo_epi8(R0, R1);
    // B15 A15 B14 A14 B13 A13 B12 A12 B11 A11 B10 A10 B9 A9 B8 A8
    __m128i R01H = _mm_unpackhi_epi8(R0, R1);

    // D7 C7 D6 C6 D5 C5 D4 C4 D3 C3 D2 C2 D1 C1 D0 C0
    __m128i R23L = _mm_unpacklo_epi8(R2, R3);
    // D15 C15 D14 C14 D13 C13 D12 C12 D11 C11 D10 C10 D9 C9 D8 C8
    __m128i R23H = _mm_unpackhi_epi8(R2, R3);

    // D3 C3 B3 A3 D2 C2 B2 A2 D1 C1 B1 A1 D0 C0 B0 A0
    __m128i Out0_3 = _mm_unpacklo_epi16(R01L, R23L);
    // D7 C7 B7 A7 D6 C6 B6 A6 D5 C5 B5 A5 D4 C4 B4 A4
    __m128i Out4_7 = _mm_unpackhi_epi16(R01L, R23L);
    // D11 C11 B11 A11 D10 C10 B10 A10 D9 C9 B9 A9 D8 C8 B8 A8
    __m128i Out8_11 = _mm_unpacklo_epi16(R01H, R23H);

    _mm_storeu_si128((__m128i*)outptr, Out0_3);
    _mm_storeu_si128((__m128i*)(outptr + 16), Out4_7);
    _mm_storeu_si128((__m128i*)(outptr + 32), Out8_11);
    inptr0 += 12;
    inptr1 += 12;
    inptr2 += 12;
    inptr3 += 12;
}
}  // namespace x86
}  // namespace megdnn
// vim: syntax=cpp.doxygen
