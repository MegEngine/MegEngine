/**
 * \file dnn/src/x86/simd_macro/immintrin.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <immintrin.h>
#ifdef __GNUC__ 
#if __GNUC__ < 8
#define _mm256_set_m128i(xmm1, xmm2)                        \
    _mm256_permute2f128_si256(_mm256_castsi128_si256(xmm1), \
                              _mm256_castsi128_si256(xmm2), 2)
#define _mm256_set_m128f(xmm1, xmm2)                     \
    _mm256_permute2f128_ps(_mm256_castps128_ps256(xmm1), \
                           _mm256_castps128_ps256(xmm2), 2)
#endif
#endif

namespace megdnn {
namespace x86 {

typedef struct __m128x2 {
    __m128 val[2];
} __m128x2;

typedef struct __m128ix2 {
    __m128i val[2];
} __m128ix2;

typedef struct __m128x4 {
    __m128 val[4];
} __m128x4;

typedef struct __m128ix4 {
    __m128i val[4];
} __m128ix4;

typedef struct __m256x2 {
    __m256 val[2];
} __m256x2;

typedef struct __m256ix2 {
    __m256i val[2];
} __m256ix2;

typedef struct __m256x4 {
    __m256 val[4];
} __m256x4;

typedef struct __m256ix4 {
    __m256i val[4];
} __m256ix4;

}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
