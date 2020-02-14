/**
 * \file dnn/src/x86/simd_macro/avx2_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <xmmintrin.h>
#include <immintrin.h>

#define MEGDNN_SIMD_NAME AVX2
#define MEGDNN_SIMD_TARGET avx2
#define MEGDNN_SIMD_ATTRIBUTE_TARGET MEGDNN_ATTRIBUTE_TARGET("avx2")
#define MEGDNN_SIMD_WIDTH 8
#define MEGDNN_SIMD_TYPE __m256
#define MEGDNN_SIMD_LOADU(addr) _mm256_loadu_ps(addr)
#define MEGDNN_SIMD_STOREU(addr, reg) _mm256_storeu_ps(addr, reg)
#define MEGDNN_SIMD_SETZERO() _mm256_setzero_ps()
#define MEGDNN_SIMD_SET1(num) _mm256_set1_ps(num)
#define MEGDNN_SIMD_FMADD(a, b, c) _mm256_add_ps(c, _mm256_mul_ps(a, b))

