/**
 * \file dnn/src/x86/matrix_mul/f32/strategy_mk8_8x8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

/**
 * \file dnn/src/x86/matrix_mul/f32/strategy_mk8_8x8.cpp
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright copyright (c) 2014-2019 megvii inc. all rights reserved.
 */

#include <immintrin.h>

#include "src/common/utils.h"
#include "src/x86/matrix_mul/common/common.h"
#include "src/x86/matrix_mul/f32/strategy.h"

using namespace megdnn;
using namespace x86;
using namespace x86::matmul;

namespace {

MEGDNN_ATTRIBUTE_TARGET("fma")
void kern_8x1(const float* a_ptr, const float* b_ptr, int LDB, size_t K,
              float* output) {
    constexpr size_t KB = 8;

    __m256 ymm0, ymm1;
    __m256 ymm4, ymm5;
    __m256 ymm8, ymm9;

    ymm0 = _mm256_loadu_ps(a_ptr);
    ymm4 = _mm256_set1_ps(*b_ptr);
    ymm8 = _mm256_mul_ps(ymm0, ymm4);

    ymm1 = _mm256_loadu_ps(a_ptr + 32);
    ymm5 = _mm256_set1_ps(*(b_ptr + 4));
    ymm9 = _mm256_mul_ps(ymm1, ymm5);

    ymm0 = _mm256_loadu_ps(a_ptr + 8);
    ymm4 = _mm256_set1_ps(*(b_ptr + 1));
    ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);

    ymm1 = _mm256_loadu_ps(a_ptr + 40);
    ymm5 = _mm256_set1_ps(*(b_ptr + 5));
    ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);

    ymm0 = _mm256_loadu_ps(a_ptr + 16);
    ymm4 = _mm256_set1_ps(*(b_ptr + 2));
    ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);

    ymm1 = _mm256_loadu_ps(a_ptr + 48);
    ymm5 = _mm256_set1_ps(*(b_ptr + 6));
    ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);

    ymm0 = _mm256_loadu_ps(a_ptr + 24);
    ymm4 = _mm256_set1_ps(*(b_ptr + 3));
    ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);

    ymm1 = _mm256_loadu_ps(a_ptr + 56);
    ymm5 = _mm256_set1_ps(*(b_ptr + 7));
    ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);

    for (size_t k = KB; k < K; k += KB) {
        a_ptr += 64;

        b_ptr += LDB;

        ymm0 = _mm256_loadu_ps(a_ptr);
        ymm4 = _mm256_set1_ps(*b_ptr);
        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);

        ymm1 = _mm256_loadu_ps(a_ptr + 32);
        ymm5 = _mm256_set1_ps(*(b_ptr + 4));
        ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);

        ymm0 = _mm256_loadu_ps(a_ptr + 8);
        ymm4 = _mm256_set1_ps(*(b_ptr + 1));
        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);

        ymm1 = _mm256_loadu_ps(a_ptr + 40);
        ymm5 = _mm256_set1_ps(*(b_ptr + 5));
        ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);

        ymm0 = _mm256_loadu_ps(a_ptr + 16);
        ymm4 = _mm256_set1_ps(*(b_ptr + 2));
        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);

        ymm1 = _mm256_loadu_ps(a_ptr + 48);
        ymm5 = _mm256_set1_ps(*(b_ptr + 6));
        ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);

        ymm0 = _mm256_loadu_ps(a_ptr + 24);
        ymm4 = _mm256_set1_ps(*(b_ptr + 3));
        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);

        ymm1 = _mm256_loadu_ps(a_ptr + 56);
        ymm5 = _mm256_set1_ps(*(b_ptr + 7));
        ymm9 = _mm256_fmadd_ps(ymm1, ymm5, ymm9);
    }

    ymm8 = _mm256_add_ps(ymm8, ymm9);
    _mm256_storeu_ps(output, ymm8);
}

MEGDNN_ATTRIBUTE_TARGET("fma")
void kern_8x2(const float* a_ptr, const float* b_ptr, int LDB, size_t K,
              float* output) {
    constexpr size_t KB = 8;

    __m256 ymm0, ymm1;
    __m256 ymm4, ymm5;
    __m256 ymm8, ymm9;
    __m256 ymm12;

    const float* brow0 = b_ptr + 8 * 0;
    const float* brow1 = b_ptr + 8 * 1;

    ymm12 = _mm256_loadu_ps(a_ptr);
    ymm1 = _mm256_set1_ps(*brow0);
    ymm8 = _mm256_mul_ps(ymm12, ymm1);

    ymm1 = _mm256_set1_ps(*brow1);
    ymm9 = _mm256_mul_ps(ymm12, ymm1);

    a_ptr += 8;

    for (size_t i = 1; i < 8; ++i) {
        ymm12 = _mm256_loadu_ps(a_ptr);
        ymm1 = _mm256_set1_ps(*(brow0 + i));
        ymm8 = _mm256_fmadd_ps(ymm12, ymm1, ymm8);

        ymm1 = _mm256_set1_ps(*(brow1 + i));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);

        a_ptr += 8;
    }

    for (size_t k = KB; k < K; k += KB) {
        ymm12 = _mm256_loadu_ps(a_ptr);

        brow0 += LDB;
        brow1 += LDB;

        ymm0 = _mm256_set1_ps(*(brow0 + 0));
        ymm1 = _mm256_set1_ps(*(brow1 + 0));

        // i = 0
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 1));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 1));

        ymm12 = _mm256_loadu_ps(a_ptr + 8);

        // i = 1
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm0 = _mm256_set1_ps(*(brow0 + 2));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);
        ymm1 = _mm256_set1_ps(*(brow1 + 2));

        ymm12 = _mm256_loadu_ps(a_ptr + 16);

        // i = 2
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 3));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 3));

        ymm12 = _mm256_loadu_ps(a_ptr + 24);

        // i = 3
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm0 = _mm256_set1_ps(*(brow0 + 4));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);
        ymm1 = _mm256_set1_ps(*(brow1 + 4));

        ymm12 = _mm256_loadu_ps(a_ptr + 32);

        // i = 4
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 5));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 5));

        ymm12 = _mm256_loadu_ps(a_ptr + 40);

        // i = 5
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm0 = _mm256_set1_ps(*(brow0 + 6));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);
        ymm1 = _mm256_set1_ps(*(brow1 + 6));

        ymm12 = _mm256_loadu_ps(a_ptr + 48);

        // i = 6
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 7));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 7));

        ymm12 = _mm256_loadu_ps(a_ptr + 56);

        // i = 7
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);

        a_ptr += 64;
    }
    _mm256_storeu_ps(output + 0, ymm8);
    _mm256_storeu_ps(output + 8, ymm9);
}

MEGDNN_ATTRIBUTE_TARGET("fma")
void kern_8x4(const float* a_ptr, const float* b_ptr, int LDB, size_t K,
              float* output) {
    constexpr size_t KB = 8;

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12;

    const float* brow0 = b_ptr + 8 * 0;
    const float* brow1 = b_ptr + 8 * 1;
    const float* brow2 = b_ptr + 8 * 2;
    const float* brow3 = b_ptr + 8 * 3;

    ymm12 = _mm256_loadu_ps(a_ptr);
    ymm1 = _mm256_set1_ps(*brow0);
    ymm8 = _mm256_mul_ps(ymm12, ymm1);

    ymm1 = _mm256_set1_ps(*brow1);
    ymm9 = _mm256_mul_ps(ymm12, ymm1);

    ymm1 = _mm256_set1_ps(*brow2);
    ymm10 = _mm256_mul_ps(ymm12, ymm1);

    ymm1 = _mm256_set1_ps(*brow3);
    ymm11 = _mm256_mul_ps(ymm12, ymm1);

    a_ptr += 8;

    for (size_t i = 1; i < 8; ++i) {
        ymm12 = _mm256_loadu_ps(a_ptr);
        ymm1 = _mm256_set1_ps(*(brow0 + i));
        ymm8 = _mm256_fmadd_ps(ymm12, ymm1, ymm8);

        ymm1 = _mm256_set1_ps(*(brow1 + i));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);

        ymm1 = _mm256_set1_ps(*(brow2 + i));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm1, ymm10);

        ymm1 = _mm256_set1_ps(*(brow3 + i));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm1, ymm11);

        a_ptr += 8;
    }

    for (size_t k = KB; k < K; k += KB) {
        ymm12 = _mm256_loadu_ps(a_ptr);

        brow0 += LDB;
        brow1 += LDB;
        brow2 += LDB;
        brow3 += LDB;

        ymm0 = _mm256_set1_ps(*(brow0 + 0));
        ymm1 = _mm256_set1_ps(*(brow1 + 0));
        ymm2 = _mm256_set1_ps(*(brow2 + 0));
        ymm3 = _mm256_set1_ps(*(brow3 + 0));

        // i = 0
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 1));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 1));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm2, ymm10);
        ymm6 = _mm256_set1_ps(*(brow2 + 1));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm3, ymm11);
        ymm7 = _mm256_set1_ps(*(brow3 + 1));

        ymm12 = _mm256_loadu_ps(a_ptr + 8);

        // i = 1
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm0 = _mm256_set1_ps(*(brow0 + 2));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);
        ymm1 = _mm256_set1_ps(*(brow1 + 2));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm6, ymm10);
        ymm2 = _mm256_set1_ps(*(brow2 + 2));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm7, ymm11);
        ymm3 = _mm256_set1_ps(*(brow3 + 2));

        ymm12 = _mm256_loadu_ps(a_ptr + 16);

        // i = 2
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 3));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 3));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm2, ymm10);
        ymm6 = _mm256_set1_ps(*(brow2 + 3));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm3, ymm11);
        ymm7 = _mm256_set1_ps(*(brow3 + 3));

        ymm12 = _mm256_loadu_ps(a_ptr + 24);

        // i = 3
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm0 = _mm256_set1_ps(*(brow0 + 4));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);
        ymm1 = _mm256_set1_ps(*(brow1 + 4));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm6, ymm10);
        ymm2 = _mm256_set1_ps(*(brow2 + 4));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm7, ymm11);
        ymm3 = _mm256_set1_ps(*(brow3 + 4));

        ymm12 = _mm256_loadu_ps(a_ptr + 32);

        // i = 4
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 5));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 5));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm2, ymm10);
        ymm6 = _mm256_set1_ps(*(brow2 + 5));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm3, ymm11);
        ymm7 = _mm256_set1_ps(*(brow3 + 5));

        ymm12 = _mm256_loadu_ps(a_ptr + 40);

        // i = 5
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm0 = _mm256_set1_ps(*(brow0 + 6));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);
        ymm1 = _mm256_set1_ps(*(brow1 + 6));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm6, ymm10);
        ymm2 = _mm256_set1_ps(*(brow2 + 6));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm7, ymm11);
        ymm3 = _mm256_set1_ps(*(brow3 + 6));

        ymm12 = _mm256_loadu_ps(a_ptr + 48);

        // i = 6
        ymm8 = _mm256_fmadd_ps(ymm12, ymm0, ymm8);
        ymm4 = _mm256_set1_ps(*(brow0 + 7));
        ymm9 = _mm256_fmadd_ps(ymm12, ymm1, ymm9);
        ymm5 = _mm256_set1_ps(*(brow1 + 7));
        ymm10 = _mm256_fmadd_ps(ymm12, ymm2, ymm10);
        ymm6 = _mm256_set1_ps(*(brow2 + 7));
        ymm11 = _mm256_fmadd_ps(ymm12, ymm3, ymm11);
        ymm7 = _mm256_set1_ps(*(brow3 + 7));

        ymm12 = _mm256_loadu_ps(a_ptr + 56);

        // i = 7
        ymm8 = _mm256_fmadd_ps(ymm12, ymm4, ymm8);
        ymm9 = _mm256_fmadd_ps(ymm12, ymm5, ymm9);
        ymm10 = _mm256_fmadd_ps(ymm12, ymm6, ymm10);
        ymm11 = _mm256_fmadd_ps(ymm12, ymm7, ymm11);

        a_ptr += 64;
    }
    _mm256_storeu_ps(output + 0, ymm8);
    _mm256_storeu_ps(output + 8, ymm9);
    _mm256_storeu_ps(output + 16, ymm10);
    _mm256_storeu_ps(output + 24, ymm11);
}

MEGDNN_ATTRIBUTE_TARGET("fma")
void kern_8x8(const float* a_ptr, const float* b_ptr, int LDB, size_t K,
              float* output) {
    constexpr size_t KB = 8;

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;

    const float* brow0 = b_ptr + 8 * 0;
    const float* brow1 = b_ptr + 8 * 1;
    const float* brow2 = b_ptr + 8 * 2;
    const float* brow3 = b_ptr + 8 * 3;
    const float* brow4 = b_ptr + 8 * 4;
    const float* brow5 = b_ptr + 8 * 5;
    const float* brow6 = b_ptr + 8 * 6;
    const float* brow7 = b_ptr + 8 * 7;

    ymm0 = _mm256_loadu_ps(a_ptr);
    ymm1 = _mm256_set1_ps(*brow0);
    ymm8 = _mm256_mul_ps(ymm0, ymm1);

    ymm1 = _mm256_set1_ps(*brow1);
    ymm9 = _mm256_mul_ps(ymm0, ymm1);

    ymm1 = _mm256_set1_ps(*brow2);
    ymm10 = _mm256_mul_ps(ymm0, ymm1);

    ymm1 = _mm256_set1_ps(*brow3);
    ymm11 = _mm256_mul_ps(ymm0, ymm1);

    ymm1 = _mm256_set1_ps(*brow4);
    ymm12 = _mm256_mul_ps(ymm0, ymm1);

    ymm1 = _mm256_set1_ps(*brow5);
    ymm13 = _mm256_mul_ps(ymm0, ymm1);

    ymm1 = _mm256_set1_ps(*brow6);
    ymm14 = _mm256_mul_ps(ymm0, ymm1);

    ymm1 = _mm256_set1_ps(*brow7);
    ymm15 = _mm256_mul_ps(ymm0, ymm1);

    a_ptr += 8;

    for (size_t i = 1; i < 8; ++i) {
        ymm0 = _mm256_loadu_ps(a_ptr);
        ymm1 = _mm256_set1_ps(*(brow0 + i));
        ymm8 = _mm256_fmadd_ps(ymm0, ymm1, ymm8);

        ymm1 = _mm256_set1_ps(*(brow1 + i));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm1, ymm9);

        ymm1 = _mm256_set1_ps(*(brow2 + i));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm1, ymm10);

        ymm1 = _mm256_set1_ps(*(brow3 + i));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm1, ymm11);

        ymm1 = _mm256_set1_ps(*(brow4 + i));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm1, ymm12);

        ymm1 = _mm256_set1_ps(*(brow5 + i));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm1, ymm13);

        ymm1 = _mm256_set1_ps(*(brow6 + i));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm1, ymm14);

        ymm1 = _mm256_set1_ps(*(brow7 + i));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);

        a_ptr += 8;
    }

    for (size_t k = KB; k < K; k += KB) {
        ymm0 = _mm256_loadu_ps(a_ptr);

        brow0 += LDB;
        brow1 += LDB;
        brow2 += LDB;
        brow3 += LDB;
        brow4 += LDB;
        brow5 += LDB;
        brow6 += LDB;
        brow7 += LDB;

        // i = 0
        ymm1 = _mm256_set1_ps(*(brow0 + 0));
        ymm2 = _mm256_set1_ps(*(brow1 + 0));
        ymm3 = _mm256_set1_ps(*(brow2 + 0));
        ymm8 = _mm256_fmadd_ps(ymm0, ymm1, ymm8);
        ymm4 = _mm256_set1_ps(*(brow3 + 0));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
        ymm5 = _mm256_set1_ps(*(brow4 + 0));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
        ymm6 = _mm256_set1_ps(*(brow5 + 0));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm4, ymm11);
        ymm7 = _mm256_set1_ps(*(brow6 + 0));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm5, ymm12);
        ymm1 = _mm256_set1_ps(*(brow7 + 0));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm6, ymm13);
        ymm2 = _mm256_set1_ps(*(brow0 + 1));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm7, ymm14);
        ymm3 = _mm256_set1_ps(*(brow1 + 1));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);
        ymm0 = _mm256_loadu_ps(a_ptr + 8);
        ymm4 = _mm256_set1_ps(*(brow2 + 1));

        // i = 1
        ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
        ymm5 = _mm256_set1_ps(*(brow3 + 1));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm3, ymm9);
        ymm6 = _mm256_set1_ps(*(brow4 + 1));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm4, ymm10);
        ymm7 = _mm256_set1_ps(*(brow5 + 1));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm5, ymm11);
        ymm1 = _mm256_set1_ps(*(brow6 + 1));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm6, ymm12);
        ymm2 = _mm256_set1_ps(*(brow7 + 1));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm7, ymm13);
        ymm3 = _mm256_set1_ps(*(brow0 + 2));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm1, ymm14);
        ymm4 = _mm256_set1_ps(*(brow1 + 2));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm2, ymm15);
        ymm0 = _mm256_loadu_ps(a_ptr + 16);
        ymm5 = _mm256_set1_ps(*(brow2 + 2));

        // i = 2
        ymm8 = _mm256_fmadd_ps(ymm0, ymm3, ymm8);
        ymm6 = _mm256_set1_ps(*(brow3 + 2));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm4, ymm9);
        ymm7 = _mm256_set1_ps(*(brow4 + 2));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm5, ymm10);
        ymm1 = _mm256_set1_ps(*(brow5 + 2));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm6, ymm11);
        ymm2 = _mm256_set1_ps(*(brow6 + 2));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm7, ymm12);
        ymm3 = _mm256_set1_ps(*(brow7 + 2));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm1, ymm13);
        ymm4 = _mm256_set1_ps(*(brow0 + 3));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm2, ymm14);
        ymm5 = _mm256_set1_ps(*(brow1 + 3));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm3, ymm15);
        ymm0 = _mm256_loadu_ps(a_ptr + 24);
        ymm6 = _mm256_set1_ps(*(brow2 + 3));

        // i = 3
        ymm8 = _mm256_fmadd_ps(ymm0, ymm4, ymm8);
        ymm7 = _mm256_set1_ps(*(brow3 + 3));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm5, ymm9);
        ymm1 = _mm256_set1_ps(*(brow4 + 3));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm6, ymm10);
        ymm2 = _mm256_set1_ps(*(brow5 + 3));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm7, ymm11);
        ymm3 = _mm256_set1_ps(*(brow6 + 3));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm1, ymm12);
        ymm4 = _mm256_set1_ps(*(brow7 + 3));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm2, ymm13);
        ymm5 = _mm256_set1_ps(*(brow0 + 4));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);
        ymm6 = _mm256_set1_ps(*(brow1 + 4));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm4, ymm15);
        ymm0 = _mm256_loadu_ps(a_ptr + 32);
        ymm7 = _mm256_set1_ps(*(brow2 + 4));

        // i = 4
        ymm8 = _mm256_fmadd_ps(ymm0, ymm5, ymm8);
        ymm1 = _mm256_set1_ps(*(brow3 + 4));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm6, ymm9);
        ymm2 = _mm256_set1_ps(*(brow4 + 4));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm7, ymm10);
        ymm3 = _mm256_set1_ps(*(brow5 + 4));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm1, ymm11);
        ymm4 = _mm256_set1_ps(*(brow6 + 4));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
        ymm5 = _mm256_set1_ps(*(brow7 + 4));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
        ymm6 = _mm256_set1_ps(*(brow0 + 5));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm4, ymm14);
        ymm7 = _mm256_set1_ps(*(brow1 + 5));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm5, ymm15);
        ymm0 = _mm256_loadu_ps(a_ptr + 40);
        ymm1 = _mm256_set1_ps(*(brow2 + 5));

        // i = 5
        ymm8 = _mm256_fmadd_ps(ymm0, ymm6, ymm8);
        ymm2 = _mm256_set1_ps(*(brow3 + 5));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm7, ymm9);
        ymm3 = _mm256_set1_ps(*(brow4 + 5));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm1, ymm10);
        ymm4 = _mm256_set1_ps(*(brow5 + 5));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm2, ymm11);
        ymm5 = _mm256_set1_ps(*(brow6 + 5));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm3, ymm12);
        ymm6 = _mm256_set1_ps(*(brow7 + 5));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm4, ymm13);
        ymm7 = _mm256_set1_ps(*(brow0 + 6));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm5, ymm14);
        ymm1 = _mm256_set1_ps(*(brow1 + 6));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm6, ymm15);
        ymm0 = _mm256_loadu_ps(a_ptr + 48);
        ymm2 = _mm256_set1_ps(*(brow2 + 6));

        // i = 6
        ymm8 = _mm256_fmadd_ps(ymm0, ymm7, ymm8);
        ymm3 = _mm256_set1_ps(*(brow3 + 6));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm1, ymm9);
        ymm4 = _mm256_set1_ps(*(brow4 + 6));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm2, ymm10);
        ymm5 = _mm256_set1_ps(*(brow5 + 6));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm3, ymm11);
        ymm6 = _mm256_set1_ps(*(brow6 + 6));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm4, ymm12);
        ymm7 = _mm256_set1_ps(*(brow7 + 6));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm5, ymm13);
        ymm1 = _mm256_set1_ps(*(brow0 + 7));
        ymm14 = _mm256_fmadd_ps(ymm0, ymm6, ymm14);
        ymm2 = _mm256_set1_ps(*(brow1 + 7));
        ymm15 = _mm256_fmadd_ps(ymm0, ymm7, ymm15);
        ymm0 = _mm256_loadu_ps(a_ptr + 56);
        ymm3 = _mm256_set1_ps(*(brow2 + 7));

        // i = 7
        ymm8 = _mm256_fmadd_ps(ymm0, ymm1, ymm8);
        ymm4 = _mm256_set1_ps(*(brow3 + 7));
        ymm9 = _mm256_fmadd_ps(ymm0, ymm2, ymm9);
        ymm5 = _mm256_set1_ps(*(brow4 + 7));
        ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
        ymm6 = _mm256_set1_ps(*(brow5 + 7));
        ymm11 = _mm256_fmadd_ps(ymm0, ymm4, ymm11);
        ymm7 = _mm256_set1_ps(*(brow6 + 7));
        ymm12 = _mm256_fmadd_ps(ymm0, ymm5, ymm12);
        ymm1 = _mm256_set1_ps(*(brow7 + 7));
        ymm13 = _mm256_fmadd_ps(ymm0, ymm6, ymm13);
        ymm14 = _mm256_fmadd_ps(ymm0, ymm7, ymm14);
        ymm15 = _mm256_fmadd_ps(ymm0, ymm1, ymm15);

        a_ptr += 64;
    }
    _mm256_storeu_ps(output + 0, ymm8);
    _mm256_storeu_ps(output + 8, ymm9);
    _mm256_storeu_ps(output + 16, ymm10);
    _mm256_storeu_ps(output + 24, ymm11);
    _mm256_storeu_ps(output + 32, ymm12);
    _mm256_storeu_ps(output + 40, ymm13);
    _mm256_storeu_ps(output + 48, ymm14);
    _mm256_storeu_ps(output + 56, ymm15);
}

}  // anonymous namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(sgemm_nopack_8x8_avx2);

void sgemm_nopack_8x8_avx2::kern(const float* A, size_t LDA, const float* B,
                                 size_t LDB, float* C, size_t LDC, size_t M,
                                 size_t K, size_t N, const float*, void*,
                                 bool trA, bool trB) const {
    constexpr static size_t MB = 8;
    constexpr static size_t KB = 8;
    constexpr static size_t NB = 8;

    megdnn_assert(!trA && !trB && M % MB == 0 && K % KB == 0);

    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)
    for (size_t m = 0; m < M; m += MB) {
        float* output = C + (m / MB) * LDC;
        const float* cur_B = B;
        for (size_t n = 0; n < N;) {
            switch (N - n) {
                case 1:
                    kern_8x1(A, cur_B, LDB, K, output);
                    cur_B += KB;
                    output += MB * 1;
                    n++;
                    break;
                case 2:
                case 3:
                    kern_8x2(A, cur_B, LDB, K, output);
                    cur_B += KB * 2;
                    output += MB * 2;
                    n += 2;
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                    kern_8x4(A, cur_B, LDB, K, output);
                    cur_B += KB * 4;
                    output += MB * 4;
                    n += 4;
                    break;
                default:
                    kern_8x8(A, cur_B, LDB, K, output);
                    cur_B += KB * NB;
                    output += MB * NB;
                    n += NB;
                    break;
            }
        }
        A += LDA;
    }
}

// vim: syntax=cpp.doxygen
