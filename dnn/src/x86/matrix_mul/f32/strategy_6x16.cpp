/**
 * \file dnn/src/x86/matrix_mul/f32/strategy_6x16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

/**
 * \file dnn/src/x86/matrix_mul/f32/strategy_6x16.cpp
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright copyright (c) 2014-2019 megvii inc. all rights reserved.
 */
#include <immintrin.h>

#include "src/common/utils.h"
#include "src/x86/avx_helper.h"
#include "src/x86/matrix_mul/common/common.h"
#include "src/x86/matrix_mul/f32/strategy.h"
#include "src/common/unroll_macro.h"

using namespace megdnn;
using namespace x86;

#define DNN_AVX2_TARGET
#if !defined(__clang__)
//! bypass gcc bug https://bugs.launchpad.net/ubuntu/+source/gcc-5/+bug/1642109
#pragma GCC target("avx2")
#else
#undef DNN_AVX2_TARGET
#define DNN_AVX2_TARGET MEGDNN_ATTRIBUTE_TARGET("avx2")
#endif

#define UNROLL_CODE(cb, i, a...) UNROLL_CALL1(i,cb,##a)
namespace {

DNN_AVX2_TARGET
void transpose_16x8_1_s(const float *inptr0, const float *inptr1,
                        const float *inptr2, const float *inptr3,
                        const float *inptr4, const float *inptr5,
                        const float *inptr6, const float *inptr7,
                        const float *inptr8, const float *inptr9,
                        const float *inptr10, const float *inptr11,
                        const float *inptr12, const float *inptr13,
                        const float *inptr14, const float *inptr15,
                        float *outptr) {
  auto ymm0 = _mm256_loadu_ps(inptr0); // A0A1A2A3A4A5A6A7
  auto ymm1 = _mm256_loadu_ps(inptr1); // B0B1B2B3B4B5B6B7
  auto ymm2 = _mm256_loadu_ps(inptr2); // C0C1C2C3C4C5C6C7
  auto ymm3 = _mm256_loadu_ps(inptr3); // D0D1D2D3D4D5D6D7
  auto ymm4 = _mm256_loadu_ps(inptr4); // E0E1E2E3E4E5E6E7
  auto ymm5 = _mm256_loadu_ps(inptr5); // F0F1F2F3F4F5F6F7
  auto ymm6 = _mm256_loadu_ps(inptr6); // G0G1G2G3G4G5G6G7
  auto ymm7 = _mm256_loadu_ps(inptr7); // H0H1H2H3H4H5H6H7

  auto ymm8 = _mm256_unpacklo_ps(ymm0, ymm2);  // A0C0A1C1A4C4A5C5
  auto ymm9 = _mm256_unpackhi_ps(ymm0, ymm2);  // A2C2A3C3A6C6A7C7
  auto ymm10 = _mm256_unpacklo_ps(ymm1, ymm3); // B0D0B1D1B4D4B5D5
  auto ymm11 = _mm256_unpackhi_ps(ymm1, ymm3); // B2D2B3D3B6D6B7D7
  auto ymm12 = _mm256_unpacklo_ps(ymm4, ymm6); // E0G0E1G1E4G4E5G5
  auto ymm13 = _mm256_unpackhi_ps(ymm4, ymm6); // E2G2E3G3E6G6E7G7
  auto ymm14 = _mm256_unpacklo_ps(ymm5, ymm7); // F0H0F1H1F4H4F5H5
  auto ymm15 = _mm256_unpackhi_ps(ymm5, ymm7); // F2H2F3H3F6H6F7H7

  ymm0 = _mm256_unpacklo_ps(ymm8, ymm10);  // A0B0C0D0A4B4C4D4
  ymm1 = _mm256_unpackhi_ps(ymm8, ymm10);  // A1B1C1D1A5B5C5D5
  ymm2 = _mm256_unpacklo_ps(ymm9, ymm11);  // A2B2C2D2A6B6C6D6
  ymm3 = _mm256_unpackhi_ps(ymm9, ymm11);  // A3B3C3D3A7B7C7D7
  ymm4 = _mm256_unpacklo_ps(ymm12, ymm14); // E0F0G0H0E4F4G4H4
  ymm5 = _mm256_unpackhi_ps(ymm12, ymm14); // E1F1G1H1E5F5G5H5
  ymm6 = _mm256_unpacklo_ps(ymm13, ymm15); // E2F2G2H2E6F6G6H6
  ymm7 = _mm256_unpackhi_ps(ymm13, ymm15); // E3F3G3H3E7F7G7H7

  ymm8 = _mm256_permute2f128_ps(ymm0, ymm4, 0x20);  // A0B0C0D0E0F0G0H0
  ymm9 = _mm256_permute2f128_ps(ymm1, ymm5, 0x20);  // A1B1C1D1E1F1G1H1
  ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 0x20); // A2B2C2D2E2F2G2H2
  ymm11 = _mm256_permute2f128_ps(ymm3, ymm7, 0x20); // A3B3C3D3E3F3G3H3
  ymm12 = _mm256_permute2f128_ps(ymm0, ymm4, 0x31); // A4B4C4D4E4F4G4H4
  ymm13 = _mm256_permute2f128_ps(ymm1, ymm5, 0x31); // A5B5C5D5E5F5G5H5
  ymm14 = _mm256_permute2f128_ps(ymm2, ymm6, 0x31); // A6B6C6D6E6F6G6H6
  ymm15 = _mm256_permute2f128_ps(ymm3, ymm7, 0x31); // A7B7C7D7E7F7G7H7

  _mm256_storeu_ps(outptr + 16 * 0, ymm8);
  _mm256_storeu_ps(outptr + 16 * 1, ymm9);
  _mm256_storeu_ps(outptr + 16 * 2, ymm10);
  _mm256_storeu_ps(outptr + 16 * 3, ymm11);
  _mm256_storeu_ps(outptr + 16 * 4, ymm12);
  _mm256_storeu_ps(outptr + 16 * 5, ymm13);
  _mm256_storeu_ps(outptr + 16 * 6, ymm14);
  _mm256_storeu_ps(outptr + 16 * 7, ymm15);
  ymm0 = _mm256_loadu_ps(inptr8);  // A0A1A2A3A4A5A6A7
  ymm1 = _mm256_loadu_ps(inptr9);  // B0B1B2B3B4B5B6B7
  ymm2 = _mm256_loadu_ps(inptr10); // C0C1C2C3C4C5C6C7
  ymm3 = _mm256_loadu_ps(inptr11); // D0D1D2D3D4D5D6D7
  ymm4 = _mm256_loadu_ps(inptr12); // E0E1E2E3E4E5E6E7
  ymm5 = _mm256_loadu_ps(inptr13); // F0F1F2F3F4F5F6F7
  ymm6 = _mm256_loadu_ps(inptr14); // G0G1G2G3G4G5G6G7
  ymm7 = _mm256_loadu_ps(inptr15); // H0H1H2H3H4H5H6H7

  ymm8 = _mm256_unpacklo_ps(ymm0, ymm2);  // A0C0A1C1A4C4A5C5
  ymm9 = _mm256_unpackhi_ps(ymm0, ymm2);  // A2C2A3C3A6C6A7C7
  ymm10 = _mm256_unpacklo_ps(ymm1, ymm3); // B0D0B1D1B4D4B5D5
  ymm11 = _mm256_unpackhi_ps(ymm1, ymm3); // B2D2B3D3B6D6B7D7
  ymm12 = _mm256_unpacklo_ps(ymm4, ymm6); // E0G0E1G1E4G4E5G5
  ymm13 = _mm256_unpackhi_ps(ymm4, ymm6); // E2G2E3G3E6G6E7G7
  ymm14 = _mm256_unpacklo_ps(ymm5, ymm7); // F0H0F1H1F4H4F5H5
  ymm15 = _mm256_unpackhi_ps(ymm5, ymm7); // F2H2F3H3F6H6F7H7

  ymm0 = _mm256_unpacklo_ps(ymm8, ymm10);  // A0B0C0D0A4B4C4D4
  ymm1 = _mm256_unpackhi_ps(ymm8, ymm10);  // A1B1C1D1A5B5C5D5
  ymm2 = _mm256_unpacklo_ps(ymm9, ymm11);  // A2B2C2D2A6B6C6D6
  ymm3 = _mm256_unpackhi_ps(ymm9, ymm11);  // A3B3C3D3A7B7C7D7
  ymm4 = _mm256_unpacklo_ps(ymm12, ymm14); // E0F0G0H0E4F4G4H4
  ymm5 = _mm256_unpackhi_ps(ymm12, ymm14); // E1F1G1H1E5F5G5H5
  ymm6 = _mm256_unpacklo_ps(ymm13, ymm15); // E2F2G2H2E6F6G6H6
  ymm7 = _mm256_unpackhi_ps(ymm13, ymm15); // E3F3G3H3E7F7G7H7

  ymm8 = _mm256_permute2f128_ps(ymm0, ymm4, 0x20);  // A0B0C0D0E0F0G0H0
  ymm9 = _mm256_permute2f128_ps(ymm1, ymm5, 0x20);  // A1B1C1D1E1F1G1H1
  ymm10 = _mm256_permute2f128_ps(ymm2, ymm6, 0x20); // A2B2C2D2E2F2G2H2
  ymm11 = _mm256_permute2f128_ps(ymm3, ymm7, 0x20); // A3B3C3D3E3F3G3H3
  ymm12 = _mm256_permute2f128_ps(ymm0, ymm4, 0x31); // A4B4C4D4E4F4G4H4
  ymm13 = _mm256_permute2f128_ps(ymm1, ymm5, 0x31); // A5B5C5D5E5F5G5H5
  ymm14 = _mm256_permute2f128_ps(ymm2, ymm6, 0x31); // A6B6C6D6E6F6G6H6
  ymm15 = _mm256_permute2f128_ps(ymm3, ymm7, 0x31); // A7B7C7D7E7F7G7H7

  _mm256_storeu_ps(outptr + 16 * 0 + 8, ymm8);
  _mm256_storeu_ps(outptr + 16 * 1 + 8, ymm9);
  _mm256_storeu_ps(outptr + 16 * 2 + 8, ymm10);
  _mm256_storeu_ps(outptr + 16 * 3 + 8, ymm11);
  _mm256_storeu_ps(outptr + 16 * 4 + 8, ymm12);
  _mm256_storeu_ps(outptr + 16 * 5 + 8, ymm13);
  _mm256_storeu_ps(outptr + 16 * 6 + 8, ymm14);
  _mm256_storeu_ps(outptr + 16 * 7 + 8, ymm15);
}

DNN_AVX2_TARGET
void transpose_16x4_1_s(const float *inptr0, const float *inptr1,
                        const float *inptr2, const float *inptr3,
                        const float *inptr4, const float *inptr5,
                        const float *inptr6, const float *inptr7,
                        const float *inptr8, const float *inptr9,
                        const float *inptr10, const float *inptr11,
                        const float *inptr12, const float *inptr13,
                        const float *inptr14, const float *inptr15,
                        float *outptr) {
  const std::uint32_t arr[8] = {0, 1, 4, 5, 2, 3, 6, 7};
  __m256i order = _mm256_loadu_si256((const __m256i *)arr);
  auto ymm0 = _mm256_loadu2_m128_emulate(inptr2, inptr0); // A0A1A2A3C0C1C2C3
  auto ymm1 = _mm256_loadu2_m128_emulate(inptr3, inptr1); // B0B1B2B3D0D1D2D3
  auto ymm2 = _mm256_loadu2_m128_emulate(inptr6, inptr4); // E0E1E2E3G0G1G2G3
  auto ymm3 = _mm256_loadu2_m128_emulate(inptr7, inptr5); // F0F1F2F3H0H1H2H3

  auto ymm4 = _mm256_unpacklo_ps(ymm0, ymm1); // A0B0A1B1C0D0C1D1
  auto ymm5 = _mm256_unpackhi_ps(ymm0, ymm1); // A2B2A3B3C2D2C3D3
  auto ymm6 = _mm256_unpacklo_ps(ymm2, ymm3); // E0F0E1F1G0H0G1H1
  auto ymm7 = _mm256_unpackhi_ps(ymm2, ymm3); // E2F2E3F3G2H2G3H3

  auto ymm8 = _mm256_permutevar8x32_ps(ymm4, order);  // A0B0C0D0A1B1C1D1
  auto ymm9 = _mm256_permutevar8x32_ps(ymm5, order);  // A2B2C2D2A3B3C3D3
  auto ymm10 = _mm256_permutevar8x32_ps(ymm6, order); // E0F0G0H0E1F1G1H1
  auto ymm11 = _mm256_permutevar8x32_ps(ymm7, order); // E2F2G2H2E3F3G3H3

  ymm0 = _mm256_permute2f128_ps(ymm8, ymm10, 0x20); // A0B0C0D0E0F0G0H0
  ymm1 = _mm256_permute2f128_ps(ymm8, ymm10, 0x31); // A1B1C1D1E1F1G1H1
  ymm2 = _mm256_permute2f128_ps(ymm9, ymm11, 0x20); // A2B2C2D2E2F2G2H2
  ymm3 = _mm256_permute2f128_ps(ymm9, ymm11, 0x31); // A3B3C3D3E3F3G3H3

  _mm256_storeu_ps(outptr + 16 * 0, ymm0);
  _mm256_storeu_ps(outptr + 16 * 1, ymm1);
  _mm256_storeu_ps(outptr + 16 * 2, ymm2);
  _mm256_storeu_ps(outptr + 16 * 3, ymm3);
  ymm0 = _mm256_loadu2_m128_emulate(inptr10, inptr8);  // A0A1A2A3C0C1C2C3
  ymm1 = _mm256_loadu2_m128_emulate(inptr11, inptr9);  // B0B1B2B3D0D1D2D3
  ymm2 = _mm256_loadu2_m128_emulate(inptr14, inptr12); // E0E1E2E3G0G1G2G3
  ymm3 = _mm256_loadu2_m128_emulate(inptr15, inptr13); // F0F1F2F3H0H1H2H3

  ymm4 = _mm256_unpacklo_ps(ymm0, ymm1); // A0B0A1B1C0D0C1D1
  ymm5 = _mm256_unpackhi_ps(ymm0, ymm1); // A2B2A3B3C2D2C3D3
  ymm6 = _mm256_unpacklo_ps(ymm2, ymm3); // E0F0E1F1G0H0G1H1
  ymm7 = _mm256_unpackhi_ps(ymm2, ymm3); // E2F2E3F3G2H2G3H3

  ymm8 = _mm256_permutevar8x32_ps(ymm4, order);  // A0B0C0D0A1B1C1D1
  ymm9 = _mm256_permutevar8x32_ps(ymm5, order);  // A2B2C2D2A3B3C3D3
  ymm10 = _mm256_permutevar8x32_ps(ymm6, order); // E0F0G0H0E1F1G1H1
  ymm11 = _mm256_permutevar8x32_ps(ymm7, order); // E2F2G2H2E3F3G3H3

  ymm0 = _mm256_permute2f128_ps(ymm8, ymm10, 0x20); // A0B0C0D0E0F0G0H0
  ymm1 = _mm256_permute2f128_ps(ymm8, ymm10, 0x31); // A1B1C1D1E1F1G1H1
  ymm2 = _mm256_permute2f128_ps(ymm9, ymm11, 0x20); // A2B2C2D2E2F2G2H2
  ymm3 = _mm256_permute2f128_ps(ymm9, ymm11, 0x31); // A3B3C3D3E3F3G3H3

  _mm256_storeu_ps(outptr + 16 * 0 + 8, ymm0);
  _mm256_storeu_ps(outptr + 16 * 1 + 8, ymm1);
  _mm256_storeu_ps(outptr + 16 * 2 + 8, ymm2);
  _mm256_storeu_ps(outptr + 16 * 3 + 8, ymm3);
}

static size_t min(size_t a, size_t b) { return a > b ? b : a; }

DNN_AVX2_TARGET
void transpose_6x16_1_s(const float *inptr0, const float *inptr1,
                        const float *inptr2, const float *inptr3,
                        const float *inptr4, const float *inptr5,
                        float *outptr) {
  auto ymm0 = _mm256_loadu_ps(inptr0 + 0); // A0A1A2A3A4A5A6A7
  auto ymm1 = _mm256_loadu_ps(inptr0 + 8); // a0a1a2a3a4a5a6a7
  auto ymm2 = _mm256_loadu_ps(inptr1 + 0); // B0B1B2B3B4B5B6B7
  auto ymm3 = _mm256_loadu_ps(inptr1 + 8); // b0b1b2b3b4b5b6b7
  auto ymm4 = _mm256_loadu_ps(inptr2 + 0); // C0C1C2C3C4C5C6C7
  auto ymm5 = _mm256_loadu_ps(inptr2 + 8); // c0c1c2c3c4c5c6c7
  auto ymm6 = _mm256_loadu_ps(inptr3 + 0); // D0D1D2D3D4D5D6D7
  auto ymm7 = _mm256_loadu_ps(inptr3 + 8); // d0d1d2d3d4d5d6d7

  auto ymm8 = _mm256_unpacklo_ps(ymm0, ymm4);  // A0C0A1C1A4C4A5C5
  auto ymm9 = _mm256_unpackhi_ps(ymm0, ymm4);  // A2C2A3C3A6C6A7C7
  auto ymm10 = _mm256_unpacklo_ps(ymm2, ymm6); // B0D0B1D1B4D4B5D5
  auto ymm11 = _mm256_unpackhi_ps(ymm2, ymm6); // B2D2B3D3B6D6B7D7

  auto ymm12 = _mm256_unpacklo_ps(ymm1, ymm5); // a0c0a1c1a4c4a5c5
  auto ymm13 = _mm256_unpackhi_ps(ymm1, ymm5); // a2c2a3c3a6c6a7c7
  auto ymm14 = _mm256_unpacklo_ps(ymm3, ymm7); // b0d0b1d1b4d4b5d5
  auto ymm15 = _mm256_unpackhi_ps(ymm3, ymm7); // b2d2b3d3b6d6b7d7

  ymm0 = _mm256_unpacklo_ps(ymm8, ymm10); // A0B0C0D0A4B4C4D4
  ymm1 = _mm256_unpackhi_ps(ymm8, ymm10); // A1B1C1D1A5B5C5D5
  ymm2 = _mm256_unpacklo_ps(ymm9, ymm11); // A2B2C2D2A6B6C6D6
  ymm3 = _mm256_unpackhi_ps(ymm9, ymm11); // A3B3C3D3A7B7C7D7

  ymm4 = _mm256_unpacklo_ps(ymm12, ymm14); // a0b0c0d0a4b4c4d4
  ymm5 = _mm256_unpackhi_ps(ymm12, ymm14); // a1b1c1d1a5b5c5d5
  ymm6 = _mm256_unpacklo_ps(ymm13, ymm15); // a2b2c2d2a6b6c6d6
  ymm7 = _mm256_unpackhi_ps(ymm13, ymm15); // a3b3c3d3a7b7c7d7

  _mm256_storeu2_m128_emulate(outptr + 6 * 4, outptr + 6 * 0, ymm0);
  _mm256_storeu2_m128_emulate(outptr + 6 * 5, outptr + 6 * 1, ymm1);
  _mm256_storeu2_m128_emulate(outptr + 6 * 6, outptr + 6 * 2, ymm2);
  _mm256_storeu2_m128_emulate(outptr + 6 * 7, outptr + 6 * 3, ymm3);
  _mm256_storeu2_m128_emulate(outptr + 6 * 12, outptr + 6 * 8, ymm4);
  _mm256_storeu2_m128_emulate(outptr + 6 * 13, outptr + 6 * 9, ymm5);
  _mm256_storeu2_m128_emulate(outptr + 6 * 14, outptr + 6 * 10, ymm6);
  _mm256_storeu2_m128_emulate(outptr + 6 * 15, outptr + 6 * 11, ymm7);

  float other[4 * 8];
  ymm8 = _mm256_loadu_ps(inptr4 + 0);  // E0E1E2E3E4E5E6E7
  ymm9 = _mm256_loadu_ps(inptr4 + 8);  // e0e1e2e3e4e5e6e7
  ymm10 = _mm256_loadu_ps(inptr5 + 0); // F0F1F2F3F4F5F6F7
  ymm11 = _mm256_loadu_ps(inptr5 + 8); // f0f1f2f3f4f5f6f7
  _mm256_storeu_ps(other, ymm8);
  _mm256_storeu_ps(other + 8, ymm9);
  _mm256_storeu_ps(other + 16, ymm10);
  _mm256_storeu_ps(other + 24, ymm11);

  for (size_t i = 0; i < 16; i++) {
    outptr[6 * i + 4] = other[i];
    outptr[6 * i + 5] = other[i + 16];
  }
}

DNN_AVX2_TARGET
void transpose_6x8_1_s(const float *inptr0, const float *inptr1,
                       const float *inptr2, const float *inptr3,
                       const float *inptr4, const float *inptr5,
                       float *outptr) {
  auto ymm0 = _mm256_loadu_ps(inptr0); // A0A1A2A3A4A5A6A7
  auto ymm1 = _mm256_loadu_ps(inptr1); // B0B1B2B3B4B5B6B7
  auto ymm2 = _mm256_loadu_ps(inptr2); // C0C1C2C3C4C5C6C7
  auto ymm3 = _mm256_loadu_ps(inptr3); // D0D1D2D3D4D5D6D7

  auto ymm4 = _mm256_unpacklo_ps(ymm0, ymm2); // A0C0A1C1A4C4A5C5
  auto ymm5 = _mm256_unpackhi_ps(ymm0, ymm2); // A2C2A3C3A6C6A7C7
  auto ymm6 = _mm256_unpacklo_ps(ymm1, ymm3); // B0D0B1D1B4D4B5D5
  auto ymm7 = _mm256_unpackhi_ps(ymm1, ymm3); // B2D2B3D3B6D6B7D7

  auto ymm8 = _mm256_unpacklo_ps(ymm4, ymm6);  // A0B0C0D0A4B4C4D4
  auto ymm9 = _mm256_unpackhi_ps(ymm4, ymm6);  // A1B1C1D1A5B5C5D5
  auto ymm10 = _mm256_unpacklo_ps(ymm5, ymm7); // A2B2C2D2A6B6C6D6
  auto ymm11 = _mm256_unpackhi_ps(ymm5, ymm7); // A3B3C3D3A7B7C7D7

  _mm256_storeu2_m128_emulate(outptr + 6 * 4, outptr + 6 * 0, ymm8);
  _mm256_storeu2_m128_emulate(outptr + 6 * 5, outptr + 6 * 1, ymm9);
  _mm256_storeu2_m128_emulate(outptr + 6 * 6, outptr + 6 * 2, ymm10);
  _mm256_storeu2_m128_emulate(outptr + 6 * 7, outptr + 6 * 3, ymm11);
  float other[16];
  auto ymm12 = _mm256_loadu_ps(inptr4); // E0E1E2E3E4E5E6E7
  auto ymm13 = _mm256_loadu_ps(inptr5); // F0F1F2F3F4F5F6F7
  _mm256_storeu_ps(other, ymm12);
  _mm256_storeu_ps(other + 8, ymm13);

  for (size_t i = 0; i < 8; i++) {
    outptr[6 * i + 4] = other[i];
    outptr[6 * i + 5] = other[8 + i];
  }
}

DNN_AVX2_TARGET
void transpose_6x4_1_s(const float *inptr0, const float *inptr1,
                       const float *inptr2, const float *inptr3,
                       const float *inptr4, const float *inptr5,
                       float *outptr) {
  const std::uint32_t arr[8] = {0, 1, 4, 5, 2, 3, 6, 7};
  __m256i order = _mm256_loadu_si256((const __m256i *)arr);
  auto ymm0 = _mm256_loadu2_m128_emulate(inptr2, inptr0); // A0A1A2A3C0C1C2C3
  auto ymm1 = _mm256_loadu2_m128_emulate(inptr3, inptr1); // B0B1B2B3D0D1D2D3
  auto ymm2 = _mm256_unpacklo_ps(ymm0, ymm1);             // A0B0A1B1C0D0C1D1
  auto ymm3 = _mm256_unpackhi_ps(ymm0, ymm1);             // A2B2A3B3C2D2C3D3
  auto ymm4 = _mm256_permutevar8x32_ps(ymm2, order);      // A0B0C0D0A1B1C1D1
  auto ymm5 = _mm256_permutevar8x32_ps(ymm3, order);      // A2B2C2D2A3B3C3D3

  _mm256_storeu2_m128_emulate(outptr + 6 * 1, outptr + 6 * 0, ymm4);
  _mm256_storeu2_m128_emulate(outptr + 6 * 3, outptr + 6 * 2, ymm5);
  float other[8];
  auto ymm6 = _mm256_loadu2_m128_emulate(inptr5, inptr4); // E0E1E2E3E4E5E6E7
  _mm256_storeu_ps(other, ymm6);

  for (size_t i = 0; i < 4; i++) {
    outptr[6 * i + 4] = other[i];
    outptr[6 * i + 5] = other[4 + i];
  }
}

DNN_AVX2_TARGET
void transpose_4x8_1_s(const float *inptr0, const float *inptr1,
                       const float *inptr2, const float *inptr3,
                       float *outptr) {
  auto ymm0 = _mm256_loadu_ps(inptr0); // A0A1A2A3A4A5A6A7
  auto ymm1 = _mm256_loadu_ps(inptr1); // B0B1B2B3B4B5B6B7
  auto ymm2 = _mm256_loadu_ps(inptr2); // C0C1C2C3C4C5C6C7
  auto ymm3 = _mm256_loadu_ps(inptr3); // D0D1D2D3D4D5D6D7

  auto ymm4 = _mm256_unpacklo_ps(ymm0, ymm2); // A0C0A1C1A4C4A5C5
  auto ymm5 = _mm256_unpackhi_ps(ymm0, ymm2); // A2C2A3C3A6C6A7C7
  auto ymm6 = _mm256_unpacklo_ps(ymm1, ymm3); // B0D0B1D1B4D4B5D5
  auto ymm7 = _mm256_unpackhi_ps(ymm1, ymm3); // B2D2B3D3B6D6B7D7

  auto ymm8 = _mm256_unpacklo_ps(ymm4, ymm6);  // A0B0C0D0A4B4C4D4
  auto ymm9 = _mm256_unpackhi_ps(ymm4, ymm6);  // A1B1C1D1A5B5C5D5
  auto ymm10 = _mm256_unpacklo_ps(ymm5, ymm7); // A2B2C2D2A6B6C6D6
  auto ymm11 = _mm256_unpackhi_ps(ymm5, ymm7); // A3B3C3D3A7B7C7D7

  ymm0 = _mm256_permute2f128_ps(ymm8, ymm9, 0x20);   // A0B0C0D0A1B1C1D1
  ymm1 = _mm256_permute2f128_ps(ymm10, ymm11, 0x20); // A2B2C2D2A3B3C3D3
  ymm2 = _mm256_permute2f128_ps(ymm8, ymm9, 0x31);   // A4B4C4D4A5B5C5D5
  ymm3 = _mm256_permute2f128_ps(ymm10, ymm11, 0x31); // A6B6C6D6A7B7C7D7

  _mm256_storeu_ps(outptr + 8 * 0, ymm0);
  _mm256_storeu_ps(outptr + 8 * 1, ymm1);
  _mm256_storeu_ps(outptr + 8 * 2, ymm2);
  _mm256_storeu_ps(outptr + 8 * 3, ymm3);
}

DNN_AVX2_TARGET
void transpose_4x4_1_s(const float *inptr0, const float *inptr1,
                       const float *inptr2, const float *inptr3,
                       float *outptr) {
  const std::uint32_t arr[8] = {0, 1, 4, 5, 2, 3, 6, 7};
  __m256i order = _mm256_loadu_si256((const __m256i *)arr);
  auto ymm0 = _mm256_loadu2_m128_emulate(inptr2, inptr0); // A0A1A2A3C0C1C2C3
  auto ymm1 = _mm256_loadu2_m128_emulate(inptr3, inptr1); // B0B1B2B3D0D1D2D3
  auto ymm2 = _mm256_unpacklo_ps(ymm0, ymm1);             // A0B0A1B1C0D0C1D1
  auto ymm3 = _mm256_unpackhi_ps(ymm0, ymm1);             // A2B2A3B3C2D2C3D3
  auto ymm4 = _mm256_permutevar8x32_ps(ymm2, order);      // A0B0C0D0A1B1C1D1
  auto ymm5 = _mm256_permutevar8x32_ps(ymm3, order);      // A2B2C2D2A3B3C3D3
  _mm256_storeu_ps(outptr, ymm4);
  _mm256_storeu_ps(outptr + 8, ymm5);
}

void transpose_2x16_1_s(const float *inptr0, const float *inptr1,
                        float *outptr) {
  for (size_t i = 0; i < 16; i++) {
    *outptr++ = inptr0[i];
    *outptr++ = inptr1[i];
  }
}
void transpose_2x8_1_s(const float *inptr0, const float *inptr1,
                       float *outptr) {
  for (size_t i = 0; i < 8; i++) {
    *outptr++ = inptr0[i];
    *outptr++ = inptr1[i];
  }
}
void transpose_2x4_1_s(const float *inptr0, const float *inptr1,
                       float *outptr) {
  for (size_t i = 0; i < 4; i++) {
    *outptr++ = inptr0[i];
    *outptr++ = inptr1[i];
  }
}

DNN_AVX2_TARGET
void interleave_1x16_1_s(const float *inptr0, float *outptr) {
  auto ymm0 = _mm256_loadu_ps(inptr0);
  auto ymm1 = _mm256_loadu_ps(inptr0 + 8);
  _mm256_storeu_ps(outptr, ymm0);
  _mm256_storeu_ps(outptr + 8, ymm1);
}

DNN_AVX2_TARGET
void interleave_8x16_1_s(const float *inptr0, const float *inptr1,
                         const float *inptr2, const float *inptr3,
                         const float *inptr4, const float *inptr5,
                         const float *inptr6, const float *inptr7,
                         float *outptr) {
  auto ymm0 = _mm256_loadu_ps(inptr0);
  auto ymm1 = _mm256_loadu_ps(inptr0 + 8);
  auto ymm2 = _mm256_loadu_ps(inptr1);
  auto ymm3 = _mm256_loadu_ps(inptr1 + 8);
  auto ymm4 = _mm256_loadu_ps(inptr2);
  auto ymm5 = _mm256_loadu_ps(inptr2 + 8);
  auto ymm6 = _mm256_loadu_ps(inptr3);
  auto ymm7 = _mm256_loadu_ps(inptr3 + 8);
  auto ymm8 = _mm256_loadu_ps(inptr4);
  auto ymm9 = _mm256_loadu_ps(inptr4 + 8);
  auto ymm10 = _mm256_loadu_ps(inptr5);
  auto ymm11 = _mm256_loadu_ps(inptr5 + 8);
  auto ymm12 = _mm256_loadu_ps(inptr6);
  auto ymm13 = _mm256_loadu_ps(inptr6 + 8);
  auto ymm14 = _mm256_loadu_ps(inptr7);
  auto ymm15 = _mm256_loadu_ps(inptr7 + 8);

  _mm256_storeu_ps(outptr + 8 * 0, ymm0);
  _mm256_storeu_ps(outptr + 8 * 1, ymm1);
  _mm256_storeu_ps(outptr + 8 * 2, ymm2);
  _mm256_storeu_ps(outptr + 8 * 3, ymm3);
  _mm256_storeu_ps(outptr + 8 * 4, ymm4);
  _mm256_storeu_ps(outptr + 8 * 5, ymm5);
  _mm256_storeu_ps(outptr + 8 * 6, ymm6);
  _mm256_storeu_ps(outptr + 8 * 7, ymm7);
  _mm256_storeu_ps(outptr + 8 * 8, ymm8);
  _mm256_storeu_ps(outptr + 8 * 9, ymm9);
  _mm256_storeu_ps(outptr + 8 * 10, ymm10);
  _mm256_storeu_ps(outptr + 8 * 11, ymm11);
  _mm256_storeu_ps(outptr + 8 * 12, ymm12);
  _mm256_storeu_ps(outptr + 8 * 13, ymm13);
  _mm256_storeu_ps(outptr + 8 * 14, ymm14);
  _mm256_storeu_ps(outptr + 8 * 15, ymm15);
}

DNN_AVX2_TARGET
void interleave_8x4_1_s(const float *inptr0, const float *inptr1,
                        const float *inptr2, const float *inptr3,
                        const float *inptr4, const float *inptr5,
                        const float *inptr6, const float *inptr7,
                        float *outptr) {
  auto ymm0 = _mm256_loadu2_m128_emulate(inptr1, inptr0); // A0A1A2A3B0B1B2B3
  auto ymm1 = _mm256_loadu2_m128_emulate(inptr3, inptr2); // C0C1C2C3D0D1D2D3
  auto ymm2 = _mm256_loadu2_m128_emulate(inptr5, inptr4); // E0E1E2E3F0F1F2F3
  auto ymm3 = _mm256_loadu2_m128_emulate(inptr7, inptr6); // G0G1G2G3H0H1H2H3
  _mm256_storeu_ps(outptr + 8 * 0, ymm0);
  _mm256_storeu_ps(outptr + 8 * 1, ymm1);
  _mm256_storeu_ps(outptr + 8 * 2, ymm2);
  _mm256_storeu_ps(outptr + 8 * 3, ymm3);
}

void interleave_8x2_1_s(const float *inptr0, const float *inptr1,
                        const float *inptr2, const float *inptr3,
                        const float *inptr4, const float *inptr5,
                        const float *inptr6, const float *inptr7,
                        float *outptr) {
#define cb(i)                                                                  \
  *outptr++ = inptr##i[0];                                                     \
  *outptr++ = inptr##i[1];
  UNROLL_CODE(cb, 8)
#undef cb
}

void interleave_1x4_1_s(const float *inptr0, float *outptr) {
  outptr[0] = inptr0[0];
  outptr[1] = inptr0[1];
  outptr[2] = inptr0[2];
  outptr[3] = inptr0[3];
}
void interleave_8x6_1_s(const float *inptr0, const float *inptr1,
                        const float *inptr2, const float *inptr3,
                        const float *inptr4, const float *inptr5,
                        const float *inptr6, const float *inptr7,
                        float *outptr) {
#define cb(i) auto xmm##i = _mm_loadu_ps(inptr##i);
  UNROLL_CODE(cb, 8)
#undef cb
#define cb(i) _mm_storeu_ps(outptr + 6 * i, xmm##i);
  UNROLL_CODE(cb, 8)
#undef cb
#define cb(i)                                                                  \
  outptr[6 * i + 4] = inptr##i[4];                                             \
  outptr[6 * i + 5] = inptr##i[5];
  UNROLL_CODE(cb, 8)
#undef cb
}

void interleave_1x6_1_s(const float *inptr0, float *outptr) {
  outptr[0] = inptr0[0];
  outptr[1] = inptr0[1];
  outptr[2] = inptr0[2];
  outptr[3] = inptr0[3];
  outptr[4] = inptr0[4];
  outptr[5] = inptr0[5];
}

void interleave_1x2_1_s(const float *inptr0, float *outptr) {
  outptr[0] = inptr0[0];
  outptr[1] = inptr0[1];
}

static inline void interleave_helper(const float *inptr, float *outptr,
                                     int unroll_k, int ksize, float val) {
  int k = 0;
  for (; k < ksize; k++) {
    *outptr++ = *inptr++;
  }
  for (; k < unroll_k; k++) {
    *outptr++ = val;
  }
}
void interleave_1(const float *inptr0, float *outptr, int unroll_k, int ksize,
                  float val) {
  for (int k = 0; k < ksize; k += unroll_k) {
    int size = min(unroll_k, ksize - k);
    interleave_helper(inptr0, outptr, unroll_k, size, val);
    inptr0 += size;
    outptr += unroll_k;
  }
}

void interleave_8(const float *inptr0, const float *inptr1, const float *inptr2,
                  const float *inptr3, const float *inptr4, const float *inptr5,
                  const float *inptr6, const float *inptr7, float *outptr,
                  int unroll_k, int ksize, float val) {
  for (int k = 0; k < ksize; k += unroll_k) {
    int size = min(unroll_k, ksize - k);
    interleave_helper(inptr0, outptr, unroll_k, size, val);
    inptr0 += size;
    outptr += unroll_k;
    interleave_helper(inptr1, outptr, unroll_k, size, val);
    inptr1 += size;
    outptr += unroll_k;
    interleave_helper(inptr2, outptr, unroll_k, size, val);
    inptr2 += size;
    outptr += unroll_k;
    interleave_helper(inptr3, outptr, unroll_k, size, val);
    inptr3 += size;
    outptr += unroll_k;
    interleave_helper(inptr4, outptr, unroll_k, size, val);
    inptr4 += size;
    outptr += unroll_k;
    interleave_helper(inptr5, outptr, unroll_k, size, val);
    inptr5 += size;
    outptr += unroll_k;
    interleave_helper(inptr6, outptr, unroll_k, size, val);
    inptr6 += size;
    outptr += unroll_k;
    interleave_helper(inptr7, outptr, unroll_k, size, val);
    inptr7 += size;
    outptr += unroll_k;
  }
}

DNN_AVX2_TARGET
MEGDNN_ATTRIBUTE_TARGET("fma")
void gemm_6x16_kern2x16(const float *packA, const float *packB, int K,
                        float *output, int LDC, bool is_first_k, int m_remain) {
  const float *cur_b = packB;
  const float *cur_a = packA;
  __m256 ymm0, ymm1, ymm2, ymm3;
  __m256 b_tmp0, b_tmp1;
  __m256 tmp;
  if (is_first_k) {
#define cb(i) ymm##i = _mm256_set1_ps(0.0f);
    UNROLL_CODE(cb, 4)
#undef cb
  } else {
    ymm0 = _mm256_loadu_ps(output + LDC * 0 + 0);
    ymm1 = _mm256_loadu_ps(output + LDC * 0 + 8);
    ymm2 = _mm256_loadu_ps(output + LDC * 1 + 0);
    ymm3 = _mm256_loadu_ps(output + LDC * 1 + 8);
  }
  b_tmp0 = _mm256_loadu_ps(cur_b);
  b_tmp1 = _mm256_loadu_ps(cur_b + 8);
  size_t i = 0;
  for (; i + 2 <= K; i += 2) {
    cur_b += 16;

#define CAL_OUPUT(i, first, second)                                            \
  tmp = _mm256_broadcast_ss(cur_a + i);                                        \
  ymm##first = _mm256_fmadd_ps(b_tmp0, tmp, ymm##first);                       \
  ymm##second = _mm256_fmadd_ps(b_tmp1, tmp, ymm##second);

    CAL_OUPUT(0, 0, 1)
    CAL_OUPUT(1, 2, 3)
    b_tmp0 = _mm256_loadu_ps(cur_b);
    b_tmp1 = _mm256_loadu_ps(cur_b + 8);
    cur_b += 16;
    CAL_OUPUT(2, 0, 1)
    CAL_OUPUT(3, 2, 3)
    cur_a += 4;
    b_tmp0 = _mm256_loadu_ps(cur_b);
    b_tmp1 = _mm256_loadu_ps(cur_b + 8);
  }
  if (i < K) {
    CAL_OUPUT(0, 0, 1)
    CAL_OUPUT(1, 2, 3)
  }
#undef CAL_OUPUT
  switch (m_remain) {
  case 2:
    _mm256_storeu_ps(output + LDC * 1 + 0, ymm2);
    _mm256_storeu_ps(output + LDC * 1 + 8, ymm3);
  case 1:
    _mm256_storeu_ps(output + LDC * 0 + 0, ymm0);
    _mm256_storeu_ps(output + LDC * 0 + 8, ymm1);
  default:
    break;
  }
}

DNN_AVX2_TARGET
MEGDNN_ATTRIBUTE_TARGET("fma")
void gemm_6x16_kern6x4(const float *packA, const float *packB, int K,
                       float *output, int LDC, bool is_first_k, int n_remain) {
  const float *cur_b = packB;
  const float *cur_a = packA;
  __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;
  __m128 tmp_a, tmp_b;
  if (is_first_k) {
    xmm0 = _mm_set1_ps(0.0f);
    xmm1 = _mm_set1_ps(0.0f);
    xmm2 = _mm_set1_ps(0.0f);
    xmm3 = _mm_set1_ps(0.0f);
    xmm4 = _mm_set1_ps(0.0f);
    xmm5 = _mm_set1_ps(0.0f);
  } else {
    xmm0 = _mm_loadu_ps(output + LDC * 0);
    xmm1 = _mm_loadu_ps(output + LDC * 1);
    xmm2 = _mm_loadu_ps(output + LDC * 2);
    xmm3 = _mm_loadu_ps(output + LDC * 3);
    xmm4 = _mm_loadu_ps(output + LDC * 4);
    xmm5 = _mm_loadu_ps(output + LDC * 5);
  }

  for (size_t i = 0; i < K; i++) {
    tmp_b = _mm_loadu_ps(cur_b);
    cur_b += 4;
    tmp_a = _mm_broadcast_ss(cur_a);
    xmm0 = _mm_fmadd_ps(tmp_a, tmp_b, xmm0);
    tmp_a = _mm_broadcast_ss(cur_a + 1);
    xmm1 = _mm_fmadd_ps(tmp_a, tmp_b, xmm1);
    tmp_a = _mm_broadcast_ss(cur_a + 2);
    xmm2 = _mm_fmadd_ps(tmp_a, tmp_b, xmm2);
    tmp_a = _mm_broadcast_ss(cur_a + 3);
    xmm3 = _mm_fmadd_ps(tmp_a, tmp_b, xmm3);
    tmp_a = _mm_broadcast_ss(cur_a + 4);
    xmm4 = _mm_fmadd_ps(tmp_a, tmp_b, xmm4);
    tmp_a = _mm_broadcast_ss(cur_a + 5);
    xmm5 = _mm_fmadd_ps(tmp_a, tmp_b, xmm5);
    cur_a += 6;
  }
  if (n_remain == 4) {
    _mm_storeu_ps(output + LDC * 0, xmm0);
    _mm_storeu_ps(output + LDC * 1, xmm1);
    _mm_storeu_ps(output + LDC * 2, xmm2);
    _mm_storeu_ps(output + LDC * 3, xmm3);
    _mm_storeu_ps(output + LDC * 4, xmm4);
    _mm_storeu_ps(output + LDC * 5, xmm5);
  } else {
    float dst[6 * 4];
    _mm_storeu_ps(dst + 4 * 0, xmm0);
    _mm_storeu_ps(dst + 4 * 1, xmm1);
    _mm_storeu_ps(dst + 4 * 2, xmm2);
    _mm_storeu_ps(dst + 4 * 3, xmm3);
    _mm_storeu_ps(dst + 4 * 4, xmm4);
    _mm_storeu_ps(dst + 4 * 5, xmm5);
    for (size_t i = 0; i < n_remain; i++) {
      for (size_t j = 0; j < 6; j++) {
        output[LDC * j + i] = dst[4 * j + i];
      }
    }
  }
}

DNN_AVX2_TARGET
MEGDNN_ATTRIBUTE_TARGET("fma")
void gemm_6x16_kern2x4(const float *packA, const float *packB, int K,
                       float *output, int LDC, bool is_first_k, int m_remain,
                       int n_remain) {
  const float *cur_b = packB;
  const float *cur_a = packA;
  __m128 xmm0, xmm1;
  __m128 tmp_a, tmp_b;
  if (is_first_k) {
    xmm0 = _mm_set1_ps(0.0f);
    xmm1 = _mm_set1_ps(0.0f);
  } else {
    xmm0 = _mm_loadu_ps(output + LDC * 0);
    xmm1 = _mm_loadu_ps(output + LDC * 1);
  }

  for (size_t i = 0; i < K; i++) {
    tmp_b = _mm_loadu_ps(cur_b);
    cur_b += 4;
    tmp_a = _mm_broadcast_ss(cur_a);
    xmm0 = _mm_fmadd_ps(tmp_a, tmp_b, xmm0);
    tmp_a = _mm_broadcast_ss(cur_a + 1);
    xmm1 = _mm_fmadd_ps(tmp_a, tmp_b, xmm1);
    cur_a += 2;
  }
  float dst[2 * 4];
  _mm_storeu_ps(dst + 4 * 0, xmm0);
  _mm_storeu_ps(dst + 4 * 1, xmm1);
  for (size_t i = 0; i < n_remain; i++) {
    for (size_t j = 0; j < m_remain; j++) {
      output[LDC * j + i] = dst[4 * j + i];
    }
  }
}

DNN_AVX2_TARGET
MEGDNN_ATTRIBUTE_TARGET("fma")
void gemm_6x16_kern6x16(const float *packA, const float *packB, int K,
                        float *output, int LDC, bool is_first_k) {
  const float *cur_b = packB;
  const float *cur_a = packA;
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10,
      ymm11;
  __m256 b_tmp0, b_tmp1;
  __m256 tmp;
  if (is_first_k) {
#define cb(i) ymm##i = _mm256_set1_ps(0.0f);
    UNROLL_CODE(cb, 12)
#undef cb
  } else {
    ymm0 = _mm256_loadu_ps(output + LDC * 0 + 0);
    ymm1 = _mm256_loadu_ps(output + LDC * 0 + 8);
    ymm2 = _mm256_loadu_ps(output + LDC * 1 + 0);
    ymm3 = _mm256_loadu_ps(output + LDC * 1 + 8);
    ymm4 = _mm256_loadu_ps(output + LDC * 2 + 0);
    ymm5 = _mm256_loadu_ps(output + LDC * 2 + 8);
    ymm6 = _mm256_loadu_ps(output + LDC * 3 + 0);
    ymm7 = _mm256_loadu_ps(output + LDC * 3 + 8);
    ymm8 = _mm256_loadu_ps(output + LDC * 4 + 0);
    ymm9 = _mm256_loadu_ps(output + LDC * 4 + 8);
    ymm10 = _mm256_loadu_ps(output + LDC * 5 + 0);
    ymm11 = _mm256_loadu_ps(output + LDC * 5 + 8);
  }
  b_tmp0 = _mm256_loadu_ps(cur_b);
  b_tmp1 = _mm256_loadu_ps(cur_b + 8);
  size_t i = 0;
  for (; i + 2 <= K; i += 2) {
    cur_b += 16;

#define CAL_OUPUT(i, first, second)                                            \
  tmp = _mm256_broadcast_ss(cur_a + i);                                        \
  ymm##first = _mm256_fmadd_ps(b_tmp0, tmp, ymm##first);                       \
  ymm##second = _mm256_fmadd_ps(b_tmp1, tmp, ymm##second);

    CAL_OUPUT(0, 0, 1)
    CAL_OUPUT(1, 2, 3)
    CAL_OUPUT(2, 4, 5)
    CAL_OUPUT(3, 6, 7)
    CAL_OUPUT(4, 8, 9)
    CAL_OUPUT(5, 10, 11)
    b_tmp0 = _mm256_loadu_ps(cur_b);
    b_tmp1 = _mm256_loadu_ps(cur_b + 8);
    cur_b += 16;
    CAL_OUPUT(6, 0, 1)
    CAL_OUPUT(7, 2, 3)
    CAL_OUPUT(8, 4, 5)
    CAL_OUPUT(9, 6, 7)
    CAL_OUPUT(10, 8, 9)
    CAL_OUPUT(11, 10, 11)
    cur_a += 12;
    b_tmp0 = _mm256_loadu_ps(cur_b);
    b_tmp1 = _mm256_loadu_ps(cur_b + 8);
  }
  if (i < K) {
    CAL_OUPUT(0, 0, 1)
    CAL_OUPUT(1, 2, 3)
    CAL_OUPUT(2, 4, 5)
    CAL_OUPUT(3, 6, 7)
    CAL_OUPUT(4, 8, 9)
    CAL_OUPUT(5, 10, 11)
  }
#undef CAL_OUPUT
  _mm256_storeu_ps(output + LDC * 0 + 0, ymm0);
  _mm256_storeu_ps(output + LDC * 0 + 8, ymm1);
  _mm256_storeu_ps(output + LDC * 1 + 0, ymm2);
  _mm256_storeu_ps(output + LDC * 1 + 8, ymm3);
  _mm256_storeu_ps(output + LDC * 2 + 0, ymm4);
  _mm256_storeu_ps(output + LDC * 2 + 8, ymm5);
  _mm256_storeu_ps(output + LDC * 3 + 0, ymm6);
  _mm256_storeu_ps(output + LDC * 3 + 8, ymm7);
  _mm256_storeu_ps(output + LDC * 4 + 0, ymm8);
  _mm256_storeu_ps(output + LDC * 4 + 8, ymm9);
  _mm256_storeu_ps(output + LDC * 5 + 0, ymm10);
  _mm256_storeu_ps(output + LDC * 5 + 8, ymm11);
}

void gemm_6x16_kern(const float *packA, const float *packB, size_t M, size_t N,
                    size_t K, float *C, size_t LDC, int is_first_k) {
  size_t n = 0;
  const int K2 = K * 2;
  const int K4 = K * 4;
  const int K6 = K * 6;
  const int K16 = K * 16;
  const int A_INTERLEAVE6 = 6;
  const int A_INTERLEAVE2 = 2;
  const int B_INTERLEAVE16 = 16;
  const int B_INTERLEAVE4 = 4;
  auto *cur_packB = packB;
  for (; n + B_INTERLEAVE16 <= N; n += B_INTERLEAVE16) {
    size_t m = 0;
    auto output = C + n;
    auto *cur_packA = packA;
    for (; m + A_INTERLEAVE6 <= M; m += A_INTERLEAVE6) {
      gemm_6x16_kern6x16(cur_packA, cur_packB, K, output, LDC, is_first_k);
      output += A_INTERLEAVE6 * LDC;
      cur_packA += K6;
    }
    for (; m < M; m += A_INTERLEAVE2) {
      gemm_6x16_kern2x16(cur_packA, cur_packB, K, output, LDC, is_first_k,
                         min(M - m, 2));
      output += A_INTERLEAVE2 * LDC;
      cur_packA += K2;
    }
    cur_packB += K16;
  }

  for (; n < N; n += B_INTERLEAVE4) {
    size_t m = 0;
    auto output = C + n;
    auto *cur_packA = packA;
    for (; m + A_INTERLEAVE6 <= M; m += A_INTERLEAVE6) {
      gemm_6x16_kern6x4(cur_packA, cur_packB, K, output, LDC, is_first_k,
                        min(N - n, 4));
      output += A_INTERLEAVE6 * LDC;
      cur_packA += K6;
    }
    for (; m < M; m += A_INTERLEAVE2) {

      gemm_6x16_kern2x4(cur_packA, cur_packB, K, output, LDC, is_first_k,
                        min(M - m, 2), min(N - n, 4));
      output += A_INTERLEAVE2 * LDC;
      cur_packA += K2;
    }
    cur_packB += K4;
  }
}

void gemm_6x16_pack_A_t(float *outptr, const float *inptr, int ldin, int x0,
                        int xmax, int k0, int kmax) {
  size_t ksize = kmax - k0;
  size_t ksize6 = ksize * 6;
  size_t ksize2 = ksize * 2;
  float *outptr_base6 = outptr;
  float *outptr_base2 = outptr_base6 + (xmax - x0) / 6 * ksize6;
  size_t k = k0;

  for (; k + 7 < kmax; k += 8) {
    const float *cur_inptr = inptr + k * ldin + k0;
#define cb(i) const float *inptr##i = cur_inptr + ldin * i;
    UNROLL_CODE(cb, 8)
#undef cb
#define cb(i) __builtin_prefetch(inptr##i, 0, 3);
    UNROLL_CODE(cb, 8)
#undef cb
    int x = x0;
    float *outptr = outptr_base6;
    for (; x + 6 <= xmax; x += 6) {
      interleave_8x6_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr);
#define cb(i) inptr##i += 6;
      UNROLL_CODE(cb, 8)
#undef cb
      outptr += ksize6;
    }
    outptr = outptr_base2;
    for (; x + 2 <= xmax; x += 2) {
      interleave_8x2_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr);
#define cb(i) inptr##i += 2;
      UNROLL_CODE(cb, 8)
#undef cb
      outptr += ksize2;
    }
    if (x < xmax) {
      interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                   inptr7, outptr, 2, xmax - x, 0);
      inptr0 += xmax - x;
      inptr1 += xmax - x;
      inptr2 += xmax - x;
      inptr3 += xmax - x;
      inptr4 += xmax - x;
      inptr5 += xmax - x;
      inptr6 += xmax - x;
      inptr7 += xmax - x;
    }
    outptr_base6 += 8 * 6;
    outptr_base2 += 8 * 2;
  }
  for (; k < kmax; k++) {
    const float *inptr0 = inptr + k * ldin + k0;
    __builtin_prefetch(inptr0, 0, 3);
    int x = x0;
    float *outptr = outptr_base6;
    for (; x + 6 <= xmax; x += 6) {
      interleave_1x6_1_s(inptr0, outptr);
      inptr0 += 6;
      outptr += ksize6;
    }
    outptr = outptr_base2;
    for (; x + 2 <= xmax; x += 2) {
      interleave_1x2_1_s(inptr0, outptr);
      inptr0 += 2;
      outptr += ksize2;
    }
    if (x < xmax) {
      interleave_1(inptr0, outptr, 2, xmax - x, 0);
      inptr0 += xmax - x;
      outptr += 2;
    }
    outptr_base6 += 6;
    outptr_base2 += 2;
  }
}

void gemm_6x16_pack_A_n(float *outptr, const float *inptr, int ldin, int y0,
                        int ymax, int k0, int kmax) {
  float zerobuff[16];
  memset(zerobuff, 0, sizeof(float) * 16);
  size_t y = y0;
  const size_t PACK_SIZE_96 = 6 * 16;
  const size_t PACK_SIZE_48 = 6 * 8;
  const size_t PACK_SIZE_24 = 6 * 4;
  const size_t PACK_SIZE_32 = 4 * 8;
  const size_t PACK_SIZE_16 = 4 * 4;
  const size_t PACK_SIZE_8 = 4 * 2;
  for (; y + 5 < ymax; y += 6) {
    const float *cur_inptr = inptr + y * ldin + k0;
#define cb(i) const float *inptr##i = cur_inptr + ldin * i;
    UNROLL_CODE(cb, 6)
#undef cb
#define cb(i) __builtin_prefetch(inptr##i, 0, 3);
    UNROLL_CODE(cb, 6)
#undef cb
    int x = (kmax - k0);
    for (; x > 15; x -= 16) {
      transpose_6x16_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                         outptr);
#define cb(i) inptr##i += 16;
      UNROLL_CODE(cb, 6)
#undef cb
      outptr += PACK_SIZE_96;
    }
    for (; x > 7; x -= 8) {
      transpose_6x8_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, outptr);
#define cb(i) inptr##i += 8;
      UNROLL_CODE(cb, 6)
#undef cb
      outptr += PACK_SIZE_48;
    }
    for (; x > 3; x -= 4) {
      transpose_6x4_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, outptr);
#define cb(i) inptr##i += 4;
      UNROLL_CODE(cb, 6)
#undef cb
      outptr += PACK_SIZE_24;
    }
    for (; x > 0; x--) {
#define cb(i) *outptr++ = *inptr##i++;
      UNROLL_CODE(cb, 6)
#undef cb
    }
  }
  for (; y < ymax; y += 2) {
    const float *cur_inptr = inptr + y * ldin + k0;
#define cb(i) const float *inptr##i = cur_inptr + ldin * i;
    UNROLL_CODE(cb, 2)
#undef cb
#define cb(i) __builtin_prefetch(inptr##i, 0, 3);
    UNROLL_CODE(cb, 2)
#undef cb
    int x = kmax - k0;
    for (; x > 15; x -= 16) {
      if ((y + 1) >= ymax) {
        inptr1 = zerobuff;
      }
      transpose_2x16_1_s(inptr0, inptr1, outptr);
#define cb(i) inptr##i += 16;
      UNROLL_CODE(cb, 2)
#undef cb
      outptr += PACK_SIZE_32;
    }
    for (; x > 7; x -= 8) {
      if ((y + 1) >= ymax) {
        inptr1 = zerobuff;
      }
      transpose_2x8_1_s(inptr0, inptr1, outptr);
#define cb(i) inptr##i += 8;
      UNROLL_CODE(cb, 2)
#undef cb
      outptr += PACK_SIZE_16;
    }
    for (; x > 3; x -= 4) {
      if ((y + 1) >= ymax) {
        inptr1 = zerobuff;
      }
      transpose_2x4_1_s(inptr0, inptr1, outptr);
#define cb(i) inptr##i += 4;
      UNROLL_CODE(cb, 2)
#undef cb
      outptr += PACK_SIZE_8;
    }
    if (x > 0) {
      if ((y + 1) >= ymax) {
        inptr1 = zerobuff;
      }
      for (size_t i = 0; i < x; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
      }
    }
  }
}

void gemm_6x16_pack_B_t(float *outptr, const float *inptr, int ldin, int y0,
                        int ymax, int k0, int kmax) {
  float zerobuff[16];
  memset(zerobuff, 0, sizeof(float) * 16);
  const size_t PACK_SIZE_128 = 8 * 16;
  const size_t PACK_SIZE_64 = 4 * 16;
  const size_t PACK_SiZE_32 = 4 * 8;
  const size_t PACK_SIZE_16 = 4 * 4;
  size_t y = y0;
  for (; y + 15 < ymax; y += 16) {
    const float *cur_inptr = inptr + y * ldin + k0;
#define cb(i) const float *inptr##i = cur_inptr + ldin * i;
    UNROLL_CODE(cb, 16)
#undef cb
#define cb(i) __builtin_prefetch(inptr##i, 0, 3);
    UNROLL_CODE(cb, 16)
#undef cb
    int x = (kmax - k0);
    for (; x > 7; x -= 8) {
      transpose_16x8_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, inptr8, inptr9, inptr10, inptr11, inptr12,
                         inptr13, inptr14, inptr15, outptr);
#define cb(i) inptr##i += 8;
      UNROLL_CODE(cb, 16)
#undef cb
      outptr += PACK_SIZE_128;
    }
    for (; x > 3; x -= 4) {
      transpose_16x4_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, inptr8, inptr9, inptr10, inptr11, inptr12,
                         inptr13, inptr14, inptr15, outptr);
#define cb(i) inptr##i += 4;
      UNROLL_CODE(cb, 16)
#undef cb
      outptr += PACK_SIZE_64;
    }
    for (; x > 0; x--) {
#define cb(i) *outptr++ = *inptr##i++;
      UNROLL_CODE(cb, 16)
#undef cb
    }
  }
  for (; y < ymax; y += 4) {
    const float *cur_inptr = inptr + y * ldin + k0;
#define cb(i) const float *inptr##i = cur_inptr + ldin * i;
    UNROLL_CODE(cb, 4)
#undef cb
#define cb(i) __builtin_prefetch(inptr##i, 0, 3);
    UNROLL_CODE(cb, 4)
#undef cb
    int x = kmax - k0;
    for (; x > 7; x -= 8) {
      if ((y + 3) >= ymax) {
        switch ((y + 3) - ymax) {
        case 2:
          inptr1 = zerobuff;
        case 1:
          inptr2 = zerobuff;
        case 0:
          inptr3 = zerobuff;
        default:
          break;
        }
      }
      transpose_4x8_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
#define cb(i) inptr##i += 8;
      UNROLL_CODE(cb, 4)
#undef cb
      outptr += PACK_SiZE_32;
    }
    for (; x > 3; x -= 4) {
      if ((y + 3) >= ymax) {
        switch ((y + 3) - ymax) {
        case 2:
          inptr1 = zerobuff;
        case 1:
          inptr2 = zerobuff;
        case 0:
          inptr3 = zerobuff;
        default:
          break;
        }
      }
      transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
#define cb(i) inptr##i += 4;
      UNROLL_CODE(cb, 4)
#undef cb
      outptr += PACK_SIZE_16;
    }
    if (x > 0) {
      if ((y + 3) >= ymax) {
        switch ((y + 3) - ymax) {
        case 2:
          inptr1 = zerobuff;
        case 1:
          inptr2 = zerobuff;
        case 0:
          inptr3 = zerobuff;
          break;
        }
      }
      for (size_t i = 0; i < x; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
      }
    }
  }
}

void gemm_6x16_pack_B_n(float *outptr, const float *inptr, int ldin, int x0,
                        int xmax, int k0, int kmax) {
  size_t ksize = kmax - k0;
  size_t ksize16 = ksize * 16;
  size_t ksize4 = ksize * 4;
  float *outptr_base16 = outptr;
  float *outptr_base4 = outptr_base16 + (xmax - x0) / 16 * ksize16;
  size_t k = k0;

  for (; k + 7 < kmax; k += 8) {
    const float *cur_inptr = inptr + k * ldin + k0;
#define cb(i) const float *inptr##i = cur_inptr + ldin * i;
    UNROLL_CODE(cb, 8)
#undef cb
#define cb(i) __builtin_prefetch(inptr##i, 0, 3);
    UNROLL_CODE(cb, 8)
#undef cb
    int x = x0;
    float *outptr = outptr_base16;
    for (; x + 16 <= xmax; x += 16) {
      interleave_8x16_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5,
                          inptr6, inptr7, outptr);
#define cb(i) inptr##i += 16;
      UNROLL_CODE(cb, 8)
#undef cb
      outptr += ksize16;
    }
    outptr = outptr_base4;
    for (; x + 4 <= xmax; x += 4) {
      interleave_8x4_1_s(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                         inptr7, outptr);
#define cb(i) inptr##i += 4;
      UNROLL_CODE(cb, 8)
#undef cb
      outptr += ksize4;
    }

    if (x < xmax) {
      interleave_8(inptr0, inptr1, inptr2, inptr3, inptr4, inptr5, inptr6,
                   inptr7, outptr, 4, xmax - x, 0);
      inptr0 += xmax - x;
      inptr1 += xmax - x;
      inptr2 += xmax - x;
      inptr3 += xmax - x;
      inptr4 += xmax - x;
      inptr5 += xmax - x;
      inptr6 += xmax - x;
      inptr7 += xmax - x;
    }
    outptr_base16 += 8 * 16;
    outptr_base4 += 8 * 4;
  }

  for (; k < kmax; k++) {
    const float *inptr0 = inptr + k * ldin + k0;
    __builtin_prefetch(inptr0, 0, 3);
    int x = x0;
    float *outptr = outptr_base16;
    for (; x + 16 <= xmax; x += 16) {
      interleave_1x16_1_s(inptr0, outptr);
      inptr0 += 16;
      outptr += ksize16;
    }
    outptr = outptr_base4;
    for (; x + 4 <= xmax; x += 4) {
      interleave_1x4_1_s(inptr0, outptr);
      inptr0 += 4;
      outptr += ksize4;
    }
    if (x < xmax) {
      interleave_1(inptr0, outptr, 4, xmax - x, 0);
      inptr0 += xmax - x;
      outptr += 4;
    }
    outptr_base16 += 16;
    outptr_base4 += 4;
  }
}
} // namespace
#undef UNROLL_CODE

namespace megdnn {
namespace x86 {
namespace matmul {
void sgemm_pack_6x16_avx2::pack_A(float *out, const float *in, int ldin, int y0,
                                  int ymax, int k0, int kmax,
                                  bool transpose_A) const {
  if (!transpose_A)
    gemm_6x16_pack_A_n(out, in, ldin, y0, ymax, k0, kmax);
  else
    gemm_6x16_pack_A_t(out, in, ldin, y0, ymax, k0, kmax);
}

void sgemm_pack_6x16_avx2::pack_B(float *out, const float *in, int ldin, int x0,
                                  int xmax, int k0, int kmax,
                                  bool transpose_B) const {
  if (!transpose_B)
    gemm_6x16_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
  else
    gemm_6x16_pack_B_t(out, in, ldin, x0, xmax, k0, kmax);
}

void sgemm_pack_6x16_avx2::kern(const float *packA, const float *packB,
                                size_t M, size_t N, size_t K, float *C,
                                size_t LDC, bool is_first_k, const float *bias,
                                float *workspace) const {
  MEGDNN_MARK_USED_VAR(bias);
  MEGDNN_MARK_USED_VAR(workspace);
  gemm_6x16_kern(packA, packB, M, N, K, C, LDC, is_first_k);
};
MEGDNN_REG_GEMM_STRATEGY_IMPL(sgemm_pack_6x16_avx2);
} // namespace matmul
} // namespace x86
} // namespace megdnn
