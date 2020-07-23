/**
 * \file dnn/src/armv7/matrix_mul/int8x8x16/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/armv7/matrix_mul/int8x8x16/kernel_4x2x16.h"
#include "src/armv7/matrix_mul/int8x8x16/kernel_4x8x8.h"
#include "src/armv7/matrix_mul/int8x8x16/kernel_mk4_8x8x4.h"
#include "src/armv7/matrix_mul/int8x8x16/strategy.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_common.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

// ===========================gemm_s8x8x16_4x2=================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8x8x16_4x2);

void gemm_s8x8x16_4x2::pack_A(dt_int8* out, const dt_int8* in, int ldin, int y0,
                              int ymax, int k0, int kmax,
                              bool transpose) const {
    if (transpose) {
        matmul_4x2x16::gemm_s8x8x16_4x2_pack_A_t(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    } else {
        matmul_4x2x16::gemm_s8x8x16_4x2_pack_A_n(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    }
}

void gemm_s8x8x16_4x2::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                              int xmax, int k0, int kmax,
                              bool transpose) const {
    if (transpose) {
        matmul_4x2x16::gemm_s8x8x16_4x2_pack_B_t(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    } else {
        matmul_4x2x16::gemm_s8x8x16_4x2_pack_B_n(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    }
}

void gemm_s8x8x16_4x2::kern(const dt_int8* packA, const dt_int8* packB,
                            size_t M, size_t N, size_t K, dt_int16* C,
                            size_t LDC, bool is_first_k, const dt_int16*,
                            dt_int16*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int8 &&
                            C_dtype.enumv() == DTypeEnum::Int16)),
                  "A: %s B: %s C: %s", A_dtype.name(), B_dtype.name(),
                  C_dtype.name());

    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 2;
    //! K is packed to times of 4
    K = round_up<size_t>(K, 16);
    const int K4 = K * 4;
    const int K2 = K * 2;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int16_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_4x2x16::kern_4x2(packA, cur_packB, K, output, LDC,
                                    is_first_k, 4, 2);
            output += B_INTERLEAVE;
            cur_packB += K2;
        }

        for (; n < N; n += B_INTERLEAVE) {
            matmul_4x2x16::kern_4x2(packA, cur_packB, K, output, LDC,
                                    is_first_k, 4, std::min<size_t>(N - n, 2));
            output += B_INTERLEAVE;
            cur_packB += K2;
        }

        packA += K4;
    }

    for (; m < M; m += 4) {
        int16_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n < N; n += B_INTERLEAVE) {
            matmul_4x2x16::kern_4x2(packA, cur_packB, K, output, LDC,
                                    is_first_k, std::min<size_t>(M - m, 4),
                                    std::min<size_t>(N - n, 2));
            output += B_INTERLEAVE;
            cur_packB += K2;
        }
        packA += K4;
    }
}

// ===========================gemm_s8x8x16_4x8==================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8x8x16_4x8);

void gemm_s8x8x16_4x8::pack_A(dt_int8* out, const dt_int8* in, int ldin, int y0,
                              int ymax, int k0, int kmax,
                              bool transpose) const {
    if (transpose) {
        matmul_4x8x8::gemm_s8x8x16_4x8_transpose_pack_A_n(out, in, ldin, y0,
                                                          ymax, k0, kmax);
    } else {
        matmul_4x8x8::gemm_s8x8x16_4x8_pack_A_n(out, in, ldin, y0, ymax, k0,
                                                kmax);
    }
}

void gemm_s8x8x16_4x8::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                              int xmax, int k0, int kmax,
                              bool transpose) const {
    if (transpose) {
        matmul_4x8x8::gemm_s8x8x16_4x8_transpose_pack_B_n(out, in, ldin, x0,
                                                          xmax, k0, kmax);
    } else {
        matmul_4x8x8::gemm_s8x8x16_4x8_pack_B_n(out, in, ldin, x0, xmax, k0,
                                                kmax);
    }
}

void gemm_s8x8x16_4x8::kern(const dt_int8* packA, const dt_int8* packB,
                            size_t M, size_t N, size_t K, dt_int16* C,
                            size_t LDC, bool is_first_k, const dt_int16*,
                            dt_int16*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int8 &&
                            C_dtype.enumv() == DTypeEnum::Int16)),
                  "A: %s B: %s C: %s", A_dtype.name(), B_dtype.name(),
                  C_dtype.name());

    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 8;
    //! K is packed to times of 8
    K = round_up<size_t>(K, 8);
    const int K4 = K * 4;
    const int K8 = K * 8;

    size_t m = 0;
    for (; m < M; m += A_INTERLEAVE) {
        int16_t* output = C + (m * LDC);
        const dt_int8* cur_packB = packB;
        size_t n = 0;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_4x8x8::kern_4x8(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K8;
        }

        for (; n < N; n += 4) {
            matmul_4x8x8::kern_4x4(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(M - m, 4),
                                   std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }
        packA += K4;
    }
}

// ===========================gemm_s8x8x16_mk4_8x8==================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8x8x16_mk4_8x8);

void gemm_s8x8x16_mk4_8x8::pack_A(dt_int16* out, const dt_int8* in, int ldin,
                                  int y0, int ymax, int k0, int kmax,
                                  bool) const {
    matmul_mk4_8x8x4::gemm_s8x8x16_mk4_8x8_pack_A_n(out, in, ldin, y0, ymax, k0,
                                                    kmax);
}

void gemm_s8x8x16_mk4_8x8::pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                  int x0, int xmax, int k0, int kmax,
                                  bool) const {
    matmul_mk4_8x8x4::gemm_s8x8x16_mk4_8x8_pack_B_n(out, in, ldin, x0, xmax, k0,
                                                    kmax);
}

void gemm_s8x8x16_mk4_8x8::kern(const dt_int16* packA, const dt_int8* packB,
                                size_t M, size_t N, size_t K, dt_int16* C,
                                size_t LDC, bool is_first_k, const dt_int16*,
                                dt_int16*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  C_dtype.enumv() == DTypeEnum::Int16 &&
                  A_dtype.enumv() == DTypeEnum::Int8);
    megdnn_assert(is_first_k == true, "only impl is_first_k");
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);
    megdnn_assert(M % 4 == 0 && K % 4 == 0, "M and K must be time of 4");

    constexpr size_t pack_size = 4;
    constexpr size_t pack_m = 8;
    constexpr size_t pack_n = 8;
    const size_t remain_n = N % pack_n;
    const size_t remain_m = M % pack_m;

    size_t m_idx = 0;
    for (; m_idx + pack_m <= M; m_idx += pack_m) {
        int16_t* output = C + (m_idx / pack_size * LDC);

        size_t n_idx = 0;
        const int8_t* cur_packB = packB;
        for (; n_idx + pack_n <= N; n_idx += pack_n) {
            matmul_mk4_8x8x4::kern_8x8(packA, cur_packB, K, output, LDC,
                                       is_first_k, pack_n);
            output += pack_n * pack_size;
            cur_packB += pack_n * K;
        }
        if (remain_n > 0) {
            matmul_mk4_8x8x4::kern_8x8(packA, cur_packB, K, output, LDC,
                                       is_first_k, remain_n);
            output += remain_n * pack_size;
            cur_packB += pack_n * K;
        }
        packA += pack_m * K;
    }
    if (remain_m > 0) {
        int16_t* output = C + (m_idx / pack_size * LDC);
        size_t n_idx = 0;
        const int8_t* cur_packB = packB;
        for (; n_idx + pack_n <= N; n_idx += pack_n) {
            matmul_mk4_8x8x4::kern_4x8(packA, cur_packB, K, output, LDC,
                                       is_first_k, pack_n);
            output += pack_n * pack_size;
            cur_packB += pack_n * K;
        }
        if (remain_n > 0) {
            matmul_mk4_8x8x4::kern_4x8(packA, cur_packB, K, output, LDC,
                                       is_first_k, remain_n);
            output += remain_n * pack_size;
            cur_packB += pack_n * K;
        }
    }
}

// vim: syntax=cpp.doxygen
