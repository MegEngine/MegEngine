/**
 * \file dnn/src/armv7/matrix_mul/int8/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/matrix_mul/int8/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/armv7/matrix_mul/int8/kernel_4x2x16.h"
#include "src/armv7/matrix_mul/int8/kernel_4x8x8.h"
#include "src/armv7/matrix_mul/int8/kernel_6x8x4.h"
#include "src/armv7/matrix_mul/int8/kernel_mk4_4x2x16.h"
#include "src/armv7/matrix_mul/int8/kernel_mk4_dot_8x4x4.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_common.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8_4x2);

// ===========================gemm_s8_4x2======================================

void gemm_s8_4x2::pack_A(dt_int8* out, const dt_int8* in, int ldin, int y0,
                         int ymax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_4x2x16::gemm_s8_4x2_pack_A_t(out, in, ldin, y0, ymax, k0, kmax);
    } else {
        matmul_4x2x16::gemm_s8_4x2_pack_A_n(out, in, ldin, y0, ymax, k0, kmax);
    }
}

void gemm_s8_4x2::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                         int xmax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_4x2x16::gemm_s8_4x2_pack_B_t(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        matmul_4x2x16::gemm_s8_4x2_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

void gemm_s8_4x2::kern(const dt_int8* packA, const dt_int8* packB, size_t M,
                       size_t N, size_t K, dt_int32* C, size_t LDC,
                       bool is_first_k, const dt_int32*, dt_int32*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int8 &&
                            C_dtype.enumv() == DTypeEnum::Int32) ||
                           (A_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                            C_dtype.enumv() == DTypeEnum::QuantizedS32)),
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
        int32_t* output = C + (m * LDC);

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
        int32_t* output = C + (m * LDC);

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

// ===========================gemm_s8_4x4======================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8_4x8);

void gemm_s8_4x8::pack_A(dt_int8* out, const dt_int8* in, int ldin, int y0,
                         int ymax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_4x8x8::gemm_s8_4x8_transpose_pack_A_n(out, in, ldin, y0, ymax,
                                                     k0, kmax);
    } else {
        matmul_4x8x8::gemm_s8_4x8_pack_A_n(out, in, ldin, y0, ymax, k0, kmax);
    }
}

void gemm_s8_4x8::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                         int xmax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_4x8x8::gemm_s8_4x8_transpose_pack_B_n(out, in, ldin, x0, xmax,
                                                     k0, kmax);
    } else {
        matmul_4x8x8::gemm_s8_4x8_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

void gemm_s8_4x8::kern(const dt_int8* packA, const dt_int8* packB, size_t M,
                       size_t N, size_t K, dt_int32* C, size_t LDC,
                       bool is_first_k, const dt_int32*, dt_int32*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int8 &&
                            C_dtype.enumv() == DTypeEnum::Int32) ||
                           (A_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                            C_dtype.enumv() == DTypeEnum::QuantizedS32)),
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
        int32_t* output = C + (m * LDC);
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

#if __ARM_FEATURE_DOTPROD
// ===========================gemm_s8_6x8======================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_dots8_6x8);
void gemm_dots8_6x8::pack_A(dt_int8* out, const dt_int8* in, int ldin, int y0,
                            int ymax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_dot_6x8x4::gemm_s8_6x8_pack_A_t(out, in, ldin, y0, ymax, k0,
                                               kmax);
    } else {
        matmul_dot_6x8x4::gemm_s8_6x8_pack_A_n(out, in, ldin, y0, ymax, k0,
                                               kmax);
    }
}

void gemm_dots8_6x8::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                            int xmax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_dot_6x8x4::gemm_s8_6x8_pack_B_t(out, in, ldin, x0, xmax, k0,
                                               kmax);
    } else {
        matmul_dot_6x8x4::gemm_s8_6x8_pack_B_n(out, in, ldin, x0, xmax, k0,
                                               kmax);
    }
}

void gemm_dots8_6x8::kern(const dt_int8* packA, const dt_int8* packB, size_t M,
                          size_t N, size_t K, dt_int32* C, size_t LDC,
                          bool is_first_k, const dt_int32* bias,
                          dt_int32* workspace) const {
    MEGDNN_MARK_USED_VAR(bias);
    constexpr size_t A_INTERLEAVE = 6;
    constexpr size_t B_INTERLEAVE = 8;
    //! K is packed to times of 4
    K = round_up<size_t>(K, 4);
    const int K4 = K * 4;
    const int K6 = K * 6;
    const int K8 = K * 8;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m * LDC);
        const dt_int8* cur_packB = packB;
        size_t n = 0;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_dot_6x8x4::kern_6x8(packA, cur_packB, K, output, LDC,
                                       is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }
        for (; n < N; n += 4) {
            size_t n_remain = std::min<size_t>(N - n, 4);
            matmul_dot_6x8x4::kern_6x4(packA, cur_packB, K, output, LDC,
                                       is_first_k, n_remain);
            output += n_remain;
            cur_packB += K4;
        }
        packA += K6;
    }
    if (m < M) {
        int32_t* output = C + (m * LDC);
        const dt_int8* cur_packB = packB;
        size_t m_remain = std::min<size_t>(M - m, 6);
        size_t n = 0;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_dot_6x8x4::kern_6x8(packA, cur_packB, K, output, LDC,
                                       is_first_k, m_remain);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }
        for (; n < N; n += 4) {
            size_t n_remain = std::min<size_t>(N - n, 4);
            matmul_dot_6x8x4::kern_6x4(packA, cur_packB, K, output, LDC,
                                       is_first_k, n_remain, m_remain);
            output += n_remain;
            cur_packB += K4;
        }
    }
}

// ===========================gemm_mk4_dots8_8x4======================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_mk4_dots8_8x4);

void gemm_mk4_dots8_8x4::pack_A(dt_int8* out, const dt_int8* in, int ldin,
                                int y0, int ymax, int k0, int kmax,
                                bool transpose) const {
    megdnn_assert(!transpose,
                  "matrix mul mk4 with transposed matrix A is not supported.");
    megdnn_assert(ymax % 4 == 0 && y0 % 4 == 0,
                  "mk4 format matmul with m is not times of 4.");
    megdnn_assert(kmax % 4 == 0 && k0 % 4 == 0,
                  "mk4 format matmul with k is not times of 4.");
    matmul_mk4_dot_8x4x4::gemm_dots8_8x4_pack_A(out, in, ldin, y0, ymax, k0,
                                                kmax);
}

void gemm_mk4_dots8_8x4::pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                int x0, int xmax, int k0, int kmax,
                                bool transpose) const {
    megdnn_assert(!transpose,
                  "matrix mul mk4 with transposed matrix B is not supported");
    megdnn_assert(kmax % 4 == 0 && k0 % 4 == 0,
                  "mk4 format matmul with k is not times of 4.");
    matmul_mk4_dot_8x4x4::gemm_dots8_8x4_pack_B(out, in, ldin, x0, xmax, k0,
                                                kmax);
}

void gemm_mk4_dots8_8x4::kern(const dt_int8* packA, const dt_int8* packB,
                              size_t M, size_t N, size_t K, dt_int32* C,
                              size_t LDC, bool is_first_k, const dt_int32* bias,
                              dt_int32* workspace) const {
    MEGDNN_MARK_USED_VAR(bias);
    constexpr size_t A_INTERLEAVE = 8;
    //! K is packed to times of 4
    K = round_up<size_t>(K, 4);
    const int K4 = K * 4;
    const int K8 = K * 8;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + ((m >> 2) * LDC);
        const dt_int8* cur_packB = packB;
        for (size_t n = 0; n < N; n += 4) {
            size_t n_remain = std::min<size_t>(N - n, 4);
            matmul_mk4_dot_8x4x4::kern_8x4(packA, cur_packB, K, output, LDC,
                                           is_first_k, n_remain);
            output += 16;
            cur_packB += K4;
        }
        packA += K8;
    }
    for (; m < M; m += 4) {
        int32_t* output = C + ((m >> 2) * LDC);
        const dt_int8* cur_packB = packB;
        for (size_t n = 0; n < N; n += 4) {
            size_t n_remain = std::min<size_t>(N - n, 4);
            matmul_mk4_dot_8x4x4::kern_4x4(packA, cur_packB, K, output, LDC,
                                           is_first_k, n_remain);
            output += 16;
            cur_packB += K4;
        }
        packA += K4;
    }
}

#endif

// ===========================gemm_mk4_s8_4x2======================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_mk4_s8_4x2);
void gemm_mk4_s8_4x2::pack_A(dt_int8* out, const dt_int8* in, int ldin, int y0,
                             int ymax, int k0, int kmax, bool transpose) const {
    megdnn_assert(!transpose);
    matmul_mk4_4x2x16::gemm_mk4_s8_4x2_pack_A(out, in, ldin, y0, ymax, k0,
                                              kmax);
}

void gemm_mk4_s8_4x2::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                             int xmax, int k0, int kmax, bool transpose) const {
    megdnn_assert(!transpose);
    matmul_mk4_4x2x16::gemm_mk4_s8_4x2_pack_B(out, in, ldin, x0, xmax, k0,
                                              kmax);
}

void gemm_mk4_s8_4x2::kern(const dt_int8* packA, const dt_int8* packB, size_t M,
                           size_t N, size_t K, dt_int32* C, size_t LDC,
                           bool is_first_k, const dt_int32*, dt_int32*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int8 &&
                            C_dtype.enumv() == DTypeEnum::Int32) ||
                           (A_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                            C_dtype.enumv() == DTypeEnum::QuantizedS32)),
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
    megdnn_assert(M % 4 == 0, "mk4 matmul with m is not times of 4");

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m / 4 * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n < N; n += B_INTERLEAVE) {
            matmul_mk4_4x2x16::kern_4x2(packA, cur_packB, K, output, is_first_k,
                                        std::min<size_t>(N - n, 2));
            output += B_INTERLEAVE * 4;
            cur_packB += K2;
        }
        packA += K4;
    }
}
// vim: syntax=cpp.doxygen
