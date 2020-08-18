/**
 * \file dnn/src/aarch64/matrix_mul/fp32/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/aarch64/matrix_mul/fp32/kernel_general_4x16.h"
#include "src/aarch64/matrix_mul/fp32/kernel_general_8x12.h"
#include "src/aarch64/matrix_mul/fp32/kernel_general_8x12_a53.h"
#include "src/aarch64/matrix_mul/fp32/kernel_general_8x12_a55.h"
#include "src/aarch64/matrix_mul/fp32/kernel_mk4_8x12.h"
#include "src/aarch64/matrix_mul/fp32/kernel_mk4_8x12_a53.h"
#include "src/aarch64/matrix_mul/fp32/kernel_mk4_8x12_a55.h"
#include "src/aarch64/matrix_mul/fp32/strategy.h"
#include "src/common/utils.h"

#if MGB_ENABLE_CPUINFO
#include "cpuinfo.h"
#endif

using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

MEGDNN_REG_GEMM_STRATEGY_IMPL(sgemm_4x16);

void sgemm_4x16::pack_A(float* out, const float* in, int ldin, int y0, int ymax,
                        int k0, int kmax, bool transpose_A) const {
    if (transpose_A) {
        matmul_general_4x16::sgemm_4x16_pack_A_t(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    } else {
        matmul_general_4x16::sgemm_4x16_pack_A_n(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    }
}

void sgemm_4x16::pack_B(float* out, const float* in, int ldin, int x0, int xmax,
                        int k0, int kmax, bool transpose_B) const {
    if (transpose_B) {
        matmul_general_4x16::sgemm_4x16_pack_B_t(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    } else {
        matmul_general_4x16::sgemm_4x16_pack_B_n(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    }
}

void sgemm_4x16::kern(const float* packA, const float* packB, size_t M,
                      size_t N, size_t K, float* C, size_t LDC, bool is_first_k,
                      const float*, float*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == C_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Float32);
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 16;
    const int K16 = K * 16;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m < M; m += A_INTERLEAVE) {
        float* output = C + (m * LDC);

        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_general_4x16::kern_4x16(packA, cur_packB, K, output, LDC,
                                           is_first_k,
                                           std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K16;
        }

        for (; n < N; n += 4) {
            matmul_general_4x16::kern_4x4(
                    packA, cur_packB, K, output, LDC, is_first_k,
                    std::min<size_t>(M - m, 4), std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }

        packA += K4;
    }
}

MEGDNN_REG_GEMM_STRATEGY_IMPL(sgemm_8x12);

void sgemm_8x12::pack_A(float* out, const float* in, int ldin, int y0, int ymax,
                        int k0, int kmax, bool transpose_A) const {
    if (transpose_A) {
        matmul_general_8x12::sgemm_8x12_pack_A_t(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    } else {
        matmul_general_8x12::sgemm_8x12_pack_A_n(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    }
}

void sgemm_8x12::pack_B(float* out, const float* in, int ldin, int x0, int xmax,
                        int k0, int kmax, bool transpose_B) const {
    if (transpose_B) {
        matmul_general_8x12::sgemm_8x12_pack_B_t(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    } else {
        matmul_general_8x12::sgemm_8x12_pack_B_n(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    }
}

template <typename gemm_class>
static inline void sgemm_8x12_helper(const float* packA, const float* packB,
                                     size_t M, size_t N, size_t K, float* C,
                                     size_t LDC, bool is_first_k) {
    constexpr size_t A_INTERLEAVE = 8;
    constexpr size_t A_INTERLEAVE4 = 4;
    constexpr size_t B_INTERLEAVE = 12;
    const int K12 = K * 12;
    const int K8 = K * 8;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m + A_INTERLEAVE <= M; m += A_INTERLEAVE) {
        float* output = C + (m * LDC);

        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE <= N; n += B_INTERLEAVE) {
            gemm_class::kern_8x12(packA, cur_packB, K, output, LDC, is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K12;
        }

        for (; n < N; n += 4) {
            gemm_class::kern_8x4(packA, cur_packB, K, output, LDC, is_first_k,
                                 std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }
        packA += K8;
    }
    for (; m < M; m += A_INTERLEAVE4) {
        float* output = C + (m * LDC);
        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            gemm_class::kern_4x12(packA, cur_packB, K, output, LDC, is_first_k,
                                  std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K12;
        }

        for (; n < N; n += 4) {
            gemm_class::kern_4x4(packA, cur_packB, K, output, LDC, is_first_k,
                                 std::min<size_t>(M - m, 4),
                                 std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }
        packA += K4;
    }
}

void sgemm_8x12::kern(const float* packA, const float* packB, size_t M,
                      size_t N, size_t K, float* C, size_t LDC, bool is_first_k,
                      const float*, float*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == C_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Float32);
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);
#if !MGB_ENABLE_CPUINFO
    sgemm_8x12_helper<matmul_general_8x12>(packA, packB, M, N, K, C, LDC,
                                           is_first_k);
#else
    auto arch = cpuinfo_get_current_core()->uarch;
    if (arch == cpuinfo_uarch_cortex_a53) {
        sgemm_8x12_helper<matmul_general_8x12_a53>(packA, packB, M, N, K, C,
                                                   LDC, is_first_k);
    } else if (arch == cpuinfo_uarch_cortex_a55) {
        sgemm_8x12_helper<matmul_general_8x12_a55>(packA, packB, M, N, K, C,
                                                   LDC, is_first_k);
    } else {
        sgemm_8x12_helper<matmul_general_8x12>(packA, packB, M, N, K, C, LDC,
                                               is_first_k);
    }
#endif
}

MEGDNN_REG_GEMM_STRATEGY_IMPL(sgemm_mk4_8x12);

void sgemm_mk4_8x12::pack_A(float* out, const float* in, int ldin, int y0,
                            int ymax, int k0, int kmax,
                            bool transpose_A) const {
    megdnn_assert(!transpose_A, "mk4 float matmul not support transpose A");
    matmul_mk4_8x12::sgemm_8x12_pack_A(out, in, ldin, y0, ymax, k0, kmax);
}

void sgemm_mk4_8x12::pack_B(float* out, const float* in, int ldin, int x0,
                            int xmax, int k0, int kmax,
                            bool transpose_B) const {
    megdnn_assert(!transpose_B, "mk4 float matmul not support transpose B");
    matmul_mk4_8x12::sgemm_8x12_pack_B(out, in, ldin, x0, xmax, k0, kmax);
}

template <typename gemm_name>
static inline void sgemm_mk4_8x12_helper(const float* packA, const float* packB,
                                         size_t M, size_t N, size_t K, float* C,
                                         size_t LDC, bool is_first_k) {
    const int K12 = K * 12;
    const int K8 = K * 8;
    const int K4 = K * 4;
    constexpr size_t PACK_C_SIZE = 4;
    constexpr size_t A_INTERLEAVE = 8;
    constexpr size_t A_INTERLEAVE4 = 4;
    constexpr size_t B_INTERLEAVE = 12;
    size_t m = 0;
    for (; m + A_INTERLEAVE <= M; m += A_INTERLEAVE) {
        float* output = C + (m / PACK_C_SIZE * LDC);

        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE <= N; n += B_INTERLEAVE) {
            gemm_name::kern_8x12(packA, cur_packB, K, output, LDC, is_first_k);
            output += B_INTERLEAVE * PACK_C_SIZE;
            cur_packB += K12;
        }

        for (; n < N; n += 4) {
            gemm_name::kern_8x4(packA, cur_packB, K, output, LDC, is_first_k,
                                std::min<size_t>(N - n, 4));
            output += 4 * PACK_C_SIZE;
            cur_packB += K4;
        }
        packA += K8;
    }
    for (; m < M; m += A_INTERLEAVE4) {
        float* output = C + (m / PACK_C_SIZE * LDC);
        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            gemm_name::kern_4x12(packA, cur_packB, K, output, LDC, is_first_k);
            output += B_INTERLEAVE * PACK_C_SIZE;
            cur_packB += K12;
        }
        for (; n < N; n += 4) {
            gemm_name::kern_4x4(packA, cur_packB, K, output, LDC, is_first_k,
                                std::min<size_t>(N - n, 4));
            output += 4 * PACK_C_SIZE;
            cur_packB += K4;
        }
        packA += K4;
    }
}
void sgemm_mk4_8x12::kern(const float* packA, const float* packB, size_t M,
                          size_t N, size_t K, float* C, size_t LDC,
                          bool is_first_k, const float*, float*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == C_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Float32);
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);
    megdnn_assert(M % 4 == 0 && K % 4 == 0, "M and K must be time of 4");
#if !MGB_ENABLE_CPUINFO
    sgemm_mk4_8x12_helper<matmul_mk4_8x12>(packA, packB, M, N, K, C, LDC,
                                           is_first_k);
#else
    auto arch = cpuinfo_get_current_core()->uarch;
    if (arch == cpuinfo_uarch_cortex_a53) {
        sgemm_mk4_8x12_helper<matmul_mk4_8x12_a53>(packA, packB, M, N, K, C,
                                                   LDC, is_first_k);
    } else if (arch == cpuinfo_uarch_cortex_a55) {
        sgemm_mk4_8x12_helper<matmul_mk4_8x12_a55>(packA, packB, M, N, K, C,
                                                   LDC, is_first_k);
    } else {
        sgemm_mk4_8x12_helper<matmul_mk4_8x12>(packA, packB, M, N, K, C, LDC,
                                               is_first_k);
    }
#endif
}

// vim: syntax=cpp.doxygen
