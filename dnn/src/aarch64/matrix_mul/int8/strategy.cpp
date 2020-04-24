/**
 * \file dnn/src/aarch64/matrix_mul/int8/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if !(__ARM_FEATURE_DOTPROD)
#include "src/aarch64/matrix_mul/int8/strategy.h"
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/aarch64/matrix_mul/int8/kernel_4x4x16.h"
#include "src/aarch64/matrix_mul/int8/kernel_8x8x8.h"
#include "src/aarch64/matrix_mul/int8/kernel_mk4_4x4x16.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

///////////////////////// gemm_s8_4x4 ////////////////////////////////////
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8_4x4);

void gemm_s8_4x4::pack_A(dt_int8* outptr, const dt_int8* inptr, int ldin,
                         int y0, int ymax, int k0, int kmax,
                         bool transpose) const {
    if (transpose) {
        matmul_4x4x16::gemm_s8_4x4_pack_B_n(outptr, inptr, ldin, y0, ymax, k0,
                                            kmax);
    } else {
        matmul_4x4x16::gemm_s8_4x4_pack_A_n(outptr, inptr, ldin, y0, ymax, k0,
                                            kmax);
    }
}

void gemm_s8_4x4::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                         int xmax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_4x4x16::gemm_s8_4x4_pack_A_n(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        matmul_4x4x16::gemm_s8_4x4_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

void gemm_s8_4x4::kern(const dt_int8* packA, const dt_int8* packB, size_t M,
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
    constexpr size_t B_INTERLEAVE = 4;
    //! K is packed to times of 4
    K = round_up<size_t>(K, 16);
    const int K4 = K * 4;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_4x4x16::kern_4x4(packA, cur_packB, K, output, LDC,
                                    is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K4;
        }

        for (; n < N; n += B_INTERLEAVE) {
            matmul_4x4x16::kern_4x4_remain(packA, cur_packB, K, output, LDC,
                                           is_first_k, 4,
                                           std::min<size_t>(N - n, 4));
            output += B_INTERLEAVE;
            cur_packB += K4;
        }

        packA += K4;
    }

    for (; m < M; m += 4) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n < N; n += B_INTERLEAVE) {
            matmul_4x4x16::kern_4x4_remain(
                    packA, cur_packB, K, output, LDC, is_first_k,
                    std::min<size_t>(M - m, 4), std::min<size_t>(N - n, 4));
            output += B_INTERLEAVE;
            cur_packB += K4;
        }
        packA += K4;
    }
}

///////////////////////// gemm_mk4_s8_4x4 ////////////////////////////////////
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_mk4_s8_4x4);

void gemm_mk4_s8_4x4::pack_A(dt_int8* outptr, const dt_int8* inptr, int ldin,
                             int y0, int ymax, int k0, int kmax,
                             bool transpose) const {
    megdnn_assert(!transpose,
                  "the gemm_mk4_s8_4x4 strategy is not support transpose A");
    matmul_mk4_4x4x16::gemm_mk4_s8_4x4_pack_A(outptr, inptr, ldin, y0, ymax, k0,
                                              kmax);
}

void gemm_mk4_s8_4x4::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                             int xmax, int k0, int kmax, bool transpose) const {
    megdnn_assert(!transpose,
                  "the gemm_mk4_s8_4x4 strategy is not support transpose B");
    matmul_mk4_4x4x16::gemm_mk4_s8_4x4_pack_B(out, in, ldin, x0, xmax, k0,
                                              kmax);
}

void gemm_mk4_s8_4x4::kern(const dt_int8* packA, const dt_int8* packB, size_t M,
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
    constexpr size_t B_INTERLEAVE = 4;
    //! K is packed to times of 4
    megdnn_assert(K % 4 == 0, "K is not time of 4");
    const size_t K4 = round_up<size_t>(K, 16) * 4;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m / 4 * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_mk4_4x4x16::kern_4x4(packA, cur_packB, K, output,
                                        is_first_k);
            output += B_INTERLEAVE * 4;
            cur_packB += K4;
        }

        if (n < N) {
            matmul_mk4_4x4x16::kern_4x4_remain(packA, cur_packB, K, output,
                                               is_first_k, N - n);
        }

        packA += K4;
    }
}


///////////////////////// gemm_s8_8x8 ////////////////////////////////////
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8_8x8);

void gemm_s8_8x8::pack_A(dt_int8* outptr, const dt_int8* inptr, int ldin,
                         int y0, int ymax, int k0, int kmax,
                         bool transpose) const {
    if (transpose) {
        matmul_8x8x8::gemm_s8_8x8_transpose_pack_A_n(outptr, inptr, ldin, y0,
                                                     ymax, k0, kmax);
    } else {
        matmul_8x8x8::gemm_s8_8x8_pack_A_n(outptr, inptr, ldin, y0, ymax, k0,
                                            kmax);
    }
}

void gemm_s8_8x8::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                         int xmax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_8x8x8::gemm_s8_8x8_transpose_pack_B_n(out, in, ldin, x0, xmax,
                                                     k0, kmax);
    } else {
        matmul_8x8x8::gemm_s8_8x8_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

void gemm_s8_8x8::kern(const dt_int8* packA, const dt_int8* packB, size_t M,
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

    constexpr size_t A_INTERLEAVE = 8;
    constexpr size_t B_INTERLEAVE = 8;
    //! K is packed to times of 4
    K = round_up<size_t>(K, 8);
    const int K8 = K * 8;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_8x8x8::kern_8x8(packA, cur_packB, K, output, LDC,
                                   is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }

        for (; n < N; n += 4) {
            matmul_8x8x8::kern_8x4(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }
        packA += K8;
    }

    for (; m < M; m += 4) {
        int32_t* output = C + (m * LDC);
        const dt_int8* cur_packB = packB;
        size_t n = 0;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_8x8x8::kern_4x8(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K8;
        }

        for (; n < N; n += 4) {
            matmul_8x8x8::kern_4x4(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(M - m, 4),
                                   std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }
        packA += K4;
    }
}
#endif

// vim: syntax=cpp.doxygen
