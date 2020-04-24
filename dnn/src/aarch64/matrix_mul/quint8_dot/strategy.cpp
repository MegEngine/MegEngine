/**
 * \file dnn/src/aarch64/matrix_mul/quint8_dot/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/matrix_mul/quint8_dot/strategy.h"
#include "megdnn/dtype.h"
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/aarch64/matrix_mul/quint8_dot/kernel_8x8x4.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

#if __ARM_FEATURE_DOTPROD
using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_u8_8x8);

void gemm_u8_8x8::pack_A(uint8_t* outptr, const uint8_t* inptr, int ldin,
                         int y0, int ymax, int k0, int kmax,
                         bool transpose) const {
    if (transpose) {
        matmul_8x8x4::gemm_u8_8x8_transpose_pack_helper(outptr, inptr, ldin, y0,
                                                        ymax, k0, kmax);
    } else {
        matmul_8x8x4::gemm_u8_8x8_interleave_pack_helper(outptr, inptr, ldin,
                                                         y0, ymax, k0, kmax);
    }
}

void gemm_u8_8x8::pack_B(uint8_t* out, const uint8_t* in, int ldin, int x0,
                         int xmax, int k0, int kmax, bool transpose) const {
    if (transpose) {
        matmul_8x8x4::gemm_u8_8x8_interleave_pack_helper(out, in, ldin, x0,
                                                         xmax, k0, kmax);
    } else {
        matmul_8x8x4::gemm_u8_8x8_transpose_pack_helper(out, in, ldin, x0, xmax,
                                                        k0, kmax);
    }
}

void gemm_u8_8x8::kern(const uint8_t* packA, const uint8_t* packB, size_t M,
                       size_t N, size_t K, dt_int32* C, size_t LDC,
                       bool is_first_k, const dt_int32*, dt_int32*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                  A_dtype.enumv() == DTypeEnum::Quantized8Asymm &&
                  C_dtype.enumv() == DTypeEnum::QuantizedS32);
    MEGDNN_MARK_USED_VAR(C_dtype);
    size_t zero_point_A = A_dtype.param<dtype::Quantized8Asymm>().zero_point;
    size_t zero_point_B = B_dtype.param<dtype::Quantized8Asymm>().zero_point;
    constexpr size_t A_INTERLEAVE = 8;
    constexpr size_t B_INTERLEAVE = 8;
    const uint32_t zAB = static_cast<uint32_t>(zero_point_A) *
                         static_cast<uint32_t>(zero_point_B) * K;
    //! K is packed to times of 4
    K = round_up<size_t>(K, 4);
    const int K8 = (K << 3);
    const int K4 = K * 4;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_uint8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_8x8x4::kern_8x8(packA, cur_packB, K, output, LDC, is_first_k,
                                   zero_point_A, zero_point_B, zAB);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }

        for (; n < N; n += 4) {
            matmul_8x8x4::kern_8x4(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(N - n, 4), zero_point_A,
                                   zero_point_B, zAB);
            output += 4;
            cur_packB += K4;
        }
        packA += K8;
    }

    for (; m < M; m += 4) {
        int32_t* output = C + (m * LDC);
        const dt_uint8* cur_packB = packB;
        size_t n = 0;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_8x8x4::kern_4x8(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(M - m, 4), zero_point_A,
                                   zero_point_B, zAB);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }

        for (; n < N; n += 4) {
            matmul_8x8x4::kern_4x4(packA, cur_packB, K, output, LDC, is_first_k,
                                   std::min<size_t>(M - m, 4),
                                   std::min<size_t>(N - n, 4), zero_point_A,
                                   zero_point_B, zAB);
            output += 4;
            cur_packB += K4;
        }
        packA += K4;
    }
}
#endif
// vim: syntax=cpp.doxygen
