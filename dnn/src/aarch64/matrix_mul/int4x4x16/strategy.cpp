/**
 * \file dnn/src/aarch64/matrix_mul/int4x4x16/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/aarch64/matrix_mul/int4x4x16/kernel_int4_8x8x8.h"
#include "src/aarch64/matrix_mul/int4x4x16/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_common.h"

using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

// ===========================gemm_s4x4x16_s4_8x8x8==================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s4x4x16_s4_8x8x8);
void gemm_s4x4x16_s4_8x8x8::pack_A(dt_int8* out, const dt_int8* in, int ldin, int y0,
                              int ymax, int k0, int kmax,
                              bool transpose) const {
    if (transpose) {
        matmul_s4_4x4x16::gemm_s4x4x16_8x8x8_interleave_pack(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    } else {
        matmul_s4_4x4x16::gemm_s4x4x16_8x8x8_transpose_pack(out, in, ldin, y0, ymax, k0,
                                                 kmax);
    }
}

void gemm_s4x4x16_s4_8x8x8::pack_B(dt_int8* out, const dt_int8* in, int ldin, int x0,
                              int xmax, int k0, int kmax,
                              bool transpose) const {
    if (transpose) {
        matmul_s4_4x4x16::gemm_s4x4x16_8x8x8_transpose_pack(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    } else {
        matmul_s4_4x4x16::gemm_s4x4x16_8x8x8_interleave_pack(out, in, ldin, x0, xmax, k0,
                                                 kmax);
    }
}

void gemm_s4x4x16_s4_8x8x8::kern(const dt_int8* packA, const dt_int8* packB,
                            size_t M, size_t N, size_t K, dt_int16* C,
                            size_t LDC, bool is_first_k, const dt_int16*,
                            dt_int16*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          (A_dtype.enumv() == DTypeEnum::QuantizedS4 &&
                           C_dtype.enumv() == DTypeEnum::QuantizedS16),
                  "A: %s B: %s C: %s", A_dtype.name(), B_dtype.name(),
                  C_dtype.name());
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 8;
    constexpr size_t B_INTERLEAVE = 8;
    //! K is packed to times of 8
    K = round_up<size_t>(K, 8);
    const int K8 = K * 8;
    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int16_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_s4_4x4x16::s4_kern_8x8(packA, cur_packB, K, output, LDC,
                                    is_first_k, A_INTERLEAVE, B_INTERLEAVE);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }

        for (; n < N; n += B_INTERLEAVE) {
            matmul_s4_4x4x16::s4_kern_8x8_remain(packA, cur_packB, K, output, LDC,
                                    is_first_k, A_INTERLEAVE,
                                    std::min<size_t>(N - n, B_INTERLEAVE));
            output += B_INTERLEAVE;
            cur_packB += K8;
        }

        packA += K8;
    }

    for (; m < M; m += A_INTERLEAVE) {
        int16_t* output = C + (m * LDC);
        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n < N; n += B_INTERLEAVE) {
            matmul_s4_4x4x16::s4_kern_8x8_remain(packA, cur_packB, K, output, LDC,
                                    is_first_k,
                                    std::min<size_t>(M - m, A_INTERLEAVE),
                                    std::min<size_t>(N - n, B_INTERLEAVE));
            output += B_INTERLEAVE;
            cur_packB += K8;
        }
        packA += K8;
    }
}


// vim: syntax=cpp.doxygen
