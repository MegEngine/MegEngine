/**
 * \file dnn/src/armv7/matrix_mul/int16x16x32/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/armv7/matrix_mul/int16x16x32/kernel_12x4x1.h"
#include "src/armv7/matrix_mul/int16x16x32/strategy.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_common.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

// ===========================gemm_s16x16x32_4x4=================================
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s16x16x32_12x4);

void gemm_s16x16x32_12x4::pack_A(dt_int16* out, const dt_int16* in, int ldin,
                                 int y0, int ymax, int k0, int kmax,
                                 bool transpose) const {
    if (transpose) {
        matmul_12x4x1::gemm_s16x16x32_12x4_transpose_pack_A_n(out, in, ldin, y0,
                                                             ymax, k0, kmax);
    } else {
        matmul_12x4x1::gemm_s16x16x32_12x4_pack_A_n(out, in, ldin, y0, ymax, k0,
                                                   kmax);
    }
}

void gemm_s16x16x32_12x4::pack_B(dt_int16* out, const dt_int16* in, int ldin,
                                 int x0, int xmax, int k0, int kmax,
                                 bool transpose) const {
    if (transpose) {
        matmul_12x4x1::gemm_s16x16x32_12x4_transpose_pack_B_n(out, in, ldin, x0,
                                                             xmax, k0, kmax);
    } else {
        matmul_12x4x1::gemm_s16x16x32_12x4_pack_B_n(out, in, ldin, x0, xmax, k0,
                                                   kmax);
    }
}

void gemm_s16x16x32_12x4::kern(const dt_int16* packA, const dt_int16* packB,
                               size_t M, size_t N, size_t K, dt_int32* C,
                               size_t LDC, bool is_first_k, const dt_int32*,
                               dt_int32*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int16 &&
                            C_dtype.enumv() == DTypeEnum::Int32)),
                  "A: %s B: %s C: %s", A_dtype.name(), B_dtype.name(),
                  C_dtype.name());

    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 12;
    constexpr size_t B_INTERLEAVE = 4;
    const int K12 = K * 12;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int16* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_12x4x1::kern_12x4(packA, cur_packB, K, output, LDC,
                                     is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K4;
        }

        if (n < N ){
            matmul_12x4x1::kern_12x123(packA, cur_packB, K, output, LDC,
                                     is_first_k, (N-n));
            output += (N-n);
            cur_packB += K4;

        }

        packA += K12;
    }

    for (; m + 3 < M; m += 4) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int16* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_12x4x1::kern_4x4(packA, cur_packB, K, output, LDC,
                                    is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K4;
        }

        if (n < N){
            int remain = N - n;
            matmul_12x4x1::kern_4x123(packA, cur_packB, K, output, LDC,
                                         is_first_k,remain);
            output += remain;
            cur_packB += K4;
        }

        packA += K4;
    }

    for (; m < M; m++) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int16* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1  < N; n += B_INTERLEAVE) {
            matmul_12x4x1::kern_1x4(packA, cur_packB, K, output, LDC,
                                    is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K4;
        }

        if (n < N) {
            int remain = N - n;
            matmul_12x4x1::kern_1x123(packA, cur_packB, K, output, LDC,
                    is_first_k,remain);
            output += remain;
            cur_packB += K4;
        }

        packA += K;
    }
}
// vim: syntax=cpp.doxygen
