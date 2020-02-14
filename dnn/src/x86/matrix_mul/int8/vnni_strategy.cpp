/**
 * \file dnn/src/x86/matrix_mul/int8/vnni_strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if MEGDNN_X86_WITH_VNNI
#include "src/common/utils.h"
#include "src/x86/matrix_mul/int8/kernel_vnni_12x32x4.h"
#include "src/x86/matrix_mul/int8/strategy.h"

using namespace megdnn;
using namespace x86;
using namespace x86::matmul;

MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_int8_vnni_12x32x4);

// ===========================gemm_s8_4x2======================================

void gemm_int8_vnni_12x32x4::pack_A(dt_uint8* out, const dt_int8* in, int ldin,
                                    int y0, int ymax, int k0, int kmax,
                                    bool transpose) const {
    if (transpose) {
        matmul_vnni_12x32x4::gemm_pack_A_t(out, in, ldin, y0, ymax, k0, kmax);
    } else {
        matmul_vnni_12x32x4::gemm_pack_A_n(out, in, ldin, y0, ymax, k0, kmax);
    }
}

void gemm_int8_vnni_12x32x4::pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                    int x0, int xmax, int k0, int kmax,
                                    bool transpose) const {
    if (transpose) {
        matmul_vnni_12x32x4::gemm_pack_B_t(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        matmul_vnni_12x32x4::gemm_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

void gemm_int8_vnni_12x32x4::kern(const dt_uint8* packA, const dt_int8* packB,
                                  size_t M, size_t N, size_t K, dt_int32* C,
                                  size_t LDC, bool is_first_k, const dt_int32*,
                                  dt_int32*) const {
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

    constexpr size_t A_INTERLEAVE = 12;
    constexpr size_t B_INTERLEAVE = 32;
    //! K is packed to times of 4
    K = round_up<size_t>(K, 4);
    const int K32 = K * 32;
    const int K16 = K * 16;
    const int K12 = K * 12;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_vnni_12x32x4::kern_12x32x4(packA, cur_packB, K, output, LDC,
                                              is_first_k);
            output += B_INTERLEAVE;
            cur_packB += K32;
        }

        for (; n < N; n += 16) {
            matmul_vnni_12x32x4::kern_12x16x4(packA, cur_packB, K, output, LDC,
                                              is_first_k,
                                              std::min<size_t>(N - n, 16));
            output += std::min<size_t>(N - n, 16);
            cur_packB += K16;
        }

        packA += K12;
    }

    for (; m < M; m += 4) {
        int32_t* output = C + (m * LDC);

        size_t n = 0;
        const dt_int8* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            matmul_vnni_12x32x4::kern_4x32x4(packA, cur_packB, K, output, LDC,
                                             is_first_k,
                                             std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K32;
        }
        for (; n < N; n += 16) {
            matmul_vnni_12x32x4::kern_4x16x4(
                    packA, cur_packB, K, output, LDC, is_first_k,
                    std::min<size_t>(M - m, 4), std::min<size_t>(N - n, 16));

            output += std::min<size_t>(N - n, 16);
            cur_packB += K16;
        }
        packA += K4;
    }
}
#endif
// vim: syntax=cpp.doxygen
