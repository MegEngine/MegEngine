/**
 * \file dnn/src/fallback/matrix_mul/generic_strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/matrix_mul/generic_strategy.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace matmul;
using namespace fallback;

MEGDNN_REG_GEMM_STRATEGY_IMPL(sgemm_8x12);

void sgemm_8x12::pack_A(float* out, const float* in, int ldin, int y0, int ymax,
                        int k0, int kmax, bool transpose_A) const {
    if (transpose_A ^ A_TRANSPOSE) {
        pack<A_INTERLEAVE, A_BLOCK, true>(out, in, ldin, y0, ymax, k0, kmax);
    } else {
        pack<A_INTERLEAVE, A_BLOCK, false>(out, in, ldin, y0, ymax, k0, kmax);
    }
}

void sgemm_8x12::pack_B(float* out, const float* in, int ldin, int x0, int xmax,
                        int k0, int kmax, bool transpose_B) const {
    if (transpose_B ^ B_TRANSPOSE) {
        pack<B_INTERLEAVE, B_BLOCK, true>(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        pack<B_INTERLEAVE, B_BLOCK, false>(out, in, ldin, x0, xmax, k0, kmax);
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
    gemm_kern(packA, packB, M, N, K, C, LDC, is_first_k, *this);
}

// vim: syntax=cpp.doxygen
