/**
 * \file dnn/src/fallback/matrix_mul/gemv.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/fallback/matrix_mul/opr_impl.h"

namespace megdnn {
namespace fallback{

template <typename itype, typename otype>
void gemv_like(const itype* A, const itype* B, otype* C, size_t M, size_t N,
               size_t K, size_t LDA, size_t LDB, size_t LDC) {
    for (size_t m = 0; m < M; ++m) {
        memset(C + m * LDC, 0, sizeof(otype) * N);
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n) {
                C[m * LDC + n] += static_cast<otype>(A[m * LDA + k]) *
                                  static_cast<otype>(B[k * LDB + n]);
            }
    }
}

template <typename itype, typename otype>
void gemv_like(const itype* A, const itype* B, otype* C, size_t M, size_t N,
               size_t K, size_t LDA, size_t LDB, size_t LDC, uint8_t zp0,
               uint8_t zp1) {
    for (size_t m = 0; m < M; ++m) {
        memset(C + m * LDC, 0, sizeof(otype) * N);
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n) {
                C[m * LDC + n] += (static_cast<otype>(A[m * LDA + k]) -
                                   static_cast<otype>(zp0)) *
                                  (static_cast<otype>(B[k * LDB + n]) -
                                   static_cast<otype>(zp1));
            }
    }
}

template <typename itype, typename otype, bool have_zp = false>
void gemm_gemv_like(const MatrixMulImpl::KernParam& kern_param) {
    const itype* A = kern_param.A<itype>();
    const itype* B = kern_param.B<itype>();
    otype* C = kern_param.C<otype>();
    size_t M = kern_param.M;
    size_t N = kern_param.N;
    size_t K = kern_param.K;
    size_t LDA = kern_param.LDA;
    size_t LDB = kern_param.LDB;
    size_t LDC = kern_param.LDC;

    if (have_zp) {
        uint8_t zp0 = kern_param.A_type.param<dtype::Quantized8Asymm>().zero_point;
        uint8_t zp1 = kern_param.B_type.param<dtype::Quantized8Asymm>().zero_point;
        gemv_like<itype, otype>(A, B, C, M, N, K, LDA, LDB, LDC, zp0, zp1);
    }
    else {
        gemv_like<itype, otype>(A, B, C, M, N, K, LDA, LDB, LDC);
    }
}

}  // namespace fallback
}  // namespace megdnn