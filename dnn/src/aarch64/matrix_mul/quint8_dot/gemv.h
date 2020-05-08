/**
 * \file dnn/src/aarch64/matrix_mul/quint8_dot/gemv.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>
#include <cstdint>

#if __ARM_FEATURE_DOTPROD
namespace megdnn {
namespace aarch64 {
namespace matmul {

bool is_gemv_like_preferred_quint8(bool transposeA, bool transposeB, size_t M,
                                   size_t N, size_t K, size_t LDA, size_t LDB,
                                   size_t LDC);

void gemv_like_quint8(const uint8_t* __restrict A, const uint8_t* __restrict B,
                      int32_t* __restrict C, size_t M, size_t N, size_t K,
                      size_t Astride, size_t Bstride, size_t Cstride,
                      uint8_t zero_point_A, uint8_t zero_point_B);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
