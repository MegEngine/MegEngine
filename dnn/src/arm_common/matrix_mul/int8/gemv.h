/**
 * \file dnn/src/arm_common/matrix_mul/int8/gemv.h
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

namespace megdnn {
namespace arm_common {

bool is_gemv_like_preferred_int8(bool transposeA, bool transposeB, size_t M,
                                 size_t N, size_t K, size_t LDA, size_t LDB,
                                 size_t LDC);

void gemv_like(const int8_t* __restrict A, const int8_t* __restrict B,
               int32_t* __restrict C, size_t M, size_t N, size_t K,
               size_t Astride, size_t Bstride, size_t Cstride);

void gemv_like_mk4(const int8_t* __restrict A, const int8_t* __restrict B,
                   int32_t* __restrict C, size_t M, size_t N, size_t K,
                   size_t Astride, size_t Bstride, size_t Cstride);

#if __ARM_FEATURE_DOTPROD
void gemv_like_mk4_dot(const int8_t* __restrict A, const int8_t* __restrict B,
                       int32_t* __restrict C, size_t M, size_t N, size_t K,
                       size_t Astride, size_t Bstride, size_t Cstride);
#endif

}  // namespace arm_common
}  // namespace megdnn


// vim: syntax=cpp.doxygen
