/**
 * \file dnn/src/fallback/matrix_mul/gi/fp32/exec_sgemv.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>

namespace megdnn {
namespace fallback {

void gi_gemv_like_mk4(
        const float* __restrict A, const float* __restrict B, float* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride);

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
