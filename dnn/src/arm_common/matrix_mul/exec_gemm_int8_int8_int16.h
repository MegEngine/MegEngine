/**
 * \file dnn/src/arm_common/matrix_mul/exec_gemm_int8_int8_int16.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstdint>
#include <cstddef>

namespace megdnn {
namespace arm_common {

///! Row-major gemm
void exec_gemm_int8_int8_int16(const int8_t* A, const int8_t* B, int16_t* C,
                               size_t M, size_t K, size_t N, size_t LDB,
                               int8_t* w0, int8_t* w1);

} // namespace arm_common
} // namespace megdnn

// vim: syntax=cpp.doxygen
