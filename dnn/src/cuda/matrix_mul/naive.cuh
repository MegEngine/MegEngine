/**
 * \file dnn/src/cuda/matrix_mul/naive.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

template <typename AType, typename BType, typename CType, typename CompType>
void exec_gemm_naive(
        const AType* A, const BType* B, CType* C, size_t m, size_t n, size_t k,
        size_t ldA, size_t ldB, size_t ldC, bool transA, bool transB,
        cudaStream_t stream);
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
