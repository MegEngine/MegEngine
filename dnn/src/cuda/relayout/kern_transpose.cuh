/**
 * \file dnn/src/cuda/relayout/kern_transpose.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
template <typename T>
void copy_by_transpose(const T* A, T* B, size_t batch, size_t m, size_t n,
                       size_t lda, size_t ldb, size_t stride_a, size_t stride_b,
                       cudaStream_t stream);
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

