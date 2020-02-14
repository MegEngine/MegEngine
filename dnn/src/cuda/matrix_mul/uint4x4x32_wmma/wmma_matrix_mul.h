/**
 * \file dnn/src/cuda/matrix_mul/uint4x4x32_wmma/wmma_matrix_mul.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
namespace matrix_mul {
void exec_wmma_matrix_mul_quint4_nt(_megdnn_tensor_in A, _megdnn_tensor_in B,
                                    _megdnn_tensor_out C,
                                    _megdnn_workspace workspace,
                                    cudaStream_t stream);
}  // namespace matrix_mul
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
