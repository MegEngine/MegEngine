/**
 * \file dnn/src/cuda/powc/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

void powc_kern(const TensorND& dest, const TensorND& src, const float* exp_f,
               const int* exp_i, cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
