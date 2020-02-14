/**
 * \file dnn/src/cuda/matrix_inverse/helper.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megcore_cdefs.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace matrix_inverse {

void check_error(const int* src_info, uint32_t n,
                 megcore::AsyncErrorInfo* dst_info, void* tracker,
                 cudaStream_t stream);

}  // namespace matrix_inverse
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
