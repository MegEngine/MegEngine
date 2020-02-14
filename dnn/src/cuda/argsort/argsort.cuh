/**
 * \file dnn/src/cuda/argsort/argsort.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include "megdnn/dtype.h"

namespace megdnn {
namespace cuda {
namespace argsort {

size_t get_fwd_workspace_in_bytes(uint32_t M, uint32_t N, DType dtype,
                                  bool is_ascending,
                                  bool iptr_src_given = false);

/*!
 * \param iptr_src pointer to indices; a range would be generated if it is null
 */
template <typename dtype>
void forward(const dtype* sptr, dtype* dptr, int* iptr, void* workspace,
             uint32_t M, uint32_t N, bool is_ascending, cudaStream_t stream,
             const int* iptr_src = NULL);

//! iterate over all supported data types
#define ARGSORT_FOREACH_CTYPE(cb) cb(float) cb(int32_t)

}  // namespace argsort
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
