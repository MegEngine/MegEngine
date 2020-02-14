/**
 * \file dnn/src/cuda/resize/resize_cv.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstdio>
#include "src/common/cv/enums.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace resize {

template <typename T>
void resize_cv(const T* src, T* dst, const size_t src_rows,
               const size_t src_cols, const size_t dst_rows,
               const size_t dst_cols, const size_t src_step,
               const size_t dst_step, size_t ch, InterpolationMode imode,
               void* workspace, cudaStream_t stream);

}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
