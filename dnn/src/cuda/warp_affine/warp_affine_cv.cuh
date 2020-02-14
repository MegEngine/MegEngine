/**
 * \file dnn/src/cuda/warp_affine/warp_affine_cv.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/cuda/utils.cuh"
#include "src/common/cv/enums.h"
#include <cstdio>

namespace megdnn {
namespace cuda {
namespace warp_affine {

template <typename T, size_t CH>
void warp_affine_cv_proxy(const T* src, T* dst, const size_t src_rows,
                          const size_t src_cols, const size_t dst_rows,
                          const size_t dst_cols, const size_t src_step,
                          const size_t dst_step, BorderMode bmode,
                          InterpolationMode imode, const float* trans,
                          T border_val, double* workspace,
                          cudaStream_t stream);

} // namespace warp_affine
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
