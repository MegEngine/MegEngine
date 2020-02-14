/**
 * \file dnn/src/cuda/warp_affine/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda_runtime_api.h>
#include "src/common/cv/enums.h"
#include "megcore_cdefs.h"

namespace megdnn {
namespace cuda {
namespace warp_affine {

// all these kernels use bilinear interpolation

template <typename ctype>
void forward_proxy(bool is_nhwc, const ctype* src, const float* mat, ctype* dst,
                   int N, int C, int IH, int IW, int OH, int OW, ctype bval,
                   BorderMode bmode, cudaStream_t stream);

}  // namespace warp_affine
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
