/**
 * \file dnn/src/cuda/remap/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <cuda_runtime_api.h>
#include "megcore_cdefs.h"
#include "src/common/cv/enums.h"
#include "src/common/opr_param_defs_enumv.cuh"

namespace megdnn {
namespace cuda {
namespace remap {

// all these kernels use LINEAR interpolation

template <typename ctype, const uint32_t format, ::BorderMode bmode>
void forward_proxy(const ctype* src, const float* map_xy, ctype* dst, int N,
                   int C, int IH, int IW, int OH, int OW, float scalar,
                   cudaStream_t stream);

template <typename ctype, const uint32_t format, ::BorderMode bmode>
void backwarddata_proxy(ctype* grad, const float* map_xy, const ctype* diff,
                        int N, int C, int IH, int IW, int OH, int OW,
                        cudaStream_t stream);

template <typename ctype, const uint32_t format, ::BorderMode bmode>
void backwardmat_proxy(const ctype* src, const float* map_xy, const ctype* diff,
                       float* grad, int N, int C, int IH, int IW, int OH,
                       int OW, float scalar, cudaStream_t stream);

}  // namespace remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
