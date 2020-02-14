/**
 * \file dnn/src/cuda/resize/common.h
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
#include "megcore_cdefs.h"
#include "src/common/cv/enums.h"

namespace megdnn {
namespace cuda {
namespace resize {

// all these kernels use bilinear interpolation

template <typename ctype>
void forward_proxy(bool is_nhwc, const ctype* src, ctype* dst, int N, int C,
                   int IH, int IW, int OH, int OW, int S_IN, int S_IC, int S_IH,
                   int S_IW, cudaStream_t stream);

template <typename ctype>
void forward_proxy_nchw4(const ctype* src, ctype* dst, int N, int C, int IH,
                         int IW, int OH, int OW, cudaStream_t stream);

void backward_data_proxy(const float* diff, float* grad, int N, int C, int IH,
                         int IW, int OH, int OW, cudaStream_t stream);

}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
