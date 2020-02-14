/**
 * \file dnn/src/cuda/images2neibs/kernel.cuh
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

namespace megdnn {
namespace cuda {
namespace images2neibs {

template <typename T>
void forward(const T *src, T *dst,
        int N, int C, int IH, int IW, int OH, int OW,
        int ph, int pw, int sh, int sw, int wh, int ww,
        cudaStream_t stream);

template <typename T>
void backward(const T *diff, T *grad,
        int N, int C, int IH, int IW, int OH, int OW,
        int ph, int pw, int sh, int sw, int wh, int ww,
        cudaStream_t stream);
       
} // namespace images2neibs
} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen

