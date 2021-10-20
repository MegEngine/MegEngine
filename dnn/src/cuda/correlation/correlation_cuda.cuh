/**
 * \file dnn/src/cuda/correlation/correlation.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace correlation {

template <typename T>
void forward_proxy(
        const int nthreads, const T* data1, const T* data2, T* dst, const int bchannels,
        const int bheight, const int bwidth, const int tchannels, const int theight,
        const int twidth, const int kernel_size, const int max_displacement,
        const int stride1, const int stride2, const int pad_size,
        const bool is_multiply, cudaStream_t stream);

template <typename T>
void backward_proxy_data1(
        const int nthreads, const T* diff, const T* data1, const T* data2, T* grad1,
        const int bchannels, const int bheight, const int bwidth, const int tchannels,
        const int theight, const int twidth, const int kernel_size,
        const int max_displacement, const int stride1, const int stride2,
        const int pad_size, const bool is_multiply, cudaStream_t stream);

template <typename T>
void backward_proxy_data2(
        const int nthreads, const T* diff, const T* data1, const T* data2, T* grad2,
        const int bchannels, const int bheight, const int bwidth, const int tchannels,
        const int theight, const int twidth, const int kernel_size,
        const int max_displacement, const int stride1, const int stride2,
        const int pad_size, const bool is_multiply, cudaStream_t stream);

}  // namespace correlation
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
