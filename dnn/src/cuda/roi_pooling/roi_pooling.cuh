/**
 * \file dnn/src/cuda/roi_pooling/roi_pooling.cuh
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
namespace roi_pooling {

template <typename T, typename Pooler>
void forward_proxy(const int nthreads, const T* bottom_data,
        const float spatial_scale, const int channels, const int height,
        const int width, const int pooled_height, const int pooled_width,
        const T* bottom_rois, T* top_data, int* argmax_data,
        cudaStream_t stream);

template <typename T, typename BwdPooler>
void backward_proxy(const int nthreads, const T* top_diff,
        const int* argmax_data, const int num_rois, const float spatial_scale,
        const int channels, const int height, const int width,
        const int pooled_height, const int pooled_width, T* bottom_diff,
        const T* bottom_rois,
        cudaStream_t stream);

} // namespace roi_pooling
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

