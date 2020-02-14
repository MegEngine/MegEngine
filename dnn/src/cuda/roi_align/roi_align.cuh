/**
 * \file dnn/src/cuda/roi_align/roi_align.cuh
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
namespace roi_align {

template <typename T, typename Pooler>
void forward_proxy(const int nthreads, const T* bottom_data,
                   const float spatial_scale, const float offset,
                   const int channels, const int height, const int width,
                   const int pooled_height, const int pooled_width,
                   const int sample_height, const int sample_width,
                   const T* bottom_rois, T* top_data, int* argmax_data,
                   cudaStream_t stream);

template <typename T, typename BwdPooler>
void backward_proxy(const int nthreads, const T* top_diff,
                    const int* argmax_data, const float spatial_scale,
                    const float offset, const int channels, const int height,
                    const int width, const int pooled_height,
                    const int pooled_width, const int sample_height,
                    const int sample_width, const T* bottom_rois,
                    T* bottom_diff, cudaStream_t stream);

}  // namespace roi_align
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen

