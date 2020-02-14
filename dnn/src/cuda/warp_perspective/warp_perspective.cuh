/**
 * \file dnn/src/cuda/warp_perspective/warp_perspective.cuh
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
namespace warp_perspective {

enum BorderMode {
    BORDER_REPLICATE,
    BORDER_REFLECT,
    BORDER_REFLECT_101,
    BORDER_WRAP,
    BORDER_CONSTANT
    BORDER_TRANSPARENT,
    BORDER_ISOLATED
};

void forward_proxy(const float *src, const float *mat, float *dst,
        int N, int C, int IH, int IW, int OH, int OW, float bval,
        BorderMode bmode, cudaStream_t stream);

void backward_data_proxy(const float *mat, const float *diff, float *grad,
        float *workspace,
        int N, int C, int IH, int IW, int OH, int OW, float bval,
        BorderMode bmode, cudaStream_t stream);
size_t get_backward_data_workspace_in_bytes(
        int N, int C, int IH, int IW, int OH, int OW,
        BorderMode bmode);

void backward_mat_proxy(
        const float *src, const float *mat, const float *diff, float *grad,
        int N, int C, int IH, int IW, int OH, int OW, float bval,
        BorderMode bmode, cudaStream_t stream);

} // namespace warp_perspective
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

