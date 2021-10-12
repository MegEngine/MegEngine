/**
 * \file dnn/src/cuda/padding/padding.cuh
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
#include "cuda_runtime.h"
#include "megdnn/basic_types.h"
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace padding {

template <typename T>
void padding_forward_proxy(
        const TensorND& src, const TensorND& dst, size_t offsets[MEGDNN_MAX_NDIM * 2],
        uint32_t mode, const float_t padding_val, cudaStream_t stream);

template <typename T>
void padding_backward_proxy(
        const TensorND& src, const TensorND& dst, size_t offsets[MEGDNN_MAX_NDIM * 2],
        uint32_t mode, cudaStream_t stream);

}  // namespace padding
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen