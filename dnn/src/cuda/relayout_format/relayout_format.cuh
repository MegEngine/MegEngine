/**
 * \file dnn/src/cuda/relayout_format/relayout_format.cuh
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

#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace relayout_format {

template <int pack_w = 1>
void relayout_format_cuda_exec(const TensorND& src, const TensorND& dst,
                               const cudaStream_t& stream,
                               const float src_scale = 1.f,
                               const float dst_scale = 1.f,
                               const uint8_t src_zero_point = 0,
                               const uint8_t dst_zero_point = 0);

bool relayout_format_cuda_usable(const TensorLayout& src_layout,
                                 const TensorLayout& dst_layout);

}  // namespace relayout_format
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
