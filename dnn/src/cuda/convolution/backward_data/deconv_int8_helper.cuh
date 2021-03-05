/**
 * \file src/cuda/convolution/backward_data/deconv_int8_helper.cuh
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
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace deconv {

void reorder_filter_nc4hw4_to_n4hwc4(int8_t* dst, const int8_t* src,
                                     uint32_t OC, uint32_t IC, uint32_t FH,
                                     uint32_t FW, cudaStream_t stream);

}  // namespace deconv
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
