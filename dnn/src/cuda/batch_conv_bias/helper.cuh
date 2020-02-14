/**
 * \file dnn/src/cuda/batch_conv_bias/helper.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/convolution_helper/parameter.cuh"

namespace megdnn {
namespace cuda {
namespace batch_conv_bias {
void compute_offset(int* offset, const convolution::ConvParam& param,
                    cudaStream_t stream);
}  // namespace batched_conv2d
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
