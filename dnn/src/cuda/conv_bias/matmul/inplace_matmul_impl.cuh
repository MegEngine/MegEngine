/**
 * \file dnn/src/cuda/conv_bias/matmul/inplace_matmul_impl.cuh
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
#include <stddef.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace conv_bias {

void exec_inplace_matmul_fwd(const float* src, const float* filter, float* dst,
                             size_t N, size_t INP_BS, size_t OUT_BS, size_t IC,
                             size_t IH, size_t IW, size_t OC, size_t OH,
                             size_t OW, size_t FH, size_t FW, size_t PH,
                             size_t PW, size_t SH, size_t SW, bool is_xcorr,
                             cudaStream_t stream);

}  // namespace conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
