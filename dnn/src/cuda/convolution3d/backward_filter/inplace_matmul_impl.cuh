/**
 * \file dnn/src/cuda/convolution3d/backward_filter/inplace_matmul_impl.cuh
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
#include <stdint.h>
#include <stddef.h>

namespace megdnn {
namespace cuda {
namespace convolution3d {

void exec_inplace_matmul_bwd_filter(
        const float *diff, const float *src, float *grad,
        size_t N, size_t INP_BS, size_t OUT_BS,
        size_t IC, size_t ID, size_t IH, size_t IW,
        size_t OC, size_t OD, size_t OH, size_t OW,
        size_t FD, size_t FH, size_t FW,
        size_t PD, size_t PH, size_t PW,
        size_t SD, size_t SH, size_t SW,
        size_t DD, size_t DH, size_t DW,
        bool is_xcorr,
        cudaStream_t stream);

} // namespace convolution
} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
