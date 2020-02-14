/**
 * \file dnn/src/cuda/convolution/im2col.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <stddef.h>
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace convolution {

//! col is of shape (ic*fh*fw, oh*ow*n)
template <typename T>
void im2col(const T *im, T *col,
        size_t N, size_t INP_BS,
        size_t IC, size_t IH, size_t IW,
        size_t FH, size_t FW,
        size_t OH, size_t OW,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        size_t DH, size_t DW,   // dilation
        cudaStream_t stream);

template <typename T>
void col2im(const T *col, T *im,
        size_t N, size_t INP_BS,
        size_t IC, size_t IH, size_t IW,
        size_t FH, size_t FW,
        size_t OH, size_t OW,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        size_t DH, size_t DW,   // dilation
        cudaStream_t stream);

} // namespace dilated_convolution
} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
