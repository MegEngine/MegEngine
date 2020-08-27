/**
 * \file src/rocm/convolution/im2col.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#pragma once

#include "hip_header.h"

namespace megdnn {
namespace rocm {
namespace convolution {

//! col is of shape (ic*fh*fw, oh*ow*n)
template <typename T>
void im2col(const T* im, T* col, size_t N, size_t INP_BS, size_t IC, size_t IH,
            size_t IW, size_t FH, size_t FW, size_t OH, size_t OW, size_t PH,
            size_t PW, size_t SH, size_t SW, size_t DH, size_t DW,  // dilation
            hipStream_t stream);

template <typename T>
void col2im(const T* col, T* im, size_t N, size_t INP_BS, size_t IC, size_t IH,
            size_t IW, size_t FH, size_t FW, size_t OH, size_t OW, size_t PH,
            size_t PW, size_t SH, size_t SW, size_t DH, size_t DW,  // dilation
            hipStream_t stream);

} // namespace convolution
} // namespace rocm
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
