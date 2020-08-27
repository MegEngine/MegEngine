/**
 * \file src/rocm/convolution/forward/inplace_matmul_impl.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include "hip_header.h"

namespace megdnn {
namespace rocm {
namespace convolution {

void exec_inplace_matmul_fwd(const float* src, const float* filter, float* dst,
                             size_t N, size_t INP_BS, size_t OUT_BS, size_t IC,
                             size_t IH, size_t IW, size_t OC, size_t OH,
                             size_t OW, size_t FH, size_t FW, size_t PH,
                             size_t PW, size_t SH, size_t SW, bool is_xcorr,
                             hipStream_t stream);

} // namespace convolution
} // namespace rocm
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
