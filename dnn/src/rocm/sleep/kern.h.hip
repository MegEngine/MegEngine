/**
 * \file src/rocm/sleep/kern.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#pragma once

#include "hip_header.h"
#include "src/rocm/utils.h.hip"

namespace megdnn {
namespace rocm {

    void sleep(hipStream_t stream, uint64_t cycles);

} // namespace rocm
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
