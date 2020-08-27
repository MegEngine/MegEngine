/**
 * \file src/rocm/relayout/relayout.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#pragma once

#include "hip_header.h"
#include "megdnn/basic_types.h"

namespace megdnn {
namespace rocm {

void copy_noncontig_general(const TensorND& dst, const TensorND& src,
                            hipStream_t stream);
}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen


