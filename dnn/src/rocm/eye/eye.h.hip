/**
 * \file src/rocm/eye/eye.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#pragma once
#include <stdint.h>
#include "hip_header.h"

namespace megdnn {
namespace rocm {
namespace eye {

template <typename T>
void exec_internal(T* dst, size_t m, size_t n, int k, hipStream_t stream);

}  // namespace eye
}  // namespace rocm
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
