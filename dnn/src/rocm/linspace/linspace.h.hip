/**
 * \file src/rocm/linspace/linspace.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hip_header.h"

namespace megdnn {
namespace rocm {
namespace linspace {

template <typename T>
void exec_internal(T *dst, double start, double step, size_t n,
        hipStream_t stream);

} // namespace linspace
} // namespace rocm
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
