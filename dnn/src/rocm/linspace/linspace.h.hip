/**
 * \file dnn/src/rocm/linspace/linspace.h.hip
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
