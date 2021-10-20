/**
 * \file dnn/src/rocm/relayout/relayout.h.hip
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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


