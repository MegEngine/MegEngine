/**
 * \file dnn/src/rocm/checksum/kern.h.hip
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

namespace megdnn {
namespace rocm {
namespace checksum {

void calc(uint32_t* dest, const uint32_t* buf, uint32_t* workspace,
          size_t nr_elem, hipStream_t stream);

size_t get_workspace_in_bytes(size_t nr_elem);

}  // namespace checksum
}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

