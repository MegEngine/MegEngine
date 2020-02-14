/**
 * \file dnn/src/cuda/sleep/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

    void sleep(cudaStream_t stream, uint64_t cycles);

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
