/**
 * \file dnn/test/cuda/local/local.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace test {

void pollute_shared_mem(cudaStream_t stream);

}  // namespace test
}  // namespace megdnn
