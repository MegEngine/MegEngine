/**
 * \file dnn/test/cuda/local/local.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./local.h"

#include <cstdio>

namespace megdnn {
namespace test {

static const int SHARED_SIZE = 12288;

__global__ void kern()
{
    __shared__ int shared[SHARED_SIZE];
    for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) {
        shared[i] = 0x7fffffff;
        shared[i] = shared[i];
    }
    __syncthreads();
}

void pollute_shared_mem(cudaStream_t stream)
{
    for (size_t i = 0; i < 256; ++i) kern<<<32, 256, 0, stream>>>();
}

} // namespace test
} // namespace megdnn
