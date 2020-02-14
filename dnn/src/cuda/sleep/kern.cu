/**
 * \file dnn/src/cuda/sleep/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"

namespace {

static __global__ void kern(uint64_t cycles) {
    uint64_t start = clock64();
    for (;;) {
        if (clock64() - start > cycles)
            return;
    }
}

}

void megdnn::cuda::sleep(cudaStream_t stream, uint64_t cycles) {
    kern<<< 1, 1, 0, stream >>>(cycles);
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen

