/**
 * \file dnn/test/cuda/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define cuda_check(expr)                                   \
    do {                                                   \
        auto ret = (expr);                                 \
        if (ret != cudaSuccess) {                          \
            fprintf(stderr, "cuda call %s failed", #expr); \
            __builtin_trap();                              \
        }                                                  \
    } while (0)

namespace megdnn {
namespace test {
bool check_compute_capability(int major, int minor);
}  // namespace test
}  // namespace megdnn

#define require_compute_capability(x, y)                       \
    do {                                                       \
        if (!megdnn::test::check_compute_capability((x), (y))) \
            return;                                            \
    } while (0)

// vim: syntax=cpp.doxygen
