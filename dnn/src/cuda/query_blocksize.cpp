/**
 * \file dnn/src/cuda/query_blocksize.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./query_blocksize.cuh"
#include "src/cuda/utils.h"

#include <mutex>
#include <unordered_map>

using namespace megdnn;
using namespace cuda;

namespace {

size_t hash_pair_combine(size_t x, size_t y) {
    return y + 0x9e3779b9 + (x << 6) + (x >> 2);
}

//! stupid committee has no pair hash. Let's do it for them
struct pairhash {
public:
    template <typename T, typename U>
    size_t operator()(const std::pair<T, U>& x) const {
        return hash_pair_combine(std::hash<T>{}(x.first),
                                 std::hash<U>{}(x.second));
    }
};
}  // anonymous namespace

LaunchConfig cuda::query_launch_config_for_kernel(const void* kern,
                                                  const SmemGetter& smem) {
    static std::mutex mtx;
    static std::unordered_map<std::pair<int, const void*>, LaunchConfig,
                              pairhash>
            cache;
    std::lock_guard<std::mutex> _lock{mtx};

    int device = -1;
    cuda_check(cudaGetDevice(&device));
    auto ins = cache.insert({{device, kern}, LaunchConfig{}});
    if (ins.second) {
        ins.first->second =
                detail::query_launch_config_for_kernel_uncached(kern, smem);
    }
    return ins.first->second;
}

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

