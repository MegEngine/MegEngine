/**
 * \file dnn/src/cuda/ptx_loader.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <mutex>
#include <unordered_map>
#include "src/cuda/ptx/uint4_int4/kern.cuh"
#include "src/cuda/utils.h"
namespace megdnn {
namespace cuda {

class PTXKernelLoader {
private:
    PTXKernelLoader() = default;
    using kernel = std::function<void(const dim3, const dim3, cudaStream_t, void**)>;

public:
    PTXKernelLoader(const PTXKernelLoader&) = delete;
    const PTXKernelLoader& operator=(const PTXKernelLoader&) = delete;
    static PTXKernelLoader& instance();

    const kernel get_kernel(const std::string& kernel_name);

    static const std::unordered_map<std::string, kernel> KERNEL_MAP;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
