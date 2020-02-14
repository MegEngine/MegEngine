/**
 * \file dnn/src/cuda/query_blocksize.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

namespace megdnn {
namespace cuda {

struct LaunchConfig {
    int grid_size;   //!< minimal grid size
    int block_size;  //!< suggested block size
};

//! get shared mem size given block size
struct SmemGetter {
    typedef int (*func_t)(int block_size, void* user_data);
    func_t func;
    void* user_data;

    SmemGetter(func_t func_ = 0, void* user_data_ = 0)
            : func(func_), user_data(user_data_) {}
};

/*!
 * \brief cudaOccupancyMaxPotentialBlockSize only available when compiled by
 *      nvcc; so we need to wrap this function and expose it to normal c++
 *
 * Note that the result is cached for kernel ptr.
 */
LaunchConfig query_launch_config_for_kernel(
        const void* kern, const SmemGetter& smem = SmemGetter());

//! return block size only
static inline int query_blocksize_for_kernel(const void* kern) {
    return query_launch_config_for_kernel(kern).block_size;
}

template <typename T>
static inline int query_blocksize_for_kernel(T kern) {
    return query_blocksize_for_kernel(reinterpret_cast<const void*>(kern));
}

namespace detail {
LaunchConfig query_launch_config_for_kernel_uncached(const void* kern,
                                                     const SmemGetter& smem);
}

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

