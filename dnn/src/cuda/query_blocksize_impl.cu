/**
 * \file dnn/src/cuda/query_blocksize_impl.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;

/*
 * Note: cudaOccupancyMaxPotentialBlockSizeVariableSMem is only available when
 * compiled by nvcc, but it is implemented as a __host__ __device__ function. So
 * we implement a device wrapper
 */
namespace {

struct SmemGetterWrapper {
    SmemGetter getter;

    __device__ __host__ int operator()(int block_size) const {
#if __CUDA_ARCH__
        // device func should never be called
        int* ptr = 0;
        *ptr = 23;
#else
        if (getter.func) {
            return getter.func(block_size, getter.user_data);
        }
#endif
        return 0;
    }
};

}  // anonymous namespace

LaunchConfig cuda::detail::query_launch_config_for_kernel_uncached(
        const void* kern, const SmemGetter& smem) {
    SmemGetterWrapper s;
    s.getter = smem;
    LaunchConfig ret;
    cuda_check(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
            &ret.grid_size, &ret.block_size, kern, s));
    return ret;
}

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

