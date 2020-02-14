/**
 * \file dnn/src/cuda/megcore/cuda_device_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megcore.h"
#include "src/common/utils.h"
#include "src/cuda/utils.h"

#include "./cuda_device_context.hpp"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#pragma message "compile with cuda " STR(CUDART_VERSION) " "

using namespace megcore;
using namespace cuda;

CUDADeviceContext::CUDADeviceContext(int device_id, unsigned int flags):
    DeviceContext(megcorePlatformCUDA, device_id, flags)
{
    int version;
    cuda_check(cudaRuntimeGetVersion(&version));
    megdnn_assert(version == CUDART_VERSION,
            "megcore compiled with cuda %d, get %d at runtime",
            CUDART_VERSION, version);
    int id = device_id;
    if (id < 0) {
        cuda_check(cudaGetDevice(&id));
    }
    cuda_check(cudaGetDeviceProperties(&prop_, id));
}

CUDADeviceContext::~CUDADeviceContext() noexcept = default;

size_t CUDADeviceContext::mem_alignment_in_bytes() const noexcept {
    return std::max(prop_.textureAlignment, prop_.texturePitchAlignment);
}

void CUDADeviceContext::activate()
{
    int id = device_id();
    if (id >= 0) {
        cuda_check(cudaSetDevice(id));
    }
}

void *CUDADeviceContext::malloc(size_t size_in_bytes)
{
    void *ptr;
    cuda_check(cudaMalloc(&ptr, size_in_bytes));
    return ptr;
}

void CUDADeviceContext::free(void *ptr)
{
    cuda_check(cudaFree(ptr));
}

// vim: syntax=cpp.doxygen
