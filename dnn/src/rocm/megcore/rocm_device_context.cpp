/**
 * \file dnn/src/rocm/megcore/rocm_device_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "megcore.h"
#include "src/common/utils.h"
#include "src/rocm/utils.h"
#include "./device_context.hpp"

#include "./rocm_device_context.hpp"

//! HIP_VERSION_MAJOR HIP_VERSION_MINOR HIP_VERSION_PATCH is defined when
//! compile with hipcc

using namespace megcore;
using namespace rocm;

std::unique_ptr<DeviceContext> megcore::make_rocm_device_context(int deviceID, unsigned int flags) {
    return std::make_unique<ROCMDeviceContext>(deviceID, flags);
}

ROCMDeviceContext::ROCMDeviceContext(int device_id, unsigned int flags):
    DeviceContext(megcorePlatformROCM, device_id, flags)
{
    int version;
    hip_check(hipRuntimeGetVersion(&version));
    int id = device_id;
    if (id < 0) {
        hip_check(hipGetDevice(&id));
    }
    hip_check(hipGetDeviceProperties(&prop_, id));
}

ROCMDeviceContext::~ROCMDeviceContext() noexcept = default;

size_t ROCMDeviceContext::mem_alignment_in_bytes() const noexcept {
    return 1u;
#if 0
    return std::max(prop_.textureAlignment, prop_.texturePitchAlignment);
#endif
}

void ROCMDeviceContext::activate()
{
    int id = device_id();
    if (id >= 0) {
        hip_check(hipSetDevice(id));
    }
}

void *ROCMDeviceContext::malloc(size_t size_in_bytes)
{
    void *ptr;
    hip_check(hipMalloc(&ptr, size_in_bytes));
    return ptr;
}

void ROCMDeviceContext::free(void *ptr)
{
    hip_check(hipFree(ptr));
}

// vim: syntax=cpp.doxygen
