/**
 * \file dnn/src/common/megcore/cpu/default_device_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"

#include "./default_device_context.hpp"
#include <cstdlib>

using namespace megcore;
using namespace megcore::cpu;
using namespace megdnn;

DefaultDeviceContext::DefaultDeviceContext(int device_id, unsigned int flags):
    DeviceContext(megcorePlatformCPU, device_id, flags)
{
    megdnn_assert(device_id == -1);
}

DefaultDeviceContext::~DefaultDeviceContext() noexcept = default;

size_t DefaultDeviceContext::mem_alignment_in_bytes() const noexcept {
    return 1;
}

void DefaultDeviceContext::activate() noexcept {
}

void *DefaultDeviceContext::malloc(size_t size_in_bytes) {
    return new uint8_t[size_in_bytes];
}

void DefaultDeviceContext::free(void *ptr) {
    delete []static_cast<uint8_t*>(ptr);
}

// vim: syntax=cpp.doxygen
