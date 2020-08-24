/**
 * \file dnn/src/cambricon/megcore/cambricon_device_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megcore.h"

#include "src/cambricon/utils.h"
#include "src/common/utils.h"

#include "src/cambricon/megcore/cambricon_device_context.hpp"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define CNRT_VERSION_STR    \
    STR(CNRT_MAJOR_VERSION) \
    "." STR(CNRT_MINOR_VERSION) "." STR(CNRT_PATCH_VERSION)

#pragma message "compile with cnrt " CNRT_VERSION_STR " "

#undef STR_HELPER
#undef STR

using namespace megcore;
using namespace cambricon;

CambriconDeviceContext::CambriconDeviceContext(int device_id,
                                               unsigned int flags,
                                               bool global_initialized)
        : DeviceContext(megcorePlatformCambricon, device_id, flags) {
    if (!global_initialized)
        init_status.init();
    unsigned int version;
    cnrt_check(cnrtGetVersion(&version));
    megdnn_assert(version == CNRT_VERSION,
                  "megcore compiled with cnrt %d, get %d at runtime",
                  CNRT_VERSION, version);
    unsigned int dev_num;
    cnrt_check(cnrtGetDeviceCount(&dev_num));
    MEGDNN_MARK_USED_VAR(dev_num);
    // check validity of device_id
    megdnn_assert(device_id >= 0 &&
                  static_cast<unsigned int>(device_id) < dev_num);
    cnrt_check(cnrtGetDeviceInfo(&device_info, device_id));
}

CambriconDeviceContext::~CambriconDeviceContext() noexcept = default;

size_t CambriconDeviceContext::mem_alignment_in_bytes() const noexcept {
    return 1;
}

void CambriconDeviceContext::activate() {
    int id = device_id();
    cnrtDev_t dev;
    cnrt_check(cnrtGetDeviceHandle(&dev, id));
    cnrt_check(cnrtSetCurrentDevice(dev));
}

void* CambriconDeviceContext::malloc(size_t size_in_bytes) {
    void* ptr;
    cnrt_check(cnrtMalloc(&ptr, size_in_bytes));
    return ptr;
}

void CambriconDeviceContext::free(void* ptr) {
    cnrt_check(cnrtFree(ptr));
}

CambriconDeviceContext::InitStatus CambriconDeviceContext::init_status;

// vim: syntax=cpp.doxygen

