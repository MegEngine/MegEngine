/**
 * \file dnn/test/common/megcore/device.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megcore.h"

#include "test/common/utils.h"
#include <gtest/gtest.h>
TEST(MegcoreCPU, DEVICE) {
    megcoreDeviceHandle_t handle;
    megcoreCreateDeviceHandle(&handle, megcorePlatformCPU, -1, 0);

    int deviceID;
    megcoreGetDeviceID(handle, &deviceID);
    ASSERT_EQ(-1, deviceID);

    megcorePlatform_t platform;
    megcoreGetPlatform(handle, &platform);
    ASSERT_EQ(megcorePlatformCPU, platform);

    unsigned int flags;
    megcoreGetDeviceFlags(handle, &flags);
    ASSERT_EQ(0u, flags);

    size_t memAlignmentInBytes;
    megcoreGetMemAlignment(handle, &memAlignmentInBytes);

    megcoreActivate(handle);

    void *ptr;
    megcoreMalloc(handle, &ptr, 256);
    megcoreFree(handle, ptr);

    megcoreDestroyDeviceHandle(handle);
}
// vim: syntax=cpp.doxygen
