/**
 * \file dnn/test/cuda/megcore/device.cpp
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
#include "./fixture.h"
#include "test/cuda/utils.h"
#include <cuda_runtime_api.h>

TEST_F(MegcoreCUDA, DEVICE)
{
    for (int id = -1; id < std::min(nr_devices(), 2); ++id) {
        megcoreDeviceHandle_t handle;
        megcoreCreateDeviceHandle(&handle, megcorePlatformCUDA,
                    id, 0);

        int deviceID;
        megcoreGetDeviceID(handle, &deviceID);
        ASSERT_EQ(id, deviceID);

        megcorePlatform_t platform;
        megcoreGetPlatform(handle, &platform);
        ASSERT_EQ(megcorePlatformCUDA, platform);

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
}

TEST_F(MegcoreCUDA, ERROR_MSG) {
#if MEGDNN_ENABLE_EXCEPTIONS
    megcoreDeviceHandle_t handle;
    ASSERT_THROW(
            megcoreCreateDeviceHandle(
                &handle, megcorePlatformCUDA, nr_devices(), 0),
            megdnn::test::MegDNNError);
    cudaGetLastError();
    cuda_check(cudaGetLastError());
#endif
}

// vim: syntax=cpp.doxygen
