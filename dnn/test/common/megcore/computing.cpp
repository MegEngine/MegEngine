/**
 * \file dnn/test/common/megcore/computing.cpp
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
TEST(MegcoreCPU, COMPUTING)
{
    megcoreDeviceHandle_t devHandle;
    megcoreCreateDeviceHandle(&devHandle, megcorePlatformCPU, -1, 0);

    megcoreComputingHandle_t compHandle;
    megcoreCreateComputingHandle(&compHandle, devHandle, 0);

    megcoreDeviceHandle_t devHandle2;
    megcoreGetDeviceHandle(compHandle, &devHandle2);
    ASSERT_EQ(devHandle, devHandle2);

    unsigned int flags;
    megcoreGetComputingFlags(compHandle, &flags);
    ASSERT_EQ(0u, flags);

    unsigned char *src, *dst;
    static const size_t N = 5;
    megcoreMalloc(devHandle, (void **)&src, N);
    megcoreMalloc(devHandle, (void **)&dst, N);
    megcoreMemset(compHandle, src, 0x0F, N);
    megcoreMemset(compHandle, dst, 0xF0, N);
    megcoreSynchronize(compHandle);
    for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(0x0F, src[i]);
        ASSERT_EQ(0xF0, dst[i]);
    }
    megcoreMemcpy(compHandle, dst, src, N, megcoreMemcpyDeviceToDevice);
    megcoreSynchronize(compHandle);
    for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(dst[i], src[i]);
    }
    megcoreFree(devHandle, src);
    megcoreFree(devHandle, dst);

    megcoreDestroyComputingHandle(compHandle);
    megcoreDestroyDeviceHandle(devHandle);
}
// vim: syntax=cpp.doxygen
