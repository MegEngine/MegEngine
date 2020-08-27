/**
 * \file dnn/test/rocm/megcore/computing.cpp
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
#include "megcore_rocm.h"

#include "test/common/utils.h"
#include "test/rocm/utils.h"
#include "./fixture.h"
#include "hip_header.h"

TEST_F(MegcoreROCM, COMPUTING)
{
    for (int id = -1; id < std::min(nr_devices(), 2); ++id) {
        megcoreDeviceHandle_t devHandle;
        megcoreCreateDeviceHandle(&devHandle,
                    megcorePlatformROCM, id, 0);
        megcoreActivate(devHandle);

        megcoreComputingHandle_t compHandle;
        megcoreCreateComputingHandle(&compHandle,
                    devHandle, 0);

        megcoreDeviceHandle_t devHandle2;
        megcoreGetDeviceHandle(compHandle, &devHandle2);
        ASSERT_EQ(devHandle, devHandle2);

        unsigned int flags;
        megcoreGetComputingFlags(compHandle, &flags);
        ASSERT_EQ(0u, flags);

        unsigned char *src, *dst;
        static const size_t N = 5;
        unsigned char src_host[N], dst_host[N];
        megcoreMalloc(devHandle, (void **)&src, N);
        megcoreMalloc(devHandle, (void **)&dst, N);
        megcoreMemset(compHandle, src, 0x0F, N);
        megcoreMemset(compHandle, dst, 0xF0, N);
        megcoreMemcpy(compHandle, src_host, src, N,
                    megcoreMemcpyDeviceToHost);
        megcoreMemcpy(compHandle, dst_host, dst, N,
                    megcoreMemcpyDeviceToHost);
        megcoreSynchronize(compHandle);
        for (size_t i = 0; i < N; ++i) {
            ASSERT_EQ(0x0F, src_host[i]);
            ASSERT_EQ(0xF0, dst_host[i]);
        }
        megcoreMemcpy(compHandle, dst, src, N,
                    megcoreMemcpyDeviceToDevice);
        megcoreMemcpy(compHandle, src_host, src, N,
                    megcoreMemcpyDeviceToHost);
        megcoreMemcpy(compHandle, dst_host, dst, N,
                    megcoreMemcpyDeviceToHost);
        megcoreSynchronize(compHandle);
        for (size_t i = 0; i < N; ++i) {
            ASSERT_EQ(dst_host[i], src_host[i]);
        }
        megcoreFree(devHandle, src);
        megcoreFree(devHandle, dst);

        megcoreDestroyComputingHandle(compHandle);
        megcoreDestroyDeviceHandle(devHandle);
    }
}

TEST_F(MegcoreROCM, STREAM)
{
    megcoreDeviceHandle_t devHandle;
    megcoreCreateDeviceHandle(&devHandle,
                megcorePlatformROCM, 0, 0);
    megcoreActivate(devHandle);

    hipStream_t stream;
    hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    megcoreComputingHandle_t compHandle;
    megcoreCreateComputingHandleWithROCMStream(&compHandle,
                devHandle, 0, stream);
    {
        hipStream_t stream2;
        megcoreGetROCMStream(compHandle, &stream2);
        ASSERT_EQ(stream, stream2);
    }

    megcoreDeviceHandle_t devHandle2;
    megcoreGetDeviceHandle(compHandle, &devHandle2);
    ASSERT_EQ(devHandle, devHandle2);

    unsigned int flags;
    megcoreGetComputingFlags(compHandle, &flags);
    ASSERT_EQ(0u, flags);

    unsigned char *src, *dst;
    static const size_t N = 5;
    unsigned char src_host[N], dst_host[N];
    megcoreMalloc(devHandle, (void **)&src, N);
    megcoreMalloc(devHandle, (void **)&dst, N);
    megcoreMemset(compHandle, src, 0x0F, N);
    megcoreMemset(compHandle, dst, 0xF0, N);
    megcoreMemcpy(compHandle, src_host, src, N,
                megcoreMemcpyDeviceToHost);
    megcoreMemcpy(compHandle, dst_host, dst, N,
                megcoreMemcpyDeviceToHost);
    megcoreSynchronize(compHandle);
    for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(0x0F, src_host[i]);
        ASSERT_EQ(0xF0, dst_host[i]);
    }
    megcoreMemcpy(compHandle, dst, src, N,
                megcoreMemcpyDeviceToDevice);
    megcoreMemcpy(compHandle, src_host, src, N,
                megcoreMemcpyDeviceToHost);
    megcoreMemcpy(compHandle, dst_host, dst, N,
                megcoreMemcpyDeviceToHost);
    megcoreSynchronize(compHandle);
    for (size_t i = 0; i < N; ++i) {
        ASSERT_EQ(dst_host[i], src_host[i]);
    }
    megcoreFree(devHandle, src);
    megcoreFree(devHandle, dst);

    megcoreDestroyComputingHandle(compHandle);
    megcoreDestroyDeviceHandle(devHandle);

    hip_check(hipStreamDestroy(stream));
}

// vim: syntax=cpp.doxygen
