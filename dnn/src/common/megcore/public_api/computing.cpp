/**
 * \file dnn/src/common/megcore/public_api/computing.cpp
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

#include "./computing.hpp"
#include "../common/computing_context.hpp"

using namespace megcore;

megcoreStatus_t megcoreCreateComputingHandle(
        megcoreComputingHandle_t *compHandle,
        megcoreDeviceHandle_t devHandle,
        unsigned int flags)
{
    auto ctx = ComputingContext::make(devHandle, flags);
    auto &H = *compHandle;
    H = new megcoreComputingContext;
    H->content = std::move(ctx);
    return megcoreSuccess;
}

megcoreStatus_t megcoreDestroyComputingHandle(
        megcoreComputingHandle_t handle)
{
    megdnn_assert(handle);
    delete handle;
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetDeviceHandle(
        megcoreComputingHandle_t compHandle,
        megcoreDeviceHandle_t *devHandle)
{
    megdnn_assert(compHandle);
    *devHandle = compHandle->content->dev_handle();
    return megcoreSuccess;
}

megcoreStatus_t megcoreGetComputingFlags(
        megcoreComputingHandle_t handle,
        unsigned int *flags)
{
    megdnn_assert(handle);
    *flags = handle->content->flags();
    return megcoreSuccess;
}

megcoreStatus_t megcoreMemcpy(megcoreComputingHandle_t handle,
        void *dst, const void *src, size_t sizeInBytes,
        megcoreMemcpyKind_t kind)
{
    megdnn_assert(handle);
    handle->content->memcpy(dst, src, sizeInBytes, kind);
    return megcoreSuccess;
}

megcoreStatus_t megcoreMemset(megcoreComputingHandle_t handle,
        void *dst, int value, size_t sizeInBytes)
{
    megdnn_assert(handle);
    handle->content->memset(dst, value, sizeInBytes);
    return megcoreSuccess;
}

megcoreStatus_t megcoreSynchronize(megcoreComputingHandle_t handle)
{
    megdnn_assert(handle);
    handle->content->synchronize();
    return megcoreSuccess;
}

// vim: syntax=cpp.doxygen
