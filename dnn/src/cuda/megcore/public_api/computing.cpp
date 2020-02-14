/**
 * \file dnn/src/cuda/megcore/public_api/computing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megcore_cuda.h"

#include "src/common/utils.h"
#include "src/common/megcore/public_api/computing.hpp"
#include "../cuda_computing_context.hpp"

using namespace megcore;

megcoreStatus_t megcore::createComputingHandleWithCUDAContext(
        megcoreComputingHandle_t *compHandle,
        megcoreDeviceHandle_t devHandle,
        unsigned int flags,
        const CudaContext& ctx)
{
    auto content = megdnn::make_unique<cuda::CUDAComputingContext>(
            devHandle, flags, ctx);
    auto &H = *compHandle;
    H = new megcoreComputingContext;
    H->content = std::move(content);
    return megcoreSuccess;
}

megcoreStatus_t megcore::getCUDAContext(megcoreComputingHandle_t handle,
        CudaContext* ctx)
{
    auto &&H = handle;
    megdnn_assert(H);
    megcoreDeviceHandle_t dev_handle = H->content->dev_handle();
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformCUDA);
    auto context = static_cast<megcore::cuda::CUDAComputingContext *>(
            H->content.get());
    *ctx = context->context();
    return megcoreSuccess;
}

// vim: syntax=cpp.doxygen

