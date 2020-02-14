/**
 * \file dnn/include/megcore_cuda.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./megcore.h"

#include <cuda_runtime_api.h>

#include "megdnn/internal/visibility_prologue.h"

namespace megcore {
struct CudaContext {
    cudaStream_t stream = nullptr;

    //! device pointer to buffer for error reporting from kernels
    AsyncErrorInfo* error_info = nullptr;

    CudaContext() = default;

    CudaContext(cudaStream_t s, AsyncErrorInfo* e) : stream{s}, error_info{e} {}
};

megcoreStatus_t createComputingHandleWithCUDAContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const CudaContext& ctx);

megcoreStatus_t getCUDAContext(megcoreComputingHandle_t handle,
                               CudaContext* ctx);

}  // namespace megcore

static inline megcoreStatus_t megcoreCreateComputingHandleWithCUDAStream(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, cudaStream_t stream) {
    megcore::CudaContext ctx;
    ctx.stream = stream;
    return megcore::createComputingHandleWithCUDAContext(compHandle, devHandle,
                                                         flags, ctx);
}

static inline megcoreStatus_t megcoreGetCUDAStream(
        megcoreComputingHandle_t handle, cudaStream_t* stream) {
    megcore::CudaContext ctx;
    auto ret = megcore::getCUDAContext(handle, &ctx);
    *stream = ctx.stream;
    return ret;
}

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
