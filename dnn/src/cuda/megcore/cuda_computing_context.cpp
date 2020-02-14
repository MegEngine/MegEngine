/**
 * \file dnn/src/cuda/megcore/cuda_computing_context.cpp
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
#include "src/cuda/utils.h"


#include "./cuda_computing_context.hpp"

using namespace megcore;
using namespace megcore::cuda;

CUDAComputingContext::CUDAComputingContext(megcoreDeviceHandle_t dev_handle,
        unsigned int flags, const CudaContext& ctx):
    ComputingContext(dev_handle, flags),
    own_stream_{ctx.stream == nullptr},
    context_{ctx}
{
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformCUDA);
    if (own_stream_) {
        cuda_check(cudaStreamCreateWithFlags(&context_.stream,
                    cudaStreamNonBlocking));
    }
}

CUDAComputingContext::~CUDAComputingContext()
{
    if (own_stream_) {
        cuda_check(cudaStreamDestroy(context_.stream));
    }
}

void CUDAComputingContext::memcpy(void *dst, const void *src,
        size_t size_in_bytes, megcoreMemcpyKind_t kind)
{
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case megcoreMemcpyDeviceToHost:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case megcoreMemcpyHostToDevice:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case megcoreMemcpyDeviceToDevice:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            megdnn_throw("bad cuda memcpy kind");
    }
    cuda_check(cudaMemcpyAsync(dst, src, size_in_bytes, cuda_kind,
                context_.stream));
}

void CUDAComputingContext::memset(void *dst, int value, size_t size_in_bytes)
{
    cuda_check(cudaMemsetAsync(dst, value, size_in_bytes, context_.stream));
}

void CUDAComputingContext::synchronize()
{
    cuda_check(cudaStreamSynchronize(context_.stream));
}


// vim: syntax=cpp.doxygen
