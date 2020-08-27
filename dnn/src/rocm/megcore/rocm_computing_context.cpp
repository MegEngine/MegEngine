/**
 * \file dnn/src/rocm/megcore/rocm_computing_context.cpp
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

#include "src/common/utils.h"
#include "src/rocm/utils.h"
#include "./computing_context.hpp"

#include "./rocm_computing_context.hpp"

using namespace megcore;
using namespace rocm;

std::unique_ptr<ComputingContext> megcore::make_rocm_computing_context(megcoreDeviceHandle_t dev_handle, unsigned int flags) {
    return std::make_unique<ROCMComputingContext>(dev_handle, flags);
}

ROCMComputingContext::ROCMComputingContext(megcoreDeviceHandle_t dev_handle,
        unsigned int flags, const ROCMContext& ctx):
    ComputingContext(dev_handle, flags),
    own_stream_{ctx.stream == nullptr},
    context_{ctx}
{
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformROCM);
    if (own_stream_) {
        hip_check(hipStreamCreateWithFlags(&context_.stream,
                    hipStreamNonBlocking));
    }
}

ROCMComputingContext::~ROCMComputingContext()
{
    if (own_stream_) {
        hip_check(hipStreamDestroy(context_.stream));
    }
}

void ROCMComputingContext::memcpy(void *dst, const void *src,
        size_t size_in_bytes, megcoreMemcpyKind_t kind)
{
    hipMemcpyKind hip_kind;
    switch (kind) {
        case megcoreMemcpyDeviceToHost:
            hip_kind = hipMemcpyDeviceToHost;
            break;
        case megcoreMemcpyHostToDevice:
            hip_kind = hipMemcpyHostToDevice;
            break;
        case megcoreMemcpyDeviceToDevice:
            hip_kind = hipMemcpyDeviceToDevice;
            break;
        default:
            megdnn_throw("bad hip memcpy kind");
    }
    hip_check(hipMemcpyAsync(dst, src, size_in_bytes, hip_kind,
                context_.stream));
}

void ROCMComputingContext::memset(void *dst, int value, size_t size_in_bytes)
{
    hip_check(hipMemsetAsync(dst, value, size_in_bytes, context_.stream));
}

void ROCMComputingContext::synchronize()
{
    hip_check(hipStreamSynchronize(context_.stream));
}


// vim: syntax=cpp.doxygen
