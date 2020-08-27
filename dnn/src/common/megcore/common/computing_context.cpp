/**
 * \file dnn/src/common/megcore/common/computing_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"

#include "./computing_context.hpp"
#include "../cpu/default_computing_context.hpp"
#if MEGDNN_WITH_CUDA
#include "src/cuda/megcore/cuda_computing_context.hpp"
#endif


#if MEGDNN_WITH_ROCM
#include "src/rocm/megcore/computing_context.hpp"
#endif

#if MEGDNN_WITH_CAMBRICON
#include "src/cambricon/megcore/cambricon_computing_context.hpp"
#endif

#if MEGDNN_WITH_ATLAS
#include "src/atlas/megcore/computing_context.hpp"
#endif

using namespace megcore;
using namespace megdnn;

std::unique_ptr<ComputingContext> ComputingContext::make(
        megcoreDeviceHandle_t dev_handle, unsigned int flags)
{
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    switch (platform) {
        case megcorePlatformCPU:
            return make_unique<cpu::DefaultComputingContext>(dev_handle, flags);
#if MEGDNN_WITH_CUDA
        case megcorePlatformCUDA:
            return make_unique<cuda::CUDAComputingContext>(dev_handle, flags);
#endif
#if MEGDNN_WITH_ROCM
        case megcorePlatformROCM:
            return make_rocm_computing_context(dev_handle, flags);
#endif
#if MEGDNN_WITH_CAMBRICON
        case megcorePlatformCambricon:
            return make_unique<cambricon::CambriconComputingContext>(dev_handle,
                                                                     flags);
#endif
#if MEGDNN_WITH_ATLAS
        case megcorePlatformAtlas:
            return make_unique<atlas::AtlasComputingContext>(dev_handle, flags);
#endif
        default:
            megdnn_throw("bad platform");
    }
}

ComputingContext::~ComputingContext() noexcept = default;

// vim: syntax=cpp.doxygen
