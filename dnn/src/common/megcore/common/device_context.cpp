/**
 * \file dnn/src/common/megcore/common/device_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./device_context.hpp"

#include "src/common/utils.h"
#include "../cpu/default_device_context.hpp"
#if MEGDNN_WITH_CUDA
#include "src/cuda/megcore/cuda_device_context.hpp"
#endif
#if MEGDNN_WITH_ROCM
#include "src/rocm/megcore/device_context.hpp"
#endif
#if MEGDNN_WITH_CAMBRICON
#include "src/cambricon/megcore/cambricon_device_context.hpp"
#endif

#if MEGDNN_WITH_ATLAS
#include "src/atlas/megcore/device_context.hpp"
#endif

using namespace megcore;
using namespace megdnn;

std::unique_ptr<DeviceContext> DeviceContext::make(megcorePlatform_t platform,
        int deviceID, unsigned int flags)
{
    switch (platform) {
        case megcorePlatformCPU:
            return make_unique<cpu::DefaultDeviceContext>(deviceID, flags);
#if MEGDNN_WITH_CUDA
        case megcorePlatformCUDA:
            return make_unique<cuda::CUDADeviceContext>(deviceID, flags);
#endif
#if MEGDNN_WITH_ROCM
        case megcorePlatformROCM:
            return make_rocm_device_context(deviceID, flags);
#endif
#if MEGDNN_WITH_CAMBRICON
        case megcorePlatformCambricon:
            return make_unique<cambricon::CambriconDeviceContext>(deviceID,
                                                                  flags);
#endif
#if MEGDNN_WITH_ATLAS
        case megcorePlatformAtlas:
            return make_unique<atlas::AtlasDeviceContext>(deviceID, flags);
#endif
        default:
            megdnn_throw("bad platform");
    }
}

DeviceContext::~DeviceContext() noexcept = default;


// vim: syntax=cpp.doxygen
