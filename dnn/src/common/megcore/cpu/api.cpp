/**
 * \file dnn/src/common/megcore/cpu/api.cpp
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

#include "./default_computing_context.hpp"
#include "../common/computing_context.hpp"
#include "../public_api/computing.hpp"

using namespace megcore;

CPUDispatcher::~CPUDispatcher() noexcept = default;

megcoreStatus_t megcoreCreateComputingHandleWithCPUDispatcher(
        megcoreComputingHandle_t *compHandle,
        megcoreDeviceHandle_t devHandle,
        const std::shared_ptr<CPUDispatcher>& dispatcher,
        unsigned int flags) {
    auto content = megdnn::make_unique<
        megcore::cpu::DefaultComputingContext>(devHandle, flags);
    auto &H = *compHandle;
    content->set_dispatcher(dispatcher);
    H = new megcoreComputingContext;
    H->content = std::move(content);
    return megcoreSuccess;
}

CPUDispatcher* megcoreGetCPUDispatcher(megcoreComputingHandle_t handle) {
    auto &&H = handle;
    megdnn_assert(H);
    // Check device handle.
    megcoreDeviceHandle_t dev_handle = H->content->dev_handle();
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform &megcorePlatformCPU);
    auto context = static_cast<megcore::cpu::DefaultComputingContext*>(
            H->content.get());
    return context->get_dispatcher();
}

// vim: syntax=cpp.doxygen
