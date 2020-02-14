/**
 * \file dnn/src/common/opr_delegate.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/opr_delegate.h"

using namespace megdnn;

const std::shared_ptr<Handle>& megdnn::inplace_cpu_handle() {
    auto make = []() {
        megcoreDeviceHandle_t dev_handle;
        megcoreCreateDeviceHandle(&dev_handle, megcorePlatformCPU);
        megcoreComputingHandle_t comp_handle;
        megcoreCreateComputingHandle(&comp_handle, dev_handle);
        auto destructor = [=]() {
            megcoreDestroyComputingHandle(comp_handle);
            megcoreDestroyDeviceHandle(dev_handle);
        };
        std::shared_ptr<Handle> handle = Handle::make(comp_handle);
        handle->set_destructor(destructor);
        return handle;
    };
    static std::shared_ptr<Handle> handle = make();
    return handle;
}

// vim: syntax=cpp.doxygen
