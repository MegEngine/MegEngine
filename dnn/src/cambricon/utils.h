/**
 * \file dnn/src/cambricon/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megcore_cdefs.h"
#include "megdnn/handle.h"
#include "src/cambricon/utils.mlu.h"
#include "src/common/utils.h"

#include "src/cambricon/handle.h"

#include <cnrt.h>

namespace megdnn {
namespace cambricon {

static inline HandleImpl* concrete_handle(Handle* handle) {
    return static_cast<cambricon::HandleImpl*>(handle);
}

static inline cnrtQueue_t cnrt_queue(Handle* handle) {
    return concrete_handle(handle)->queue();
}

//! get device info of current active device
cnrtDeviceInfo_t current_device_info();

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen

