/**
 * \file dnn/src/common/megcore/public_api/device.hpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megcore.h"
#include "../common/device_context.hpp"
#include <memory>

struct megcoreDeviceContext {
    std::unique_ptr<megcore::DeviceContext> content;
};

// vim: syntax=cpp.doxygen
