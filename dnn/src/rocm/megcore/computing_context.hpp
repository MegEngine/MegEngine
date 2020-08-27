/**
 * \file dnn/src/rocm/megcore/computing_context.hpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/common/megcore/common/computing_context.hpp"
#include <memory>

namespace megcore {
std::unique_ptr<ComputingContext> make_rocm_computing_context(megcoreDeviceHandle_t dev_handle, unsigned int flags);
}
