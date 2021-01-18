/**
 * \file dnn/src/common/cpuinfo_arch_vendor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "src/common/utils.h"
#if defined(MGB_ENABLE_CPUINFO_CHECK) && MGB_ENABLE_CPUINFO

#include <cpuinfo.h>

namespace megdnn {

const char* vendor_to_string(enum cpuinfo_vendor vendor);
const char* uarch_to_string(enum cpuinfo_uarch uarch);

}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
