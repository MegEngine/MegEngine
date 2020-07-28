/**
 * \file dnn/src/arm_common/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/handle.h"
#if MGB_ENABLE_CPUINFO
#include "cpuinfo.h"
#endif

namespace megdnn {
namespace arm_common {

class HandleImpl: public fallback::HandleImpl {
    public:
        HandleImpl(megcoreComputingHandle_t computing_handle,
                HandleType type = HandleType::ARM_COMMON):
            fallback::HandleImpl::HandleImpl(computing_handle, type)
        {
            #if MGB_ENABLE_CPUINFO
            cpuinfo_initialize();
            #endif
        }

        template <typename Opr>
        std::unique_ptr<Opr> create_operator();
};

} // namespace arm_common
} // namespace megdnn

// vim: syntax=cpp.doxygen
