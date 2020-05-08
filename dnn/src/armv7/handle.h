/**
 * \file dnn/src/armv7/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/arm_common/handle.h"

namespace megdnn {
namespace armv7 {

class HandleImpl: public arm_common::HandleImpl {
    public:
        HandleImpl(megcoreComputingHandle_t computing_handle,
                HandleType type = HandleType::ARMV7):
            arm_common::HandleImpl::HandleImpl(computing_handle, type)
        {
        }

        template <typename Opr>
        std::unique_ptr<Opr> create_operator();
};

} // namespace armv7
} // namespace megdnn

// vim: syntax=cpp.doxygen


