/**
 * \file dnn/src/aarch64/handle.h
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
namespace aarch64 {

class HandleImpl: public arm_common::HandleImpl {
    public:
        HandleImpl(megcoreComputingHandle_t computing_handle,
                HandleType type = HandleType::AARCH64):
            arm_common::HandleImpl::HandleImpl(computing_handle, type)
        {}

        template <typename Opr>
        std::unique_ptr<Opr> create_operator();
};

} // namespace aarch64
} // namespace megdnn

// vim: syntax=cpp.doxygen


