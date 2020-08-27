/**
 * \file dnn/src/rocm/add_update/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs.h"
#include "src/common/add_update_helper.h"
#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

class AddUpdateForwardImpl final : public AddUpdateForwardHelper {
    void exec_noncontig(_megdnn_tensor_inout dest, _megdnn_tensor_in delta);

public:
    using AddUpdateForwardHelper::AddUpdateForwardHelper;

    void exec(_megdnn_tensor_inout dest, _megdnn_tensor_in delta) override;

    bool is_thread_safe() const override { return true; }
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
