/**
 * \file dnn/src/cuda/check_non_finite/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs/utils.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class CheckNonFiniteImpl final : public CheckNonFinite {
public:
    using CheckNonFinite::CheckNonFinite;

    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;

    bool is_thread_safe() const override { return true; }

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
