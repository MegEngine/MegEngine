/**
 * \file dnn/src/naive/check_non_finite/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class CheckNonFiniteImpl final : public CheckNonFinite {
    size_t _get_workspace_in_bytes() override { return 0; }

public:
    using CheckNonFinite::CheckNonFinite;

    bool is_thread_safe() const override { return true; }

    size_t get_workspace_in_bytes(const TensorNDArray&, const TensorLayout&) override {
        m_size = 0;
        return _get_workspace_in_bytes();
    }

    void exec(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
