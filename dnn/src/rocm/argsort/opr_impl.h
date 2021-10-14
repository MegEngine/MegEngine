/**
 * \file dnn/src/rocm/argsort/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace rocm {

class ArgsortForwardImpl final : public ArgsortForward {
public:
    using ArgsortForward::ArgsortForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst,
            const TensorLayout& indices) override;
};

class ArgsortBackwardImpl final : public ArgsortBackward {
public:
    using ArgsortBackward::ArgsortBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in indices, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
