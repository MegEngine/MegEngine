/**
 * \file dnn/src/cuda/dot/opr_impl.h
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

#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
namespace cuda {

class DotForwardImpl final : public DotForward {
public:
    using DotForward::DotForward;
    void exec(
            _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return sizeof(float);
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
