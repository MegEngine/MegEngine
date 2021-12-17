/**
 * \file dnn/src/naive/softmax/opr_impl.h
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
namespace naive {

class SoftmaxForwardImpl final : public SoftmaxForward {
public:
    using SoftmaxForward::SoftmaxForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout&) override {
        return src.span().dist_byte() * 2;
    }
};

class SoftmaxBackwardImpl final : public SoftmaxBackward {
public:
    using SoftmaxBackward::SoftmaxBackward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad_x,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout&,
            const TensorLayout&) override {
        return src.span().dist_byte() * 3;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen