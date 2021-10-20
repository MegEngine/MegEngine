/**
 * \file dnn/src/cuda/fake_quant/opr_impl.h
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
#include "megdnn/oprs.h"
#include "src/cuda/utils.h"
namespace megdnn {
namespace cuda {
class FakeQuantForwardImpl : public FakeQuantForward {
public:
    using FakeQuantForward::FakeQuantForward;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out output,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }

private:
    void exec_noncontig(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out output);
};

class FakeQuantBackwardImpl : public FakeQuantBackward {
public:
    using FakeQuantBackward::FakeQuantBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    void exec_noncontig(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out grad);
};

}  // namespace cuda
}  // namespace megdnn
