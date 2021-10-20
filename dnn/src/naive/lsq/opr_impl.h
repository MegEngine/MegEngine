/**
 * \file dnn/src/naive/lsq/opr_impl.h
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

namespace megdnn {
namespace naive {

class LSQForwardImpl final : public LSQForward {
public:
    using LSQForward::LSQForward;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out output, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& /* input */, const TensorLayout& /* scale */,
            const TensorLayout& /* zero_point */, const TensorLayout& /* grad_scale */,
            const TensorLayout& /* output */) override {
        return 0;
    }
};

class LSQBackwardImpl final : public LSQBackward {
public:
    using LSQBackward::LSQBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& /* diff */, const TensorLayout& /* input */,
            const TensorLayout& /* scale */, const TensorLayout& /* zero_point */,
            const TensorLayout& /* grad_scale */, const TensorLayout& /* grad_x */,
            const TensorLayout& /* grad_s */) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn
