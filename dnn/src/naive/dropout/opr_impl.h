/**
 * \file dnn/src/naive/dropout/opr_impl.h
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
#include "src/naive/rng/opr_impl.h"

namespace megdnn {
namespace naive {

class DropoutForwardImpl final : public DropoutForward {
    Xoroshiro128plus m_rng;

public:
    using DropoutForward::DropoutForward;
    void exec(
            _megdnn_tensor_in inp, _megdnn_tensor_out oup, _megdnn_tensor_out mask,
            _megdnn_workspace workspace) override;
    size_t get_mask_size_in_bytes(const TensorLayout& inp) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class DropoutBackwardImpl final : public DropoutBackward {
public:
    using DropoutBackward::DropoutBackward;
    void exec(
            _megdnn_tensor_in doup, _megdnn_tensor_in mask, _megdnn_tensor_out dinp,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
