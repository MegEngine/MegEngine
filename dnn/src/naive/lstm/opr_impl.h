/**
 * \file dnn/src/naive/lstm/opr_impl.h
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

class LSTMImpl : public LSTM {
public:
    using LSTM::LSTM;

    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in hx, _megdnn_tensor_in cx,
            _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
            _megdnn_tensor_out hy, _megdnn_tensor_out cy,
            _megdnn_tensor_out reserve_space, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
            const TensorLayout& flatten_weights, const TensorLayout& output,
            const TensorLayout& hy, const TensorLayout& cy,
            const TensorLayout& reserve_space) override;
    size_t get_reserve_size_in_bytes(const TensorLayout& input) override;

    bool is_thread_safe() const override { return true; }
};

class LSTMBackwardImpl : public LSTMBackward {
public:
    using LSTMBackward::LSTMBackward;

    virtual void exec(
            _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in hx,
            _megdnn_tensor_in cx, _megdnn_tensor_in dy, _megdnn_tensor_in dhy,
            _megdnn_tensor_in dcy, _megdnn_tensor_in flatten_weights,
            _megdnn_tensor_in reserve_space, _megdnn_tensor_out dx,
            _megdnn_tensor_out dhx, _megdnn_tensor_out dcx, _megdnn_tensor_out dw,
            _megdnn_workspace workspace) override;

    bool is_thread_safe() const override { return true; }

    virtual size_t get_workspace_in_bytes(
            const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
            const TensorLayout& cx, const TensorLayout& dy, const TensorLayout& dhy,
            const TensorLayout& dcy, const TensorLayout& flatten_weights,
            const TensorLayout& reserve_space, const TensorLayout& dx,
            const TensorLayout& dhx, const TensorLayout& dcx,
            const TensorLayout& dw) override;
};

}  // namespace naive
}  // namespace megdnn