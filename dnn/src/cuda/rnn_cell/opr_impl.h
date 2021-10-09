/**
 * \file dnn/src/cuda/rnn_cell/opr_impl.h
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
namespace cuda {

class RNNCellImpl : public RNNCell {
public:
    using RNNCell::RNNCell;
    void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
            _megdnn_tensor_in bias_ih, _megdnn_tensor_in hx,
            _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& dst) override;
    /*
    private:
        void add_bias(_megdnn_tensor_in A,
                _megdnn_tensor_in B,
                        _megdnn_tensor_in bias,
                _megdnn_tensor_out C);
    */
};

}  // namespace cuda
}  // namespace megdnn