/**
 * \file dnn/src/common/lstm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void LSTM::deduce_layout(
        const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
        const TensorLayout& /*flatten_weights*/, TensorLayout& output, TensorLayout& hy,
        TensorLayout& cy, TensorLayout& reserve_space) {
    // input: [seq_len, batch_size, input_size]
    // hx: [D * num_layers, batch_size, hidden_size]
    size_t seq_len = input.shape[0];
    size_t batch_size = input.shape[1];
    size_t D = param().bidirectional ? 2 : 1;
    size_t hidden_size = hx.shape[2];
    output = TensorLayout(
            TensorShape{seq_len, batch_size, D * hidden_size}, input.dtype);
    hy = TensorLayout(hx);
    cy = TensorLayout(cx);
    reserve_space = {{get_reserve_size_in_bytes(input)}, input.dtype};
}

void LSTM::check_exec(
        const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
        const TensorLayout& flatten_weights, const TensorLayout& output,
        const TensorLayout& hy, const TensorLayout& cy,
        const TensorLayout& /*reserve_space*/, size_t /*workspace_in_bytes*/) {
    auto errmsg = [&]() {
        std::string msg;
        msg.append("input=");
        msg.append(input.to_string());
        msg.append(", output=");
        msg.append(output.to_string());
        msg.append(", hx=");
        msg.append(hx.to_string());
        msg.append(", cx=");
        msg.append(cx.to_string());
        msg.append(", hy=");
        msg.append(hy.to_string());
        msg.append(", cy=");
        msg.append(cy.to_string());
        msg.append(", flatten_weights=");
        msg.append(flatten_weights.to_string());
        msg.append(", hidden_size=");
        msg.append(std::to_string(param().hidden_size));
        msg.append(", num_layers=");
        msg.append(std::to_string(param().num_layers));
        msg.append(", bidirectional=");
        msg.append(std::to_string(param().bidirectional));
        return msg;
    };
    size_t D = param().bidirectional ? 2 : 1;
    size_t b = param().bias ? 1 : 0;
    size_t num_layers = param().num_layers;
    size_t input_size = input.shape[2];
    size_t gate_hidden_size = 4 * param().hidden_size;
    // first layer{ weight_ih_l[k][_reverse].shape = (4*hidden_size, input_size)
    //              weight_hh_l[k][_reverse].shape = (4*hidden_size, hidden_size)}
    // other layers{ weight_ih_l[k][_reverse].shape = (4*hidden_size, num_directions *
    // hidden_size)
    //               weight_hh_l[k][_reverse].shape = (4*hidden_size, hidden_size)}
    // bias:  2 * num_directions * num_layers
    // size_dim1 = D * first layer + (layer -1) * other layer + bias
    size_t size_dim1 = D * (input_size + param().hidden_size) +
                       (num_layers - 1) * D * ((D + 1) * param().hidden_size) +
                       b * 2 * D * num_layers;
#define ASSERT_BRIEF(_content) megdnn_assert(_content, "%s", errmsg().c_str());

    ASSERT_BRIEF(input.ndim == 3)
    ASSERT_BRIEF(output.ndim == 3)
    ASSERT_BRIEF(flatten_weights.shape[0] == gate_hidden_size)
    ASSERT_BRIEF(flatten_weights.shape[0] == size_dim1)
    ASSERT_BRIEF(output.shape[0] == input.shape[0])
    ASSERT_BRIEF(output.shape[1] == input.shape[1])
    ASSERT_BRIEF(output.shape[2] == D * param().hidden_size)
    ASSERT_BRIEF(hx.ndim == 3)
    ASSERT_BRIEF(hx.shape[0] == D * num_layers)
    ASSERT_BRIEF(hx.shape[1] == input.shape[1])  // batch_size
    ASSERT_BRIEF(hx.shape[2] == param().hidden_size)
    ASSERT_BRIEF(cx.ndim == 3)
    ASSERT_BRIEF(cx.shape[0] == D * num_layers)
    ASSERT_BRIEF(cx.shape[1] == input.shape[1])  // batch_size
    ASSERT_BRIEF(cx.shape[2] == param().hidden_size)
    ASSERT_BRIEF(hy.ndim == 3)
    ASSERT_BRIEF(hy.shape[0] == D * num_layers)
    ASSERT_BRIEF(hy.shape[1] == input.shape[1])  // batch_size
    ASSERT_BRIEF(hy.shape[2] == param().hidden_size)
    ASSERT_BRIEF(cy.ndim == 3)
    ASSERT_BRIEF(cy.shape[0] == D * num_layers)
    ASSERT_BRIEF(cy.shape[1] == input.shape[1])  // batch_size
    ASSERT_BRIEF(cy.shape[2] == param().hidden_size)
#undef ASSERT_BRIEF
}

void LSTMBackward::deduce_layout(
        const TensorLayout& x, const TensorLayout& /*y*/, const TensorLayout& hx,
        const TensorLayout& cx, const TensorLayout& /*dy*/, const TensorLayout& /*dhy*/,
        const TensorLayout& /*dcy*/, const TensorLayout& flatten_weights,
        const TensorLayout& /*reserve_space*/, TensorLayout& dx, TensorLayout& dhx,
        TensorLayout& dcx, TensorLayout& dw) {
    dx = x;
    dhx = hx;
    dcx = cx;
    dw = flatten_weights;
}

// TODO: add shape check of BWD
void LSTMBackward::check_exec(
        const TensorLayout& /*x*/, const TensorLayout& /*y*/,
        const TensorLayout& /*hx*/, const TensorLayout& /*cx*/,
        const TensorLayout& /*dy*/, const TensorLayout& /*dhy*/,
        const TensorLayout& /*dcy*/, const TensorLayout& /*flatten_weights*/,
        const TensorLayout& /*reserve_space*/, const TensorLayout& /*dx*/,
        const TensorLayout& /*dhx*/, const TensorLayout& /*dcx*/,
        const TensorLayout& /*dw*/, size_t /*workspace_in_bytes*/) {}

}  // namespace megdnn