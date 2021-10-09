/**
 * \file dnn/test/common/rnn.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <vector>

#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace rnn {
struct TestArg {
    param::RNN param;
    TensorShape input, hx, flatten_weights;
    TestArg(param::RNN param, TensorShape input, TensorShape hx,
            TensorShape flatten_weights)
            : param(param), input(input), hx(hx), flatten_weights(flatten_weights) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    size_t batch_size = 2;
    size_t input_size = 3;
    size_t hidden_size = 2;
    size_t seq_len = 2;
    size_t gate_hidden_size = hidden_size;
    param::RNN param;
    param.num_layers = 1;
    param.bidirectional = false;
    param.bias = false;
    param.hidden_size = hidden_size;
    param.nonlineMode = param::RNN::NonlineMode::RELU;

    args.emplace_back(
            param, TensorShape{seq_len, batch_size, input_size},
            TensorShape{batch_size, hidden_size},
            TensorShape{gate_hidden_size, input_size + hidden_size});
    return args;
}

}  // namespace rnn
}  // namespace test
}  // namespace megdnn