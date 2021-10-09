/**
 * \file dnn/test/naive/rnn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/rnn.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

/*TEST_F(NAIVE, RNN) {
    std::vector<rnn::TestArg> args = rnn::get_args();
    Checker<RNN> checker(handle());
    for (auto&& arg : args) {
                checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_dtype(3, dtype::Float32())
                .set_dtype(4, dtype::Float32())
                .set_dtype(5, dtype::Float32())
                .execs({arg.input, arg.hx, arg.flatten_weights, {}, {}, {}});
    }
}*/

TEST_F(NAIVE, RNN_HAND_MADE) {
    Checker<RNN> checker(handle(), false);
    size_t batch_size = 2;
    size_t input_size = 3;
    size_t hidden_size = 2;
    size_t seq_len = 2;
    size_t gate_hidden_size = hidden_size;
    RNN::Param param;
    param.num_layers = 1;
    param.bidirectional = false;
    param.bias = false;
    param.hidden_size = hidden_size;
    param.nonlineMode = param::RNN::NonlineMode::RELU;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {seq_len, batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),  // input
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {2, 1, 3, 5}),  // hx
                    TensorValue(
                            {gate_hidden_size, input_size + hidden_size},
                            dtype::Float32(),
                            {3, 6, 1, 3, 2, 7, 9, 3, 5, 1}),  // weights
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {seq_len, batch_size, hidden_size}, dtype::Float32(),
                            {39, 39, 90, 84, 300, 216, 546, 366}),  // output
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {21, 11, 42, 20}),  // hy
                    TensorValue(
                            {1, 2, 2, 2}, dtype::Float32(),
                            {2, 1, 3, 5, 21, 11, 42, 20})  // reserve space
            });
}

}  // namespace test
}  // namespace megdnn
