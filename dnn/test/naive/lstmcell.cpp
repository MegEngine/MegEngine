/**
 * \file dnn/test/naive/lstmcell.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {
TEST_F(NAIVE, LSTMCELL) {
    Checker<LSTMCell> checker(handle(), true);
    for (size_t batch : {1, 4})
        for (size_t n : {3, 4, 5, 23, 100})
            for (size_t out : {3, 6, 25, 100}) {
                checker.exec(
                        {{batch, n},
                         {out * 4, n},
                         {1, out * 4},
                         {batch, out},
                         {out * 4, out},
                         {1, out * 4},
                         {batch, out},
                         {},
                         {},
                         {}});
            }
    size_t batch_size = 2;
    size_t input_size = 3;
    size_t hidden_size = 2;
    checker.exect(
            Testcase{
                    TensorValue(
                            {batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6}),  // input
                    TensorValue(
                            {4 * hidden_size, input_size}, dtype::Float32(),
                            {
                                    0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                                    0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                                    0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                                    0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                            }),  // weight_ih
                    TensorValue(
                            {4 * hidden_size}, dtype::Float32(),
                            {0, 0, 0, 0, 0, 0, 0, 0}),  // bias_ih
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {1, 2, 3, 4}),  // hx
                    TensorValue(
                            {4 * hidden_size, hidden_size}, dtype::Float32(),
                            {0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                             0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                             0.3535, 0.3535}),  // weight_hh
                    TensorValue(
                            {4 * hidden_size}, dtype::Float32(),
                            {0, 0, 0, 0, 0, 0, 0, 0}),  // bias_hh
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {2, 3, 4, 5}),  // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {0.9541, 0.9593, 0.9995, 0.9996}),  // hy
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {2.8771, 3.8373, 4.9979, 5.9975}),  // cy
                    TensorValue(
                            {batch_size, 4 * hidden_size}, dtype::Float32(),
                            {3.18198, 3.18198, 7.7781, 7.7781, 3.18198, 3.18198,
                             7.77817, 7.77817, 3.18198, 3.18198, 7.77817, 7.77817,
                             3.18198, 3.18198, 7.77817, 7.77817}),  // cy
            });
    batch_size = 2;
    input_size = 2;
    hidden_size = 1;
    checker.exect(
            Testcase{
                    TensorValue(
                            {batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4}),  // input
                    TensorValue(
                            {4 * hidden_size, input_size}, dtype::Float32(),
                            {0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                             0.3535}),  // weight_ih
                    TensorValue(
                            {4 * hidden_size}, dtype::Float32(),
                            {0.3535, 0.3535, 0.3535, 0.3535}),  // bias_ih
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(), {1, 2}),  // hx
                    TensorValue(
                            {4 * hidden_size, hidden_size}, dtype::Float32(),
                            {0.3535, 0.3535, 0.3535, 0.3535}),  // weight_hh
                    TensorValue(
                            {4 * hidden_size}, dtype::Float32(),
                            {0.3535, 0.3535, 0.3535, 0.3535}),  // bias_hh
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(), {4, 5}),  // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {0.8927, 0.9799}),  // hy
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {4.4393, 5.8788}),  // cy
                    TensorValue(
                            {batch_size, 4 * hidden_size}, dtype::Float32(),
                            {2.1210, 3.8885, 2.1210, 3.8885, 2.1210, 3.8885, 2.1210,
                             3.8885}),  // gates
            });
}
}  // namespace test
}  // namespace megdnn