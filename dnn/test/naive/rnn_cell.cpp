#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, RNNCELL) {
    Checker<RNNCell> checker(handle(), false);
    for (size_t batch : {1, 4})
        for (size_t inp : {3, 4, 5, 23, 100})
            for (size_t hidden : {3, 6, 25, 100}) {
                checker.exec(
                        {{batch, inp},
                         {hidden, inp},
                         {1, hidden},
                         {batch, hidden},
                         {hidden, hidden},
                         {1, hidden},
                         {}});
            }
    size_t batch_size = 2;
    size_t input_size = 3;
    size_t hidden_size = 2;
    RNNCell::Param param;
    param.nonlineMode = param::RNNCell::NonlineMode::TANH;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6}),  // input
                    TensorValue(
                            {hidden_size, input_size}, dtype::Float32(),
                            {0.3535, 0.3535, 0.3535, 0.3535, 0.3535,
                             0.3535}),  // weight_ih
                    TensorValue({1, hidden_size}, dtype::Float32(), {0, 0}),  // bias_ih
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {1, 2, 3, 4}),  // hx
                    TensorValue(
                            {hidden_size, hidden_size}, dtype::Float32(),
                            {0.3535, 0.3535, 0.3535, 0.3535}),  // weight_hh
                    TensorValue({1, hidden_size}, dtype::Float32(), {0, 0}),  // bias_hh
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {0.9966, 0.9966, 1.0, 1.0}),  // dst
            });
    batch_size = 2;
    input_size = 2;
    hidden_size = 1;
    param.nonlineMode = param::RNNCell::NonlineMode::RELU;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4}),  // input
                    TensorValue(
                            {hidden_size, input_size}, dtype::Float32(),
                            {0.3535, 0.3535}),  // weight_ih
                    TensorValue(
                            {1, hidden_size}, dtype::Float32(), {0.3535}),  // bias_ih
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {-1, -2}),  // hx
                    TensorValue(
                            {hidden_size, hidden_size}, dtype::Float32(),
                            {0.3535}),  // weight_hh
                    TensorValue(
                            {1, hidden_size}, dtype::Float32(), {0.3535}),  // bias_hh
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {1.414, 2.4745}),  // hy
            });
}

}  // namespace test
}  // namespace megdnn