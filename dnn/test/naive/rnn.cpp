#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, RNN_FORWARD) {
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
                            {-0.66536, 0.08049, 0.12008, 0.63423, 1.37801, 0.02591,
                             0.09153, 0.82866, -1.70429, -1.26624, -0.06421,
                             0.35816}),  // input
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {-3.19544, -1.24232, 1.99512, -0.25692}),  // hx
                    TensorValue(
                            {gate_hidden_size, input_size + hidden_size},
                            dtype::Float32(),
                            {0.35355, 0.35355, 0.35355, 0.35355, 0.35355, 0.35355,
                             0.35355, 0.35355, 0.35355, 0.35355}),  // flattern weights
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {seq_len, batch_size, hidden_size}, dtype::Float32(),
                            {0.0, 0.0, 1.3351, 1.3351, 0.0, 0.0, 0.6003,
                             0.6003}),  // output
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {0.0, 0.0, 0.6003, 0.6003}),  // hy
                    TensorValue(
                            {1, 2, 2, 2}, dtype::Float32(),
                            {0.0, 0.0, 1.33512, 1.33512, 0.0, 0.0, 0.60031,
                             0.60031})  // reserve space
            });
    param.num_layers = 2;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {seq_len, batch_size, input_size}, dtype::Float32(),
                            {-0.66536, 0.08049, 0.12008, 0.63423, 1.37801, 0.02591,
                             0.09153, 0.82866, -1.70429, -1.26624, -0.06421,
                             0.35816}),  // input
                    TensorValue(
                            {2, batch_size, hidden_size}, dtype::Float32(),
                            {-3.19544, -1.24232, 1.99512, -0.25692, -3.19544, -1.24232,
                             1.99512, -0.25692}),  // hx
                    TensorValue(
                            {2, 9}, dtype::Float32(),
                            {0.35355, 0.35355, 0.35355, 0.35355, 0.35355, 0.35355,
                             0.35355, 0.35355, 0.35355, 0.35355, 0.35355, 0.35355,
                             0.35355, 0.35355, 0.35355, 0.35355, 0.35355,
                             0.35355}),  // weights
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {seq_len, batch_size, hidden_size}, dtype::Float32(),
                            {0.0, 0.0, 1.5586, 1.5586, 0.0, 0.0, 1.5266,
                             1.5266}),  // output
                    TensorValue(
                            {2, batch_size, hidden_size}, dtype::Float32(),
                            {0.0, 0.0, 0.6003, 0.6003, 0.0, 0.0, 1.5266,
                             1.5266}),  // hy
                    TensorValue(
                            {2, 2, 2, 2}, dtype::Float32(),
                            {0.0, 0.0, 1.33512, 1.33512, 0.0, 0.0, 0.60031, 0.60031,
                             0.0, 0.0, 1.55861, 1.55861, 0.0, 0.0, 1.52658,
                             1.52658})  // reserve space
            });
}

}  // namespace test
}  // namespace megdnn
