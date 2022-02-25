// #include "test/common/lstm.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, LSTM_FORWARD) {
    Checker<LSTM> checker(handle(), true);
    size_t batch_size = 2;
    size_t input_size = 3;
    size_t hidden_size = 2;
    size_t seq_len = 2;
    size_t gate_hidden_size = 4 * hidden_size;
    LSTM::Param param;
    param.num_layers = 1;
    param.bidirectional = false;
    param.bias = false;
    param.hidden_size = hidden_size;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {seq_len, batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),  // input
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {1, 2, 3, 4}),  // hx
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {2, 3, 4, 5}),  // cx
                    TensorValue(
                            {gate_hidden_size, input_size + hidden_size},
                            dtype::Float32(),
                            {3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 3, 6,
                             1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 9, 3, 5, 1,
                             9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1}),  // flattern weights
                    {},
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {seq_len, batch_size, hidden_size}, dtype::Float32(),
                            {0.9951, 0.9993, 0.9999, 1.0000, 0.9993, 0.9999, 1.0000,
                             1.0000}),  // output
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {0.9993, 0.9999, 1.0000, 1.0000}),  // hy
                    TensorValue(
                            {batch_size, hidden_size}, dtype::Float32(),
                            {4.0000, 5.0000, 6.0000, 7.0000}),  // cy
                    TensorValue(
                            {2, 2, 2, 2}, dtype::Float32(),
                            {0.995054, 0.999328, 0.99990, 0.999987, 3., 4., 5., 6.,
                             0.999329, 0.999328, 0.99990, 1., 4., 5., 6.,
                             7.})  // reserve space
            });
    param.bidirectional = true;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {seq_len, batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),  // input
                    TensorValue(
                            {2, batch_size, hidden_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6, 7, 8}),  // hx
                    TensorValue(
                            {2, batch_size, hidden_size}, dtype::Float32(),
                            {2, 3, 4, 5, 6, 7, 8, 9}),  // cx
                    TensorValue(
                            {gate_hidden_size, 2 * (input_size + hidden_size)},
                            dtype::Float32(),
                            {3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 3, 6, 1, 3, 2,
                             7, 2, 1, 3, 2, 1, 1, 9, 3, 5, 1, 9, 3, 5, 1, 9, 3,
                             5, 1, 9, 3, 5, 1, 3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1,
                             1, 3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 9, 3, 5, 1,
                             9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1}),  // flattern weights
                    {},
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {seq_len, batch_size, 2 * hidden_size}, dtype::Float32(),
                            {0.9951, 0.9993, 1.0000, 1.0000, 0.9999, 1.0000, 1.0000,
                             1.0000, 0.9993, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000,
                             1.0000, 1.0000}),  // output
                    TensorValue(
                            {2, batch_size, hidden_size}, dtype::Float32(),
                            {0.9993, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                             1.0000}),  // hy
                    TensorValue(
                            {2, batch_size, hidden_size}, dtype::Float32(),
                            {4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000, 10.0000,
                             11.0000}),  // cy
                    TensorValue(
                            {4, 2, 2, 2}, dtype::Float32(),
                            {0.995054, 0.999328, 0.99990,  0.999987, 3.,      4.,
                             5.,       6.,       0.999329, 0.999328, 0.99990, 1.,
                             4.,       5.,       6.,       7.,       1.,      0.999328,
                             0.99990,  0.999987, 7.,       8.,       9.,      10.,
                             0.999329, 0.999328, 0.99990,  1.,       8.,      9.,
                             10.,      11.})  // reserve space
            });
    param.num_layers = 2;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {seq_len, batch_size, input_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),  // input
                    TensorValue(
                            {4, batch_size, hidden_size}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}),  // hx
                    TensorValue(
                            {4, batch_size, hidden_size}, dtype::Float32(),
                            {2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9}),  // cx
                    TensorValue(
                            {8, 22}, dtype::Float32(),
                            {
                                    3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 3, 6, 1, 3,
                                    2, 7, 2, 1, 3, 2, 1, 1, 9, 3, 5, 1, 9, 3, 5, 1,
                                    9, 3, 5, 1, 9, 3, 5, 1, 3, 6, 1, 3, 2, 7, 2, 1,
                                    3, 2, 1, 1, 3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1,
                                    9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1,
                                    3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 2, 7, 2, 1,
                                    3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 2, 7, 2, 1,
                                    3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 2, 7, 2, 1,
                                    3, 6, 1, 3, 2, 7, 2, 1, 3, 2, 1, 1, 2, 7, 2, 1,
                                    9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1,
                                    9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1, 9, 3, 5, 1,
                            }),  // flattern weights
                    {},
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {seq_len, batch_size, 2 * hidden_size}, dtype::Float32(),
                            {0.9951, 0.9993, 1.0000, 1.0000, 0.9999, 1.0000, 1.0000,
                             1.0000, 0.9993, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000,
                             1.0000, 1.0000}),  // output
                    TensorValue(
                            {4, batch_size, hidden_size}, dtype::Float32(),
                            {0.9993, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                             1.0000, 0.9993, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000,
                             1.0000, 1.0000}),  // hy
                    TensorValue(
                            {4, batch_size, hidden_size}, dtype::Float32(),
                            {4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000, 10.0000,
                             11.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000,
                             10.0000, 11.0000}),  // cy
                    TensorValue(
                            {8, 2, 2, 2}, dtype::Float32(),
                            {
                                    0.995054, 0.999328, 0.99990,  0.999987, 3.,
                                    4.,       5.,       6.,       0.999329, 0.999328,
                                    0.99990,  1.,       4.,       5.,       6.,
                                    7.,       1.,       0.999328, 0.99990,  0.999987,
                                    7.,       8.,       9.,       10.,      0.999329,
                                    0.999328, 0.99990,  1.,       8.,       9.,
                                    10.,      11.,      0.995054, 0.999328, 0.99990,
                                    0.999987, 3.,       4.,       5.,       6.,
                                    0.999329, 0.999328, 0.99990,  1.,       4.,
                                    5.,       6.,       7.,       1.,       0.999328,
                                    0.99990,  0.999987, 7.,       8.,       9.,
                                    10.,      0.999329, 0.999328, 0.99990,  1.,
                                    8.,       9.,       10.,      11.,
                            })  // reserve space
            });
}

}  // namespace test
}  // namespace megdnn