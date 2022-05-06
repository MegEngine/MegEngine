#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, SOFTMAX_FORWARD) {
    Checker<Softmax> checker(handle(), /* check_dispatch */ false);

    Softmax::Param param{0};

    TensorND input = TensorValue(
            {2, 2, 2, 2}, dtype::Float32(),
            {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});

    TensorND output = TensorValue(
            {2, 2, 2, 2}, dtype::Float32(),
            {0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.9997,
             0.9997, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997});

    checker.set_param(param).exect(Testcase{input, {}}, Testcase{{}, output});
}

TEST_F(NAIVE, SOFTMAX_BACKWARD) {
    Checker<SoftmaxBackward> checker(handle(), /* check_dispatch */ false);

    Softmax::Param param{0};

    TensorND input = TensorValue(
            {2, 2, 2, 2}, dtype::Float32(),
            {0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.9997,
             0.9997, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997, 0.9997});

    TensorND diff = TensorValue(
            {2, 2, 2, 2}, dtype::Float32(),
            {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});

    TensorND output = TensorValue(
            {2, 2, 2, 2}, dtype::Float32(),
            {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});

    checker.set_param(param).exect(Testcase{input, diff, {}}, Testcase{{}, {}, output});
}

TEST_F(NAIVE, SOFTMAX_FORWARD_NHWCD4) {
    Checker<Softmax> checker(handle(), false);
    Softmax::Param param{0};

    TensorND input1 = TensorValue(
            {1, 2, 1, 2, 4}, dtype::Float32(),
            {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15});
    TensorND output1 = TensorValue(
            {1, 2, 1, 2, 4}, dtype::Float32(),
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    checker.set_param(param).exect(Testcase{input1, {}}, Testcase{{}, output1});

    TensorND input2 = TensorValue(
            {2, 2, 1, 2, 4}, dtype::Float32(),
            {0,  4,  8,  12, 1,  5,  9,  13, 2,  6,  10, 14, 3,  7,  11, 15,
             16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31});
    TensorND output2 = TensorValue(
            {2, 2, 1, 2, 4}, dtype::Float32(),
            {1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01});
    checker.set_param(param).exect(Testcase{input2, {}}, Testcase{{}, output2});
}

TEST_F(NAIVE, SOFTMAX_BACKWARD_NHWCD4) {
    Checker<SoftmaxBackward> checker(handle(), false);
    Softmax::Param param{0};

    TensorND input = TensorValue(
            {2, 2, 1, 2, 4}, dtype::Float32(),
            {1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             1.12535162e-07, 1.12535162e-07, 1.12535162e-07, 1.12535162e-07,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01,
             9.99999887e-01, 9.99999887e-01, 9.99999887e-01, 9.99999887e-01});

    TensorND diff = TensorValue(
            {2, 2, 1, 2, 4}, dtype::Float32(),
            {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.});

    TensorND output = TensorValue(
            {2, 2, 1, 2, 4}, dtype::Float32(),
            {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});

    checker.set_param(param).exect(Testcase{input, diff, {}}, Testcase{{}, {}, output});
}
