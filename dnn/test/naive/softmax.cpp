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
