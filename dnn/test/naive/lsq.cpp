#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, LSQ_FORWARD) {
    Checker<LSQ> checker(handle(), /* check_dispatch */ false);

    param::LSQ param;

    param.qmin = -127;
    param.qmax = 127;

    TensorND input = TensorValue(
            {2, 2, 2, 2}, dtype::Float32(),
            {0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 8});

    TensorND scale_shape = TensorValue({1}, dtype::Float32(), {2});

    TensorND zero_point = TensorValue({1}, dtype::Float32(), {1});

    TensorND grad_scale = TensorValue({1}, dtype::Float32(), {0.5});

    TensorND output = TensorValue(
            {2, 2, 2, 2}, dtype::Float32(),
            {0, 2, 4, 4, 2, 2, 4, 6, 4, 4, 6, 8, 4, 6, 8, 8});

    checker.set_param(param).exect(
            Testcase{input, scale_shape, zero_point, grad_scale, {}},
            Testcase{{}, {}, {}, {}, output});
}
