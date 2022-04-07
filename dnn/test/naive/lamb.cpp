#include "test/common/lamb.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, LAMBUpdate) {
    Checker<LAMBUpdate> checker(handle(), false);
    LAMBUpdate::Param param;
    param.beta_1 = 0;
    param.beta_2 = 0;
    param.eps = 0;
    param.weight_decay = 0;
    param.lr = 1;
    param.step = 1;
    param.bias_correction = true;
    param.always_adapt = false;

    TensorND m_t_1 = TensorValue({2}, dtype::Float32(), {1, 1});
    TensorND v_t_1 = TensorValue({2}, dtype::Float32(), {1, 1});
    TensorND param_lamb = TensorValue({2}, dtype::Float32(), {1, 1});
    TensorND grad = TensorValue({2}, dtype::Float16(), {1, 1});

    TensorND m_t = TensorValue({2}, dtype::Float32(), {1, 1});
    TensorND v_t = TensorValue({2}, dtype::Float32(), {1, 1});
    TensorND new_param = TensorValue({2}, dtype::Float32(), {0, 0});
    checker.set_param(param).exect(
            Testcase{m_t_1, v_t_1, param_lamb, grad, {}, {}, {}},
            Testcase{{}, {}, {}, {}, m_t, v_t, new_param});
}
