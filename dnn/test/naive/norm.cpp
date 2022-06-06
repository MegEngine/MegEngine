#include "test/common/norm.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {
TEST_F(NAIVE, L2NORM_FP32_DIM0) {
    Checker<Norm> checker(handle(), false);
    Norm::Param param;
    param.p = 2;
    param.dim = 0;
    checker.set_param(param);
    checker.exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
            });
}
TEST_F(NAIVE, L2NORM_FP32_DIM1) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.p = 2;
    param.dim = 1;
    checker.set_param(param);
    checker.exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 1, 3, 4}, dtype::Float32(),
                            {12.000, 13.0384, 14.1421, 15.2971, 16.4924, 17.7200,
                             18.9737, 20.2485, 21.5407, 22.8473, 24.1661, 25.4951}),
            });
}
TEST_F(NAIVE, L2NORM_FP32_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.p = 2;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 2, 3, 1}, dtype::Float32(),
                            {3.7417, 11.2250, 19.1311, 27.0924, 35.0714, 43.0581})});
}
// l2, fp16
TEST_F(NAIVE, L2NORM_FP16_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.p = 2;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float16(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 2, 3, 1}, dtype::Float16(),
                            {3.7422, 11.2266, 19.1250, 27.0938, 35.0625, 43.0625})});
}
// l1, fp32,fp16
TEST_F(NAIVE, L1NORM_FP32_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.p = 1;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 2, 3, 1}, dtype::Float32(), {6, 22, 38, 54, 70, 86}),
            });
}
TEST_F(NAIVE, L1NORM_FP16_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.p = 1;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float16(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {1, 2, 3, 1}, dtype::Float16(), {6, 22, 38, 54, 70, 86}),
            });
}
// l0, fp32,fp16
TEST_F(NAIVE, L0NORM_FP32_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.p = 0;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue({1, 2, 3, 1}, dtype::Float32(), {3, 4, 4, 4, 4, 4}),
            });
}
TEST_F(NAIVE, L0NORM_FP16_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.p = 0;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float16(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue({1, 2, 3, 1}, dtype::Float16(), {3, 4, 4, 4, 4, 4}),
            });
}
// inf
TEST_F(NAIVE, INF_NORM_FP32_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    using Mode = Norm::Param::Mode;

    param.dim = 3;
    param.mode = Mode::INF_NORM;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue({1, 2, 3, 1}, dtype::Float32(), {3, 7, 11, 15, 19, 23}),
            });
}
TEST_F(NAIVE, INF_NORM_FP16_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    using Mode = Norm::Param::Mode;

    param.dim = 3;
    param.mode = Mode::INF_NORM;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float16(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue({1, 2, 3, 1}, dtype::Float16(), {3, 7, 11, 15, 19, 23}),
            });
}
// -inf
TEST_F(NAIVE, NEG_INF_NORM_FP32_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.mode = Norm::Param::Mode::NEG_INF_NORM;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float32(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue({1, 2, 3, 1}, dtype::Float32(), {0, 4, 8, 12, 16, 20}),
            });
}
TEST_F(NAIVE, NEG_INF_NORM_FP16_DIM3) {
    Checker<Norm> checker(handle());
    Norm::Param param;
    param.mode = Norm::Param::Mode::NEG_INF_NORM;
    param.dim = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 2, 3, 4}, dtype::Float16(),
                            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}),
                    {}},
            Testcase{
                    {},
                    TensorValue({1, 2, 3, 1}, dtype::Float16(), {0, 4, 8, 12, 16, 20}),
            });
}

}  // namespace test
}  // namespace megdnn