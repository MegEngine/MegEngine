#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, GENERALNORM_FORWARD) {
    Checker<GeneralNorm> checker(handle(), true);

    GeneralNorm::Param param;
    param.affine = true;

    param.normalized_axis = 0;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {3.3179, 0.109, -0.5855, 0.2566, -1.2897, 1.2683, -2.0587,
                             0.0711, -0.1169, 0.2509, -0.2393, 0.0876}),  // input
                    TensorValue({2}, dtype::Float32(), {1., 1.}),         // hx
                    TensorValue({2}, dtype::Float32(), {0., 0.}),         // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {0.999999, 0.986361, -0.999909, 0.669477, -0.999982,
                             0.999986, -0.999999, -0.986361, 0.999909, -0.669477,
                             0.999982, -0.999986}),  // output
                    TensorValue(
                            {3, 2, 1}, dtype::Float32(),
                            {0.629600, 0.090050, -0.351200, 0.253750, -0.764500,
                             0.677950}),  // mean
                    TensorValue(
                            {3, 2, 1}, dtype::Float32(),
                            {0.371982, 52.050690, 4.267644, 234.904434, 1.904002,
                             1.693886})  // var
            });
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {3.3179, 0.109, -0.5855, 0.2566, -1.2897, 1.2683, -2.0587,
                             0.0711, -0.1169, 0.2509, -0.2393, 0.0876}),  // input
                    TensorValue({2}, dtype::Float32(), {1., 1.}),         // hx
                    TensorValue({2}, dtype::Float32(), {1., 1.}),         // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {1.999999, 1.986361, 0.000091, 1.669477, 0.000018, 1.999986,
                             0.000001, 0.013639, 1.999909, 0.330523, 1.999982,
                             0.000014}),  // output
                    TensorValue(
                            {3, 2, 1}, dtype::Float32(),
                            {0.629600, 0.090050, -0.351200, 0.253750, -0.764500,
                             0.677950}),  // mean
                    TensorValue(
                            {3, 2, 1}, dtype::Float32(),
                            {0.371982, 52.050690, 4.267644, 234.904434, 1.904002,
                             1.693886})  // var
            });

    param.normalized_axis = 1;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {3.3179, 0.109, -0.5855, 0.2566, -1.2897, 1.2683, -2.0587,
                             0.0711, -0.1169, 0.2509, -0.2393, 0.0876}),  // input
                    TensorValue({3}, dtype::Float32(), {1., 1., 1.}),     // hx
                    TensorValue({3}, dtype::Float32(), {0., 0., 0.}),     // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {1.399909, -0.845471, -0.526212, -0.559011, -0.873697,
                             1.404483, -1.411963, -0.805719, 0.774906, 1.408264,
                             0.637058, -0.602545}),  // output
                    TensorValue(
                            {2, 2, 1}, dtype::Float32(),
                            {0.480900, 0.544633, -0.804967, 0.136533}),  // mean
                    TensorValue(
                            {2, 2, 1}, dtype::Float32(),
                            {0.493447, 1.940786, 1.126207, 12.313588}),  // var
            });

    param.normalized_axis = 2;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {3.3179, 0.109, -0.5855, 0.2566, -1.2897, 1.2683, -2.0587,
                             0.0711, -0.1169, 0.2509, -0.2393, 0.0876}),  // input
                    TensorValue({2}, dtype::Float32(), {1., 1.}),         // hx
                    TensorValue({2}, dtype::Float32(), {0., 0.}),         // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {0.999998, -0.999998, -0.999972, 0.999972, -0.999997,
                             0.999997, -0.999996, 0.999996, -0.999852, 0.999852,
                             -0.999813, 0.999813}),  // output
                    TensorValue(
                            {2, 3, 1}, dtype::Float32(),
                            {1.713450, -0.164450, -0.010700, -0.993800, 0.067000,
                             -0.075850}),  // mean
                    TensorValue(
                            {2, 3, 1}, dtype::Float32(),
                            {0.623265, 2.374948, 0.781858, 0.939051, 5.436934,
                             6.116934}),  // var
            });

    param.normalized_axis = 3;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {3.3179, 0.109, -0.5855, 0.2566, -1.2897, 1.2683, -2.0587,
                             0.0711, -0.1169, 0.2509, -0.2393, 0.0876}),  // input
                    TensorValue({1}, dtype::Float32(), {1.}),             // hx
                    TensorValue({1}, dtype::Float32(), {0.}),             // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                             0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                             0.000000}),  // output
                    TensorValue(
                            {2, 3, 2}, dtype::Float32(),
                            {3.317900, 0.109000, -0.585500, 0.256600, -1.289700,
                             1.268300, -2.058700, 0.071100, -0.116900, 0.250900,
                             -0.239300, 0.087600}),  // mean
                    TensorValue(
                            {2, 3, 2}, dtype::Float32(),
                            {316.227783, 316.227783, 316.227783, 316.227783, 316.227783,
                             316.227783, 316.227783, 316.227783, 316.227783, 316.227783,
                             316.227783, 316.227783}),  // var 1.0 / sqrt(p.eps)
            });
}

}  // namespace test
}  // namespace megdnn
