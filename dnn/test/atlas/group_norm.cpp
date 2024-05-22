#include "test/common/fill.h"

#include "test/atlas/fixture.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, GROUPNORM_FORWARD) {
    Checker<GroupNorm> checker(handle_atlas(), true);

    GroupNorm::Param param;
    param.affine = true;
    param.group = 3;
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
                            {1., -1., -1., 1., -1., 1., -1., 1., -0.9999, 0.9999,
                             -0.9998, 0.9998}),  // output
                    TensorValue(
                            {2, 3}, dtype::Float32(),
                            {1.7135, -0.1645, -0.0107, -0.9938, 0.067,
                             -0.0758}),  // mean
                    TensorValue(
                            {2, 3}, dtype::Float32(),
                            {2.5742, 0.1772, 1.6358, 1.1340, 0.0338, 0.0267}),  // var
            });
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 3, 1, 2}, dtype::Float32(),
                            {-2.4348, -1.7948, 0.5223, 0.0932, -0.2955,
                             -0.0492}),                                // input
                    TensorValue({3}, dtype::Float32(), {1., 1., 1.}),  // hx
                    TensorValue({3}, dtype::Float32(), {0., 0., 0.}),  // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {1, 3, 1, 2}, dtype::Float32(),
                            {-0.9999, 0.9999, 0.9999, -0.9999, -0.9997,
                             0.9997}),  // output
                    TensorValue(
                            {1, 3}, dtype::Float32(),
                            {-2.1148, 0.3077, -0.1724}),  // mean
                    TensorValue(
                            {1, 3}, dtype::Float32(), {0.1023, 0.0460, 0.0151}),  // var
            });

    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 3, 1, 2}, dtype::Float32(),
                            {-2.4348, -1.7948, 0.5223, 0.0932, -0.2955,
                             -0.0492}),                                         // input
                    TensorValue({1, 3, 1, 1}, dtype::Float32(), {1., 1., 1.}),  // hx
                    TensorValue({1, 3, 1, 1}, dtype::Float32(), {0., 0., 0.}),  // cx
                    {},
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue(
                            {1, 3, 1, 2}, dtype::Float32(),
                            {-0.9999, 0.9999, 0.9999, -0.9999, -0.9997,
                             0.9997}),  // output
                    TensorValue(
                            {1, 3}, dtype::Float32(),
                            {-2.1148, 0.3077, -0.1724}),  // mean
                    TensorValue(
                            {1, 3}, dtype::Float32(), {0.1023, 0.0460, 0.0151}),  // var
            });
}
}  // namespace test
}  // namespace megdnn