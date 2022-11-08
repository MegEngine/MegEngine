#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, MASKEDFILL) {
    Checker<MaskedFill> checker(handle(), true);

    MaskedFill::Param param;
    param.value = 0.2;

    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {3.3179, 0.109, -0.5855, 0.2566, -1.2897, 1.2683, -2.0587,
                             0.0711, -0.1169, 0.2509, -0.2393, 0.0876}),  // input
                    TensorValue({2}, dtype::Bool(), {false, true}),       // hx
                    {}},
            Testcase{
                    {},
                    {},
                    TensorValue(
                            {2, 3, 2, 1}, dtype::Float32(),
                            {3.3179, 0.109, -0.5855, 0.2566, -1.2897, 1.2683, 0.2, 0.2,
                             0.2, 0.2, 0.2, 0.2}),  // output
            });
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {1, 3, 1, 2}, dtype::Float32(),
                            {-2.4348, -1.7948, 0.5223, 0.0932, -0.2955,
                             -0.0492}),                                        // input
                    TensorValue({1, 3}, dtype::Bool(), {false, true, false}),  // hx
                    {},
            },
            Testcase{
                    {},
                    {},
                    TensorValue(
                            {1, 3, 1, 2}, dtype::Float32(),
                            {-2.4348, -1.7948, 0.2, 0.2, -0.2955, -0.0492}),
            });
}

}  // namespace test
}  // namespace megdnn
