#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, WHERE) {
    Checker<Where> checker(handle());

    checker.exect(
            Testcase{
                    TensorValue({1, 2, 2}, dtype::Bool(), {true, false, false, true}),
                    TensorValue({1, 2, 2}, dtype::Float32(), {1, 2, 3, 4}),
                    TensorValue({1, 2, 2}, dtype::Float32(), {5, 6, 7, 8}),
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    TensorValue({1, 2, 2}, dtype::Float32(), {1, 6, 7, 4})});
}

TEST_F(NAIVE, WHEREBACKWARD) {
    Checker<WhereBackward> checker(handle());

    checker.exect(
            Testcase{
                    TensorValue({1, 2, 2}, dtype::Float32(), {5, 6, 7, 8}),
                    TensorValue({1, 2, 2}, dtype::Bool(), {true, false, false, true}),
                    {},
                    {}},
            Testcase{
                    {},
                    {},
                    TensorValue({1, 2, 2}, dtype::Float32(), {5, 0, 0, 8}),
                    TensorValue({1, 2, 2}, dtype::Float32(), {0, 6, 7, 0})});
}

}  // namespace test
}  // namespace megdnn
