#include "test/naive/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, CHECK_NON_FINITE_BASIC) {
    Checker<CheckNonFinite> checker(handle(), false);
    checker.exect(
            Testcase{
                    TensorValue({4}, dtype::Float32(), {1.1, 2.2, 3.3, 4.3}),
                    TensorValue({4}, dtype::Float32(), {1.1, 2.2, 3.3, 4.3}),
                    {}},
            Testcase{{}, {}, TensorValue({1}, dtype::Int32(), {0})});
    checker.exect(
            Testcase{
                    TensorValue({4}, dtype::Float32(), {1.1, 2.2, 3.3, 4.3}),
                    TensorValue(
                            {4}, dtype::Float32(),
                            {1.1f, 2.2f, 3.3f, std::numeric_limits<float>::infinity()}),
                    {}},
            Testcase{{}, {}, TensorValue({1}, dtype::Int32(), {1})});
    checker.exect(
            Testcase{
                    TensorValue({4}, dtype::Float32(), {1.1, 2.2, 3.3, 4.3}),
                    TensorValue(
                            {4}, dtype::Float32(),
                            {1.1f, 2.2f, 3.3f,
                             std::numeric_limits<float>::quiet_NaN()}),
                    {}},
            Testcase{{}, {}, TensorValue({1}, dtype::Int32(), {1})});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
