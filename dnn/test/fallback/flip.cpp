#include <gtest/gtest.h>

#include "megdnn.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/flip.h"
#include "test/common/task_record_check.h"
#include "test/common/tensor.h"
#include "test/fallback/fixture.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, FLIP) {
    using namespace flip;
    std::vector<TestArg> args = get_args();
    Checker<Flip> checker(handle());
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto&& arg : args) {
        checker.execs({arg.src, {}});
    }
}
TEST_F(FALLBACK, FLIP_RECORD) {
    using namespace flip;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<Flip> checker(0);
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto&& arg : args) {
        checker.execs({arg.src, {}});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
