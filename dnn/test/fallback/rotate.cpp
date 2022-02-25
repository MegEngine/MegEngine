#include <gtest/gtest.h>

#include "megdnn.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rotate.h"
#include "test/common/task_record_check.h"
#include "test/common/tensor.h"
#include "test/fallback/fixture.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, ROTATE) {
    using namespace rotate;
    std::vector<TestArg> args = get_args();
    Checker<Rotate> checker(handle());
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto&& arg : args) {
        checker.set_dtype(0, arg.dtype).set_dtype(1, arg.dtype).execs({arg.src, {}});
    }
}

TEST_F(FALLBACK, ROTATE_RECORD) {
    using namespace rotate;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<Rotate> checker(1);
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto&& arg : args) {
        checker.set_dtype(0, arg.dtype).set_dtype(1, arg.dtype).execs({arg.src, {}});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
