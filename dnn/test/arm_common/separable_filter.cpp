#include "test/common/separable_filter.h"
#include "test/arm_common/fixture.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/task_record_check.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON, SEPARABLE_FILTER) {
    using namespace separable_filter;
    std::vector<TestArg> args = get_args();
    Checker<SeparableFilter> checker(handle());
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }

    checker.set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Uint8())
            .set_epsilon(1 + 1e-3);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }
}

TEST_F(ARM_COMMON, SEPARABLE_FILTER_RECORD) {
    using namespace separable_filter;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<SeparableFilter> checker(0);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }

    checker.set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Uint8())
            .set_epsilon(1 + 1e-3);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
