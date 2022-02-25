#include "test/common/separable_filter.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/x86/fixture.h"
namespace megdnn {
namespace test {

TEST_F(X86, SEPARABLE_FILTER) {
    using namespace separable_filter;
    std::vector<TestArg> args = get_args();
    Checker<SeparableFilter> checker(handle());

    ConstValue rng(2);
    checker.set_rng(0, &rng).set_rng(1, &rng).set_rng(2, &rng);
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

TEST_F(X86, SEPARABLE_FILTER_RECORD) {
    using namespace separable_filter;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<SeparableFilter> checker(0);
    auto arg = args[0];
    ConstValue rng(2);
    checker.set_rng(0, &rng).set_rng(1, &rng).set_rng(2, &rng);
    checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});

    checker.set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Uint8())
            .set_epsilon(1 + 1e-3);
    checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
