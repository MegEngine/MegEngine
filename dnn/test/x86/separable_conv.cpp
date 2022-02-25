#include "test/common/separable_conv.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/x86/fixture.h"
namespace megdnn {
namespace test {

TEST_F(X86, SEPARABLE_CONV) {
    using namespace separable_conv;
    std::vector<TestArg> args = get_args();
    Checker<SeparableConvForward> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }
}
TEST_F(X86, SEPARABLE_CONV_RECORD) {
    using namespace separable_conv;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<SeparableConvForward> checker(0);

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
