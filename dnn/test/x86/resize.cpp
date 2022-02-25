#include "test/common/resize.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/x86/fixture.h"
namespace megdnn {
namespace test {

TEST_F(X86, RESIZE_CV) {
    using namespace resize;
    std::vector<TestArg> args = get_cv_args();
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Uint8())
                .set_epsilon(1 + 1e-3)
                .execs({arg.src, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}
TEST_F(X86, RESIZE_CV_RECORD) {
    using namespace resize;
    std::vector<TestArg> args = get_cv_args();
    TaskRecordChecker<Resize> checker(0);
    auto arg = args[0];
    checker.set_param(arg.param)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Uint8())
            .set_epsilon(1 + 1e-3)
            .execs({arg.src, arg.dst});

    checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({arg.src, arg.dst});
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
