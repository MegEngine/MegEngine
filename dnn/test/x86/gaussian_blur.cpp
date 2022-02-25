#include "test/common/gaussian_blur.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/x86/fixture.h"
namespace megdnn {
namespace test {

TEST_F(X86, GAUSSIAN_BLUR) {
    using namespace gaussian_blur;
    std::vector<TestArg> args = get_args();
    Checker<GaussianBlur> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, {}});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(1 + 1e-3)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Uint8())
                .execs({arg.src, {}});
    }
}

TEST_F(X86, GAUSSIAN_BLUR_RECORD) {
    using namespace gaussian_blur;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<GaussianBlur> checker(0);
    auto arg = args[0];
    checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({arg.src, {}});

    checker.set_param(arg.param)
            .set_epsilon(1 + 1e-3)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Uint8())
            .execs({arg.src, {}});
}
}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
