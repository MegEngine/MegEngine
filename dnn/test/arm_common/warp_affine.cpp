#include "test/common/warp_affine.h"
#include "test/arm_common/fixture.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON_MULTI_THREADS, WARP_AFFINE_CV) {
    using namespace warp_affine;
    std::vector<TestArg> args = get_cv_args();
    Checker<WarpAffine> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(1 + 1e-3)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Uint8())
                .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .execs({arg.src, arg.trans, arg.dst});
    }
}

TEST_F(ARM_COMMON_MULTI_THREADS, WARP_AFFINE_CV_RECORD) {
    using namespace warp_affine;
    std::vector<TestArg> args = get_cv_args();
    TaskRecordChecker<WarpAffine> checker(0);

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(1 + 1e-3)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Uint8())
                .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .execs({arg.src, arg.trans, arg.dst});
    }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
