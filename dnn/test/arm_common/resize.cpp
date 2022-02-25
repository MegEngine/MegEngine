#include "test/common/resize.h"
#include "test/arm_common/fixture.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"

namespace megdnn {
namespace test {

using namespace resize;

static void set_nchw_args(IMode imode, std::vector<TestArg>& args) {
    param::Resize param;
    param.format = param::Resize::Format::NCHW;
    param.imode = imode;
    rep(n, 4ul) rep(c, 4ul) rep(ih, 4ul) rep(iw, 4ul) rep(oh, 4ul) rep(ow, 4ul)
            args.emplace_back(
                    param, TensorShape{n + 1ul, c + 1ul, ih + 1ul, iw + 1ul},
                    TensorShape{n + 1ul, c + 1ul, oh + 1ul, ow + 1ul});
    args.emplace_back(param, TensorShape{1, 1, 10, 10}, TensorShape{1, 1, 20, 20});
    args.emplace_back(param, TensorShape{1, 1, 10, 10}, TensorShape{1, 1, 7, 9});
    args.emplace_back(param, TensorShape{2, 2, 3, 4}, TensorShape{2, 2, 6, 8});
    args.emplace_back(param, TensorShape{1, 2, 6, 8}, TensorShape{1, 2, 3, 4});
}

TEST_F(ARM_COMMON, RESIZE_CV) {
    std::vector<TestArg> args = get_cv_args();
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(1 + 1e-3)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Uint8())
                .execs({arg.src, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

TEST_F(ARM_COMMON, RESIZE_CV_RECORD) {
    std::vector<TestArg> args = get_cv_args();
    TaskRecordChecker<Resize> checker(0);

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(1 + 1e-3)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Uint8())
                .execs({arg.src, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON, RESIZE_NCHW_FP16) {
    std::vector<TestArg> args;
    set_nchw_args(IMode::INTER_LINEAR, args);
    set_nchw_args(IMode::INTER_NEAREST, args);
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(0.01)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .execs({arg.src, arg.dst});
    }
}
#endif

TEST_F(ARM_COMMON, RESIZE_NCHW_FP32) {
    std::vector<TestArg> args;
    set_nchw_args(IMode::INTER_LINEAR, args);
    set_nchw_args(IMode::INTER_NEAREST, args);
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

TEST_F(ARM_COMMON, RESIZE_NCHW44_FP32) {
    std::vector<TestArg> args = get_nchw44_args();
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON, RESIZE_NCHW88_FP16) {
    std::vector<TestArg> args = get_nchw88_args();
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_epsilon(0.01)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .execs({arg.src, arg.dst});
    }
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
