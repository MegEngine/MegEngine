#include "test/common/resize.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {
namespace resize {

TEST_F(CAMBRICON, RESIZE_CV) {
    using namespace resize;
    std::vector<TestArg> args = get_cv_args();
    Checker<Resize> checker(handle_cambricon());

    for (auto&& arg : args) {
        if (arg.param.imode == param::Resize::InterpolationMode::INTER_NEAREST ||
            arg.param.imode == param::Resize::InterpolationMode::INTER_LINEAR ||
            arg.param.imode == param::Resize::InterpolationMode::INTER_CUBIC) {
            checker.set_param(arg.param)
                    .set_dtype(0, dtype::Float32())
                    .set_dtype(1, dtype::Float32())
                    .set_epsilon(1e-3)
                    .execs({arg.src, arg.dst});
        }
    }
}

TEST_F(CAMBRICON, RESIZE_NHWC) {
    using namespace resize;
    std::vector<TestArg> args;

    param::Resize param;
    param.format = param::Resize::Format::NHWC;
    param.imode = param::Resize::InterpolationMode::LINEAR;
    args.emplace_back(param, TensorShape{1, 1, 4, 5}, TensorShape{1, 1, 8, 5});
    args.emplace_back(param, TensorShape{2, 6, 4, 5}, TensorShape{2, 3, 8, 5});
    args.emplace_back(param, TensorShape{1, 2, 2, 2}, TensorShape{1, 4, 3, 2});

    Checker<ResizeBackward> checkerBackward(handle_cambricon());

    for (auto&& arg : args) {
        checkerBackward.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_epsilon(1e-3)
                .execs({arg.src, arg.dst});
    }

    for (auto&& arg : args) {
        checkerBackward.set_param(arg.param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_epsilon(1e-3)
                .execs({arg.src, arg.dst});
    }

    Checker<ResizeForward> checkerForward(handle_cambricon());
    for (auto&& arg : args) {
        checkerForward.set_param(arg.param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_epsilon(1e-3)
                .execs({arg.src, arg.dst});
    }
    for (auto&& arg : args) {
        checkerForward.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_epsilon(1e-3)
                .execs({arg.src, arg.dst});
    }
}

}  // namespace resize
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
