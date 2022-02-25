#include "test/common/resize.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/fallback/fixture.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, RESIZE_CV) {
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
TEST_F(FALLBACK, RESIZE_CV_RECORD) {
    using namespace resize;
    std::vector<TestArg> args = get_cv_args();
    TaskRecordChecker<Resize> checker(1);

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

TEST_F(FALLBACK, RESIZE) {
    using namespace resize;
    std::vector<TestArg> args = get_args();
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
TEST_F(FALLBACK, RESIZE_RECORD) {
    using namespace resize;
    std::vector<TestArg> args = get_args();
    TaskRecordChecker<Resize> checker(1);

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

TEST_F(FALLBACK, RESIZE_NCHW_WITH_STRIDE) {
    param::Resize param;
    param.format = param::Resize::Format::NCHW;
    param.imode = param::Resize::InterpolationMode::LINEAR;
    Checker<Resize> checker(handle());
    checker.set_epsilon(1 + 1e-3).set_param(param);

    auto run = [&](TensorShape src_shape, std::vector<ptrdiff_t> src_layout,
                   TensorShape dst_shape, DType dtype) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).execl(
                {{src_shape, src_layout, dtype}, {dst_shape, dtype}});
    };

    for (DType& dtype : std::vector<DType>{dtype::Float32(), dtype::Uint8()}) {
        run({2, 3, 4, 4}, {256, 32, 8, 1}, {2, 3, 3, 3}, dtype);
        run({1, 3, 4, 3}, {105, 35, 7, 2}, {1, 3, 5, 5}, dtype);
        run({2, 3, 4, 4}, {-256, 32, -8, 1}, {2, 3, 3, 3}, dtype);
        run({2, 3, 4, 4}, {256, -32, 8, -1}, {2, 3, 3, 3}, dtype);
        run({2, 3, 4, 4}, {-256, -32, -8, -1}, {2, 3, 3, 3}, dtype);
    }
}
TEST_F(FALLBACK, RESIZE_NCHW_WITH_STRIDE_RECORD) {
    param::Resize param;
    param.format = param::Resize::Format::NCHW;
    param.imode = param::Resize::InterpolationMode::LINEAR;
    TaskRecordChecker<Resize> checker(1);
    checker.set_epsilon(1 + 1e-3).set_param(param);

    auto run = [&](TensorShape src_shape, std::vector<ptrdiff_t> src_layout,
                   TensorShape dst_shape, DType dtype) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).execl(
                {{src_shape, src_layout, dtype}, {dst_shape, dtype}});
    };

    for (DType& dtype : std::vector<DType>{dtype::Float32(), dtype::Uint8()}) {
        run({2, 3, 4, 4}, {256, 32, 8, 1}, {2, 3, 3, 3}, dtype);
        run({1, 3, 4, 3}, {105, 35, 7, 2}, {1, 3, 5, 5}, dtype);
        run({2, 3, 4, 4}, {-256, 32, -8, 1}, {2, 3, 3, 3}, dtype);
        run({2, 3, 4, 4}, {256, -32, 8, -1}, {2, 3, 3, 3}, dtype);
        run({2, 3, 4, 4}, {-256, -32, -8, -1}, {2, 3, 3, 3}, dtype);
    }
}

TEST_F(FALLBACK, RESIZE_NCHW4) {
    using namespace resize;
    auto args = get_nchw4_args();
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::QuantizedS8(1.0f))
                .set_dtype(1, dtype::QuantizedS8(1.0f))
                .set_epsilon(1 + 1e-3)
                .execs({arg.src, arg.dst});
    }
}
TEST_F(FALLBACK, RESIZE_NCHW4_RECORD) {
    using namespace resize;
    auto args = get_nchw4_args();
    TaskRecordChecker<Resize> checker(1);

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::QuantizedS8(1.0f))
                .set_dtype(1, dtype::QuantizedS8(1.0f))
                .set_epsilon(1 + 1e-3)
                .execs({arg.src, arg.dst});
    }
}

namespace {
static void set_nchw_args(resize::IMode imode, std::vector<resize::TestArg>& args) {
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
}  // namespace

TEST_F(FALLBACK, RESIZE_NCHW_FP32) {
    std::vector<resize::TestArg> args;
    set_nchw_args(resize::IMode::INTER_LINEAR, args);
    set_nchw_args(resize::IMode::INTER_NEAREST, args);
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

TEST_F(FALLBACK, RESIZE_NCHW44_FP32) {
    std::vector<resize::TestArg> args = resize::get_nchw44_args();
    Checker<Resize> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
