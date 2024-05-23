#include "test/common/resize.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {
namespace resize {

TEST_F(ATLAS, RESIZE_BACKWARD) {
    IMode modes[] = {IMode::NEAREST, IMode::LINEAR};
    DType dtypes[] = {dtype::Float32(), dtype::Float16()};
    for (auto imode : modes) {
        for (auto type : dtypes) {
            Checker<ResizeBackward> checker(handle_atlas());
            param::Resize param;
            param.format = param::Resize::Format::NCHW;
            param.imode = imode;
            checker.set_param(param);
            checker.set_dtype(0, type);
            checker.set_dtype(1, type);
            if (type == dtype::Float16()) {
                checker.set_epsilon(1e-2);
            }

            checker.execs({{2, 3, 4, 5}, {2, 3, 8, 9}});
            checker.execs({{2, 5, 8, 9}, {2, 5, 4, 5}});
            checker.execs({{2, 5, 8, 5}, {2, 5, 4, 9}});
            checker.execs({{2, 5, 4, 9}, {2, 5, 8, 5}});
        }
    }
    // TODO: skip NHWC temporarily
    // for (auto imode : modes) {
    //     for (auto type : dtypes) {
    //         Checker<ResizeBackward> checker(handle_atlas());
    //         param::Resize param;
    //         param.format = param::Resize::Format::NHWC;
    //         param.imode = imode;
    //         checker.set_param(param);
    //         checker.set_dtype(0, type);
    //         checker.set_dtype(1, type);

    //         checker.execs({{2, 4, 5, 3}, {2, 8, 9, 3}});
    //         checker.execs({{2, 8, 9, 5}, {2, 4, 5, 5}});
    //         checker.execs({{2, 8, 5, 5}, {2, 4, 9, 5}});
    //         checker.execs({{2, 4, 9, 5}, {2, 8, 5, 5}});
    //     }
    // }
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
    args.emplace_back(param, TensorShape{1, 1, 2, 2}, TensorShape{1, 1, 4, 4});
    args.emplace_back(param, TensorShape{1, 1, 10, 10}, TensorShape{1, 1, 20, 20});
    args.emplace_back(param, TensorShape{1, 1, 10, 10}, TensorShape{1, 1, 7, 9});
    args.emplace_back(param, TensorShape{2, 2, 3, 4}, TensorShape{2, 2, 6, 8});
    args.emplace_back(param, TensorShape{1, 2, 6, 8}, TensorShape{1, 2, 3, 4});
}
}  // namespace

TEST_F(ATLAS, RESIZE) {
    using namespace resize;
    std::vector<resize::TestArg> args;
    set_nchw_args(resize::IMode::INTER_LINEAR, args);
    set_nchw_args(resize::IMode::INTER_NEAREST, args);
    Checker<Resize> checker(handle_atlas());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
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

}  // namespace resize
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen