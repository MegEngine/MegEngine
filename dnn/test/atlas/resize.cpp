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

}  // namespace resize
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen