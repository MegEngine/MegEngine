#include "test/common/gaussian_blur.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, GAUSSIAN_BLUR) {
    using namespace gaussian_blur;
    std::vector<TestArg> args = get_args();
    Checker<GaussianBlur> checker(handle_cuda());

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

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
