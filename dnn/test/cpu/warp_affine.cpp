#include "test/common/warp_affine.h"
#include "test/common/checker.h"
#include "test/cpu/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CPU, WARP_AFFINE_CV) {
    using namespace warp_affine;
    std::vector<TestArg> args = get_cv_args();
    Checker<WarpAffine> checker(handle());

    for (auto&& arg : args) {
        checker.set_param(arg.param)
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
