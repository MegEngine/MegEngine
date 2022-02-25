#include "test/armv7/fixture.h"

#include "test/common/checker.h"
#include "test/common/pooling.h"

namespace megdnn {
namespace test {

TEST_F(ARMV7, POOLING) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        Checker<Pooling> checker(handle());
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
