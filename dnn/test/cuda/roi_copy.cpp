#include <gtest/gtest.h>

#include "megdnn.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/roi_copy.h"
#include "test/common/tensor.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, ROICOPY) {
    using namespace roi_copy;
    std::vector<TestArg> args = get_args();
    Checker<ROICopy> checker(handle_cuda());
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, {}});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
