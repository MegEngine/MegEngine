#include "test/aarch64/fixture.h"

#include "test/common/checker.h"
#include "test/common/pooling.h"
#include "test/common/task_record_check.h"

namespace megdnn {
namespace test {

TEST_F(AARCH64, POOLING) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        Checker<Pooling> checker(handle());
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}

TEST_F(AARCH64, POOLING_RECORD) {
    TaskRecordChecker<Pooling> checker(0);
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
