#include "test/fallback/fixture.h"

#include "test/common/checker.h"
#include "test/common/task_record_check.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, SPLIT) {
    Checker<Split> checker(handle());
    using Param = Split::Param;
    for (size_t axis = 0; axis < 4; ++axis) {
        Param param;
        param.axis = axis;
        TensorShapeArray shapes(5, TensorShape({2, 3, 4, 5}));
        shapes[0].shape[axis] = 10;
        for (size_t i = 1; i < 5; ++i) {
            shapes[i].shape[axis] = i;
        }
        checker.set_param(param).exec(shapes);
    }
}
TEST_F(FALLBACK, SPLIT_RECORD) {
    TaskRecordChecker<Split> checker(1);
    using Param = Split::Param;
    for (size_t axis = 0; axis < 4; ++axis) {
        Param param;
        param.axis = axis;
        TensorShapeArray shapes(5, TensorShape({2, 3, 4, 5}));
        shapes[0].shape[axis] = 10;
        for (size_t i = 1; i < 5; ++i) {
            shapes[i].shape[axis] = i;
        }
        checker.set_param(param).exec(shapes);
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
