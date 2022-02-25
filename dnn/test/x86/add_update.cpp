#include "test/common/checker.h"
#include "test/common/resize.h"
#include "test/common/rng.h"
#include "test/common/task_record_check.h"
#include "test/x86/fixture.h"
namespace megdnn {
namespace test {

TEST_F(X86, ADD_UPDATE) {
    Checker<AddUpdate> checker(handle());

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{1, 3, 5, 5}, {1, 3, 5, 5}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {2, 3, 4}});
    checker.set_param({2, -1, 3})
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 2}, {2, 3, 2}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{1, 3, 5, 5}, {1, 3, 1, 1}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {1}});
}

TEST_F(X86, ADD_UPDATE_RECORD) {
    TaskRecordChecker<AddUpdate> checker(0);

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{1, 3, 5, 5}, {1, 3, 5, 5}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {2, 3, 4}});
    checker.set_param({2, -1, 3})
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 2}, {2, 3, 2}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{1, 3, 5, 5}, {1, 3, 1, 1}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {1}});
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
