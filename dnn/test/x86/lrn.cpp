#include "test/x86/fixture.h"

#include "test/common/checker.h"
#include "test/common/local.h"
#include "test/common/task_record_check.h"
namespace megdnn {
namespace test {

TEST_F(X86, LRN) {
    Checker<LRNForward> checker(handle());
    checker.execs({{2, 11, 12, 13}, {}});
    for (size_t w = 10; w <= 50; ++w) {
        checker.execs({{2, w, 12, 13}, {}});
    }
}
TEST_F(X86, LRN_RECORD) {
    TaskRecordChecker<LRNForward> checker(0);
    checker.execs({{2, 11, 12, 13}, {}});
    for (size_t w = 10; w <= 50; w += 10) {
        checker.execs({{2, w, 12, 13}, {}});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
