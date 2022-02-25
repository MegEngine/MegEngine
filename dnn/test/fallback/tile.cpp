#include "test/fallback/fixture.h"

#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/common/tile_repeat.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, TILE) {
    Checker<TileForward> checker(handle());
    auto args = tile_repeat::get_args();
    for (auto&& arg : args) {
        checker.set_param(arg.param()).execs({arg.src, {}});
    }
}
TEST_F(FALLBACK, TILE_REOCRD) {
    TaskRecordChecker<TileForward> checker(1);
    auto args = tile_repeat::get_args();
    for (auto&& arg : args) {
        checker.set_param(arg.param()).execs({arg.src, {}});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
