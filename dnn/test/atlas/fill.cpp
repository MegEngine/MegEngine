#include "test/common/fill.h"

#include "test/atlas/fixture.h"

namespace megdnn {
namespace test {
namespace fill {

TEST_F(ATLAS, FILL_F32) {
    run_fill_test(handle_atlas(), dtype::Float32{});
}

TEST_F(ATLAS, FILL_I32) {
    run_fill_test(handle_atlas(), dtype::Int32{});
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(ATLAS, FILL_F16) {
    run_fill_test(handle_atlas(), dtype::Float16{});
}
#endif

}  // namespace fill
}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
