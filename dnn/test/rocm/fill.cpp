#include "test/common/fill.h"

#include "test/rocm/fixture.h"

namespace megdnn {
namespace test {
namespace fill {

TEST_F(ROCM, FILL_F32) {
    run_fill_test(handle_rocm(), dtype::Float32{});
}

TEST_F(ROCM, FILL_I32) {
    run_fill_test(handle_rocm(), dtype::Int32{});
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(ROCM, FILL_F16) {
    run_fill_test(handle_rocm(), dtype::Float16{});
}
#endif

}  // namespace fill
}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
