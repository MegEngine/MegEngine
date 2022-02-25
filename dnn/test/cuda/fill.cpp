#include "test/common/fill.h"

#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {
namespace fill {

TEST_F(CUDA, FILL_F32) {
    run_fill_test(handle_cuda(), dtype::Float32{});
}

TEST_F(CUDA, FILL_I32) {
    run_fill_test(handle_cuda(), dtype::Int32{});
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(CUDA, FILL_F16) {
    run_fill_test(handle_cuda(), dtype::Float16{});
}
#endif

}  // namespace fill
}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
