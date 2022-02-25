#include "test/common/powc.h"

#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, POW_C_F32) {
    run_powc_test(handle_cuda(), dtype::Float32{});
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(CUDA, POW_C_F16) {
    run_powc_test(handle_cuda(), dtype::Float16{});
}
#endif

// vim: syntax=cpp.doxygen
