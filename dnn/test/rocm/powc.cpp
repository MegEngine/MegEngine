#include "test/common/powc.h"

#include "test/rocm/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(ROCM, POW_C_F32) {
    run_powc_test(handle_rocm(), dtype::Float32{});
}

#if !MEGDNN_DISABLE_FLOAT16
//! FIXME: powc for rocm has bugs
TEST_F(ROCM, POW_C_F16) {
    run_powc_test(handle_rocm(), dtype::Float16{});
}
#endif

// vim: syntax=cpp.doxygen
