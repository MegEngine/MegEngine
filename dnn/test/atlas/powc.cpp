#include "test/common/powc.h"

#include "test/atlas/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, POW_C_F32) {
    run_powc_test(handle_atlas(), dtype::Float32{}, true);
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(ATLAS, POW_C_F16) {
    run_powc_test(handle_atlas(), dtype::Float16{}, true);
}
#endif

// // vim: syntax=cpp.doxygen
