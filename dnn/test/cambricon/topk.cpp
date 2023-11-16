#include "test/common/topk.h"
#include "test/cambricon/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CAMBRICON, TOP_K) {
    run_topk_test<dtype::Float32>(handle_cambricon(), false);
}
TEST_F(CAMBRICON, TOP_K_I32) {
    run_topk_test<dtype::Int32>(handle_cambricon(), false);
}
#if !MEGDNN_DISABLE_FLOAT16
TEST_F(CAMBRICON, TOP_K_F16) {
    run_topk_test<dtype::Float16>(handle_cambricon(), false);
}
#endif

// vim: syntax=cpp.doxygen
