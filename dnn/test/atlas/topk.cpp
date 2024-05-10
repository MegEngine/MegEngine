#include "test/common/topk.h"
#include "test/atlas/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, TOP_K) {
    run_topk_test<dtype::Float32>(handle_atlas(), true);
}
TEST_F(ATLAS, TOP_K_I32) {
    run_topk_test<dtype::Int32>(handle_atlas(), true);
}
#if !MEGDNN_DISABLE_FLOAT16
TEST_F(ATLAS, TOP_K_F16) {
    run_topk_test<dtype::Float16>(handle_atlas(), true);
}
#endif

// vim: syntax=cpp.doxygen
