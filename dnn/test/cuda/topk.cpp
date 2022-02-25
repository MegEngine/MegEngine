#include "test/common/topk.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

/*
 * !!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!
 * The kernels are indepedently developed and tested in the
 * MegDNN/expr/cuda_topk directory. Here we only check some common cases.
 */

TEST_F(CUDA, TOP_K) {
    run_topk_test<dtype::Float32>(handle_cuda());
}
TEST_F(CUDA, TOP_K_I32) {
    run_topk_test<dtype::Int32>(handle_cuda());
}
#if !MEGDNN_DISABLE_FLOAT16
TEST_F(CUDA, TOP_K_F16) {
    run_topk_test<dtype::Float16>(handle_cuda());
}
#endif

// vim: syntax=cpp.doxygen
