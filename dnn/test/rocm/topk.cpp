#include "hcc_detail/hcc_defs_prologue.h"

#include "test/common/topk.h"
#include "test/rocm/fixture.h"

using namespace megdnn;
using namespace test;

/*
 * !!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!
 * The kernels are indepedently developed and tested in the
 * MegDNN/expr/cuda_topk directory. Here we only check some common cases.
 */

TEST_F(ROCM, TOP_K) {
    run_topk_test<dtype::Float32>(handle_rocm());
}
TEST_F(ROCM, TOP_K_I32) {
    run_topk_test<dtype::Int32>(handle_rocm());
}

// vim: syntax=cpp.doxygen
