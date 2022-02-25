#include "test/cuda/fixture.h"

#include "megdnn/oprs/linalg.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/common/svd.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, SINGULAR_VALUE_DECOMPOSITION) {
    auto opr_naive = handle_naive()->create_operator<SVDForward>();
    auto opr_cuda = handle_cuda()->create_operator<SVDForward>();
    auto testcases = SVDTestcase::make();
    for (auto& t : testcases) {
        auto cuda_result = t.run(opr_cuda.get());

        bool old_compute_nv = t.m_param.compute_uv;
        t.m_param.compute_uv = false;
        auto naive_result = t.run(opr_naive.get());
        t.m_param.compute_uv = old_compute_nv;

        MEGDNN_ASSERT_TENSOR_EQ(*naive_result.s, *cuda_result.s);
        if (t.m_param.compute_uv) {
            MEGDNN_ASSERT_TENSOR_EQ(*cuda_result.recovered_mat, t.m_mat);
        }
    }
}

// vim: syntax=cpp.doxygen
