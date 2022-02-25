#include "test/common/cond_take.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, COND_TAKE) {
    auto opr_naive = handle_naive()->create_operator<CondTake>();
    auto opr_cuda = handle_cuda()->create_operator<CondTake>();

    size_t tot_size = 0;
    for (auto&& i : CondTakeTestcase::make()) {
        auto ret_naive = i.run(opr_naive.get()), ret_cuda = i.run(opr_cuda.get());
        MEGDNN_ASSERT_TENSOR_EQ(*ret_naive.first, *ret_cuda.first);
        MEGDNN_ASSERT_TENSOR_EQ(*ret_naive.second, *ret_cuda.second);
        tot_size += ret_naive.first->layout.total_nr_elems();
    }
    ASSERT_GT(tot_size, (size_t)0);
}

// vim: syntax=cpp.doxygen
