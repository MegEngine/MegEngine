#include "test/common/cond_take.h"
#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, COND_TAKE) {
    auto opr_naive = handle_naive()->create_operator<CondTake>();
    auto opr_atlas = handle_atlas()->create_operator<CondTake>();

    size_t tot_size = 0;
    for (auto&& i : CondTakeTestcase::make()) {
        auto ret_naive = i.run(opr_naive.get()), ret_atlas = i.run(opr_atlas.get());
        MEGDNN_ASSERT_TENSOR_EQ(*ret_naive.first, *ret_atlas.first);
        MEGDNN_ASSERT_TENSOR_EQ(*ret_naive.second, *ret_atlas.second);
        tot_size += ret_naive.first->layout.total_nr_elems();
    }
    ASSERT_GT(tot_size, (size_t)0);
}

// vim: syntax=cpp.doxygen
