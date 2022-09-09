#include "test/common/non_zero.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;
TEST_F(CUDA, NONZERO) {
    std::vector<NonZeroTestcase> test_cases = NonZeroTestcase::make();
    auto opr_cuda = handle_cuda()->create_operator<NonZero>();
    auto opr_naive = handle_naive()->create_operator<NonZero>();
    for (NonZeroTestcase& test_case : test_cases) {
        NonZeroTestcase::CUDAResult data = test_case.run_cuda(opr_cuda.get());
        NonZeroTestcase::CUDAResult data_naive = test_case.run_cuda(opr_naive.get());

        std::vector<int> result = test_case.correct_answer;
        MEGDNN_ASSERT_TENSOR_EQ(*data, *data_naive);
    }
}