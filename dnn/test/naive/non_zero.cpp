#include "test/common/non_zero.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;
TEST_F(NAIVE, NONZERO) {
    std::vector<NonZeroTestcase> test_cases = NonZeroTestcase::make();
    auto opr_naive = handle()->create_operator<NonZero>();
    for (NonZeroTestcase& test_case : test_cases) {
        NonZeroTestcase::Result data = test_case.run_naive(opr_naive.get());
        int ndim = test_case.m_data.layout.ndim;
        std::vector<int> result = test_case.correct_answer;
        NonZeroTestcase::Assert(result, ndim, data);
    }
}
