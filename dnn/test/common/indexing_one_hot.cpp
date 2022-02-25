#include "test/common/indexing_one_hot.h"
#include "test/common/checker.h"

#include "megdnn/oprs/general.h"

using namespace megdnn;
using namespace test;

void test::run_indexing_one_hot_test(
        Handle* handle, const thin_function<void()>& fail_test) {
    Checker<IndexingOneHot> checker(handle);
    UniformIntRNG rng_idx{0, 7};
    checker.set_param({2}).set_dtype(1, dtype::Int32{}).set_rng(1, &rng_idx);
    checker.execs({{10, 4, 8, 9}, {10, 4, 9}, {}});
    if (fail_test) {
        rng_idx = {100, 200};
        checker.set_expect_exec_fail(fail_test).execs({{10, 4, 8, 9}, {10, 4, 9}, {}});
    }
}

void test::run_indexing_set_one_hot_test(Handle* handle) {
    Checker<IndexingSetOneHot> checker(handle);
    UniformIntRNG rng_idx{0, 7};
    checker.set_param({2}).set_dtype(1, dtype::Int32{}).set_rng(1, &rng_idx);
    checker.execs({{10, 4, 8, 9}, {10, 4, 9}, {10, 4, 1, 9}});
}

// vim: syntax=cpp.doxygen
