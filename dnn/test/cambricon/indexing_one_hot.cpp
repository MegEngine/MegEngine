#include "test/common/indexing_one_hot.h"
#include "megdnn/oprs.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(CAMBRICON, INDEXING_ONE_HOT) {
    run_indexing_one_hot_test(handle_cambricon());
}

TEST_F(CAMBRICON, INDEXING_SET_ONE_HOT) {
    run_indexing_set_one_hot_test(handle_cambricon());
}
