#include "test/common/indexing_one_hot.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"

#include "megdnn/oprs/general.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, INDEXING_ONE_HOT) {
    run_indexing_one_hot_test(handle_atlas());
}

TEST_F(ATLAS, INDEXING_SET_ONE_HOT) {
    run_indexing_set_one_hot_test(handle_atlas());
}