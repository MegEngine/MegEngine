#include <gtest/gtest.h>
#include <string.h>
#include <memory>
#include "./test_common.h"
#include "test_options.h"
using namespace lar;
DECLARE_bool(lite);
DECLARE_int32(iter);
DECLARE_bool(fast_run);
namespace {
BOOL_OPTION_WRAP(fast_run);
BOOL_OPTION_WRAP(lite);
INT32_OPTION_WRAP(iter, 10);
}  // anonymous namespace

TEST(TestLarFastRun, FAST_RUN) {
    double heristic_time = 0, fast_run_time = 0;
    lite::Timer timer("profile");
    {
        std::string model_path = "./shufflenet.mge";
        timer.reset_start();
        TEST_INT32_OPTION(iter, 1);
        heristic_time = timer.get_used_time();
    }

    {
        std::string model_path = "./shufflenet.mge";
        timer.reset_start();
        DEFINE_INT32_WRAP(iter, 1);
        TEST_BOOL_OPTION(fast_run);
        fast_run_time = timer.get_used_time();
    }
    //! profile time is longer than excute time
    ASSERT_LT(heristic_time, fast_run_time);
}

TEST(TestLarFastRun, FAST_RUN_LITE) {
    double heristic_time = 0, fast_run_time = 0;
    lite::Timer timer("profile");
    //! clear profile cache
    auto profile_cache = mgb::PersistentCache::inst().get_cache();
    DEFINE_BOOL_WRAP(lite);
    {
        std::string model_path = "./shufflenet.mge";
        timer.reset_start();
        TEST_INT32_OPTION(iter, 1);
        heristic_time = timer.get_used_time();
    }

    {
        std::string model_path = "./shufflenet.mge";
        timer.reset_start();
        DEFINE_INT32_WRAP(iter, 1);
        TEST_BOOL_OPTION(fast_run);
        fast_run_time = timer.get_used_time();
    }
    //! profile time is longer than excute time
    ASSERT_LT(heristic_time, fast_run_time);
}