#include <gtest/gtest.h>
#include <string.h>
#include <memory>
#include "test_options.h"

using namespace lar;
DECLARE_bool(lite);
DECLARE_bool(cpu);
#if LITE_WITH_CUDA
DECLARE_bool(cuda);
#endif

DECLARE_bool(enable_nchw4);
DECLARE_bool(enable_chwn4);
DECLARE_bool(enable_nchw44);
DECLARE_bool(enable_nchw88);
DECLARE_bool(enable_nchw32);
DECLARE_bool(enable_nchw64);
DECLARE_bool(enable_nhwcd4);
DECLARE_bool(enable_nchw44_dot);
DECLARE_bool(fast_run);
namespace {
BOOL_OPTION_WRAP(enable_nchw4);
BOOL_OPTION_WRAP(enable_chwn4);
BOOL_OPTION_WRAP(enable_nchw44);
BOOL_OPTION_WRAP(enable_nchw88);
BOOL_OPTION_WRAP(enable_nchw32);
BOOL_OPTION_WRAP(enable_nchw64);
BOOL_OPTION_WRAP(enable_nhwcd4);
BOOL_OPTION_WRAP(enable_nchw44_dot);
BOOL_OPTION_WRAP(fast_run);

BOOL_OPTION_WRAP(lite);
BOOL_OPTION_WRAP(cpu);
#if LITE_WITH_CUDA
BOOL_OPTION_WRAP(cuda);
#endif
}  // anonymous namespace

TEST(TestLarLayout, X86_CPU) {
    DEFINE_WRAP(cpu);
    std::string model_path = "./shufflenet.mge";

    TEST_BOOL_OPTION(enable_nchw4);
    TEST_BOOL_OPTION(enable_chwn4);
    TEST_BOOL_OPTION(enable_nchw44);
    TEST_BOOL_OPTION(enable_nchw44_dot);
    TEST_BOOL_OPTION(enable_nchw64);
    TEST_BOOL_OPTION(enable_nchw32);
    TEST_BOOL_OPTION(enable_nchw88);
}

TEST(TestLarLayout, X86_CPU_LITE) {
    DEFINE_WRAP(cpu);
    DEFINE_WRAP(lite);
    std::string model_path = "./shufflenet.mge";

    TEST_BOOL_OPTION(enable_nchw4);
    TEST_BOOL_OPTION(enable_nchw44);
    TEST_BOOL_OPTION(enable_nchw44_dot);
    TEST_BOOL_OPTION(enable_nchw64);
    TEST_BOOL_OPTION(enable_nchw32);
    TEST_BOOL_OPTION(enable_nchw88);
}

TEST(TestLarLayoutFastRun, CPU_LITE) {
    DEFINE_WRAP(cpu);
    DEFINE_WRAP(lite);
    std::string model_path = "./shufflenet.mge";
    {
        DEFINE_WRAP(enable_nchw44);
        DEFINE_WRAP(fast_run);
        run_NormalStrategy(model_path);
    }
}
#if LITE_WITH_CUDA
TEST(TestLarLayout, CUDA) {
    DEFINE_WRAP(cuda);
    std::string model_path = "./shufflenet.mge";
    TEST_BOOL_OPTION(enable_nchw4);
    TEST_BOOL_OPTION(enable_chwn4);
    TEST_BOOL_OPTION(enable_nchw64);
    TEST_BOOL_OPTION(enable_nchw32);

    FLAGS_cuda = false;
}

TEST(TestLarLayout, CUDA_LITE) {
    DEFINE_WRAP(cuda);
    DEFINE_WRAP(lite);
    std::string model_path = "./shufflenet.mge";

    TEST_BOOL_OPTION(enable_nchw4);
    TEST_BOOL_OPTION(enable_nchw64);
    TEST_BOOL_OPTION(enable_nchw32);
}
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
