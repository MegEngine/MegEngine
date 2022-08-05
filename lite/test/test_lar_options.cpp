#include <gtest/gtest.h>
#include <string.h>
#include <memory>
#include "test_common.h"
#include "test_options.h"

using namespace lar;
DECLARE_bool(lite);
DECLARE_bool(cpu);
DECLARE_bool(optimize_for_inference);
#if LITE_WITH_CUDA
DECLARE_bool(cuda);
#endif

namespace {

BOOL_OPTION_WRAP(optimize_for_inference);

BOOL_OPTION_WRAP(lite);
BOOL_OPTION_WRAP(cpu);
#if LITE_WITH_CUDA
BOOL_OPTION_WRAP(cuda);
#endif
}  // anonymous namespace

TEST(TestLarOption, OPTIMIZE_FOR_INFERENCE) {
    DEFINE_BOOL_WRAP(cpu);
    std::string model_path = "./shufflenet.mge";

    TEST_BOOL_OPTION(optimize_for_inference);
}

#if LITE_WITH_OPENCL
TEST(TestLarOption, OPTIMIZE_FOR_INFERENCE_OPENCL) {
    REQUIRE_OPENCL();
    DEFINE_BOOL_WRAP(opencl);
    std::string model_path = "./shufflenet.mge";

    TEST_BOOL_OPTION(optimize_for_inference);
}
#endif

#if LITE_WITH_CUDA
TEST(TestLarOption, OPTIMIZE_FOR_INFERENCE_CUDA) {
    REQUIRE_CUDA();
    DEFINE_BOOL_WRAP(cuda);
    std::string model_path = "./shufflenet.mge";

    TEST_BOOL_OPTION(optimize_for_inference);
}
#endif
