#include <gtest/gtest.h>
#include <string.h>
#include <memory>
#include "test_options.h"

using namespace lar;
DECLARE_bool(lite);
DECLARE_string(input);
DECLARE_int32(batch_size);
DECLARE_int32(iter);
namespace {
STRING_OPTION_WRAP(input, "");
INT32_OPTION_WRAP(batch_size, -1);
BOOL_OPTION_WRAP(lite);
INT32_OPTION_WRAP(iter, 10);
}  // anonymous namespace

TEST(TestLarIO, INPUT) {
    DEFINE_INT32_WRAP(iter, 1);
    {
        std::string model_path = "./resnet50.mge";
        TEST_STRING_OPTION(input, "data:./resnet50_input.npy");
    }
    {
        std::string model_path = "./add_demo.mge";
        TEST_STRING_OPTION(input, "data:add_demo_input.json");
    }
    {
        std::string model_path = "./resnet50_uint8.mge";
        TEST_STRING_OPTION(input, "data:./cat.ppm");
    }
    {
        std::string model_path = "./add_demo.mge";
        TEST_STRING_OPTION(input, "data:[2.0,3.0,4.0]");
    }
    {
        std::string model_path = "./shufflenet.mge";
        TEST_STRING_OPTION(input, "data:{2,3,224,224}");
    }
    {
        std::string model_path = "./resnet50_b10.mdl";
        TEST_INT32_OPTION(batch_size, 1);
        TEST_INT32_OPTION(batch_size, 5);
        TEST_INT32_OPTION(batch_size, 11);
    }
}

TEST(TestLarIO, INPUT_LITE) {
    DEFINE_INT32_WRAP(iter, 1);
    DEFINE_BOOL_WRAP(lite);
    {
        std::string model_path = "./resnet50.mge";
        TEST_STRING_OPTION(input, "data:./resnet50_input.npy");
    }
    {
        std::string model_path = "./add_demo.mge";
        TEST_STRING_OPTION(input, "data:add_demo_input.json");
    }
    {
        std::string model_path = "./resnet50_uint8.mge";
        TEST_STRING_OPTION(input, "data:./cat.ppm");
    }
    {
        std::string model_path = "./add_demo.mge";
        TEST_STRING_OPTION(input, "data:[2.0,3.0,4.0]");
    }
    {
        std::string model_path = "./shufflenet.mge";
        TEST_STRING_OPTION(input, "data:{2,3,224,224}");
    }
    {
        std::string model_path = "./resnet50_b10.mdl";
        TEST_INT32_OPTION(batch_size, 1);
        TEST_INT32_OPTION(batch_size, 5);
        TEST_INT32_OPTION(batch_size, 11);
    }
}