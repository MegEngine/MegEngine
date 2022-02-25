#pragma once

#include <gtest/gtest.h>
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/opr_param_defs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {
namespace convolution3d {

struct TestArg {
    param::Convolution3D param;
    TensorShape src, filter;
    TestArg(param::Convolution3D param, TensorShape src, TensorShape filter)
            : param(param), src(src), filter(filter) {}
};

std::vector<TestArg> get_args();
std::vector<TestArg> get_1x1x1_args();
std::vector<TestArg> get_dilated_args();
std::vector<TestArg> get_chanwise_args();
std::vector<TestArg> get_speed_test_args();
//! \param stage 0 for fwd, 1 for bwd data, 2 for bwd filter
using ConvEPSGetter = std::function<float(bool f16, int stage, const char* algo_name)>;

//! check for various conv configurations (dilation, group, stride, padding)
//! and run all usable algorithms
void test_conv_config_combinations(
        Handle* handle, bool test_int8, bool test_backward, bool is_cuda,
        ConvEPSGetter conv_eps_getter = [](bool f16, int, const char*) -> float {
            return f16 ? 1e-1 : 1e-3;
        });

}  // namespace convolution3d
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
