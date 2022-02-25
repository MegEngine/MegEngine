#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace separable_conv {

struct TestArg {
    param::SeparableConv param;
    TensorShape src, filter_x, filter_y;
    TestArg(param::SeparableConv param, TensorShape src, TensorShape filter_x,
            TensorShape filter_y)
            : param(param), src(src), filter_x(filter_x), filter_y(filter_y) {}
};

std::vector<TestArg> get_args();

}  // namespace separable_conv
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
