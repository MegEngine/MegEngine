#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace conv_pooling {

struct TestArg {
    param::ConvPooling param;
    TensorShape src, filter, bias;
    TestArg(param::ConvPooling param, TensorShape src, TensorShape filter,
            TensorShape bias)
            : param(param), src(src), filter(filter), bias(bias) {}
};

std::vector<TestArg> get_args();

}  // namespace conv_pooling
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
