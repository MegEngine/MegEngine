
#pragma once
#include <iostream>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace norm {

struct TestArg {
    param::Norm param;
    TensorShape src;
    TestArg(param::Norm param, TensorShape src) : param(param), src(src) {}
};

}  // namespace norm
}  // namespace test
}  // namespace megdnn
