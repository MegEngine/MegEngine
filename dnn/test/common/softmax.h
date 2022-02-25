#pragma once
#include <cstddef>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace softmax {

struct TestArg {
    param::Softmax param;
    TensorShape ishape;
    TestArg(param::Softmax param, TensorShape ishape) : param(param), ishape(ishape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    using Param = param::Softmax;

    for (int32_t axis = 0; axis < 5; axis++) {
        args.emplace_back(Param{axis}, TensorShape{2, 23, 32, 30, 17});
    }

    return args;
}

}  // namespace softmax
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen