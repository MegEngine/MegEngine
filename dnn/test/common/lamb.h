#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace lamb {

struct TestArg {
    param::LAMBUpdate param;
    TensorShape src;
    TestArg(param::LAMBUpdate param, TensorShape src) : param(param), src(src) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    param::LAMBUpdate cur_param;
    cur_param.beta_1 = 0.9;
    cur_param.beta_2 = 0.999;
    cur_param.eps = 1e-8;
    cur_param.weight_decay = 0;
    cur_param.lr = 6.25e-5;
    cur_param.bias_correction = true;
    cur_param.always_adapt = false;
    args.emplace_back(
            cur_param, TensorShape{
                               1280,
                       });
    args.emplace_back(cur_param, TensorShape{1280, 1280});
    args.emplace_back(cur_param, TensorShape{1280, 3, 224, 224});
    return args;
}

}  // namespace lamb
}  // namespace test
}  // namespace megdnn
