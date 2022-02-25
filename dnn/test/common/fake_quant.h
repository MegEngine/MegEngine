#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace fake_quant {

struct TestArg {
    param::FakeQuant param;
    TensorShape ishape;
    TensorShape scale_shape;
    TensorShape zeropoint_shape;
    TestArg(param::FakeQuant param, TensorShape ishape, TensorShape scale_shape,
            TensorShape zeropoint_shape)
            : param(param),
              ishape(ishape),
              scale_shape(scale_shape),
              zeropoint_shape(zeropoint_shape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    param::FakeQuant cur_param;

    cur_param.qmin = -128;
    cur_param.qmax = 128;

    for (size_t i = 10; i < 40; i += 2) {
        args.emplace_back(
                cur_param, TensorShape{10, 64, i, i}, TensorShape{1}, TensorShape{1});
    }

    for (size_t m : {1, 10})
        for (size_t n : {1, 10})
            for (size_t j : {1, 10})
                for (size_t k : {1, 10}) {
                    args.emplace_back(
                            cur_param, TensorShape{10, 64, 10, 10},
                            TensorShape{10, 64, m, n}, TensorShape{10, 64, j, k});
                }
    return args;
}

}  // namespace fake_quant
}  // namespace test

}  // namespace megdnn