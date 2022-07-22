#pragma once
#include <cstddef>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace adaptive_pooling {

struct TestArg {
    param::AdaptivePooling param;
    TensorShape ishape;
    TensorShape oshape;
    TestArg(param::AdaptivePooling param, TensorShape ishape, TensorShape oshape)
            : param(param), ishape(ishape), oshape(oshape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    using Param = param::AdaptivePooling;
    using Mode = param::AdaptivePooling::Mode;

    for (size_t i = 36; i < 40; ++i) {
        args.emplace_back(
                Param{Mode::AVERAGE}, TensorShape{2, 3, i, i + 1},
                TensorShape{2, 3, i - 4, i - 2});
        args.emplace_back(
                Param{Mode::MAX}, TensorShape{2, 3, i, i + 1},
                TensorShape{2, 3, i - 4, i - 2});
    }

    for (size_t i = 5; i < 10; ++i) {
        args.emplace_back(
                Param{Mode::AVERAGE}, TensorShape{2, 3, i, i + 1},
                TensorShape{2, 3, i - 3, i - 2});
        args.emplace_back(
                Param{Mode::MAX}, TensorShape{2, 3, i, i + 1},
                TensorShape{2, 3, i - 3, i - 2});
    }
    return args;
}

inline std::vector<TestArg> get_args_nchw44() {
    std::vector<TestArg> args;
    using Param = param::AdaptivePooling;
    using Mode = param::AdaptivePooling::Mode;

    for (size_t i = 36; i < 40; ++i) {
        args.emplace_back(
                Param{Mode::AVERAGE, Param::Format::NCHW44},
                TensorShape{2, 3, i, i + 1, 4}, TensorShape{2, 3, i - 4, i - 2, 4});
        args.emplace_back(
                Param{Mode::MAX, Param::Format::NCHW44}, TensorShape{2, 3, i, i + 1, 4},
                TensorShape{2, 3, i - 4, i - 2, 4});
        args.emplace_back(
                Param{Mode::AVERAGE, Param::Format::NCHW44},
                TensorShape{2, 3, i, i + 1, 4}, TensorShape{2, 3, 1, 1, 4});
        args.emplace_back(
                Param{Mode::MAX, Param::Format::NCHW44}, TensorShape{2, 3, i, i + 1, 4},
                TensorShape{2, 3, 1, 1, 4});
    }

    for (size_t i = 5; i < 10; ++i) {
        args.emplace_back(
                Param{Mode::AVERAGE, Param::Format::NCHW44},
                TensorShape{2, 3, i, i + 1, 4}, TensorShape{2, 3, i - 3, i - 2, 4});
        args.emplace_back(
                Param{Mode::MAX, Param::Format::NCHW44}, TensorShape{2, 3, i, i + 1, 4},
                TensorShape{2, 3, i - 3, i - 2, 4});
    }
    return args;
}
}  // namespace adaptive_pooling
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
