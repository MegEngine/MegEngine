#pragma once
#include <iostream>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

#include "./rng.h"
namespace megdnn {
namespace test {
namespace remap {

struct TestArg {
    param::Remap param;
    TensorShape src;
    TensorShape map_xy;
    TensorShape dst;
    TestArg(param::Remap param_, TensorShape src_, TensorShape map_xy_,
            TensorShape dst_)
            : param(param_), src(src_), map_xy(map_xy_), dst(dst_) {}
};

static inline std::vector<TestArg> get_nchw_args() {
    std::vector<TestArg> args;

    param::Remap param;
    std::vector<param::Remap::Format> format_vec = {param::Remap::Format::NCHW};
    std::vector<param::Remap::InterpolationMode> interp_mode_vec = {
            param::Remap::InterpolationMode::NEAREST,
            param::Remap::InterpolationMode::LINEAR};
    std::vector<param::Remap::BorderMode> border_mode_vec = {
            param::Remap::BorderMode::CONSTANT, param::Remap::BorderMode::REFLECT_101,
            param::Remap::BorderMode::REFLECT, param::Remap::BorderMode::WRAP,
            param::Remap::BorderMode::REPLICATE};

    // current do not test this.
    std::vector<float> scalar;
    for (auto fmt : format_vec) {
        for (auto interp_mode : interp_mode_vec) {
            for (auto border_type : border_mode_vec) {
                param.format = fmt;
                param.imode = interp_mode;
                param.border_type = border_type;
                args.emplace_back(
                        param, TensorShape{70000, 1, 2, 2}, TensorShape{70000, 2, 2, 2},
                        TensorShape{70000, 1, 2, 2});

                args.emplace_back(
                        param, TensorShape{1, 1, 2, 2}, TensorShape{1, 2, 2, 2},
                        TensorShape{1, 1, 2, 2});

                args.emplace_back(
                        param, TensorShape{1, 3, 2, 2}, TensorShape{1, 2, 2, 2},
                        TensorShape{1, 3, 2, 2});

                args.emplace_back(
                        param, TensorShape{1, 10, 100, 100},
                        TensorShape{1, 100, 100, 2}, TensorShape{1, 10, 100, 100});

                args.emplace_back(
                        param, TensorShape{2, 4, 100, 200}, TensorShape{2, 100, 200, 2},
                        TensorShape{2, 4, 100, 200});

                args.emplace_back(
                        param, TensorShape{2, 4, 100, 200}, TensorShape{2, 20, 30, 2},
                        TensorShape{2, 4, 20, 30});

                args.emplace_back(
                        param, TensorShape{2, 4, 10, 10}, TensorShape{2, 20, 30, 2},
                        TensorShape{2, 4, 20, 30});
            }
        }
    }
    return args;
}

static inline std::vector<TestArg> get_nhwcd4_args() {
    std::vector<TestArg> args;

    param::Remap param;
    param.format = param::Remap::Format::NHWCD4;
    param.imode = param::Remap::InterpolationMode::LINEAR;
    param.border_type = param::Remap::BorderMode::CONSTANT;
    // FIXME: when fractional part of bval is not zero, naive and opencl bankend may
    // have different rounding result
    param.scalar = 77;
    args.emplace_back(
            param, TensorShape{2, 2, 1, 3, 4}, TensorShape{2, 4, 6, 2},
            TensorShape{2, 4, 1, 6, 4});
    args.emplace_back(
            param, TensorShape{2, 4, 1, 6, 4}, TensorShape{2, 2, 3, 2},
            TensorShape{2, 2, 1, 3, 4});

    param.imode = param::Remap::InterpolationMode::NEAREST;
    args.emplace_back(
            param, TensorShape{2, 2, 1, 3, 4}, TensorShape{2, 4, 6, 2},
            TensorShape{2, 4, 1, 6, 4});
    args.emplace_back(
            param, TensorShape{2, 4, 1, 6, 4}, TensorShape{2, 2, 3, 2},
            TensorShape{2, 2, 1, 3, 4});

    return args;
}

static inline std::vector<TestArg> get_nhwc_args() {
    std::vector<TestArg> args;

    param::Remap param;
    std::vector<param::Remap::Format> format_vec = {param::Remap::Format::NHWC};
    std::vector<param::Remap::InterpolationMode> interp_mode_vec = {
            param::Remap::InterpolationMode::NEAREST,
            param::Remap::InterpolationMode::LINEAR};
    std::vector<param::Remap::BorderMode> border_mode_vec = {
            param::Remap::BorderMode::CONSTANT, param::Remap::BorderMode::REFLECT_101,
            param::Remap::BorderMode::REFLECT, param::Remap::BorderMode::WRAP,
            param::Remap::BorderMode::REPLICATE};
    // current do not test this.
    std::vector<float> scalar;
    for (auto fmt : format_vec) {
        for (auto interp_mode : interp_mode_vec) {
            for (auto border_type : border_mode_vec) {
                param.format = fmt;
                param.imode = interp_mode;
                param.border_type = border_type;
                param.scalar = 12.f;
                args.emplace_back(
                        param, TensorShape{70000, 2, 2, 1}, TensorShape{70000, 2, 2, 2},
                        TensorShape{70000, 2, 2, 1});

                args.emplace_back(
                        param, TensorShape{1, 2, 2, 1}, TensorShape{1, 2, 2, 2},
                        TensorShape{1, 2, 2, 1});

                args.emplace_back(
                        param, TensorShape{1, 2, 2, 3}, TensorShape{1, 2, 2, 2},
                        TensorShape{1, 2, 2, 3});

                args.emplace_back(
                        param, TensorShape{1, 2, 2, 66}, TensorShape{1, 2, 2, 2},
                        TensorShape{1, 2, 2, 66});

                args.emplace_back(
                        param, TensorShape{1, 100, 100, 10},
                        TensorShape{1, 100, 100, 2}, TensorShape{1, 100, 100, 10});

                args.emplace_back(
                        param, TensorShape{2, 100, 200, 4}, TensorShape{2, 100, 200, 2},
                        TensorShape{2, 100, 200, 4});

                args.emplace_back(
                        param, TensorShape{2, 100, 200, 4}, TensorShape{2, 20, 30, 2},
                        TensorShape{2, 20, 30, 4});

                args.emplace_back(
                        param, TensorShape{2, 10, 10, 4}, TensorShape{2, 20, 30, 2},
                        TensorShape{2, 20, 30, 4});
            }
        }
    }
    return args;
}

}  // namespace remap
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
