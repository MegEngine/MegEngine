/**
 * \file dnn/test/cuda/warp_perspective.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/warp_perspective.h"
#include "test/common/opr_proxy.h"
#include "test/cuda/utils.h"

namespace {

using namespace megdnn;
using namespace test;

class NanMatRNG : public RNG {
    void gen(const TensorND& tensor_) override {
        auto& gen = RandomState::generator();
        std::uniform_real_distribution<dt_float32> pdist3(1.9f, 2.1f);
        std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
        std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
        std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
        std::uniform_real_distribution<dt_float32> ndist3(-2.1f, -1.9f);
        std::uniform_real_distribution<dt_float32> ndisth(-0.6f, -0.4f);
        std::uniform_int_distribution<int> dice(0, 5);
        float* ptr = tensor_.ptr<dt_float32>();
        auto N = tensor_.layout.shape[0];
        for (size_t n = 0; n < N; ++n) {
            for (size_t i = 0; i < 9; ++i) {
                switch (dice(gen)) {
                    case 0:
                        ptr[i] = pdist3(gen);
                        break;
                    case 1:
                        ptr[i] = pdist(gen);
                        break;
                    case 2:
                        ptr[i] = pdisth(gen);
                        break;
                    case 3:
                        ptr[i] = ndist(gen);
                        break;
                    case 4:
                        ptr[i] = ndist3(gen);
                        break;
                    case 5:
                        ptr[i] = ndisth(gen);
                        break;
                }
            }
            ptr[6] = 1;
            ptr[7] = -1;
            ptr[8] = 5;
            ptr += 9;
        }
    }
};

}  // anonymous namespace

namespace megdnn {
namespace test {

// FIXME test WARP_PERSPECTIVE_CV failed here
#if 0
TEST_F(CUDA, WARP_PERSPECTIVE_CV) {
    //! format = NHWC
    Checker<WarpPerspective> checker(handle_cuda());
    param::WarpPerspective param;
    class ResizeMatRNG: public RNG {
        void gen(const TensorND &tensor_) override
        {
            auto &gen = RandomState::generator();
            std::uniform_real_distribution<dt_float32> pdist3(1.9f, 3.1f);
            std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
            std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
            std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
            std::uniform_real_distribution<dt_float32> ndist3(-3.1f, -1.9f);
            std::uniform_real_distribution<dt_float32> ndisth(-0.6f, -0.4f);
            std::uniform_int_distribution<int> dice(0, 5);
            float *ptr = tensor_.ptr<dt_float32>();
            auto N = tensor_.layout.shape[0];
            for (size_t n = 0; n < N; ++n) {
                for (size_t i = 0; i < 9; ++i) {
                    switch (dice(gen)) {
                        case 0:
                            ptr[i] = pdist3(gen);
                            break;
                        case 1:
                            ptr[i] = pdist(gen);
                            break;
                        case 2:
                            ptr[i] = pdisth(gen);
                            break;
                        case 3:
                            ptr[i] = ndist(gen);
                            break;
                        case 4:
                            ptr[i] = ndist3(gen);
                            break;
                        case 5:
                            ptr[i] = ndisth(gen);
                            break;
                    }
                }
                // is resize?
                if (n & 1) {
                    ptr[1] = 0;
                    ptr[3] = 0;
                    ptr[6] = ptr[7] = 0;
                }
                ptr += 9;
            }
        }
    } rng;
    checker.set_rng(1, &rng);
    using BMode = param::WarpPerspective::BorderMode;
    param.format = param::WarpPerspective::Format::NHWC;
    // naive and cuda uses different algorithms and different border handling
    checker.set_epsilon(2.001).set_max_avg_error(4e-2);
    for (auto mode: {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT,
            BMode::WRAP, BMode::CONSTANT})
    {
        param.bmode = mode;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }

    auto args = warp_perspective::get_cv_args();
    for (auto &&arg : args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .execs({arg.src, arg.trans, arg.dst});
    }
    for (auto &&arg : args) {
        checker.set_param(arg.param)
            .set_epsilon(242.001)
            .set_max_avg_error(3.0)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Uint8())
            .execs({arg.src, arg.trans, arg.dst});
    }

    // resize nan case
    UniformFloatRNG rng_zero(0, 0);
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        // no invalid mem access is enough; no need to check value
        checker.set_expect_exec_fail([](){});
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }
}
#endif

TEST_F(CUDA, WARP_PERSPECTIVE_FORWARD) {
    using Param = WarpPerspective::Param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        WarpPerspective::Param param;
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NHWC;
        checker.set_param(param);
        checker.set_epsilon(0.15).set_max_avg_error(4e-2);
        checker.execs({{2, 10, 11, 3}, {2, 3, 3}, {2, 11, 12, 3}});
        checker.execs({{2200, 10, 11, 3}, {2200, 3, 3}, {2200, 11, 12, 3}});
        checker.set_epsilon(1e-3);
        checker.execs({{20, 10, 11, 123}, {20, 3, 3}, {20, 11, 12, 123}});

        param.format = Param::Format::NCHW;
        checker.set_param(param);
        checker.execs({{2, 3, 10, 11}, {2, 3, 3}, {2, 3, 11, 12}});
        checker.execs({{20, 3000, 10, 11}, {20, 3, 3}, {20, 3000, 11, 12}});
        checker.execs({{22000, 3, 10, 11}, {22000, 3, 3}, {22000, 3, 11, 12}});
    }
    // nan case
    NanMatRNG rng_nan;
    UniformFloatRNG rng_zero(0, 0);
    for (auto rng : std::vector<RNG*>{&rng_nan, &rng_zero}) {
        param::WarpPerspective param;
        param.bmode = param::WarpPerspective::BorderMode::CONSTANT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_rng(1, rng);
        param.border_val = 1.737;
        checker.set_param(param);
        // no invalid mem access is enough; no need to check value
        checker.set_expect_exec_fail([]() {});
        checker.exec({{1000, 2, 10, 11}, {1000, 3, 3}, {1000, 2, 12, 13}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_FORWARD_INTMAX) {
    require_compute_capability(6, 0);
    using Param = WarpPerspective::Param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    for (auto bmode : {WarpPerspective::BorderMode::REPLICATE}) {
        WarpPerspective::Param param;
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NHWC;
        checker.set_param(param);
        checker.set_epsilon(0.15).set_max_avg_error(4e-2);
        size_t n = (INT_MAX) / (512 * 512 * 3);
        checker.execs(
                {{n + 1, 512, 512, 3}, {n + 1, 3, 3}, {n + 1, 25, 25, 3}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_FORWARD_FP16) {
    using Param = WarpPerspective::Param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float16());
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        WarpPerspective::Param param;
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NHWC;
        checker.set_param(param);
        checker.set_epsilon(2.1).set_max_avg_error(4e-2);
        checker.execs({{2, 10, 11, 3}, {2, 3, 3}, {2, 11, 12, 3}});
        checker.execs({{2200, 10, 11, 3}, {2200, 3, 3}, {2200, 11, 12, 3}});
        checker.set_epsilon(1e-3);
        checker.execs({{20, 10, 11, 123}, {20, 3, 3}, {20, 11, 12, 123}});

        param.format = Param::Format::NCHW;
        checker.set_param(param);
        checker.execs({{2, 3, 10, 11}, {2, 3, 3}, {2, 3, 11, 12}});
        checker.execs({{20, 3000, 10, 11}, {20, 3, 3}, {20, 3000, 11, 12}});
        checker.execs({{22000, 3, 10, 11}, {22000, 3, 3}, {22000, 3, 11, 12}});
    }
    // nan case
    NanMatRNG rng_nan;
    UniformFloatRNG rng_zero(0, 0);
    for (auto rng : std::vector<RNG*>{&rng_nan, &rng_zero}) {
        param::WarpPerspective param;
        param.bmode = param::WarpPerspective::BorderMode::CONSTANT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_rng(1, rng);
        param.border_val = 1.737;
        checker.set_param(param);
        // no invalid mem access is enough; no need to check value
        checker.set_expect_exec_fail([]() {});
        checker.exec({{1000, 2, 10, 11}, {1000, 3, 3}, {1000, 2, 12, 13}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_NCHW4) {
    using Param = WarpPerspective::Param;
    WarpPerspective::Param param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    checker.set_dtype(2, dtype::QuantizedS8(0.1f));
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NCHW4;
        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{2, 1, 10, 11, 4}, {2, 3, 3}, {2, 1, 11, 12, 4}});
        checker.execs({{20, 300, 10, 11, 4}, {20, 3, 3}, {20, 300, 11, 12, 4}});
        checker.execs(
                {{2200, 3, 10, 11, 4}, {2200, 3, 3}, {2200, 3, 11, 12, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 25, 51, 4}});
        checker.execs({{1, 1, 25, 510, 4}, {1, 3, 3}, {1, 1, 25, 25, 4}});
        checker.execs({{1, 1, 25, 25, 4}, {1, 3, 3}, {1, 1, 51, 51, 4}});
        checker.execs({{1, 1, 51, 51, 4}, {1, 3, 3}, {1, 1, 25, 25, 4}});
    }
    {
        Checker<WarpPerspective, WarpPerspectiveMatIdxProxy> checker(
                handle_cuda());
        constexpr int N_SRC = 5;
        UniformIntRNG mat_idx_rng{0, N_SRC - 1};
        checker.set_dtype(0, dtype::QuantizedS8(0.1f));
        checker.set_rng(1, &rng);
        checker.set_dtype(2, dtype::Int32());
        checker.set_rng(2, &mat_idx_rng);
        checker.set_dtype(3, dtype::QuantizedS8(0.1f));
        param.bmode = WarpPerspective::Param::BorderMode::REFLECT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs(
                {{N_SRC, 3, 10, 11, 4}, {2, 3, 3}, {2}, {2, 3, 11, 12, 4}});
        checker.execs({{N_SRC, 14, 17, 13, 4},
                       {123, 3, 3},
                       {123},
                       {123, 14, 16, 15, 4}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_NCHW_NCHW4_IC_SMALL) {
    using Param = WarpPerspective::Param;
    WarpPerspective::Param param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;

    param.format = Param::Format::NCHW_NCHW4_IC_SMALL;

    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Quantized8Asymm(0.1f, 128));
    checker.set_dtype(2, dtype::QuantizedS8(0.1f));
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{2, 3, 10, 11}, {2, 3, 3}, {2, 1, 11, 12, 4}});
        checker.execs({{1, 3, 25, 510}, {1, 3, 3}, {1, 1, 25, 25, 4}});
        checker.execs({{1, 3, 25, 25}, {1, 3, 3}, {1, 1, 51, 51, 4}});
        checker.execs({{1, 3, 51, 51}, {1, 3, 3}, {1, 1, 25, 25, 4}});
    }
    {
        Checker<WarpPerspective, WarpPerspectiveMatIdxProxy> checker(
                handle_cuda());
        constexpr int N_SRC = 5;
        UniformIntRNG mat_idx_rng{0, N_SRC - 1};
        checker.set_dtype(0, dtype::Quantized8Asymm(0.1f, 128));
        checker.set_rng(1, &rng);
        checker.set_dtype(2, dtype::Int32());
        checker.set_rng(2, &mat_idx_rng);
        checker.set_dtype(3, dtype::QuantizedS8(0.1f));
        param.bmode = WarpPerspective::Param::BorderMode::REFLECT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{N_SRC, 3, 10, 11}, {2, 3, 3}, {2}, {2, 1, 11, 12, 4}});
        checker.execs(
                {{N_SRC, 3, 17, 13}, {123, 3, 3}, {123}, {123, 1, 16, 15, 4}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_NHWC_NCHW4_IC_SMALL) {
    using Param = WarpPerspective::Param;
    WarpPerspective::Param param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;

    param.format = Param::Format::NHWC_NCHW4_IC_SMALL;

    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(2, dtype::QuantizedS8(1.f));
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{2, 10, 11, 3}, {2, 3, 3}, {2, 1, 11, 12, 4}});
        checker.execs({{1, 25, 510, 3}, {1, 3, 3}, {1, 1, 25, 25, 4}});
        checker.execs({{1, 25, 25, 3}, {1, 3, 3}, {1, 1, 51, 51, 4}});
        checker.execs({{1, 51, 51, 3}, {1, 3, 3}, {1, 1, 25, 25, 4}});
    }
    {
        Checker<WarpPerspective, WarpPerspectiveMatIdxProxy> checker(
                handle_cuda());
        constexpr int N_SRC = 5;
        UniformIntRNG mat_idx_rng{0, N_SRC - 1};
        checker.set_dtype(0, dtype::Uint8());
        checker.set_rng(1, &rng);
        checker.set_dtype(2, dtype::Int32());
        checker.set_rng(2, &mat_idx_rng);
        checker.set_dtype(3, dtype::QuantizedS8(1.f));
        param.bmode = WarpPerspective::Param::BorderMode::REFLECT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{N_SRC, 10, 11, 3}, {2, 3, 3}, {2}, {2, 1, 11, 12, 4}});
        checker.execs(
                {{N_SRC, 17, 13, 3}, {123, 3, 3}, {123}, {123, 1, 16, 15, 4}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_NHWC_NCHW) {
    using Param = WarpPerspective::Param;
    WarpPerspective::Param param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;

    param.format = Param::Format::NHWC_NCHW;

    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::Uint8());
    checker.set_dtype(2, dtype::Float32());
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{2, 10, 11, 3}, {2, 3, 3}, {2, 3, 11, 12}});
        checker.execs({{1, 25, 510, 3}, {1, 3, 3}, {1, 3, 25, 25}});
        checker.execs({{1, 25, 25, 3}, {1, 3, 3}, {1, 3, 51, 51}});
        checker.execs({{1, 51, 51, 3}, {1, 3, 3}, {1, 3, 25, 25}});
    }
    {
        Checker<WarpPerspective, WarpPerspectiveMatIdxProxy> checker(
                handle_cuda());
        constexpr int N_SRC = 5;
        UniformIntRNG mat_idx_rng{0, N_SRC - 1};
        checker.set_dtype(0, dtype::Uint8());
        checker.set_rng(1, &rng);
        checker.set_dtype(2, dtype::Int32());
        checker.set_rng(2, &mat_idx_rng);
        checker.set_dtype(3, dtype::Float32());
        param.bmode = WarpPerspective::Param::BorderMode::REFLECT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{N_SRC, 10, 11, 3}, {2, 3, 3}, {2}, {2, 3, 11, 12}});
        checker.execs(
                {{N_SRC, 17, 13, 3}, {123, 3, 3}, {123}, {123, 3, 16, 15}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_FORWARD_NCHW_INT8) {
    warp_perspective::run_int8_test(handle_cuda());
}

TEST_F(CUDA, WARP_PERSPECTIVE_BACKWARD_DATA) {
    Checker<WarpPerspectiveBackwardData> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(0, &rng);
    for (int i = 0; i < 1; ++i) {
        for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                           WarpPerspective::BorderMode::REFLECT,
                           WarpPerspective::BorderMode::REPLICATE,
                           WarpPerspective::BorderMode::CONSTANT}) {
            WarpPerspective::Param param;
            param.border_val = 0.3f;
            param.bmode = bmode;
            param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
            checker.set_param(param);
            checker.execs({{2, 3, 3}, {2, 3, 11, 12}, {2, 3, 10, 11}});
            checker.execs(
                    {{22000, 3, 3}, {22000, 3, 11, 12}, {22000, 3, 10, 11}});
        }
    }
    // nan case
    NanMatRNG rng_nan;
    UniformFloatRNG rng_zero(0, 0);
    for (auto rng : std::vector<RNG*>{&rng_nan, &rng_zero}) {
        param::WarpPerspective param;
        param.bmode = param::WarpPerspective::BorderMode::CONSTANT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_rng(0, rng);
        param.border_val = 1.737;
        checker.set_param(param);
        // no invalid mem access is enough; no need to check value
        checker.set_expect_exec_fail([]() {});
        checker.exec({{1000, 3, 3}, {1000, 2, 10, 11}, {1000, 2, 12, 13}});
    }

    {
        Checker<WarpPerspectiveBackwardData, WarpPerspectiveMatIdxProxy>
                checker(handle_cuda());
        constexpr int N_SRC = 5;
        UniformIntRNG mat_idx_rng{0, N_SRC - 1};
        checker.set_rng(0, &rng);
        checker.set_dtype(1, dtype::Int32());
        checker.set_rng(1, &mat_idx_rng);
        param::WarpPerspective param;
        param.bmode = param::WarpPerspective::BorderMode::REFLECT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{2, 3, 3}, {2}, {2, 12, 11, 12}, {N_SRC, 12, 10, 11}});
        checker.execs(
                {{123, 3, 3}, {123}, {123, 56, 16, 15}, {N_SRC, 56, 17, 13}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_BACKWARD_MAT) {
    Checker<WarpPerspectiveBackwardMat> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    for (int i = 0; i < 1; ++i) {
        for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                           WarpPerspective::BorderMode::REFLECT,
                           WarpPerspective::BorderMode::REPLICATE,
                           WarpPerspective::BorderMode::CONSTANT}) {
            WarpPerspective::Param param;
            param.border_val = 0.3f;
            param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
            param.bmode = bmode;
            checker.set_param(param);
            checker.set_epsilon(1e-2);
            checker.execs({{1000, 3, 11, 12},
                           {1000, 3, 3},
                           {1000, 3, 10, 11},
                           {1000, 3, 3}});
        }
    }
    // nan case
    NanMatRNG rng_nan;
    UniformFloatRNG rng_zero(0, 0);
    for (auto rng : std::vector<RNG*>{&rng_nan, &rng_zero}) {
        param::WarpPerspective param;
        param.bmode = param::WarpPerspective::BorderMode::CONSTANT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_rng(1, rng);
        param.border_val = 1.737;
        checker.set_param(param);
        // no invalid mem access is enough; no need to check value
        checker.set_expect_exec_fail([]() {});
        checker.exec({{1000, 2, 10, 11},
                      {1000, 3, 3},
                      {1000, 2, 12, 13},
                      {1000, 3, 3}});
    }
    {
        Checker<WarpPerspectiveBackwardMat, WarpPerspectiveMatIdxProxy> checker(
                handle_cuda());
        constexpr int N_SRC = 5;
        UniformIntRNG mat_idx_rng{0, N_SRC - 1};
        checker.set_rng(1, &rng);
        checker.set_dtype(2, dtype::Int32());
        checker.set_rng(2, &mat_idx_rng);
        param::WarpPerspective param;
        param.bmode = param::WarpPerspective::BorderMode::REFLECT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        checker.set_param(param);
        checker.set_epsilon(1 + 1e-3);
        checker.execs({{N_SRC, 12, 10, 11},
                       {2, 3, 3},
                       {2},
                       {2, 12, 11, 12},
                       {2, 3, 3}});
        checker.execs({{N_SRC, 56, 17, 13},
                       {123, 3, 3},
                       {123},
                       {123, 56, 16, 15},
                       {123, 3, 3}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_FORWARD_BFLOAT16) {
    using Param = WarpPerspective::Param;
    Checker<WarpPerspectiveForward> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::BFloat16())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::BFloat16());
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        WarpPerspective::Param param;
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NHWC;
        checker.set_param(param);
        checker.set_epsilon(2.1).set_max_avg_error(4e-2);
        checker.execs({{2, 10, 11, 3}, {2, 3, 3}, {2, 11, 12, 3}});

        param.format = Param::Format::NCHW;
        checker.set_param(param);
        checker.execs({{2, 3, 10, 11}, {2, 3, 3}, {2, 3, 11, 12}});
        checker.execs({{20, 3000, 10, 11}, {20, 3, 3}, {20, 3000, 11, 12}});
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_BACKWARD_DATA_BFLOAT16) {
    Checker<WarpPerspectiveBackwardData> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(0, &rng)
            .set_epsilon(1e-1)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::BFloat16())
            .set_dtype(2, dtype::BFloat16());
    for (int i = 0; i < 1; ++i) {
        for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                           WarpPerspective::BorderMode::REFLECT,
                           WarpPerspective::BorderMode::REPLICATE,
                           WarpPerspective::BorderMode::CONSTANT}) {
            WarpPerspective::Param param;
            param.border_val = 0.3f;
            param.bmode = bmode;
            param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
            checker.set_param(param);
            checker.execs({{2, 3, 3}, {2, 3, 11, 12}, {2, 3, 10, 11}});
        }
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_BACKWARD_MAT_BFLOAT16) {
    Checker<WarpPerspectiveBackwardMat> checker(handle_cuda());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng)
            .set_epsilon(1e-2)
            .set_dtype(0, dtype::BFloat16())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::BFloat16())
            .set_dtype(3, dtype::Float32());
    for (int i = 0; i < 1; ++i) {
        for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                           WarpPerspective::BorderMode::REFLECT,
                           WarpPerspective::BorderMode::REPLICATE,
                           WarpPerspective::BorderMode::CONSTANT}) {
            WarpPerspective::Param param;
            param.border_val = 0.3f;
            param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
            param.bmode = bmode;
            checker.set_param(param);
            checker.execs({{1000, 3, 11, 12},
                           {1000, 3, 3},
                           {1000, 3, 10, 11},
                           {1000, 3, 3}});
        }
    }
}

TEST_F(CUDA, WARP_PERSPECTIVE_MAT_IDX) {
    warp_perspective::run_mat_idx_test(handle_cuda());
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(CUDA, BENCHMARK_WARP_PERSPECTIVE_NCHW4) {
    Benchmarker<WarpPerspective> benchmarker(handle_cuda());
    using Param = param::WarpPerspective;
    WarpPerspectiveMatRNG rng;
    benchmarker.set_rng(1, &rng);
    Param param;

    auto run = [&benchmarker, &param](const megdnn::TensorShapeArray& shapes) {
        benchmarker.set_param(param);
        auto used = benchmarker.execs(shapes);
        printf("format %s, run %s->%s used: %f ms %f GBPS %f Gflops\n",
               param.format == Param::Format::NCHW ? "NCHW" : "NCHW4",
               shapes[0].to_string().c_str(), shapes[2].to_string().c_str(),
               used,
               shapes[2].total_nr_elems() *
                       (4.f + 1.f + shapes[1].total_nr_elems()) /
                       (1024 * 1024 * 1024) / used * 1e3,
               shapes[2].total_nr_elems() * (4.f + 3.f) / (1024 * 1024 * 1024) /
                       used * 1e3);
    };
    param.format = Param::Format::NCHW;
    benchmarker.set_dtype(0, dtype::Int8());
    benchmarker.set_dtype(2, dtype::Int8());
    run({TensorShape{1, 100, 256, 256}, {1, 3, 3}, {1, 100, 256, 5120}});
    run({TensorShape{1, 100, 256, 5120}, {1, 3, 3}, {1, 100, 256, 256}});
    run({TensorShape{1, 100, 256, 256}, {1, 3, 3}, {1, 100, 512, 512}});
    run({TensorShape{1, 100, 512, 512}, {1, 3, 3}, {1, 100, 256, 256}});

    param.format = Param::Format::NCHW4;
    benchmarker.set_dtype(0, dtype::QuantizedS8(1.0f));
    benchmarker.set_dtype(2, dtype::QuantizedS8(1.0f));
    run({TensorShape{1, 25, 256, 256, 4}, {1, 3, 3}, {1, 25, 256, 5120, 4}});
    run({TensorShape{1, 25, 256, 5120, 4}, {1, 3, 3}, {1, 25, 256, 256, 4}});
    run({TensorShape{1, 25, 256, 256, 4}, {1, 3, 3}, {1, 25, 512, 512, 4}});
    run({TensorShape{1, 25, 512, 512, 4}, {1, 3, 3}, {1, 25, 256, 256, 4}});
}

#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
