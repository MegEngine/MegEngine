/**
 * \file dnn/test/arm_common/reduce.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"

using namespace megdnn;
using namespace test;

TEST_F(ARM_COMMON, REDUCE) {
    using Param = Reduce::Param;
    using Mode = Param::Mode;
    Checker<Reduce> checker(handle());
    UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
    checker.set_rng(0, &rng);
    struct Config {
        Param param;
        DType dtype;
        TensorShape shape;
        Config(Param param, DType dtype, TensorShape shape)
                : param(param), dtype(dtype), shape(shape) {}
    };
    std::vector<Config> configs;
    for (auto mode : {Mode::MEAN, Mode::MAX, Mode::MIN})
        for (auto dtype : std::vector<DType>{
                     dtype::Float32(), dtype::Float16(),
                     dtype::QuantizedS8(1.3f),
                     dtype::Quantized8Asymm(1.3f, static_cast<uint8_t>(3))})
            for (int32_t axis : {0, 1, 2}) {
                for (size_t A : {1, 3, 5}) {
                    for (size_t B : {4, 6, 9, 16, 33, 45}) {
                        for (size_t C : {4, 6, 9, 16, 33, 45}) {
                            TensorShape shape{A, B, C};
                            Param param(mode, axis);
                            Config config(param, dtype, shape);
                            configs.push_back(config);
                        }
                    }
                }
            }
    for (auto&& config : configs) {
        auto&& dtype = config.dtype;
        auto&& param = config.param;
        auto&& shape = config.shape;

        checker.set_dtype(0, dtype).set_param(param).execs({shape, {}});
    }
    configs.clear();
    for (auto mode : {Mode::SUM, Mode::PRODUCT, Mode::SUM_SQR})
        for (auto dtype :
             std::vector<DType>{dtype::Float32(), dtype::Float16()})
            for (int32_t axis : {0, 1, 2}) {
                for (size_t A : {1, 3, 5}) {
                    for (size_t B : {4, 6, 9, 16, 33, 45}) {
                        for (size_t C : {4, 6, 9, 16, 33, 45}) {
                            TensorShape shape{A, B, C};
                            Param param(mode, axis);
                            Config config(param, dtype, shape);
                            configs.push_back(config);
                        }
                    }
                }
            }

    UniformFloatRNG rng_float(-2, 2);
    checker.set_rng(0, &rng_float);
    checker.set_epsilon(1e-1);
    for (auto&& config : configs) {
        auto&& dtype = config.dtype;
        auto&& param = config.param;
        auto&& shape = config.shape;
        if(dtype == dtype::Float16())
            checker.set_epsilon(1e-1);
        else
            checker.set_epsilon(1e-3);

        checker.set_dtype(0, dtype).set_param(param).execs({shape, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ARM_COMMON, BENCHMARK_REDUCE) {
    auto run = [&](size_t A, size_t B, size_t C, size_t axis,
            megdnn::param::Reduce::Mode mode, megdnn::DType& dtype) {
        auto handle_fallback = create_cpu_handle(1);
        Benchmarker<Reduce> benchmarker(handle());
        Benchmarker<Reduce> benchmarker_fallback(handle_fallback.get());
        benchmarker_fallback.set_display(false);
        benchmarker.set_display(false);
        constexpr size_t RUNS = 50;
        benchmarker_fallback.set_times(RUNS);
        benchmarker.set_times(RUNS);
        param::Reduce param;
        param.axis = axis;
        param.mode = mode;
        benchmarker.set_param(param);
        benchmarker_fallback.set_param(param);

        TensorLayout src({A, B, C}, dtype), dst;
        auto opr = handle()->create_operator<Reduce>();
        opr->param() = param;
        opr->deduce_layout(src, dst);

        auto bench = [&](const char* msg) {
                auto cur = benchmarker.execs({src, dst}) / RUNS;
                auto fallback =
                        benchmarker_fallback.execs({src, dst}) / RUNS;
                float computation =
                        src.total_nr_elems() / 1024.0 / 1024.0 / 1024.0 * 1e3;
                printf("run %s->%s %s: fallback: %fms %fGflops "
                       "cur: %fms %fGflops speedup=%f\n",
                       src.to_string().c_str(), dst.to_string().c_str(), msg,
                       fallback, computation / fallback, cur, computation / cur,
                       fallback / cur);
        };

        benchmarker_fallback.set_dtype(0, dtype);
        benchmarker.set_dtype(0, dtype);
        bench(dtype.name());
    };

    for (auto mode : {param::Reduce::Mode::MEAN, param::Reduce::Mode::MAX,
            param::Reduce::Mode::MIN})
        for (int32_t axis : {1, 2}) {
            if (mode == param::Reduce::Mode::MEAN)
                printf("testcase mean %s\n", axis == 2 ? "c == 1" : "c > 1");
            else if (mode == param::Reduce::Mode::MAX)
                printf("testcase max %s\n", axis == 2 ? "c == 1" : "c > 1");
            else if (mode == param::Reduce::Mode::MIN)
                printf("testcase min %s\n", axis == 2 ? "c == 1" : "c > 1");
            for (auto dtype :
                 std::vector<megdnn::DType>{dtype::Float16(), dtype::Float32(),
                                            dtype::QuantizedS8(4.2f),
                                            dtype::Quantized8Asymm(3.2f, static_cast<uint8_t>(10))}) {
                run(1, 1024, 49, axis, mode, dtype);
                run(2, 10, 10000, axis, mode, dtype);
                run(2, 100, 10000, axis, mode, dtype);
                run(2, 10, 100000, axis, mode, dtype);
            }
        }
    for (auto mode : {param::Reduce::Mode::SUM, param::Reduce::Mode::PRODUCT,
                      param::Reduce::Mode::SUM_SQR})
        for (int32_t axis : {1, 2}) {
            if (mode == param::Reduce::Mode::SUM)
                printf("testcase sum %s\n", axis == 2 ? "c == 1" : "c > 1");
            else if (mode == param::Reduce::Mode::PRODUCT)
                printf("testcase product %s\n", axis == 2 ? "c == 1" : "c > 1");
            else if (mode == param::Reduce::Mode::SUM_SQR)
                printf("testcase sum SumSqr %s\n",
                       axis == 2 ? "c == 1" : "c > 1");
            for (auto dtype : std::vector<megdnn::DType>{dtype::Float16(),
                                                         dtype::Float32()}) {
                run(1, 1024, 49, axis, mode, dtype);
                run(2, 10, 10000, axis, mode, dtype);
                run(2, 100, 10000, axis, mode, dtype);
                run(2, 10, 100000, axis, mode, dtype);
            }
        }
}
#endif
// vim: syntax=cpp.doxygen
