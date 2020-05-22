/**
 * \file dnn/test/arm_common/warp_perspective.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <string>
#include <vector>
#include "test/arm_common/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/common/rng.h"

#include "test/common/warp_perspective.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON, WARP_PERSPECTIVE_CV) {
    //! Just for the format NHWC
    Checker<WarpPerspective> checker(handle());
    param::WarpPerspective param;
    class ResizeMatRNG : public RNG {
        void gen(const TensorND& tensor_) override {
            auto& gen = RandomState::generator();
            std::uniform_real_distribution<dt_float32> pdist3(1.9f, 3.1f);
            std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
            std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
            std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
            std::uniform_real_distribution<dt_float32> ndist3(-3.1f, -1.9f);
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

    using BMode = param::WarpPerspective::BorderMode;
    param.format = param::WarpPerspective::Format::NHWC;
    // add for nearest test
    param.imode = param::WarpPerspective::InterpolationMode::NEAREST;
    for (auto mode : {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT,
                      BMode::WRAP, BMode::CONSTANT}) {
        param.bmode = mode;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{10, 128, 108, 3}, {10, 3, 3}, {10, 56, 128, 3}});
    }
    // resize nan case
    UniformFloatRNG rng_zero(0, 0);
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }

    // add linear test
    param.imode = param::WarpPerspective::InterpolationMode::INTER_LINEAR;
    for (auto mode : {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT,
                      BMode::WRAP, BMode::CONSTANT}) {
        param.bmode = mode;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{10, 128, 108, 3}, {10, 3, 3}, {10, 56, 128, 3}});
    }
    // resize nan case
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }

    auto args = warp_perspective::get_cv_args();
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Uint8())
                .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .execs({arg.src, arg.trans, arg.dst});
    }
}

TEST_F(ARM_COMMON_MULTI_THREADS, WARP_PERSPECTIVE_CV) {
    //! Just for the format NHWC
    Checker<WarpPerspective> checker(handle());
    param::WarpPerspective param;
    class ResizeMatRNG : public RNG {
        void gen(const TensorND& tensor_) override {
            auto& gen = RandomState::generator();
            std::uniform_real_distribution<dt_float32> pdist3(1.9f, 3.1f);
            std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
            std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
            std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
            std::uniform_real_distribution<dt_float32> ndist3(-3.1f, -1.9f);
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

    using BMode = param::WarpPerspective::BorderMode;
    param.format = param::WarpPerspective::Format::NHWC;
    // add for nearest test
    param.imode = param::WarpPerspective::InterpolationMode::NEAREST;
    for (auto mode : {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT,
                      BMode::WRAP, BMode::CONSTANT}) {
        param.bmode = mode;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{10, 128, 108, 3}, {10, 3, 3}, {10, 56, 128, 3}});
    }
    // resize nan case
    UniformFloatRNG rng_zero(0, 0);
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }

    // add linear test
    param.imode = param::WarpPerspective::InterpolationMode::INTER_LINEAR;
    for (auto mode : {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT,
                      BMode::WRAP, BMode::CONSTANT}) {
        param.bmode = mode;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{10, 128, 108, 3}, {10, 3, 3}, {10, 56, 128, 3}});
    }
    // resize nan case
    checker.set_rng(1, &rng_zero);
    {
        param.bmode = BMode::CONSTANT;
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{1000, 2, 10, 3}, {1000, 3, 3}, {1000, 2, 12, 3}});
    }

    auto args = warp_perspective::get_cv_args();
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Uint8())
                .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .execs({arg.src, arg.trans, arg.dst});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ARM_COMMON, BENCHMARK_WARP_PERSPECTIVE_FORWARD) {
    Benchmarker<WarpPerspectiveForward> benchmarker(handle());
    auto handle_naive = create_cpu_handle(2);
    Benchmarker<WarpPerspectiveForward> benchmarker_naive(handle_naive.get());
    constexpr size_t NR_RUN = 50;

    using BMode = param::WarpPerspective::BorderMode;
    using IMode = param::WarpPerspective::InterpolationMode;

    WarpPerspective::Param param;
    param.border_val = 0.3f;
    param.format = param::WarpPerspective::Format::NHWC;

    auto run = [&](size_t N, size_t C, size_t IH, size_t IW, size_t OH,
                   size_t OW, size_t scale) {
        printf("src={%zu, %zu, %zu, %zu}, dst={%zu, %zu, %zu, %zu}\n", N, IH,
               IW, C, N, OH, OW, C);
        auto time_ms =
                benchmarker.exec({{N, IH, IW, C}, {N, 3, 3}, {N, OH, OW, C}}) /
                NR_RUN;
        auto time_naive_ms =
                benchmarker_naive.exec(
                        {{N, IH, IW, C}, {N, 3, 3}, {N, OH, OW, C}}) /
                NR_RUN;
        auto bandwidth = N * C * (scale * OH * OW) * dtype::Float32().size();
        printf("aarch64: %.3f, perf: %.3f GBPS naive: %.3f, perf %.3f GBPS "
               "speedup: %f\n",
               time_ms, bandwidth / time_ms / 1e6, time_naive_ms,
               bandwidth / time_naive_ms / 1e6, time_naive_ms / time_ms);
    };

    std::vector<std::string> bmodestringmap = {
            "REPLICATE", "REFLECT", "REFLECT_101", "WARP", "CONSTANT"};

    std::vector<std::string> imodestringmap = {"NEAREST", "INTER_LINEAR"};
    size_t scales[2] = {2, 5};

    for (auto imode : {IMode::NEAREST, IMode::INTER_LINEAR}) {
        for (auto bmode : {BMode::REFLECT_101, BMode::REPLICATE, BMode::REFLECT,
                           BMode::WRAP, BMode::CONSTANT}) {
            param.imode = imode;
            param.bmode = bmode;
            benchmarker.set_param(param).set_display(false).set_times(NR_RUN);
            benchmarker_naive.set_param(param).set_display(false).set_times(
                    NR_RUN);
            size_t scale = scales[(int)imode];
            printf("\n\n\n warpperspective InterpolationMode::%s "
                   "BorderMode::%s start\n",
                   imodestringmap[(int)imode].c_str(),
                   bmodestringmap[(int)bmode].c_str());
            for (auto&& shape :
                 std::vector<std::pair<size_t, size_t>>{{700, 490},
                                                        {500, 334},
                                                        {472, 342},
                                                        {448, 306},
                                                        {626, 412},
                                                        {140, 144},
                                                        {120, 128},
                                                        {180, 176}}) {
                for (size_t ch : {1, 2, 3}) {
                    run(1, ch, shape.first, shape.second, 256, 256, scale);
                }
            }
        }
    }
}

namespace {
void benchmark_impl(const typename WarpPerspective::Param& param,
                    std::vector<SmallVector<TensorShape>> shapes, size_t RUNS,
                    TaskExecutorConfig&& multi_thread_config,
                    TaskExecutorConfig&& single_thread_config) {
    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle =
                create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker =
                Benchmarker<WarpPerspective>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS).set_display(false).set_param(param);
        for (auto shape : shapes) {
            multi_thread_times.push_back(benchmarker.exec(shape) / RUNS);
        }
    }
    {
        auto single_thread_handle =
                create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker =
                Benchmarker<WarpPerspective>(single_thread_handle.get());
        benchmarker.set_times(RUNS).set_display(false).set_param(param);
        for (auto shape : shapes) {
            single_thread_times.push_back(benchmarker.exec(shape) / RUNS);
        }
    }
    printf("Benchmark : Multi threads  %zu, ", multi_thread_config.nr_thread);
    printf("core_ids:");
    for (size_t i = 0; i < multi_thread_config.affinity_core_set.size(); i++) {
        printf("%zu ", multi_thread_config.affinity_core_set[i]);
    }
    printf(", Single thread core_id %zu\n",
           single_thread_config.affinity_core_set[0]);
    for (size_t i = 0; i < shapes.size(); i++) {
        auto shape = shapes[i];
        printf("Case: ");
        for (auto sh : shape)
            printf("%s ", sh.to_string().c_str());
        printf("%zu threads time: %f,\n single thread time: "
               "%f. spead up = %f, speedup/cores=%f\n",
               multi_thread_config.nr_thread, multi_thread_times[i],
               single_thread_times[i],
               single_thread_times[i] / multi_thread_times[i],
               single_thread_times[i] / multi_thread_times[i] /
                       multi_thread_config.nr_thread);
    }
}
}  // namespace

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_WARP_PERSPECTIVE) {
    constexpr size_t RUNS = 50;
    using BMode = param::WarpPerspective::BorderMode;
    using IMode = param::WarpPerspective::InterpolationMode;

    WarpPerspective::Param param;
    param.border_val = 0.3f;
    param.format = param::WarpPerspective::Format::NHWC;
    param.imode = IMode::INTER_LINEAR;
    param.bmode = BMode::REPLICATE;

    std::vector<SmallVector<TensorShape>> shapes;
    auto bench_case = [&](size_t N, size_t H, size_t W, size_t C) {
        SmallVector<TensorShape> shape{
                {N, H, W, C}, {N, 3, 3}, {N, 224, 224, C}};
        shapes.push_back(shape);
    };
    bench_case(1, 700, 490, 1);
    bench_case(1, 700, 490, 2);
    bench_case(1, 700, 490, 3);
    bench_case(1, 500, 334, 1);
    bench_case(1, 500, 334, 2);
    bench_case(1, 500, 334, 3);
    bench_case(1, 140, 144, 1);
    bench_case(1, 140, 144, 2);
    bench_case(1, 140, 114, 3);

    printf("Benchmark warp perspective\n");
    benchmark_impl(param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {4}});
    benchmark_impl(param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {7}});
    benchmark_impl(param, shapes, RUNS, {2, {4, 5}}, {1, {4}});
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
