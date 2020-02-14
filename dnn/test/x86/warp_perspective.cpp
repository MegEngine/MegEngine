/**
 * \file dnn/test/x86/warp_perspective.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/x86/fixture.h"

#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/random_state.h"
#include "test/common/rng.h"
#include "test/common/warp_perspective.h"
#include "test/common/warp_affine.h"

namespace megdnn {
namespace test {

TEST_F(X86, WARP_PERSPECTIVE_MAT_IDX) {
    warp_perspective::run_mat_idx_test(handle());
}

TEST_F(X86_MULTI_THREADS, WARP_PERSPECTIVE_MAT_IDX) {
    warp_perspective::run_mat_idx_test(handle());
}

TEST_F(X86_MULTI_THREADS, WARP_AFFINE_CV)
{
    using namespace warp_affine;
    std::vector<TestArg> args = get_cv_args();
    Checker<WarpAffine> checker(handle());

    for (auto &&arg : args) {
        checker.set_param(arg.param)
            .set_epsilon(1 + 1e-3)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Uint8())
            .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .execs({arg.src, arg.trans, arg.dst});
    }

}

#if MEGDNN_WITH_BENCHMARK
namespace {
template<typename Opr>
void benchmark_impl(const typename Opr::Param& param,
                    std::vector<SmallVector<TensorShape>> shapes, size_t RUNS,
                    TaskExecutorConfig&& multi_thread_config,
                    TaskExecutorConfig&& single_thread_config) {
    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle =
                create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker = Benchmarker<Opr>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS).set_display(false).set_param(param);
        for (auto shape : shapes) {
            multi_thread_times.push_back(benchmarker.exec(shape) / RUNS);
        }
    }
    {
        auto single_thread_handle =
                create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker = Benchmarker<Opr>(single_thread_handle.get());
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

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_WARP_PERSPECTIVE) {
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
    benchmark_impl<WarpPerspective>(param, shapes, RUNS, {4, {4, 5, 6, 7}},
                                    {1, {4}});
    benchmark_impl<WarpPerspective>(param, shapes, RUNS, {4, {4, 5, 6, 7}},
                                    {1, {7}});
    benchmark_impl<WarpPerspective>(param, shapes, RUNS, {2, {4, 5}}, {1, {4}});
}

TEST_F(X86_BENCHMARK_MULTI_THREADS, BENCHMARK_WARP_AFFINE) {
    constexpr size_t RUNS = 50;
    using BMode = param::WarpAffine::BorderMode;
    using IMode = param::WarpAffine::InterpolationMode;

    WarpAffine::Param param;
    param.border_val = 0.3f;
    param.format = param::WarpAffine::Format::NHWC;
    param.imode = IMode::LINEAR;
    param.border_mode = BMode::BORDER_CONSTANT;

    std::vector<SmallVector<TensorShape>> shapes;
    auto bench_case = [&](size_t N, size_t H, size_t W, size_t C) {
        SmallVector<TensorShape> shape{
                {N, H, W, C}, {N, 2, 3}, {N, 224, 224, C}};
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
    benchmark_impl<WarpAffine>(param, shapes, RUNS, {4, {4, 5, 6, 7}},
                                    {1, {4}});
    benchmark_impl<WarpAffine>(param, shapes, RUNS, {4, {4, 5, 6, 7}},
                                    {1, {7}});
    benchmark_impl<WarpAffine>(param, shapes, RUNS, {2, {4, 5}}, {1, {4}});
}

#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
