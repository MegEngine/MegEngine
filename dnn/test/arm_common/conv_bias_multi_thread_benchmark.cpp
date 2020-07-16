/**
 * \file dnn/test/arm_common/conv_bias_multi_thread_benchmark.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/arm_common/fixture.h"
#include "test/common/benchmarker.h"
#include "test/common/conv_bias.h"

using namespace megdnn;
using namespace test;
using namespace conv_bias;
#if MEGDNN_WITH_BENCHMARK
namespace {
void benchmark_impl(const param::ConvBias param,
                    std::vector<std::pair<SmallVector<TensorShape>, float>>&
                            shapes_and_computation,
                    const std::string algo_name, size_t RUNS,
                    TaskExecutorConfig&& multi_thread_config,
                    TaskExecutorConfig&& single_thread_config,
                    std::vector<DType>& data_type) {
    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle =
                create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_param(param)
                .set_dtype(0, data_type[0])
                .set_dtype(1, data_type[1])
                .set_dtype(2, data_type[2])
                .set_dtype(4, data_type[3])
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(
                                algo_name.c_str()));
        for (auto shape : shapes_and_computation) {
            multi_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    {
        auto single_thread_handle =
                create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker = Benchmarker<ConvBias>(single_thread_handle.get());
        benchmarker.set_times(RUNS)
                .set_display(false)
                .set_param(param)
                .set_dtype(0, data_type[0])
                .set_dtype(1, data_type[1])
                .set_dtype(2, data_type[2])
                .set_dtype(4, data_type[3])
                .set_before_exec_callback(
                        conv_bias::ConvBiasAlgoChecker<ConvBias>(
                                algo_name.c_str()));
        for (auto shape : shapes_and_computation) {
            single_thread_times.push_back(benchmarker.exec(shape.first) / RUNS);
        }
    }
    printf("Benchmark : Multi threads  %zu, ", multi_thread_config.nr_thread);
    printf("core_ids:");
    for (size_t i = 0; i < multi_thread_config.affinity_core_set.size(); i++) {
        printf("%zu ", multi_thread_config.affinity_core_set[i]);
    }
    printf(", Single thread core_id %zu\n",
           single_thread_config.affinity_core_set[0]);
    for (size_t i = 0; i < shapes_and_computation.size(); i++) {
        auto shapes = shapes_and_computation[i];
        printf("Bench case: ");
        for (auto&& shape : shapes.first) {
            printf("%s ", shape.to_string().c_str());
        }
        float computations = shapes.second;
        printf("%zu threads gflops: %f,\n single thread gflops: "
               "%f. spead up = %f, speedup/cores=%f\n",
               multi_thread_config.nr_thread,
               computations / multi_thread_times[i],
               computations / single_thread_times[i],
               single_thread_times[i] / multi_thread_times[i],
               single_thread_times[i] / multi_thread_times[i] /
                       multi_thread_config.nr_thread);
    }
}
}  // namespace

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_DIRECTF32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 32);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 32);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 32);
    bench_case(1, 32, 32, 80, 80, 3, 4);
    bench_case(1, 32, 32, 80, 80, 3, 32);

    std::string algo_name = "F32DIRECT";
    printf("Benchmark F32DIRECT_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "F32DIRECT";
    printf("Benchmark F32DIRECT_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_DIRECTF32_STR1) {
    constexpr size_t RUNS = 50;
    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 32);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 32);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 32);
    bench_case(1, 32, 32, 80, 80, 3, 4);
    bench_case(1, 32, 32, 80, 80, 3, 32);

    std::string algo_name = "F32STRD1";
    printf("Benchmark F32STRD1_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "F32STRD1";
    printf("Benchmark F32STRD1_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_DIRECTF32_STR2) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 2);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 2);

    std::string algo_name = "F32STRD2";
    printf("Benchmark F32STRD2_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "F32STRD2";
    printf("Benchmark F32STRD2_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 2);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_DIRECTF16) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 32);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 32);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 32);
    bench_case(1, 32, 32, 80, 80, 3, 4);
    bench_case(1, 32, 32, 80, 80, 3, 32);

    std::string algo_name = "F16DIRECT";
    printf("Benchmark F16DIRECT_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Float16(), dtype::Float16(),
                                    dtype::Float16(), dtype::Float16()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "F16DIRECT";
    printf("Benchmark F16DIRECT_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_DIRECTF16_STR1) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 32);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 32);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 32);
    bench_case(1, 32, 32, 80, 80, 3, 4);
    bench_case(1, 32, 32, 80, 80, 3, 32);

    std::string algo_name = "F16STRD1";
    printf("Benchmark F16STRD1_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Float16(), dtype::Float16(),
                                    dtype::Float16(), dtype::Float16()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "F16STRD1";
    printf("Benchmark F16STRD1_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
#endif
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_DIRECT_INT8x8x16) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 32);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 32);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 32);
    bench_case(1, 32, 32, 80, 80, 3, 4);
    bench_case(1, 32, 32, 80, 80, 3, 32);

    std::string algo_name = "I8816DIRECT";
    printf("Benchmark I8816DIRECT_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int16(), dtype::Int16()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "I8816DIRECT";
    printf("Benchmark I8816DIRECT_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_DIRECT_INT8x8x16_STR2) {
    constexpr size_t RUNS = 50;
    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, (H + 2 * P - FS) / S + 1,
                        (W + 2 * P - FS) / S + 1};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 2);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 2);

    std::string algo_name = "I8816STRD2";
    printf("Benchmark I8816STRD2_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int16(), dtype::Int16()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "I8816STRD2";
    printf("Benchmark I8816STRD2_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 2);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_INT8_INT8_INT8_STRIDE1) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 1);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 1);

    std::string algo_name = "S8STRD1";
    printf("Benchmark S8STRD1_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "S8STRD1";
    printf("Benchmark S8STRD1_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_INT8_NCHW44) {
    constexpr size_t RUNS = 40;
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S,
                          bool is_nchw = false) {
        param::ConvBias param;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        param.pad_h = P;
        param.pad_w = P;
        param.stride_h = S;
        param.stride_w = S;
        param.sparse = param::ConvBias::Sparse::DENSE;
        param.format = param::ConvBias::Format::NCHW44;
        auto OH = (H + 2 * P - FS) / static_cast<size_t>(S) + 1;
        auto OW = (W + 2 * P - FS) / static_cast<size_t>(S) + 1;
        TensorShape src = {N, IC / 4, H, W, 4};
        TensorShape filter = {OC / 4, IC / 4, FS, FS, 4, 4};
        if (group > 1) {
            filter = {group, OC / group / 4, IC / group / 4, FS, FS, 4, 4};
            param.sparse = param::ConvBias::Sparse::GROUP;
        }
        if (is_nchw) {
            src = {N, IC, H, W};
            filter = {OC / 4, FS, FS, IC, 4};
        }
        TensorShape bias = {1, OC / 4, 1, 1, 4};
        TensorShape dst = {N, OC / 4, OH, OW, 4};

        SmallVector<TensorShape> shapes{src, filter, bias, {}, dst};
        float computations =
                (((IC / group) * FS * FS + 1) * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        std::vector<std::pair<SmallVector<TensorShape>, float>> shape_arg = {
                std::make_pair(shapes, computations)};
        benchmark_impl(param, shape_arg, ".+", RUNS, {4, {4, 5, 6, 7}},
                       {1, {7}}, data_type);
    };
    bench_case(1, 3, 64, 224, 224, 7, 1, 3, 2, true);
    bench_case(1, 64, 64, 56, 56, 3, 1, 1, 1);
    bench_case(1, 128, 128, 28, 28, 3, 1, 1, 1);
    bench_case(1, 256, 256, 14, 14, 3, 1, 1, 1);
    bench_case(1, 512, 512, 7, 7, 3, 1, 1, 1);

    bench_case(1, 64, 64, 56, 56, 3, 4, 1, 1);
    bench_case(1, 128, 128, 28, 28, 3, 4, 1, 1);
    bench_case(1, 256, 256, 14, 14, 3, 4, 1, 1);
    bench_case(1, 512, 512, 7, 7, 3, 4, 1, 1);

    bench_case(1, 4, 64, 224, 224, 7, 1, 1, 2);
    bench_case(1, 256, 128, 56, 56, 3, 1, 1, 2);
    bench_case(1, 512, 256, 28, 28, 3, 1, 1, 2);
    bench_case(1, 4, 32, 224, 224, 3, 1, 1, 2);

    bench_case(1, 256, 128, 56, 56, 3, 4, 1, 2);
    bench_case(1, 512, 256, 28, 28, 3, 4, 1, 2);
}

#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_INT8_NCHW44_DOT) {
    constexpr size_t RUNS = 40;
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S,
                          bool is_nchw = false) {
        param::ConvBias param;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        param.pad_h = P;
        param.pad_w = P;
        param.stride_h = S;
        param.stride_w = S;
        param.sparse = param::ConvBias::Sparse::DENSE;
        param.format = param::ConvBias::Format::NCHW44_DOT;
        auto OH = (H + 2 * P - FS) / static_cast<size_t>(S) + 1;
        auto OW = (W + 2 * P - FS) / static_cast<size_t>(S) + 1;
        TensorShape src = {N, IC / 4, H, W, 4};
        TensorShape filter = {OC / 4, IC / 4, FS, FS, 4, 4};
        if (group > 1) {
            filter = {group, OC / group / 4, IC / group / 4, FS, FS, 4, 4};
            param.sparse = param::ConvBias::Sparse::GROUP;
        }
        if (is_nchw) {
            src = {N, IC, H, W};
            filter = {OC / 4, FS, FS, IC, 4};
        }
        TensorShape bias = {1, OC / 4, 1, 1, 4};
        TensorShape dst = {N, OC / 4, OH, OW, 4};

        SmallVector<TensorShape> shapes{src, filter, bias, {}, dst};
        float computations =
                (((IC / group) * FS * FS + 1) * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        std::vector<std::pair<SmallVector<TensorShape>, float>> shape_arg = {
                std::make_pair(shapes, computations)};
        benchmark_impl(param, shape_arg, ".+", RUNS, {4, {4, 5, 6, 7}},
                       {1, {7}}, data_type);
    };
    bench_case(1, 64, 64, 56, 56, 3, 1, 1, 1);
    bench_case(1, 128, 128, 28, 28, 3, 1, 1, 1);
    bench_case(1, 256, 256, 14, 14, 3, 1, 1, 1);
    bench_case(1, 512, 512, 7, 7, 3, 1, 1, 1);

    bench_case(1, 64, 64, 56, 56, 3, 4, 1, 1);
    bench_case(1, 128, 128, 28, 28, 3, 4, 1, 1);
    bench_case(1, 256, 256, 14, 14, 3, 4, 1, 1);
    bench_case(1, 512, 512, 7, 7, 3, 4, 1, 1);

}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_INT8_NCHW44_DOT_S2) {
    constexpr size_t RUNS = 40;
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S,
                          bool is_nchw = false) {
        param::ConvBias param;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        param.pad_h = P;
        param.pad_w = P;
        param.stride_h = S;
        param.stride_w = S;
        param.sparse = param::ConvBias::Sparse::DENSE;
        param.format = param::ConvBias::Format::NCHW44_DOT;
        auto OH = (H + 2 * P - FS) / static_cast<size_t>(S) + 1;
        auto OW = (W + 2 * P - FS) / static_cast<size_t>(S) + 1;
        TensorShape src = {N, IC / 4, H, W, 4};
        TensorShape filter = {OC / 4, IC / 4, FS, FS, 4, 4};
        if (group > 1) {
            filter = {group, OC / group / 4, IC / group / 4, FS, FS, 4, 4};
            param.sparse = param::ConvBias::Sparse::GROUP;
        }
        if (is_nchw) {
            src = {N, IC, H, W};
            filter = {OC / 4, FS, FS, IC, 4};
        }
        TensorShape bias = {1, OC / 4, 1, 1, 4};
        TensorShape dst = {N, OC / 4, OH, OW, 4};

        SmallVector<TensorShape> shapes{src, filter, bias, {}, dst};
        float computations =
                (((IC / group) * FS * FS + 1) * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        std::vector<std::pair<SmallVector<TensorShape>, float>> shape_arg = {
                std::make_pair(shapes, computations)};
        benchmark_impl(param, shape_arg, ".+", RUNS, {4, {4, 5, 6, 7}},
                       {1, {7}}, data_type);
    };
    bench_case(1, 64, 64, 56, 56, 3, 1, 1, 2);
    bench_case(1, 64, 64, 128, 128, 3, 1, 1, 2);
    bench_case(1, 64, 64, 256, 256, 3, 1, 1, 2);
    bench_case(1, 64, 64, 156, 156, 3, 1, 1, 2);
    bench_case(1, 128, 128, 28, 28, 3, 1, 1, 2);
    bench_case(1, 256, 256, 14, 14, 3, 1, 1, 2);
    bench_case(1, 512, 512, 7, 7, 3, 1, 1, 2);

    bench_case(1, 64, 64, 56, 56, 3, 4, 1, 2);
    bench_case(1, 128, 128, 28, 28, 3, 4, 1, 2);
    bench_case(1, 256, 256, 14, 14, 3, 4, 1, 2);
    bench_case(1, 512, 512, 7, 7, 3, 4, 1, 2);

}


#endif
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_FLOAT_NCHW44) {
    constexpr size_t RUNS = 40;
    std::vector<DType> data_type = {
            dtype::Float32(), dtype::Float32(),
            dtype::Float32(), dtype::Float32()};
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S,
                          bool is_nchw = false) {
        param::ConvBias param;
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        param.pad_h = P;
        param.pad_w = P;
        param.stride_h = S;
        param.stride_w = S;
        param.sparse = param::ConvBias::Sparse::DENSE;
        param.format = param::ConvBias::Format::NCHW44;
        auto OH = (H + 2 * P - FS) / static_cast<size_t>(S) + 1;
        auto OW = (W + 2 * P - FS) / static_cast<size_t>(S) + 1;
        TensorShape src = {N, IC / 4, H, W, 4};
        TensorShape filter = {OC / 4, IC / 4, FS, FS, 4, 4};
        if (group > 1) {
            filter = {group, OC / group / 4, IC / group / 4, FS, FS, 4, 4};
            param.sparse = param::ConvBias::Sparse::GROUP;
        }
        if (is_nchw) {
            src = {N, IC, H, W};
            filter = {OC / 4, FS, FS, IC, 4};
        }
        TensorShape bias = {1, OC / 4, 1, 1, 4};
        TensorShape dst = {N, OC / 4, OH, OW, 4};

        SmallVector<TensorShape> shapes{src, filter, bias, {}, dst};
        float computations =
                (((IC / group) * FS * FS + 1) * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        std::vector<std::pair<SmallVector<TensorShape>, float>> shape_arg = {
                std::make_pair(shapes, computations)};
        benchmark_impl(param, shape_arg, ".+", RUNS, {4, {4, 5, 6, 7}},
                       {1, {7}}, data_type);
    };
    bench_case(1, 64, 64, 56, 56, 3, 1, 1, 2);
    bench_case(1, 128, 128, 28, 28, 3, 1, 1, 2);
    bench_case(1, 256, 256, 14, 14, 3, 1, 1, 2);
    bench_case(1, 512, 512, 7, 7, 3, 1, 1, 2);

    bench_case(1, 64, 64, 56, 56, 3, 4, 1, 2);
    bench_case(1, 128, 128, 28, 28, 3, 4, 1, 2);
    bench_case(1, 256, 256, 14, 14, 3, 4, 1, 2);
    bench_case(1, 512, 512, 7, 7, 3, 4, 1, 2);
    
    bench_case(1, 64, 64, 56*2, 56*2, 3, 4, 1, 2);
    bench_case(1, 128, 128, 28*2, 28*2, 3, 4, 1, 2);
    bench_case(1, 256, 256, 14*2, 14*2, 3, 4, 1, 2);
    bench_case(1, 512, 512, 7*2, 7*2, 3, 4, 1, 2);
}




TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_INT8_INT8_INT8_STRIDE2) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 2);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 2);

    std::string algo_name = "S8STRD2";
    printf("Benchmark S8STRD2_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "S8STRD2";
    printf("Benchmark S8STRD2_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 2);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_INT8_INT8_INT8_STRIDE1_WITHDOTPROD) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 1);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 1);

    std::string algo_name = "ARMDOTS8STRD1";
    printf("Benchmark ARMDOTS8STRD1_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "ARMDOTS8STRD1";
    printf("Benchmark ARMDOTS8STRD1_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_INT8_INT8_INT8_STRIDE2_WITHDOTPROD) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 2);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 2);

    std::string algo_name = "ARMDOTS8STRD2";
    printf("Benchmark ARMDOTS8STRD2_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "ARMDOTS8STRD2";
    printf("Benchmark ARMDOTS8STRD2_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 2);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
#endif

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_QUINT8_QUINT8_QUINT8_STRIDE1) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 1);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 1);

    std::string algo_name = "QU8STRD1";
    printf("Benchmark QU8STRD1_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Quantized8Asymm(0.2f, 100),
                                    dtype::Quantized8Asymm(0.2f, 120),
                                    dtype::QuantizedS32(0.04f),
                                    dtype::Quantized8Asymm(1.4f, 110)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "QU8STRD1";
    printf("Benchmark QU8STRD1_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_QUINT8_QUINT8_QUINT8_STRIDE2) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 2);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 2);

    std::string algo_name = "QU8STRD2";
    printf("Benchmark QU8STRD2_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Quantized8Asymm(0.2f, 100),
                                    dtype::Quantized8Asymm(0.2f, 120),
                                    dtype::QuantizedS32(0.04f),
                                    dtype::Quantized8Asymm(1.4f, 110)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "QU8STRD2";
    printf("Benchmark QU8STRD2_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 2);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 2);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 2);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 2);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_QUINT8_QUINT8_QUINT8_STRIDE1_WITHDOTPROD) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, (H + 2 * P - FS) / S + 1,
                        (W + 2 * P - FS) / S + 1};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4, 1, 1);
    bench_case(1, 32, 32, 200, 200, 3, 32, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 4, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 32, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 4, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 32, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 4, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 32, 1, 1);

    std::string algo_name = "ARMDOTU8STRD1";
    printf("Benchmark ARMDOTU8STRD1_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Quantized8Asymm(0.2f, 100),
                                    dtype::Quantized8Asymm(0.2f, 120),
                                    dtype::QuantizedS32(0.04f),
                                    dtype::Quantized8Asymm(1.4f, 110)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "ARMDOTU8STRD1";
    printf("Benchmark ARMDOTS8STRD1_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 3, 1, 1, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1, 1, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1, 1, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1, 1, 1);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_QUINT8_QUINT8_QUINT8_STRIDE2_WITHDOTPROD) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 2;
    param.stride_w = 2;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group, size_t P, size_t S) {
        SmallVector<TensorShape> shapes{
                {N, IC, H, W},
                {group, OC / group, IC / group, FS, FS},
                {1, OC, 1, 1},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1}};
        TensorShape dst{N, OC, (H + 2 * P - FS) / S + 1,
                        (W + 2 * P - FS) / S + 1};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 5, 4, 1, 2);
    bench_case(1, 32, 32, 200, 200, 5, 32, 1, 2);
    bench_case(1, 32, 32, 128, 128, 5, 4, 1, 2);
    bench_case(1, 32, 32, 128, 128, 5, 32, 1, 2);
    bench_case(1, 32, 32, 100, 100, 5, 4, 1, 2);
    bench_case(1, 32, 32, 100, 100, 5, 32, 1, 2);
    bench_case(1, 32, 32, 80, 80, 5, 4, 1, 2);
    bench_case(1, 32, 32, 80, 80, 5, 32, 1, 2);

    std::string algo_name = "ARMDOTU8STRD2";
    printf("Benchmark ARMDOTU8STRD2_LARGE_GROUP algo\n");
    std::vector<DType> data_type = {dtype::Quantized8Asymm(0.2f, 100),
                                    dtype::Quantized8Asymm(0.2f, 120),
                                    dtype::QuantizedS32(0.04f),
                                    dtype::Quantized8Asymm(1.4f, 110)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();

    algo_name = "ARMDOTU8STRD2";
    printf("Benchmark ARMDOTU8STRD2_SMALL_GROUP algo\n");
    bench_case(1, 32, 32, 200, 200, 5, 1, 1, 2);
    bench_case(1, 32, 32, 128, 128, 5, 1, 1, 2);
    bench_case(1, 32, 32, 100, 100, 5, 1, 1, 2);
    bench_case(1, 32, 32, 80, 80, 5, 1, 1, 2);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}
#endif

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_WINOGRAD_F32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 4);

    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 512, 256, 14, 14, 3, 1);
    bench_case(1, 512, 128, 14, 14, 3, 1);
    bench_case(1, 512, 64, 14, 14, 3, 1);

    bench_case(1, 512, 512, 7, 7, 3, 1);
    bench_case(1, 512, 256, 7, 7, 3, 1);
    bench_case(1, 512, 128, 7, 7, 3, 1);
    bench_case(1, 512, 64, 7, 7, 3, 1);

    std::string algo_name;
#if MEGDNN_AARCH64
    algo_name = "WINOGRAD:AARCH64_F32_MK4_4x16:4:2";
#else
    algo_name = "WINOGRAD:ARMV7_F32_MK4_4x8:4:2";
#endif
    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};
    printf("Benchmark WINOGRAD_F32_MK4 algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_WINOGRAD_INT8) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {group, OC / group, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 4);
    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 4);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 4);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 4);

    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 512, 256, 14, 14, 3, 1);
    bench_case(1, 512, 128, 14, 14, 3, 1);
    bench_case(1, 512, 64, 14, 14, 3, 1);

    bench_case(1, 512, 512, 7, 7, 3, 1);
    bench_case(1, 512, 256, 7, 7, 3, 1);
    bench_case(1, 512, 128, 7, 7, 3, 1);
    bench_case(1, 512, 64, 7, 7, 3, 1);

    std::string algo_name;
#if MEGDNN_AARCH64
    algo_name = "WINOGRAD:AARCH64_INT16X16X32_MK8_8X8:8:2:32";
#else
    algo_name = "WINOGRAD:ARMV7_INT16X16X32_MK8_4X8:8:2:32";
#endif


    std::vector<DType> data_type = {dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
                                   dtype::QuantizedS32(6.25f) ,dtype::QuantizedS8(60.25f) };
    printf("Benchmark WINOGRAD_IN8_MK8 algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_WINOGRAD_NCHW44_INT8_MK8) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::DENSE;
    param.format = param::ConvBias::Format::NCHW44;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC / 4, H, W, 4},
                                        {OC / 4, IC / 4, FS, FS, 4, 4},
                                        {1, OC / 4, 1, 1, 4},
                                        {},
                                        {N, OC / 4, H, W, 4}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);

    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 512, 256, 14, 14, 3, 1);
    bench_case(1, 512, 128, 14, 14, 3, 1);
    bench_case(1, 512, 64, 14, 14, 3, 1);

    bench_case(1, 512, 512, 7, 7, 3, 1);
    bench_case(1, 512, 256, 7, 7, 3, 1);
    bench_case(1, 512, 128, 7, 7, 3, 1);
    bench_case(1, 512, 64, 7, 7, 3, 1);

    std::string algo_name;
#if MEGDNN_AARCH64
    algo_name = "WINOGRAD_NCHW44:AARCH64_INT16X16X32_MK8_8X8:8:2:32";
#else
    algo_name = "WINOGRAD_NCHW44:ARMV7_INT16X16X32_MK8_4X8:8:2:32";
#endif

    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    printf("Benchmark WINOGRAD_INT8_MK8 algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_WINOGRAD_NCHW44_INT8_COMP_F32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::DENSE;  // GROUP;
    param.format = param::ConvBias::Format::NCHW44;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC / 4, H, W, 4},
                                        {OC / 4, IC / 4, FS, FS, 4, 4},
                                        {1, OC / 4, 1, 1, 4},
                                        {},
                                        {N, OC / 4, H, W, 4}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 200, 200, 3, 1);
    bench_case(1, 32, 32, 128, 128, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);

    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 512, 256, 14, 14, 3, 1);
    bench_case(1, 512, 128, 14, 14, 3, 1);
    bench_case(1, 512, 64, 14, 14, 3, 1);

    bench_case(1, 512, 512, 7, 7, 3, 1);
    bench_case(1, 512, 256, 7, 7, 3, 1);
    bench_case(1, 512, 128, 7, 7, 3, 1);
    bench_case(1, 512, 64, 7, 7, 3, 1);

    std::string algo_name;
#if MEGDNN_AARCH64
    algo_name = "WINOGRAD_NCHW44:AARCH64_F32_MK4_4x16:4:2:32";
#else
    algo_name = "WINOGRAD_NCHW44:ARMV7_F32_MK4_4x8:4:2:32";
#endif

    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    printf("Benchmark WINOGRAD_INT8_NCHW44_MK4_COMP_F32 algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_IM2COL_FP32) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group) {
        SmallVector<TensorShape> shapes{{N, IC, H, W},
                                        {OC, IC / group, FS, FS},
                                        {1, OC, 1, 1},
                                        {},
                                        {N, OC, H, W}};
        TensorShape dst{N, OC, H, W};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };
    std::vector<DType> data_type = {dtype::Float32(), dtype::Float32(),
                                    dtype::Float32(), dtype::Float32()};
    bench_case(1, 32, 32, 300, 300, 3, 1);
    bench_case(1, 32, 32, 400, 400, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    bench_case(1, 32, 64, 200, 200, 3, 1);
    bench_case(1, 32, 64, 128, 128, 3, 1);
    bench_case(1, 32, 64, 100, 100, 3, 1);
    bench_case(1, 32, 64, 80, 80, 3, 1);
    bench_case(1, 32, 128, 200, 200, 3, 1);
    bench_case(1, 32, 128, 128, 128, 3, 1);
    bench_case(1, 32, 128, 100, 100, 3, 1);
    bench_case(1, 32, 128, 80, 80, 3, 1);

    bench_case(1, 64, 32, 7, 7, 3, 1);
    bench_case(1, 64, 64, 7, 7, 3, 1);
    bench_case(1, 64, 128, 7, 7, 3, 1);
    bench_case(1, 64, 256, 7, 7, 3, 1);
    bench_case(1, 64, 512, 7, 7, 3, 1);
    bench_case(1, 64, 1024, 7, 7, 3, 1);

    bench_case(1, 64, 32, 14, 14, 3, 1);
    bench_case(1, 64, 64, 14, 14, 3, 1);
    bench_case(1, 64, 128, 14, 14, 3, 1);
    bench_case(1, 64, 256, 14, 14, 3, 1);
    bench_case(1, 64, 512, 14, 14, 3, 1);

    bench_case(1, 64, 1024, 14, 14, 3, 1);
    bench_case(1, 128, 128, 14, 14, 3, 1);
    bench_case(1, 128, 256, 14, 14, 3, 1);
    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 256, 512, 14, 14, 3, 1);
    bench_case(1, 512, 1024, 14, 14, 3, 1);
    bench_case(1, 1024, 1024, 14, 14, 3, 1);
    std::string algo_name = "IM2COLMATMUL:AARCH64_F32K8X12X1:96";
    printf("Benchmark IM2COLMATMUL:AARCH64_F32K8X12X1algo:96\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    algo_name = "IM2COLMATMUL:AARCH64_F32K8X12X1:192";
    printf("Benchmark IM2COLMATMUL:AARCH64_F32K8X12X1algo:192\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    algo_name = "IM2COLMATMUL:AARCH64_F32K8X12X1:384";
    printf("Benchmark IM2COLMATMUL:AARCH64_F32K8X12X1algo:384\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();
}
TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CHANNEL_WISE_INT8_INT8_INT8_STRIDE1) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;
    param.format = param::ConvBias::Format::NCHW44;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t H, size_t W, size_t FS,
                          size_t P) {
        size_t group = IC;
        size_t OC = IC;
        size_t S = 1;
        SmallVector<TensorShape> shapes{
                {N, IC, H, W, 4},
                {group, 1, 1, FS, FS, 4},
                {1, OC, 1, 1, 4},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1, 4}};
        TensorShape dst{N, OC, (H + 2 * P - FS) / S + 1,
                        (W + 2 * P - FS) / S + 1, 4};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };
    bench_case(1, 128, 200, 200, 3, 1);
    bench_case(1, 128, 128, 128, 3, 1);
    bench_case(1, 128, 100, 100, 3, 1);
    bench_case(1, 128, 80, 80, 3, 1);
    bench_case(1, 128, 56, 56, 3, 1);
    bench_case(1, 128, 28, 28, 3, 1);
    bench_case(1, 128, 14, 14, 3, 1);

    bench_case(1, 64, 200, 200, 3, 1);
    bench_case(1, 64, 128, 128, 3, 1);
    bench_case(1, 64, 100, 100, 3, 1);
    bench_case(1, 64, 80, 80, 3, 1);
    bench_case(1, 64, 56, 56, 3, 1);
    bench_case(1, 64, 28, 28, 3, 1);
    bench_case(1, 64, 14, 14, 3, 1);

    bench_case(1, 32, 200, 200, 3, 1);
    bench_case(1, 32, 128, 128, 3, 1);
    bench_case(1, 32, 100, 100, 3, 1);
    bench_case(1, 32, 80, 80, 3, 1);
    bench_case(1, 32, 56, 56, 3, 1);
    bench_case(1, 32, 28, 28, 3, 1);
    bench_case(1, 32, 14, 14, 3, 1);

    std::string algo_name = "S8_CHAN_WISE_STRD1_NCHW44";
    printf("Benchmarker S8_CHAN_WISE_STRD1_NCHW44 algo\n");
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
            dtype::QuantizedS32(6.25f), dtype::QuantizedS8(60.25f)};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CHANNEL_WISE_INT8_INT8_INT16_STRIDE1) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::GROUP;
    param.format = param::ConvBias::Format::NCHW44;

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t H, size_t W, size_t FS,
                          size_t P) {
        size_t group = IC;
        size_t OC = IC;
        size_t S = 1;
        SmallVector<TensorShape> shapes{
                {N, IC, H, W, 4},
                {group, 1, 1, FS, FS, 4},
                {1, OC, 1, 1, 4},
                {},
                {N, OC, (H + 2 * P - FS) / S + 1, (W + 2 * P - FS) / S + 1, 4}};
        TensorShape dst{N, OC, (H + 2 * P - FS) / S + 1,
                        (W + 2 * P - FS) / S + 1, 4};
        float computations =
                ((IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };
    bench_case(1, 128, 200, 200, 3, 1);
    bench_case(1, 128, 128, 128, 3, 1);
    bench_case(1, 128, 100, 100, 3, 1);
    bench_case(1, 128, 80, 80, 3, 1);
    bench_case(1, 128, 56, 56, 3, 1);
    bench_case(1, 128, 28, 28, 3, 1);
    bench_case(1, 128, 14, 14, 3, 1);

    bench_case(1, 64, 200, 200, 3, 1);
    bench_case(1, 64, 128, 128, 3, 1);
    bench_case(1, 64, 100, 100, 3, 1);
    bench_case(1, 64, 80, 80, 3, 1);
    bench_case(1, 64, 56, 56, 3, 1);
    bench_case(1, 64, 28, 28, 3, 1);
    bench_case(1, 64, 14, 14, 3, 1);

    bench_case(1, 32, 200, 200, 3, 1);
    bench_case(1, 32, 128, 128, 3, 1);
    bench_case(1, 32, 100, 100, 3, 1);
    bench_case(1, 32, 80, 80, 3, 1);
    bench_case(1, 32, 56, 56, 3, 1);
    bench_case(1, 32, 28, 28, 3, 1);
    bench_case(1, 32, 14, 14, 3, 1);

    std::string algo_name = "S8x8x16_CHAN_WISE_STRD1_STRD2_NCHW44";
    printf("Benchmarker S8x8x16_CHAN_WISE_STRD1_STRD2_NCHW44 algo\n");
    std::vector<DType> data_type = {dtype::Int8(), dtype::Int8(),
                                    dtype::Int16(), dtype::Int16()};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
}


TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_IM2COL_NCHW44_INT8x8x32_STRIDE1) {
    constexpr size_t RUNS = 50;

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::IDENTITY;
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.sparse = param::ConvBias::Sparse::DENSE;
    param.format = param::ConvBias::Format::NCHW44;
    

    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t N, size_t IC, size_t OC, size_t H, size_t W,
                          size_t FS, size_t group=1) {
        SmallVector<TensorShape> shapes{{N, IC, H, W,4},
                                        {OC, IC / group, FS, FS,4,4},
                                        {/*1, OC, 1, 1*/},
                                        {},
                                        {N, OC, H, W,4}};
        TensorShape dst{N, OC, H, W,4};
        float computations =
                ((4 * IC / group) * FS * FS * dst.total_nr_elems() * 2 +
                 dst.total_nr_elems()) *
                1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };

    bench_case(1, 32, 32, 300, 300, 3, 1);
    bench_case(1, 32, 32, 400, 400, 3, 1);
    bench_case(1, 32, 32, 100, 100, 3, 1);
    bench_case(1, 32, 32, 80, 80, 3, 1);
    bench_case(1, 32, 64, 200, 200, 3, 1);
    bench_case(1, 32, 64, 128, 128, 3, 1);
    bench_case(1, 32, 64, 100, 100, 3, 1);
    bench_case(1, 32, 64, 80, 80, 3, 1);
    bench_case(1, 32, 128, 200, 200, 3, 1);
    bench_case(1, 32, 128, 128, 128, 3, 1);
    bench_case(1, 32, 128, 100, 100, 3, 1);
    bench_case(1, 32, 128, 80, 80, 3, 1);
#if 1
    bench_case(1, 64, 32, 7, 7, 3, 1);
    bench_case(1, 64, 64, 7, 7, 3, 1);
    bench_case(1, 64, 128, 7, 7, 3, 1);
    bench_case(1, 64, 256, 7, 7, 3, 1);
    bench_case(1, 64, 512, 7, 7, 3, 1);
    bench_case(1, 64, 1024, 7, 7, 3, 1);

    bench_case(1, 64, 32, 14, 14, 3, 1);
    bench_case(1, 64, 64, 14, 14, 3, 1);
    bench_case(1, 64, 128, 14, 14, 3, 1);
    bench_case(1, 64, 256, 14, 14, 3, 1);
    bench_case(1, 64, 512, 14, 14, 3, 1);

    bench_case(1, 64, 1024, 14, 14, 3, 1);
    bench_case(1, 128, 128, 14, 14, 3, 1);
    bench_case(1, 128, 256, 14, 14, 3, 1);
    bench_case(1, 512, 512, 14, 14, 3, 1);
    bench_case(1, 256, 512, 14, 14, 3, 1);
    bench_case(1, 512, 1024, 14, 14, 3, 1);
    bench_case(1, 1024, 1024, 14, 14, 3, 1);
#endif
    std::string algo_name = "IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96";
    printf("Benchmarker  IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:96  algo\n");
    std::vector<DType> data_type = {
            dtype::QuantizedS8(2.5f), dtype::QuantizedS8(2.5f),
           dtype::QuantizedS32(6.25f),  {}};
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);


    
    algo_name = "IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:192";
    printf("Benchmarker  IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:192  algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);

    algo_name = "IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:384";
    printf("Benchmarker  IM2COLMATMUL:AARCH64_INT8X8X32_MK4_4X4X16:384  algo\n");
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);

}

#endif

/*================== BENCHMARK MULTITHREAD CONV1X1 =====================*/
#if MEGDNN_WITH_BENCHMARK

namespace {
std::vector<std::pair<SmallVector<TensorShape>, float>>
get_conv1x1_multithread_benchmark_args() {
    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation;
    auto bench_case = [&](size_t IC, size_t OC, size_t H, size_t W) {
        SmallVector<TensorShape> shapes{{1, IC, H, W},
                                        {OC, IC, 1, 1},
                                        {1, OC, 1, 1},
                                        {},
                                        {1, OC, H, W}};
        TensorShape dst{1, OC, H, W};
        float computations =
                (IC * dst.total_nr_elems() * 2 + dst.total_nr_elems()) * 1e-6;
        shapes_and_computation.push_back(std::make_pair(shapes, computations));
    };
    bench_case(32, 32, 300, 300);
    bench_case(32, 32, 400, 400);
    bench_case(32, 32, 100, 100);
    bench_case(32, 32, 80, 80);
    bench_case(32, 64, 200, 200);
    bench_case(32, 64, 128, 128);
    bench_case(32, 64, 100, 100);
    bench_case(32, 64, 80, 80);
    bench_case(32, 128, 200, 200);
    bench_case(32, 128, 128, 128);
    bench_case(32, 128, 100, 100);
    bench_case(32, 128, 80, 80);

    bench_case(64, 32, 7, 7);
    bench_case(64, 64, 7, 7);
    bench_case(64, 128, 7, 7);
    bench_case(64, 256, 7, 7);
    bench_case(64, 512, 7, 7);
    bench_case(64, 1024, 7, 7);

    bench_case(64, 32, 14, 14);
    bench_case(64, 64, 14, 14);
    bench_case(64, 128, 14, 14);
    bench_case(64, 256, 14, 14);
    bench_case(64, 512, 14, 14);

    bench_case(64, 1024, 14, 14);
    bench_case(128, 128, 14, 14);
    bench_case(128, 256, 14, 14);
    bench_case(512, 512, 14, 14);
    bench_case(256, 512, 14, 14);
    bench_case(512, 1024, 14, 14);
    bench_case(1024, 1024, 14, 14);
    return shapes_and_computation;
}

void conv1x1_multithread_benchmark(const char* algo_name, DType stype,
                                   DType ftype, DType btype, DType dtype) {
    constexpr size_t RUNS = 50;
    std::vector<std::pair<SmallVector<TensorShape>, float>>
            shapes_and_computation = get_conv1x1_multithread_benchmark_args();

    std::vector<DType> data_type = {stype, ftype, btype, dtype};

    param::ConvBias param;
    param.nonlineMode = param::ConvBias::NonlineMode::RELU;
    param.pad_h = 0;
    param.pad_w = 0;
    param.stride_h = 1;
    param.stride_w = 1;

    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {4}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS,
                   {4, {4, 5, 6, 7}}, {1, {7}}, data_type);
    benchmark_impl(param, shapes_and_computation, algo_name, RUNS, {2, {4, 5}},
                   {1, {4}}, data_type);
    shapes_and_computation.clear();
}
}  // namespace

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_CONVBIAS_CONV1X1_S1_FP32) {
#if MEGDNN_AARCH64
    conv1x1_multithread_benchmark("CONV1x1:AARCH64_F32K8X12X1:8",
                                  dtype::Float32(), dtype::Float32(),
                                  dtype::Float32(), dtype::Float32());
#else
    conv1x1_multithread_benchmark("CONV1x1:ARMV7_F32:8", dtype::Float32(),
                                  dtype::Float32(), dtype::Float32(),
                                  dtype::Float32());
#endif
}

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS,
       BENCHMARK_CONVBIAS_CONV1X1_S1_QUANTIZEDASYM) {
    dtype::Quantized8Asymm stype(0.2f, 100);
    dtype::Quantized8Asymm ftype(0.2f, 120);
    dtype::QuantizedS32 btype(0.04f);
    dtype::Quantized8Asymm dtype(1.4f, 110);
#if MEGDNN_AARCH64
#if __ARM_FEATURE_DOTPROD
    conv1x1_multithread_benchmark("CONV1x1:AARCH64_QUINT8_K8X8X4_DOTPROD:8",
                                  stype, ftype, btype, dtype);
#else
    conv1x1_multithread_benchmark("CONV1x1:AARCH64_QUINT8_K8X8X8:8", stype,
                                  ftype, btype, dtype);
#endif
#else
    conv1x1_multithread_benchmark("CONV1x1:ARMV7_QUINT8_K4X8X8:8", stype, ftype,
                                  btype, dtype);
#endif
}

#endif

// vim: syntax=cpp.doxygen
