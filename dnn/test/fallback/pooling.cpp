/**
 * \file dnn/test/fallback/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/fallback/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/pooling.h"
#include "test/common/rng.h"
#include "test/common/task_record_check.h"

namespace megdnn {
namespace test {

namespace {
std::vector<std::pair<param::Pooling, TensorShapeArray>> get_nchw44_pool_args(
        size_t filter, size_t stride) {
    constexpr size_t ic_step = 4;
    std::vector<std::pair<param::Pooling, TensorShapeArray>> args;

    for (size_t n : {1, 2})
        for (size_t c : {4, 8})
            for (size_t ih : {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
                for (size_t iw : {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})
                    for (size_t ph : {0, 1, 2})
                        for (size_t pw : {0, 1, 2})
                            for (auto mode :
                                 {param::Pooling::Mode::MAX,
                                  param::Pooling::Mode::AVERAGE})
                                if (ih + 2 * ph >= filter && iw + 2 * pw >= filter &&
                                    filter > ph && filter > pw) {
                                    param::Pooling param;
                                    param.mode = mode;
                                    param.format = param::Pooling::Format::NCHW44;
                                    param.pad_h = ph;
                                    param.pad_w = pw;
                                    param.stride_h = param.stride_w = stride;
                                    param.window_h = param.window_w = filter;
                                    args.emplace_back(std::make_pair(
                                            param,
                                            TensorShapeArray{
                                                    {n, c / ic_step, ih, iw, ic_step},
                                                    {}}));
                                }
    return args;
}

void run_pooling_check(
        Handle* handle, std::vector<std::pair<param::Pooling, TensorShapeArray>> args,
        bool is_int8) {
    Checker<Pooling> checker(handle);
    UniformIntRNG rng_int8{INT8_MIN >> 1, INT8_MAX >> 1};
    UniformIntRNG rng_fp32{-10, 10};
    if (is_int8) {
        checker.set_dtype(0, dtype::QuantizedS8(1.1f));
        checker.set_rng(0, &rng_int8);
    } else {
        checker.set_rng(0, &rng_fp32);
    }
    for (auto arg : args) {
        checker.set_param(arg.first).exec(arg.second);
    }
}
}  // namespace

TEST_F(FALLBACK_MULTI_THREADS, POOLING_GI_NCHW44_FP32) {
    for (auto filter : {2, 3, 4, 5})
        for (auto stride : {1, 2}) {
            run_pooling_check(handle(), get_nchw44_pool_args(filter, stride), false);
        }
}

TEST_F(FALLBACK, POOLING_GI) {
    using Param = param::Pooling;
    // clang-format off
    for (size_t ih: {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
    for (size_t iw: {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
    for (size_t p: {1, 2})
    {
        Param param;
        param.mode = Param::Mode::MAX;
        param.window_h = param.window_w = 3;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        Checker<Pooling> checker(handle());
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});

        param.mode = Param::Mode::AVERAGE;
        param.window_h = param.window_w = 3;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});

        param.mode = Param::Mode::MAX;
        param.window_h = param.window_w = 4;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});

        param.mode = Param::Mode::MAX;
        param.window_h = param.window_w = 5;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        if (ih + p * 2 >= 5 && iw + p * 2 >= 5)
            checker.set_param(param).exec({{2, 3, ih, iw}, {}});
    }
    for (size_t ih: {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
    for (size_t iw: {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
    for (size_t p: {1, 2})
    {
        Param param;
        param.mode = Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
        param.window_h = param.window_w = 3;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = p;
        Checker<Pooling> checker(handle());
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});
    }
    // clang-format on
}

TEST_F(FALLBACK, POOLING_GI_RECORD) {
    using Param = param::Pooling;
    TaskRecordChecker<Pooling> checker(0);
    // clang-format off
    for (size_t ih: {2, 3, 5, 7, 11, 13, 17})
    for (size_t iw: {2, 3, 5, 7, 11, 13, 17})
    for (size_t p: {1, 2})
    {
        Param param;
        param.mode = Param::Mode::MAX;
        param.window_h = param.window_w = 3;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});

        param.mode = Param::Mode::AVERAGE;
        param.window_h = param.window_w = 3;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});

        param.mode = Param::Mode::MAX;
        param.window_h = param.window_w = 4;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});

        param.mode = Param::Mode::MAX;
        param.window_h = param.window_w = 5;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = p;
        if (ih + p * 2 >= 5 && iw + p * 2 >= 5)
            checker.set_param(param).exec({{2, 3, ih, iw}, {}});
    }
    for (size_t ih: {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
    for (size_t iw: {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
    for (size_t p: {1, 2})
    {
        Param param;
        param.mode = Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
        param.window_h = param.window_w = 3;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = p;
        Checker<Pooling> checker(handle());
        checker.set_param(param).exec({{2, 3, ih, iw}, {}});
    }
    // clang-format on
}

TEST_F(FALLBACK_MULTI_THREADS, POOLING_GI_RECORD) {
    using Param = param::Pooling;
    TaskRecordChecker<Pooling> checker(0);
    for (size_t ih : {2, 3, 5, 7, 11, 13, 17})
        for (size_t iw : {2, 3, 5, 7, 11, 13, 17})
            for (size_t p : {1, 2}) {
                Param param;
                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 3;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                param.mode = Param::Mode::AVERAGE;
                param.window_h = param.window_w = 3;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 4;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 5;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                if (ih + p * 2 >= 5 && iw + p * 2 >= 5)
                    checker.set_param(param).exec({{2, 3, ih, iw}, {}});
            }
}

TEST_F(FALLBACK_MULTI_THREADS, POOLING_GI_W9_w13_NCHW44) {
    UniformIntRNG rng{-10, 10};
    Checker<Pooling> checker(handle());
    checker.set_rng(0, &rng);
    // clang-format off
    for (size_t ih: {20, 15})
    for (size_t iw: {15, 20})
    for (size_t kernel: {9, 13})
    for (size_t pad: {4, 6})
    for(auto mode: {param::Pooling::Mode::MAX, param::Pooling::Mode::AVERAGE})
    if (kernel > pad)
    {
        param::Pooling param;
        param.mode = mode;
        param.format = param::Pooling::Format::NCHW44;
        param.pad_h = pad;
        param.pad_w = pad;
        param.stride_h = param.stride_w = 1;
        param.window_h = param.window_w = kernel ;
        checker.set_param(param).exec(TensorShapeArray{{2, 8, ih, iw, 4}, {}});
    }
    // clang-format on
}

TEST_F(FALLBACK_MULTI_THREADS, POOLING_GI_FALLBACK) {
    using Param = param::Pooling;
    for (size_t ih : {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
        for (size_t iw : {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
            for (size_t p : {1, 2}) {
                Param param;
                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 3;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                Checker<Pooling> checker(handle());
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});
            }
}

TEST_F(FALLBACK_MULTI_THREADS, POOLING_GI) {
    using Param = param::Pooling;
    for (size_t ih : {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
        for (size_t iw : {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
            for (size_t p : {1, 2}) {
                Param param;
                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 3;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                Checker<Pooling> checker(handle());
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                param.mode = Param::Mode::AVERAGE;
                param.window_h = param.window_w = 3;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 4;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 5;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                if (ih + p * 2 >= 5 && iw + p * 2 >= 5)
                    checker.set_param(param).exec({{2, 3, ih, iw}, {}});
            }
}

#if MEGDNN_WITH_BENCHMARK
namespace {
void benchmark_nchw44_fp32(Handle* handle) {
    using Param = param::Pooling;
    auto run = [&](size_t n, size_t c, size_t h, size_t w, size_t filter, size_t stride,
                   size_t pad, Param::Mode mode) {
        Param param;
        param.window_h = param.window_w = filter;
        param.stride_h = param.stride_w = stride;
        param.pad_h = param.pad_w = pad;
        param.format = Param::Format::NCHW;
        param.mode = mode;
        TensorShape nchw_shape = {n, c, h, w};
        TensorShape nchw44_shape = {n, c / 4, h, w, 4};
        TensorLayout dst_layout;
        auto opr = handle->create_operator<Pooling>();
        opr->param() = param;
        opr->deduce_layout({nchw_shape, dtype::Float32()}, dst_layout);
        float calc_amount =
                dst_layout.total_nr_elems() * param.window_h * param.window_w;

        Benchmarker<Pooling> benchmarker_float_nchw(handle);
        Benchmarker<Pooling> benchmarker_float_nchw44(handle);
        Benchmarker<Pooling> benchmarker_int_nchw44(handle);
        size_t RUN = 500;
        auto t1 = benchmarker_float_nchw.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec({nchw_shape, {}});

        param.format = Param::Format::NCHW44;
        auto t2 = benchmarker_int_nchw44.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .execl({{nchw44_shape, dtype::QuantizedS8(1.0)},
                                  {{}, dtype::QuantizedS8(1.0)}});
        auto t3 = benchmarker_float_nchw44.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec({nchw44_shape, {}});

        printf("{%zu %zu %zu %zu} filter = %zu, stride = %zu pad = %zu\n"
               "nchw_fp32={%.3f ms, %.3f Mflops},  "
               "nchw44_int={%.3f ms, %.3f Mflops},  "
               "nchw44_fp32={%.3f ms, %.3f Mflops, speed_up %f}\n\n",
               n, c, h, w, filter, stride, pad, t1 / RUN,
               calc_amount / (t1 / RUN * 1000), t2 / RUN,
               calc_amount / (t2 / RUN * 1000), t3 / RUN,
               calc_amount / (t3 / RUN * 1000), t1 / t3);
    };
    // Resnet50
    run(1, 64, 112, 112, 3, 2, 1, param::Pooling::Mode::MAX);
    run(1, 2048, 7, 7, 7, 1, 0, param::Pooling::Mode::AVERAGE);

    // VGG16
    run(1, 64, 224, 224, 2, 2, 0, param::Pooling::Mode::MAX);
    run(1, 128, 112, 112, 2, 2, 0, param::Pooling::Mode::MAX);
    run(1, 256, 56, 56, 2, 2, 0, param::Pooling::Mode::MAX);
    run(1, 512, 28, 28, 2, 2, 0, param::Pooling::Mode::MAX);
    run(1, 512, 14, 14, 2, 2, 0, param::Pooling::Mode::MAX);
}
}  // namespace

TEST_F(FALLBACK, BENCHMARK_POOLING_GI_NCHW44_FP32) {
    benchmark_nchw44_fp32(handle());
}

TEST_F(FALLBACK_MULTI_THREADS, BENCHMARK_POOLING_GI_NCHW44_FP32) {
    benchmark_nchw44_fp32(handle());
}
TEST_F(FALLBACK, BENCHMARK_POOLING_GI_W4x4_S2x2) {
    using Param = param::Pooling;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        std::cout << "N:" << shapes[0][0] << " "
                  << "IC:" << shapes[0][1] << " "
                  << "IH:" << shapes[0][2] << " "
                  << "IW:" << shapes[0][3] << std::endl;
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<Pooling> benchmarker_naive(handle_naive.get());
        Benchmarker<Pooling> benchmarker_float(handle());
        size_t RUN = 10;
        auto t1 = benchmarker_naive.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec(shapes);
        auto t2 = benchmarker_float.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec(shapes);
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Pooling>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()}, dst_layout);
        float calc_amount =
                dst_layout.total_nr_elems() * param.window_h * param.window_w;
        printf("naive={%.3fms, %.3fMflops}, neon={%.3fms, %.3fMflops}\n", t1 / RUN,
               calc_amount / (t1 / RUN * 1000), t2 / RUN,
               calc_amount / (t2 / RUN * 1000));
    };
    Param param;
    param.window_h = param.window_w = 4;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;
    std::cout << "4x4 with 2x2 stride max pooling:" << std::endl;
    run({{1, 24, 160, 128}, {}}, param);
    run({{1, 4, 240, 135}, {}}, param);
    run({{1, 32, 120, 67}, {}}, param);
    run({{1, 64, 60, 33}, {}}, param);
}

TEST_F(FALLBACK, BENCHMARK_POOLING_GI_W5x5_S2x2) {
    using Param = param::Pooling;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        std::cout << "N:" << shapes[0][0] << " "
                  << "IC:" << shapes[0][1] << " "
                  << "IH:" << shapes[0][2] << " "
                  << "IW:" << shapes[0][3] << std::endl;
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<Pooling> benchmarker_naive(handle_naive.get());
        Benchmarker<Pooling> benchmarker_float(handle());
        size_t RUN = 10;
        auto t1 = benchmarker_naive.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec(shapes);
        auto t2 = benchmarker_float.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec(shapes);
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Pooling>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()}, dst_layout);
        float calc_amount =
                dst_layout.total_nr_elems() * param.window_h * param.window_w;
        printf("naive={%.3fms, %.3fMflops}, neon={%.3fms, %.3fMflops}\n", t1 / RUN,
               calc_amount / (t1 / RUN * 1000), t2 / RUN,
               calc_amount / (t2 / RUN * 1000));
    };
    Param param;
    param.window_h = param.window_w = 5;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;
    std::cout << "5x5 with 2x2 stride max pooling:" << std::endl;
    run({{1, 24, 160, 128}, {}}, param);
    run({{1, 4, 240, 135}, {}}, param);
    run({{1, 32, 120, 67}, {}}, param);
    run({{1, 64, 60, 33}, {}}, param);
}
namespace {
template <typename Opr>
void benchmark_impl(
        const typename Opr::Param& param, std::vector<SmallVector<TensorShape>> shapes,
        size_t RUNS, TaskExecutorConfig&& multi_thread_config,
        TaskExecutorConfig&& single_thread_config, DType data_type) {
    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle = create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker = Benchmarker<Opr>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS).set_display(false).set_param(param);
        benchmarker.set_dtype(0, data_type);
        for (auto shape : shapes) {
            multi_thread_times.push_back(benchmarker.exec(shape) / RUNS);
        }
    }
    {
        auto single_thread_handle = create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker = Benchmarker<Opr>(single_thread_handle.get());
        benchmarker.set_times(RUNS).set_display(false).set_param(param);
        benchmarker.set_dtype(0, data_type);
        for (auto shape : shapes) {
            single_thread_times.push_back(benchmarker.exec(shape) / RUNS);
        }
    }
    printf("Benchmark : Multi threads  %zu, ", multi_thread_config.nr_thread);
    printf("core_ids:");
    for (size_t i = 0; i < multi_thread_config.affinity_core_set.size(); i++) {
        printf("%zu ", multi_thread_config.affinity_core_set[i]);
    }
    printf(", Single thread core_id %zu\n", single_thread_config.affinity_core_set[0]);
    for (size_t i = 0; i < shapes.size(); i++) {
        auto shape = shapes[i];
        printf("Case: ");
        for (auto sh : shape)
            printf("%s ", sh.to_string().c_str());
        printf("%zu threads time: %f,\n single thread time: "
               "%f. spead up = %f, speedup/cores=%f\n",
               multi_thread_config.nr_thread, multi_thread_times[i],
               single_thread_times[i], single_thread_times[i] / multi_thread_times[i],
               single_thread_times[i] / multi_thread_times[i] /
                       multi_thread_config.nr_thread);
    }
}
}  // namespace

TEST_F(FALLBACK_MULTI_THREADS, BENCHMARK_POOLING_GI) {
    constexpr size_t RUNS = 50;

    using Param = param::Pooling;
    Param param;
    param.window_h = param.window_w = 3;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;

    std::vector<SmallVector<TensorShape>> shapes;

    shapes.push_back({{32, 32, 215, 215}, {}});
    shapes.push_back({{32, 32, 128, 128}, {}});
    shapes.push_back({{8, 256, 100, 100}, {}});
    shapes.push_back({{1, 256, 100, 100}, {}});
    shapes.push_back({{1, 32, 100, 100}, {}});
    shapes.push_back({{1, 256, 80, 80}, {}});
    shapes.push_back({{1, 256, 60, 60}, {}});
    shapes.push_back({{1, 256, 30, 30}, {}});

    param.window_h = param.window_w = 3;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;
    printf("Benchmark POOLING kernel:%d*%d stride:%d,mode %d\n", param.window_h,
           param.window_w, param.stride_h, static_cast<int>(param.mode));
    benchmark_impl<Pooling>(
            param, shapes, RUNS, {4, {0, 1, 2, 3}}, {1, {0}}, dtype::Float32());
    benchmark_impl<Pooling>(
            param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {4}}, dtype::Float32());
    benchmark_impl<Pooling>(
            param, shapes, RUNS, {2, {0, 1}}, {1, {0}}, dtype::Float32());
}

TEST_F(FALLBACK_MULTI_THREADS, BENCHMARK_POOLING_GI_NCHW44) {
    constexpr size_t RUNS = 50;

    using Param = param::Pooling;
    Param param;
    param.pad_h = param.pad_w = 0;
    param.mode = Param::Mode::MAX;
    std::vector<SmallVector<TensorShape>> shapes;
    std::vector<std::vector<size_t>> filter_and_stride = {
            {2, 1}, {2, 2}, {3, 1}, {3, 2}, {4, 1}, {4, 2}, {5, 1}, {5, 2}};

    for (auto mode : {param::Pooling::Mode::MAX, param::Pooling::Mode::AVERAGE}) {
        for (auto filter : filter_and_stride) {
            shapes.push_back({{1, 32 * 4, 215, 215}, {}});
            shapes.push_back({{1, 32 * 4, 128, 128}, {}});
            shapes.push_back({{1, 16 * 4, 56, 56}, {}});

            param.mode = mode;
            param.window_h = param.window_w = filter[0];
            param.stride_h = param.stride_w = filter[1];
            param.format = Param::Format::NCHW;
            printf("NCHW Benchmark POOLING kernel:%d*%d stride:%d,mode %d\n",
                   param.window_h, param.window_h, param.stride_h,
                   static_cast<int>(param.mode));
            benchmark_impl<Pooling>(
                    param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
                    dtype::QuantizedS8(1.1f));
            shapes.clear();
            shapes.push_back({{1, 32, 215, 215, 4}, {}});
            shapes.push_back({{1, 32, 128, 128, 4}, {}});
            shapes.push_back({{1, 16, 56, 56, 4}, {}});

            param.format = Param::Format::NCHW44;
            printf("NCHW44 Benchmark POOLING kernel:%d*%d stride:%d,mode %d\n",
                   param.window_h, param.window_w, param.stride_h,
                   static_cast<int>(param.mode));
            benchmark_impl<Pooling>(
                    param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {4}},
                    dtype::QuantizedS8(1.1f));
            shapes.clear();
        }
    }
}
#endif

}  // namespace test
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
