/**
 * \file dnn/test/x86/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/pooling.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/task_record_check.h"
#include "test/x86/fixture.h"
namespace megdnn {
namespace test {

TEST_F(X86, POOLING) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        Checker<Pooling> checker(handle());
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}

TEST_F(X86, POOLING_RECORD) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        TaskRecordChecker<Pooling> checker(0);
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}

TEST_F(X86, S1POOLING88) {
    Checker<Pooling> checker(handle());
    auto run = [&](size_t WH, size_t WW, size_t PH, size_t PW, size_t SH, size_t SW,
                   size_t N, size_t C, size_t H, size_t W) {
        Pooling::Param param;
        param.format = param::Pooling::Format::NCHW88;
        param.window_h = WH;
        param.window_w = WW;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_w = SW;
        param.stride_h = SH;
        param.mode = param::Pooling::Mode::MAX;
        checker.set_param(param);
        checker.execs({{N, C, H, W, 8}, {}});
    };

    for (size_t wh = 10; wh < 15; ++wh) {
        for (size_t ww = 10; ww < 15; ++ww) {
            for (size_t n : {1, 2, 4}) {
                for (size_t c : {1, 4}) {
                    for (size_t h : {10, 13, 20}) {
                        for (size_t w : {10, 13, 20}) {
                            run(wh, ww, wh / 2, ww / 2, 1, 1, n, c, h, w);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(X86_MULTI_THREADS, S1POOLING88) {
    Checker<Pooling> checker(handle());
    auto run = [&](size_t WH, size_t WW, size_t PH, size_t PW, size_t SH, size_t SW,
                   size_t N, size_t C, size_t H, size_t W) {
        Pooling::Param param;
        param.format = param::Pooling::Format::NCHW88;
        param.window_h = WH;
        param.window_w = WW;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_w = SW;
        param.stride_h = SH;
        param.mode = param::Pooling::Mode::MAX;
        checker.set_param(param);
        checker.execs({{N, C, H, W, 8}, {}});
    };

    for (size_t wh = 10; wh < 15; ++wh) {
        for (size_t ww = 10; ww < 15; ++ww) {
            for (size_t n : {1, 2, 4}) {
                for (size_t c : {1, 4}) {
                    for (size_t h : {10, 13, 20}) {
                        for (size_t w : {10, 13, 20}) {
                            run(wh, ww, wh / 2, ww / 2, 1, 1, n, c, h, w);
                        }
                    }
                }
            }
        }
    }
}

#if MEGDNN_X86_WITH_MKL_DNN
TEST_F(X86, POOLING88) {
    Checker<Pooling> checker(handle());
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        arg.ishape.ndim = 5;
        arg.ishape[1] = (arg.ishape[1] + 7) / 8;
        arg.ishape[4] = 8;
        arg.param.format = param::Pooling::Format::NCHW88;
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}
TEST_F(X86, POOLING88_RECORD) {
    TaskRecordChecker<Pooling> checker(0);
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        arg.ishape.ndim = 5;
        arg.ishape[1] = (arg.ishape[1] + 7) / 8;
        arg.ishape[4] = 8;
        arg.param.format = param::Pooling::Format::NCHW88;
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}
TEST_F(X86_MULTI_THREADS, POOLING88) {
    Checker<Pooling> checker(handle());
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        arg.ishape.ndim = 5;
        arg.ishape[1] = (arg.ishape[1] + 7) / 8;
        arg.ishape[4] = 8;
        arg.param.format = param::Pooling::Format::NCHW88;
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}
#endif
#if MEGDNN_WITH_BENCHMARK
static void test_x86_megdnn_pooling(Handle* handle) {
    constexpr size_t RUNS = 50;
    auto rng = std::make_unique<UniformIntRNG>(-127, 127);

    Benchmarker<Pooling> benchmarker_pooling(handle);
    benchmarker_pooling.set_times(RUNS)
            .set_dtype(0, dtype::QuantizedS8(1.2))
            .set_display(false)
            .set_rng(0, rng.get());
    auto run = [&](uint32_t pad, uint32_t stride, uint32_t window_size,
                   size_t in_number, size_t in_channel, size_t in_height,
                   size_t in_width) {
        TensorLayout dst_layout;
        auto opr = handle->create_operator<Pooling>();
        opr->param() = {param::Pooling::Mode::MAX,
                        pad,
                        pad,
                        stride,
                        stride,
                        window_size,
                        window_size};

        TensorShape shape{in_number, in_channel, in_height, in_width};
        opr->deduce_layout({shape, dtype::Int8{}}, dst_layout);
        float computation =
                dst_layout.total_nr_elems() * window_size * window_size * 1e-9;

        auto pooling_used = benchmarker_pooling
                                    .set_param(
                                            {param::Pooling::Mode::MAX, pad, pad,
                                             stride, stride, window_size, window_size})
                                    .exec(TensorShapeArray{shape, {}}) /
                            RUNS;
        float through_put = computation / pooling_used * 1e3;
        std::cout << "{" << pad << "," << stride << "," << window_size << ","
                  << in_number << "," << in_channel << "," << in_height << ","
                  << in_width << "} "
                  << "use time " << pooling_used << "ms, "
                  << "through_put " << through_put << "Gops, " << std::endl;
    };
    for (auto widows_size : {2, 3})
        for (auto stride : {2})
            for (auto pad : {2})
                for (auto n : {1, 3, 4})
                    for (auto c : {1, 32, 64})
                        for (auto h_w : {12, 32, 64}) {
                            run(pad, stride, widows_size, n, c, h_w, h_w);
                        }
}
TEST_F(X86, BENCHMARK_POOLING) {
    test_x86_megdnn_pooling(handle());
}
TEST_F(X86_MULTI_THREADS, BENCHMARK_POOLING) {
    test_x86_megdnn_pooling(handle());
}
TEST_F(X86, BENCHMARK_POOLING_MAX_S1_NCHW88) {
    constexpr size_t RUNS = 50;
    auto x86_handle = handle();
    Benchmarker<Pooling> benchmarker_pooling(x86_handle);
    benchmarker_pooling.set_times(RUNS);
    auto run = [&](uint32_t pad, uint32_t stride, uint32_t window_size,
                   size_t in_number, size_t in_channel, size_t in_height,
                   size_t in_width) {
        auto opr = x86_handle->create_operator<Pooling>();
        opr->param() = {param::Pooling::Mode::MAX,
                        pad,
                        pad,
                        stride,
                        stride,
                        window_size,
                        window_size};
        opr->param().format = param::Pooling::Format::NCHW88;

        TensorShape shape{in_number, in_channel / 8, in_height, in_width, 8};
        TensorLayout dst_layout;
        opr->deduce_layout({shape, dtype::Float32()}, dst_layout);
        float computation =
                dst_layout.total_nr_elems() * window_size * window_size * 1e-9;

        auto pooling_used = benchmarker_pooling.set_param(opr->param())
                                    .exec(TensorShapeArray{shape, {}}) /
                            RUNS;
        float through_put = computation / pooling_used * 1e3;
        printf("profiling max pooling NCHW88 {%zu,%zu,%zu,%zu,8}\nuse time : "
               "%f ms\nthrough_put : %f Gflops\n",
               in_number, in_channel / 8, in_height, in_width, pooling_used,
               through_put);
    };
    run(6, 1, 13, 1, 32 * 8, 20, 20);
}

#endif
#if MEGDNN_X86_WITH_MKL_DNN
TEST_F(X86, POOLING_INT8) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        Checker<Pooling> checker(handle());
        auto rng = std::make_unique<UniformIntRNG>(-127, 127);
        checker.set_dtype(0, dtype::Int8()).set_rng(0, rng.get());
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}

TEST_F(X86, POOLING_INT8_RECORD) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        Checker<Pooling> checker(handle());
        auto rng = std::make_unique<UniformIntRNG>(-127, 127);
        checker.set_dtype(0, dtype::Int8()).set_rng(0, rng.get());
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}
TEST_F(X86_MULTI_THREADS, POOLING_INT8) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        Checker<Pooling> checker(handle());
        auto rng = std::make_unique<UniformIntRNG>(-127, 127);
        checker.set_dtype(0, dtype::Int8()).set_rng(0, rng.get());
        checker.set_param(arg.param).exec(TensorShapeArray{arg.ishape, {}});
    }
}
#endif
}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
