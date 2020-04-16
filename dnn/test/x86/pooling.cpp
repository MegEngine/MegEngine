/**
 * \file dnn/test/x86/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/pooling.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
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

        auto pooling_used =
                benchmarker_pooling
                        .set_param({param::Pooling::Mode::MAX, pad, pad, stride,
                                    stride, window_size, window_size})
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
