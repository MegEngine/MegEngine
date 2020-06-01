/**
 * \file dnn/test/cpu/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "test/common/convolution.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"

using namespace megdnn;
using namespace test;

namespace {

Convolution::Param gconv_param(Convolution::Param p) {
    p.sparse = Convolution::Param::Sparse::GROUP;
    return p;
}

} // anonymous namespace

#define CONVOLUTION_ARG_DIV_SIZE 100
TEST_F(CPU, CONVOLUTION_0) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    auto loop_size = args.size();
    ASSERT_GT(loop_size, CONVOLUTION_ARG_DIV_SIZE);
    Checker<Convolution> checker(handle());
    for (unsigned int i = 0; i < CONVOLUTION_ARG_DIV_SIZE; i++) {
        checker.set_param(args[i].param)
                .execs({args[i].src, args[i].filter, {}});
    }
}

#define CONVOLUTION1_ARG_LOOP_END_TIME (CONVOLUTION_ARG_DIV_SIZE + 205)

TEST_F(CPU, CONVOLUTION_1) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    auto loop_size = args.size();
    ASSERT_GT(loop_size, CONVOLUTION_ARG_DIV_SIZE);
    ASSERT_GT(loop_size, CONVOLUTION1_ARG_LOOP_END_TIME);
    Checker<Convolution> checker(handle());
    for (unsigned int i = CONVOLUTION_ARG_DIV_SIZE;
         i < CONVOLUTION1_ARG_LOOP_END_TIME; i++) {
        checker.set_param(args[i].param)
                .execs({args[i].src, args[i].filter, {}});
    }
}

#define CONVOLUTION2_ARG_LOOP_END_TIME (CONVOLUTION1_ARG_LOOP_END_TIME + 200)
TEST_F(CPU, CONVOLUTION_2) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    auto loop_size = args.size();
    ASSERT_GT(loop_size, CONVOLUTION2_ARG_LOOP_END_TIME);
    Checker<Convolution> checker(handle());
    for (unsigned int i = CONVOLUTION1_ARG_LOOP_END_TIME;
         i < CONVOLUTION2_ARG_LOOP_END_TIME; i++) {
        checker.set_param(args[i].param)
                .execs({args[i].src, args[i].filter, {}});
    }
}

TEST_F(CPU, CONVOLUTION_3) {
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    auto loop_size = args.size();
    ASSERT_GT(loop_size, CONVOLUTION2_ARG_LOOP_END_TIME);
    Checker<Convolution> checker(handle());
    for (unsigned int i = CONVOLUTION2_ARG_LOOP_END_TIME; i < loop_size; i++) {
        checker.set_param(args[i].param)
                .execs({args[i].src, args[i].filter, {}});
    }
}

#undef CONVOLUTION_ARG_DIV_SIZE
#undef CONVOLUTION1_ARG_LOOP_END_TIME
#undef CONVOLUTION2_ARG_LOOP_END_TIME

#define CB_CONV_CONFIG_COMBINATIONS(KSIZE)                                \
    TEST_F(CPU, CONV_CONFIG_COMBINATIONS_KSIZE_1_KSIZE_##KSIZE) {         \
        convolution::test_conv_config_combinations(KSIZE, handle(), true, \
                                                   false, false);         \
    }

// FIXME: only test ksize=1, will crash on IOS, so we tmp test ksize_1##other_ksize
CB_CONV_CONFIG_COMBINATIONS(2);
CB_CONV_CONFIG_COMBINATIONS(3);
CB_CONV_CONFIG_COMBINATIONS(5);
#undef CB_CONV_CONFIG_COMBINATIONS

#if MEGDNN_WITH_BENCHMARK
TEST_F(CPU, BENCHMARK_CONVOLUTION)
{
    using TestArg = convolution::TestArg;
    using Param = param::Convolution;
    std::vector<TestArg> args;
    // case 1: detection-like (padding x stride x kernel_size)
    // clang-format off
    for (size_t has_pad = 0; has_pad < 2; ++has_pad)
    for (uint32_t stride = 1; stride <= 2; ++stride)
    for (std::pair<size_t, size_t> kersize :
         std::vector<std::pair<size_t, size_t>>{
                 {2, 2}, {3, 3}, {5, 5}, {7, 7}}) {
        uint32_t pad_h, pad_w;
        if (has_pad)
            pad_h = kersize.first / 2;
        else
            pad_h = 0;
        if (has_pad)
            pad_w = kersize.second / 2;
        else
            pad_w = 0;
        auto param = Param{Param::Mode::CROSS_CORRELATION, pad_h, pad_w,
                           stride, stride};
        {
            auto arg = TestArg{param,
                               {2, 3, 320, 240},
                               {4, 3, kersize.first, kersize.second}};
            args.push_back(arg);
        }
    }
    // clang-format on
    Checker<Convolution> checker(handle());
    checker.set_perf_check(true).set_perf_check_threshold(2.0);
    for (auto &&arg: args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter, {}});
    }
}

#endif

TEST_F(CPU, CHANWISE_CONVOLUTION)
{
    constexpr auto M = Convolution::Mode::CROSS_CORRELATION;
    Checker<Convolution> checker(handle());
    checker.set_param(gconv_param({M, 0, 0, 1, 1})).
        execs({{1, 1, 2, 2}, {1, 1, 1, 2, 2}, {}}).
        execs({{1, 1, 5, 5}, {1, 1, 1, 2, 2}, {}}).
        execs({{2, 2, 5, 5}, {2, 3, 1, 2, 2}, {2, 6, 4, 4}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1})).
        execs({{2, 2, 5, 5}, {2, 1, 1, 2, 2}, {}});

    checker.set_param(gconv_param({M, 2, 3, 2, 1})).
        execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 20, 30, 4, 5})).
        execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});
}

TEST_F(CPU, CHANWISE_CONVOLUTION_INT8_INT8_INT16)
{
    constexpr auto M = Convolution::Mode::CROSS_CORRELATION;
    Checker<Convolution> checker(handle());

    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int16());

    checker.set_param(gconv_param({M, 0, 0, 1, 1, 1, 1})).
        execs({{1, 1, 2, 2}, {1, 1, 1, 2, 2}, {}}).
        execs({{1, 1, 5, 5}, {1, 1, 1, 2, 2}, {}}).
        execs({{2, 2, 5, 5}, {2, 3, 1, 2, 2}, {2, 6, 4, 4}});

    checker.set_param(gconv_param({M, 1, 1, 1, 1, 1, 1})).
        execs({{2, 2, 5, 5}, {2, 1, 1, 2, 2}, {}});

    checker.set_param(gconv_param({M, 2, 3, 2, 1, 1, 1})).
        execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});

    // padding larger than kern
    checker.set_param(gconv_param({M, 20, 30, 4, 5, 1, 1})).
        execs({{32, 12, 20, 10}, {12, 2, 1, 4, 5}, {}});

    // clang-format off
    for (uint32_t s : {1, 2})
    for (uint32_t p : {0, 1})
    for (size_t kh : {2, 3, 5})
    for (size_t kw : {kh, kh + 1})
    for (size_t ic : {5})
    for (size_t oc : {3})
    for (size_t h = 20; h <= 60; h += 7)
    for (size_t w : {h, h + 1}) {
        checker.set_param(gconv_param({M, p, p, s, s, 1, 1}))
                .execs({{2, ic, h, w}, {ic, oc, 1, kh, kw}, {}});
    }
    // clang-format on
}

TEST_F(CPU, GROUP_CONV)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t /* OH */, size_t /* OW */,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t group)
    {
        Checker<Convolution> checker(handle());
        Convolution::Param param;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(gconv_param(param)).exec({{N, IC, IH, IW},
                {group, OCg, ICg, FH, FW}, {}});
    };
    // normal case
    run(2, 64, 7, 7,
            3, 3,
            32, 5, 5,
            0, 0,
            1, 1,
            1);
    // padded case
    run(2, 32, 7, 7,
            3, 3,
            64, 7, 7,
            1, 1,
            1, 1,
            4);
    // strided case
    run(2, 32, 7, 7,
            3, 3,
            64, 3, 3,
            0, 0,
            2, 2,
            8);
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CPU, BENCHMARK_7X7_CONVOLUTION)
{
    using Param = param::Convolution;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<Convolution> benchmarker_naive(handle_naive.get());
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 10;
        auto tfloat = benchmarker_float.set_display(false)
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        auto tnaive = benchmarker_naive.set_display(false)
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        printf("src: %s filter: %s dst: %s naive=%.3fms float=%.3fms\n",
               shapes[0].to_string().c_str(), shapes[1].to_string().c_str(),
               shapes[2].to_string().c_str(), tnaive / RUN, tfloat / RUN);
    };
    Param param;
    param.stride_h = 2;
    param.stride_w = 2;
    param.pad_h = 3;
    param.pad_w = 3;

    // clang-format off
    for (size_t ic : {1, 3, 8, 16, 24}) {
    for (size_t oc : {8, 16}) {
    for (size_t h : {128, 224, 256, 512}) {
    for (size_t w : {128, 224, 256, 512}) {
        run({{1, ic, h, w}, {oc, ic, 7, 7}, {1, oc, h / 2, w / 2}}, param);
    } } } }
    // clang-format on
    // Used in FaceModel
    //run({{2, 3, 512, 512}, {8, 3, 7, 7}, {2, 8, 256, 256}}, param);
    //run({{2, 3, 128, 128}, {16, 3, 7, 7}, {2, 16, 64, 64}}, param);
    //run({{2, 3, 224, 224}, {32, 3, 7, 7}, {2, 32, 112, 112}}, param);
}
#endif

// vim: syntax=cpp.doxygen
