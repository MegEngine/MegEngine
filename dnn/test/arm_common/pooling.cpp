/**
 * \file dnn/test/arm_common/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"
#include "test/arm_common/fixture.h"

#include "test/common/pooling.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON, POOLING)
{
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
    // clang-format on
}

TEST_F(ARM_COMMON, POOLING_INT8_W2x2_S2x2)
{
    // clang-format off
    for (size_t ih: {2, 3, 7, 13, 52, 53, 54, 55})
    for (size_t iw: {2, 3, 6, 14, 53, 54, 55, 56})
    for (size_t ph: {0, 1})
    for (size_t pw: {0, 1})
    if (ih+2*ph >= 3 && iw+2*pw >= 3)
    {
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::Int8());
        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 2;
        param.window_h = param.window_w = 2;
        checker.set_param(param).exec(TensorShapeArray{{2, 3, ih, iw}, {}});
    }
    // clang-format on
}

TEST_F(ARM_COMMON, POOLING_INT8_W3x3_S2x2)
{
    // clang-format off
    for (size_t ih: {2, 3, 7, 13, 52, 53, 54, 55})
    for (size_t iw: {2, 3, 6, 14, 53, 54, 55, 56})
    for (size_t ph: {0, 1, 2})
    for (size_t pw: {0, 1, 2})
    if (ih+2*ph >= 3 && iw+2*pw >= 3)
    {
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::Int8());
        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 2;
        param.window_h = param.window_w = 3;
        checker.set_param(param).exec(TensorShapeArray{{2, 3, ih, iw}, {}});
    }
    // clang-format on
}

TEST_F(ARM_COMMON, POOLING_MAX_W3x3_S2x2_NCHW44)
{
    // clang-format off
    for (size_t ih: {3, 5, 10})
    for (size_t iw: {3, 5, 7, 9, 15, 20})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 3 && iw+2*pw >= 3)
    {
        UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::QuantizedS8(1.1f));
        checker.set_rng(0,&rng);

        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.format = param::Pooling::Format::NCHW44;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 2;
        param.window_h = param.window_w = 3;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}

TEST_F(ARM_COMMON, POOLING_MAX_W3x3_S1x1_NCHW44)
{
    // clang-format off
    for (size_t ih: {3, 5, 10})
    for (size_t iw: {3, 5, 7, 9, 15, 20})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 3 && iw+2*pw >= 3)
    {
        UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::QuantizedS8(1.1f));
        checker.set_rng(0,&rng);

        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.format = param::Pooling::Format::NCHW44;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 1;
        param.window_h = param.window_w = 3;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}

TEST_F(ARM_COMMON, POOLING_MAX_W2x2_S1x1_NCHW44)
{
    // clang-format off
    for (size_t ih: {2, 5, 10, 17})
    for (size_t iw: {2, 6, 8, 16, 26})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 2 && iw+2*pw >= 2)
    {
        UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::QuantizedS8(1.1f));
        checker.set_rng(0,&rng);

        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.format = param::Pooling::Format::NCHW44;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 1;
        param.window_h = param.window_w = 2;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}
TEST_F(ARM_COMMON, POOLING_MAX_W2x2_S2x2_NCHW44)
{
    // clang-format off
    for (size_t ih: {2, 5, 10, 17})
    for (size_t iw: {2, 6, 8, 16, 26})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 2 && iw+2*pw >= 2)
    {
        UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::QuantizedS8(1.1f));
        checker.set_rng(0,&rng);

        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.format = param::Pooling::Format::NCHW44;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 2;
        param.window_h = param.window_w = 2;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}
TEST_F(ARM_COMMON, POOLING_MAX_W4x4_S1x1_NCHW44)
{
    // clang-format off
    for (size_t ih: {4, 7, 10, 17, 20})
    for (size_t iw: {4, 8, 10, 21, 32})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 2 && iw+2*pw >= 2)
    {
        UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::QuantizedS8(1.1f));
        checker.set_rng(0,&rng);

        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.format = param::Pooling::Format::NCHW44;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 1;
        param.window_h = param.window_w = 4;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}
TEST_F(ARM_COMMON, POOLING_MAX_W4x4_S2x2_NCHW44)
{
    // clang-format off
    for (size_t ih: {4, 10, 18, 25, 30})
    for (size_t iw: {4, 12, 17, 20, 25})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 2 && iw+2*pw >= 2)
    {
        UniformIntRNG rng{INT8_MIN >> 1, INT8_MAX >> 1};
        Checker<Pooling> checker(handle());
        checker.set_dtype(0, dtype::QuantizedS8(1.1f));
        checker.set_rng(0,&rng);

        param::Pooling param;
        param.mode = param::Pooling::Mode::MAX;
        param.format = param::Pooling::Format::NCHW44;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = 2;
        param.window_h = param.window_w = 4;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON, POOLING_FP16) {
    Checker<Pooling> checker(handle());
    checker.set_dtype(0, dtype::Float16{})
            .set_dtype(1, dtype::Float16{})
            .set_epsilon(3e-3);

    using Param = param::Pooling;
    for (size_t ih : {2, 3, 5, 7, 11, 13, 17, 19, 23})
        for (size_t iw : {2, 3, 5, 7, 11, 13, 17, 19, 23})
            for (auto mode : {Param::Mode::AVERAGE, Param::Mode::MAX}) {
                for (size_t window : {2, 3}) {
                    Param param;
                    param.mode = mode;
                    param.window_h = param.window_w = window;
                    param.stride_h = param.stride_w = 1;
                    param.pad_h = param.pad_w = window / 2;
                    //! test for SH == 1 && SW == 1 && FH == FW (FH == 2 || FH
                    //! == 3)
                    checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                    //! test for SH = SW = 2 && FH = FW = 2
                    param.stride_h = param.stride_w = 2;
                    checker.set_param(param).exec({{2, 3, ih, iw}, {}});
                }
            }

    //! test for SH == 2 && SW == 2 && FH == FW == 3 max pooling
    for (size_t ih : {2, 3, 7, 13, 52, 53, 54, 55})
        for (size_t iw : {2, 3, 6, 14, 53, 54, 55, 56})
            for (size_t ph : {0, 1, 2})
                for (size_t pw : {0, 1, 2})
                    if (ih + 2 * ph >= 3 && iw + 2 * pw >= 3) {
                        param::Pooling param;
                        param.mode = param::Pooling::Mode::MAX;
                        param.pad_h = ph;
                        param.pad_w = pw;
                        param.stride_h = param.stride_w = 2;
                        param.window_h = param.window_w = 3;
                        checker.set_param(param).exec(
                                TensorShapeArray{{2, 3, ih, iw}, {}});
                    }

    //! test for SH == 2 && SW == 2 && FH = FW = 4 max pooling
    for (size_t ih :
         {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
        for (size_t iw :
             {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
            for (size_t p : {1, 2}) {
                Param param;
                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 4;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});
            }

    //! test for SH == 2 && SW == 2 && FH = FW = 5 max pooling
    for (size_t ih :
         {3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
        for (size_t iw :
             {3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
            for (size_t p : {1, 2}) {
                Param param;
                param.mode = Param::Mode::MAX;
                param.window_h = param.window_w = 5;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = p;
                checker.set_param(param).exec({{2, 3, ih, iw}, {}});
            }
}
#endif

TEST_F(ARM_COMMON, POOLING_QUANTIZED) {
    Checker<Pooling> checker(handle());
    UniformIntRNG rng1{INT8_MIN >> 1, INT8_MAX >> 1};
    UniformIntRNG rng2{0, UINT8_MAX >> 1};

    using Param = param::Pooling;

    for (auto type : std::vector<DType>{
                 dtype::QuantizedS8(1.1f),
                 dtype::Quantized8Asymm(1.1f, static_cast<uint8_t>(3))}) {
        if (type.enumv() == DTypeEnum::QuantizedS8) {
            checker.set_rng(0, &rng1);
        } else {
            megdnn_assert(type.enumv() == DTypeEnum::Quantized8Asymm);
            checker.set_rng(0, &rng2);
        }
        for (size_t ih : {2, 3, 5, 7, 11, 13, 17, 19, 23, 33, 49})
            for (size_t iw : {2, 3, 5, 7, 11, 13, 17, 19, 23, 33, 49})
                for (auto mode : {Param::Mode::AVERAGE, Param::Mode::MAX}) {
                    for (size_t window : {2, 3}) {
                        Param param;
                        param.mode = mode;
                        param.window_h = param.window_w = window;
                        param.stride_h = param.stride_w = 1;
                        param.pad_h = param.pad_w = window / 2;
                        //! test for SH == 1 && SW == 1 && FH == FW (FH == 2 ||
                        //! FH
                        //! == 3)
                        checker.set_param(param).exec({{2, 3, ih, iw}, {}});

                        //! test for SH = SW = 2 && FH = FW = 2
                        param.stride_h = param.stride_w = 2;
                        checker.set_param(param).exec({{2, 3, ih, iw}, {}});
                    }
                }

        //! test for SH == 2 && SW == 2 && FH == FW == 3 max pooling
        for (size_t ih : {2, 3, 7, 13, 52, 53, 54, 55})
            for (size_t iw : {2, 3, 6, 14, 53, 54, 55, 56})
                for (size_t ph : {0, 1, 2})
                    for (size_t pw : {0, 1, 2})
                        if (ih + 2 * ph >= 3 && iw + 2 * pw >= 3) {
                            param::Pooling param;
                            param.mode = param::Pooling::Mode::MAX;
                            param.pad_h = ph;
                            param.pad_w = pw;
                            param.window_h = param.window_w = 3;
                            param.stride_h = param.stride_w = 2;
                            checker.set_param(param).exec(
                                    TensorShapeArray{{2, 3, ih, iw}, {}});
                        }

        //! test for SH == 2 && SW == 2 && FH == FW == 4 max pooling
        for (size_t ih :
             {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
            for (size_t iw :
                 {2, 3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
                for (size_t p : {1, 2}) {
                    Param param;
                    param.mode = Param::Mode::MAX;
                    param.window_h = param.window_w = 4;
                    param.stride_h = param.stride_w = 2;
                    param.pad_h = param.pad_w = p;
                    checker.set_param(param).exec({{2, 3, ih, iw}, {}});
                }

        //! test for SH == 2 && SW == 2 && FH == FW == 5 max pooling
        for (size_t ih :
             {3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
            for (size_t iw :
                 {3, 5, 7, 11, 13, 17, 19, 23, 24, 25, 26, 27, 28, 29, 30})
                for (size_t p : {1, 2}) {
                    Param param;
                    param.mode = Param::Mode::MAX;
                    param.window_h = param.window_w = 5;
                    param.stride_h = param.stride_w = 2;
                    param.pad_h = param.pad_w = p;
                    checker.set_param(param).exec({{2, 3, ih, iw}, {}});
                }
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ARM_COMMON, BENCHMARK_POOLING_INT8_W3x3_S2x2)
{
    using Param = param::Pooling;
    auto run = [&](const TensorShapeArray &shapes,
            Param param) {
        auto handle_naive = create_cpu_handle(2);
        TensorLayoutArray layouts;
        layouts.emplace_back(shapes[0], dtype::Int8());
        layouts.emplace_back(shapes[1], dtype::Int8());
        Benchmarker<Pooling> benchmarker_naive(handle_naive.get());
        Benchmarker<Pooling> benchmarker_float(handle());
        Benchmarker<Pooling> benchmarker_int(handle());
        size_t RUN = 10;
        auto t1 = benchmarker_naive.set_display(false).set_times(RUN).
            set_param(param).exec(shapes);
        auto t2 = benchmarker_float.set_display(false).set_times(RUN).
            set_param(param).exec(shapes);
        auto t3 = benchmarker_int.set_display(false).set_times(RUN).
            set_param(param).execl(layouts);
        printf("naive=%.3fms float=%.3fms, int=%.3fms\n",
                t1 / RUN, t2 / RUN, t3 / RUN);
        auto speedup = t2/t3;
        ASSERT_GE(speedup, 2.0);
    };
    Param param;
    param.window_h = param.window_w = 3;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;
    std::cout << "3x3 with 2x2 stride max pooling:" << std::endl;
    run({{1, 3, 640, 480}, {}}, param);
}

TEST_F(ARM_COMMON, BENCHMARK_POOLING_W4x4_S2x2)
{
    using Param = param::Pooling;
    auto run = [&](const TensorShapeArray &shapes,
            Param param) {
        std::cout << "N:" << shapes[0][0] << " "
                  << "IC:" << shapes[0][1] << " "
                  << "IH:" << shapes[0][2] << " "
                  << "IW:" << shapes[0][3] << std::endl;
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<Pooling> benchmarker_naive(handle_naive.get());
        Benchmarker<Pooling> benchmarker_float(handle());
        size_t RUN = 10;
        auto t1 = benchmarker_naive.set_display(false).set_times(RUN).
            set_param(param).exec(shapes);
        auto t2 = benchmarker_float.set_display(false).set_times(RUN).
            set_param(param).exec(shapes);
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Pooling>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()}, dst_layout);
        float calc_amount = dst_layout.total_nr_elems() *
                            param.window_h * param.window_w;
        printf("naive={%.3fms, %.3fMflops}, neon={%.3fms, %.3fMflops}\n",
               t1 / RUN, calc_amount / (t1 / RUN * 1000),
               t2 / RUN, calc_amount / (t2 / RUN * 1000));
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

TEST_F(ARM_COMMON, BENCHMARK_POOLING_W5x5_S2x2)
{
    using Param = param::Pooling;
    auto run = [&](const TensorShapeArray &shapes,
            Param param) {
        std::cout << "N:" << shapes[0][0] << " "
                  << "IC:" << shapes[0][1] << " "
                  << "IH:" << shapes[0][2] << " "
                  << "IW:" << shapes[0][3] << std::endl;
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<Pooling> benchmarker_naive(handle_naive.get());
        Benchmarker<Pooling> benchmarker_float(handle());
        size_t RUN = 10;
        auto t1 = benchmarker_naive.set_display(false).set_times(RUN).
            set_param(param).exec(shapes);
        auto t2 = benchmarker_float.set_display(false).set_times(RUN).
            set_param(param).exec(shapes);
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Pooling>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()}, dst_layout);
        float calc_amount = dst_layout.total_nr_elems() *
                            param.window_h * param.window_w;
        printf("naive={%.3fms, %.3fMflops}, neon={%.3fms, %.3fMflops}\n",
               t1 / RUN, calc_amount / (t1 / RUN * 1000),
               t2 / RUN, calc_amount / (t2 / RUN * 1000));
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


TEST_F(ARM_COMMON, BENCHMARK_POOLING_FP16) {
    using Param = param::Pooling;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        TensorLayoutArray layouts;
        layouts.emplace_back(shapes[0], dtype::Float16());
        layouts.emplace_back(shapes[1], dtype::Float16());
        Benchmarker<Pooling> benchmarker_float(handle());
        Benchmarker<Pooling> benchmarker_half(handle());
        size_t RUN = 10;
        auto tf = benchmarker_float.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec(shapes) /
                  RUN;
        auto th = benchmarker_half.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .execl(layouts) /
                  RUN;
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Pooling>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()}, dst_layout);

        float computations = dst_layout.total_nr_elems() * param.window_h *
                             param.window_w / (1024.f * 1024 * 1024);
        printf("float=%.3fms %f gflops, float16=%.3fms %f gflops speedup: %f\n",
               tf, computations / tf * 1e3, th, computations / th * 1e3,
               tf / th);
    };
    Param param;
    param.window_h = param.window_w = 2;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    printf("2x2 with 1x1 stride max pooling:\n");
    run({{1, 3, 640, 480}, {}}, param);

    for (size_t oh : {640, 128})
        for (size_t ow : {480, 112}) {
            param.window_h = param.window_w = 3;
            param.stride_h = param.stride_w = 2;
            param.pad_h = param.pad_w = 1;
            param.mode = Param::Mode::AVERAGE;
            printf("3x3 with 2x2 stride average pooling.\n");
            run({{1, 3, oh, ow}, {}}, param);

            for (size_t pw : {2, 3, 4, 5}) {
                param.window_h = param.window_w = pw;
                param.stride_h = param.stride_w = 2;
                param.pad_h = param.pad_w = 1;
                param.mode = Param::Mode::MAX;
                printf("%zux%zu with 2x2 stride max pooling:\n", pw, pw);
                run({{1, 3, oh, ow}, {}}, param);
            }
        }
}

TEST_F(ARM_COMMON, BENCHMARK_POOLING_QUANTIZED) {
    using Param = param::Pooling;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        TensorLayoutArray layouts;
        layouts.emplace_back(shapes[0], dtype::QuantizedS8(1.1f));
        layouts.emplace_back(shapes[1], dtype::QuantizedS8(1.1f));
        Benchmarker<Pooling> benchmarker_int(handle());
        Benchmarker<Pooling> benchmarker_naive(handle_naive.get());
        size_t RUN = 10;
        auto time_int = benchmarker_int.set_display(false)
                                .set_times(RUN)
                                .set_param(param)
                                .exec(shapes) /
                        RUN;
        auto time_naive = benchmarker_naive.set_display(false)
                                  .set_times(RUN)
                                  .set_param(param)
                                  .execl(layouts) /
                          RUN;
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Pooling>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::QuantizedS8(1.1f)}, dst_layout);

        float computations = dst_layout.total_nr_elems() * param.window_h *
                             param.window_w / (1024.f * 1024 * 1024);
        printf("naive=%.3fms %f gflops, int8=%.3fms %f gflops speedup: %f\n",
               time_naive, computations / time_naive * 1e3, time_int,
               computations / time_int * 1e3, time_naive / time_int);
    };
    Param param;
    param.window_h = param.window_w = 2;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    printf("2x2 with 1x1 stride max pooling:\n");
    run({{1, 3, 640, 480}, {}}, param);

    // clang-format off
    for (size_t oh : {640, 128})
    for (size_t ow : {480, 112})
    for (size_t pw : {2, 3, 4, 5}) {
        param.window_h = param.window_w = pw;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = 1;
        printf("%zux%zu with 2x2 stride max pooling:\n", pw, pw);
        run({{1, 3, oh, ow}, {}}, param);
    }
    // clang-format on
}
#endif

}  // namespace test
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
