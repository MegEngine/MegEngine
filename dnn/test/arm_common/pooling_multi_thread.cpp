/**
 * \file dnn/test/arm_common/pooling_multi_thread.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"

#include "test/common/pooling.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

/*********************** mutli threads *********************************/
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING) {
    using Param = param::Pooling;
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
}
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W3x3_S2x2_NCHW44)
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
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W3x3_S1x1_NCHW44)
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

TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W2x2_S1x1_NCHW44)
{
    // clang-format off
    for (size_t ih: {2, 5, 10, 17})
    for (size_t iw: {2, 6, 8, 16, 26})
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
        param.window_h = param.window_w = 2;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W2x2_S2x2_NCHW44)
{
    // clang-format off
    for (size_t ih: {2, 5, 10, 17})
    for (size_t iw: {2, 6, 8, 16, 26})
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
        param.window_h = param.window_w = 2;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W4x4_S1x1_NCHW44)
{
    // clang-format off
    for (size_t ih: {4, 7, 10, 17, 20})
    for (size_t iw: {4, 8, 10, 21, 32})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 4 && iw+2*pw >= 4)
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
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W4x4_S2x2_NCHW44)
{
    // clang-format off
    for (size_t ih: {4, 10, 18, 25, 30})
    for (size_t iw: {4, 12, 17, 20, 25})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 4 && iw+2*pw >= 4)
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
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W5x5_S1x1_NCHW44)
{
    // clang-format off
    for (size_t ih: {5, 9, 19, 20, 39})
    for (size_t iw: {5, 12, 23, 27, 39})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 5 && iw+2*pw >= 5)
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
        param.window_h = param.window_w = 5;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_MAX_W5x5_S2x2_NCHW44)
{
    // clang-format off
    for (size_t ih: {5, 9, 19, 20, 39})
    for (size_t iw: {5, 12, 23, 27, 39})
    for (size_t ph: {0})
    for (size_t pw: {0})
    if (ih+2*ph >= 5 && iw+2*pw >= 5)
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
        param.window_h = param.window_w = 5;
        checker.set_param(param).exec(TensorShapeArray{{2, 2, ih, iw, 4}, {}});
    }
    // clang-format on
}

TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_INT8_W3x3_S2x2)
{
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
        checker.set_param(param).exec(TensorShapeArray{
                {2, 3, ih, iw}, {}});
    }
}

TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_INT8_W2x2_S2x2)
{
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
        checker.set_param(param).exec(TensorShapeArray{
                {2, 3, ih, iw}, {}});
    }
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_FP16) {
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

TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_QUANTIZED) {
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
TEST_F(ARM_COMMON_MULTI_THREADS, POOLING_FALLBACK) {
    using Param = param::Pooling;
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
    }
}

#if MEGDNN_WITH_BENCHMARK
namespace {
template <typename Opr>
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

TEST_F(ARM_COMMON_BENCHMARK_MULTI_THREADS, BENCHMARK_POOLING) {
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
           param.stride_h, param.pad_h, static_cast<int>(param.mode));
    benchmark_impl<Pooling>(param, shapes, RUNS, {4, {0, 1, 2, 3}}, {1, {0}});
    benchmark_impl<Pooling>(param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {4}});
    benchmark_impl<Pooling>(param, shapes, RUNS, {2, {0, 1}}, {1, {0}});
}
#endif

}  // namespace test
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
