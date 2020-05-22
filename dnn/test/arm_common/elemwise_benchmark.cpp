/**
 * \file dnn/test/arm_common/elemwise_benchmark.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#if MEGDNN_WITH_BENCHMARK
#include "test/arm_common/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

#define TEST_IN_DIFF_DISTRUBUTION(proportion_of_inf, dataset_number) \
    max_val = 88.3762626647949f / (1 - proportion_of_inf);           \
    UniformFloatRNG rng##dataset_number(0.f, max_val);               \
    B.set_rng(0, &rng##dataset_number);                              \
    B.execs({{355600}, {}});

TEST_F(ARM_COMMON, BENCHMARK_ELEM_UNARY_FLOATONLY) {
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;
    // UniformFloatWithZeroRNG rng(80, 100, 0.1);
    printf("Test Optr exp(x)\n");
    B.set_param(Mode::EXP);
    B.execs({{355600}, {}});

    B.set_param(Mode::EXP);
    B.execs({{355600}, {}});
    float max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 1)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 2)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 3)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 4)

    printf("Test Optr tanh(x)\n");
    B.set_param(Mode::TANH);
    B.execs({{355600}, {}});

    B.set_param(Mode::TANH);
    B.execs({{355600}, {}});
    max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 5)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 6)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 7)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 8)

    printf("Test Optr fast_tanh(x)\n");
    B.set_param(Mode::FAST_TANH);
    B.execs({{355600}, {}});

    printf("Test Optr sigmoid(x)\n");
    B.set_param(Mode::SIGMOID);
    B.execs({{355600}, {}});
    TEST_IN_DIFF_DISTRUBUTION(0.25, 9)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 10)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 11)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 12)

    B.set_param(Mode::SIGMOID);
    B.execs({{355600}, {}});
    max_val = 0;
    TEST_IN_DIFF_DISTRUBUTION(0.25, 13)
    TEST_IN_DIFF_DISTRUBUTION(0.5, 14)
    TEST_IN_DIFF_DISTRUBUTION(0.75, 15)
    TEST_IN_DIFF_DISTRUBUTION(0.9999, 16)
}

TEST_F(ARM_COMMON, BENCHMARK_ELEMWISE_UNARY) {
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;

    const size_t RUN_TIMES = 10;
    B.set_times(RUN_TIMES).set_display(false);

    auto run_unary = [&](const TensorShape& shape, param::Elemwise::Mode mode,
                         const char* mode_str, DType dtype) {
        B.set_param(mode).set_dtype(0, dtype);
        float time = B.execs({shape, {}}) / RUN_TIMES;
        float computations =
                shape.total_nr_elems() * 2 / (1024.f * 1024.f * 1024.f);
        printf("%s(%s):\tlayout(%s)\ttime(%fms)\tbandwidth(%fGBps)\n", mode_str,
               dtype.name(), shape.to_string().c_str(), time,
               computations * dtype.size() / time * 1e3);
    };
#define RUN(shape, mode, dtype) run_unary(shape, mode, #mode, dtype);

#define BENCHMARK_CASES_INT(shape, dtype) \
    RUN(shape, Mode::RELU, dtype)         \
    RUN(shape, Mode::ABS, dtype)

#define BENCHMARK_CASES_FLOAT(shape, dtype) \
    BENCHMARK_CASES_INT(shape, dtype)       \
    RUN(shape, Mode::SIGMOID, dtype)        \
    RUN(shape, Mode::EXP, dtype)            \
    RUN(shape, Mode::TANH, dtype)           \
    RUN(shape, Mode::FAST_TANH, dtype)

    TensorShape shape = {10, 50, 10, 100};
    BENCHMARK_CASES_INT(shape, dtype::Int32());
    BENCHMARK_CASES_INT(shape, dtype::Int16());
    BENCHMARK_CASES_INT(shape, dtype::Int8());
    BENCHMARK_CASES_FLOAT(shape, dtype::Float32());
#undef BENCHMARK_CASES_INT
#undef BENCHMARK_CASES_FLOAT
#undef RUN
}

TEST_F(ARM_COMMON, BENCHMARK_ELEMWISE_UNARY_MULTI_TYPE) {
    Benchmarker<ElemwiseMultiType> B(handle());
    using Mode = ElemwiseMultiType::Param::Mode;

    const size_t RUN_TIMES = 20;
    B.set_times(RUN_TIMES).set_display(false);

    auto run_unary = [&](const TensorShape& shape, Mode mode,
                         const char* mode_str, DType src_dtype,
                         DType dst_dtype) {
        B.set_param(mode).set_dtype(0, src_dtype).set_dtype(1, dst_dtype);
        float time = B.execs({shape, {}}) / RUN_TIMES;
        float computations =
                shape.total_nr_elems() * 2 / (1024.f * 1024.f * 1024.f);
        printf("type %s %s(%s) to %s \ttime(%fms)\tbandwidth(%fGBps)\n",
               mode_str, src_dtype.name(), shape.to_string().c_str(),
               dst_dtype.name(), time,
               computations * src_dtype.size() / time * 1e3);
    };

#define RUN(shape, mode, src_dtype, dst_dtye) \
    run_unary(shape, mode, #mode, src_dtype, dst_dtye);

#define BENCHMARK_CASES_INT(shape, src_dtype, dst_dtye) \
    RUN(shape, Mode::QRELU, src_dtype, dst_dtye)        \
    RUN(shape, Mode::QABS, src_dtype, dst_dtye)

    TensorShape shape = {10, 50, 10, 100};
    BENCHMARK_CASES_INT(shape, dtype::QuantizedS32(62.5f),
                        dtype::QuantizedS8(2.5f));
#undef BENCHMARK_CASES_INT
#undef BENCHMARK_CASES_FLOAT
#undef RUN
}

TEST_F(ARM_COMMON, BENCHMARK_ELEMWISE_BINARY) {
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;

    const size_t RUN_TIMES = 10;
    B.set_times(RUN_TIMES).set_display(false);

    auto run_binary = [&](const TensorShape& shape0, const TensorShape& shape1,
                          param::Elemwise::Mode mode, const char* mode_str,
                          DType dtype) {
        B.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);
        float time = B.execs({shape0, shape1, {}}) / RUN_TIMES;
        float bandwidth =
                (shape0.total_nr_elems() + shape1.total_nr_elems() +
                 std::max(shape0.total_nr_elems(), shape1.total_nr_elems())) /
                (1024.f * 1024.f * 1024.f) * dtype.size() / time * 1e3;
        printf("%s(%s):\tlayout(%s %s)\ttime(%fms)\tbandwidth(%fGBps)\n",
               mode_str, dtype.name(), shape0.to_string().c_str(),
               shape1.to_string().c_str(), time, bandwidth);
    };
#define RUN(shape0, shape1, mode, dtype) \
    run_binary(shape0, shape1, mode, #mode, dtype);

#define BENCHMARK_CASES_INT(shape0, shape1, dtype) \
    RUN(shape0, shape1, Mode::ADD, dtype)          \
    RUN(shape0, shape1, Mode::MIN, dtype)          \
    RUN(shape0, shape1, Mode::MAX, dtype)          \
    RUN(shape0, shape1, Mode::SUB, dtype)          \
    RUN(shape0, shape1, Mode::MUL, dtype)          \
    RUN(shape0, shape1, Mode::FUSE_ADD_RELU, dtype)

#define BENCHMARK_CASES_FLOAT(shape0, shape1, dtype)   \
    BENCHMARK_CASES_INT(shape0, shape1, dtype)         \
    RUN(shape0, shape1, Mode::TRUE_DIV, dtype)         \
    RUN(shape0, shape1, Mode::FUSE_ADD_SIGMOID, dtype) \
    RUN(shape0, shape1, Mode::FUSE_ADD_TANH, dtype)

#define BENCHMARK_CASES_EVERY_DTYPE(shape0, shape1)      \
    BENCHMARK_CASES_INT(shape0, shape1, dtype::Int32()); \
    BENCHMARK_CASES_INT(shape0, shape1, dtype::Int16()); \
    BENCHMARK_CASES_INT(shape0, shape1, dtype::Int8());  \
    BENCHMARK_CASES_FLOAT(shape0, shape1, dtype::Float32());

    TensorShape shape0 = {10, 50, 10, 100};
    TensorShape shape1 = {10, 50, 10, 100};
    BENCHMARK_CASES_EVERY_DTYPE(shape0, shape1);

    shape1 = {1, 50, 1, 1};
    BENCHMARK_CASES_EVERY_DTYPE(shape0, shape1);

    shape1 = {1, 1, 1, 1};
    BENCHMARK_CASES_EVERY_DTYPE(shape0, shape1);
#undef BENCHMARK_CASES_EVERY_DTYPE
#undef BENCHMARK_CASES_FLOAT
#undef BENCHMARK_CASES_INT
#undef RUN
}

TEST_F(ARM_COMMON, BENCHMARK_ELEMWISE_TERNARY) {
    Benchmarker<ElemwiseForward> B(handle());
    using Mode = ElemwiseForward::Param::Mode;

    const size_t RUN_TIMES = 10;
    B.set_times(RUN_TIMES).set_display(false);

    auto run_ternary = [&](const TensorShape& shape0, const TensorShape& shape1,
                           const TensorShape& shape2,
                           param::Elemwise::Mode mode, const char* mode_str,
                           DType dtype) {
        B.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(
                2, dtype);
        float time = B.execs({shape0, shape1, shape2, {}}) / RUN_TIMES;
        float bandwidth = (shape0.total_nr_elems() * 2 +
                           shape1.total_nr_elems() + shape2.total_nr_elems()) /
                          (1024.f * 1024.f * 1024.f) * dtype.size() / time *
                          1e3;
        printf("%s(%s):\tlayout(%s %s %s)\ttime(%fms)\tbandwidth(%fGBps)\n",
               mode_str, dtype.name(), shape0.to_string().c_str(),
               shape1.to_string().c_str(), shape2.to_string().c_str(), time,
               bandwidth);
    };

    TensorShape shape = {10, 50, 10, 100};
    run_ternary(shape, shape, shape, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Int32());
    run_ternary(shape, shape, shape, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Int16());
    run_ternary(shape, shape, shape, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Int8());
    run_ternary(shape, shape, shape, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Float32());
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    run_ternary(shape, {1}, {1}, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Float32());
    run_ternary(shape, {1}, {1}, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Float16());
    run_ternary({1}, shape, {1}, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Float32());
    run_ternary({1}, shape, {1}, Mode::FUSE_MUL_ADD3, "FUSE_MUL_ADD3",
                dtype::Float16());
#endif
}
#endif
