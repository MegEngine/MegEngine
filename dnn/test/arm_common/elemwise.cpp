/**
 * \file dnn/test/arm_common/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/common/elemwise.h"
#include "test/arm_common/fixture.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"

#include "megdnn/oprs/general.h"

using namespace megdnn;
using namespace test;

template <typename tag>
class ARM_ELEMWISE : public ARM_COMMON {};
TYPED_TEST_CASE(ARM_ELEMWISE, elemwise::test_types);
TYPED_TEST(ARM_ELEMWISE, run) {
    elemwise::run_test<TypeParam>(this->handle());
}

#define TERNARY_COMPLATE_TEST_CASE(_optr)                               \
    printf("Check binary optr %s by all cases.\n", #_optr);             \
    checker.set_param(Mode::_optr)                                      \
            .execs({{3, 4, 7}, {3, 4, 7}, {3, 4, 7}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}, {}});     \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 4, 1}, {3, 4, 7}, {1, 4, 1}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}, {}});     \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {1, 7}, {}}); \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 2, 1}, {1, 2, 2}, {1, 2, 1}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 2, 2}, {1, 2, 2}, {1, 1, 1}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{3, 4, 1}, {3, 4, 1}, {3, 4, 1}, {}});              \
    checker.set_param(Mode::_optr).execs({{3, 4, 5}, {1}, {1}, {}});    \
    checker.set_param(Mode::_optr).execs({{1}, {3, 4, 5}, {1}, {}});

#define BUILD_TERNARY_COMPLATE_TEST_CASE \
    TERNARY_COMPLATE_TEST_CASE(FUSE_MUL_ADD3)

TEST_F(ARM_COMMON, ELEMWISE_FORWARD_TERNARY) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());
    // case int
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int8());
    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    checker.set_dtype(2, dtype::Int16());
    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());

    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // case half
    UniformFloatRNG rng_float16(1, 10);
    checker.set_rng(0, &rng_float16);
    checker.set_epsilon(1e-2);
    checker.set_dtype(0, dtype::Float16());
    checker.set_dtype(1, dtype::Float16());
    checker.set_dtype(2, dtype::Float16());

    // BUILD_TERNARY_TEST_CASE
    BUILD_TERNARY_COMPLATE_TEST_CASE
#endif
}

TEST_F(ARM_COMMON, ELEMWISE_FORWARD_NCHW44_INT8_INT16_INT32) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    auto run = [&]() {
        // VEC_BCAST101x not PowOp
        checker.set_param(Mode::ADD).execs(
                {{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH)
                .execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH)
                .execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH)
                .execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH)
                .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::RMULH)
                .execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        // BCAST101x_VEC not PowOp
        checker.set_param(Mode::ADD).execs(
                {{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::ADD).execs(
                {{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
    };
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    run();
    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    run();
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    run();
}

TEST_F(ARM_COMMON, ELEMWISE_FORWARD_NCHW44_FP32) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());

    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});

    auto run = [&](Mode mode) {
        // VEC_BCAST101x
        checker.set_param(mode).execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(mode).execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(mode).execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(mode).execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(mode).execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        // BCAST101x_VEC not powOp
        checker.set_param(mode).execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.set_param(mode).execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.set_param(mode).execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.set_param(mode).execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(mode).execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
    };
    run(Mode::ADD);
    run(Mode::FUSE_ADD_H_SWISH);
    run(Mode::FUSE_ADD_RELU);
    run(Mode::MAX);
    run(Mode::MIN);
    run(Mode::MUL);
    run(Mode::SUB);
    run(Mode::TRUE_DIV);
    run(Mode::POW);
}

#if MEGDNN_WITH_BENCHMARK
namespace {
void run_elemwise_benchmark(const TensorShapeArray& shapes,
                            param::Elemwise::Mode mode, const char* mode_str,
                            DType type, Handle* handle_bench) {
    auto handle_fallback = create_cpu_handle(1);
    Benchmarker<Elemwise> benchmarker_bench(handle_bench);
    Benchmarker<Elemwise> benchmarker_fallback(handle_fallback.get());

    float throughput = 0;
    SmallVector<TensorLayout> layouts;
    std::string src_strs;
    for (size_t i = 0; i < shapes.size(); i++) {
        layouts.emplace_back(shapes[i], type);
        throughput += layouts.back().span().dist_byte();
        src_strs += layouts.back().to_string();
        if (i != shapes.size() - 1) {
            src_strs += ",";
        }
    }
    constexpr size_t RUN = 50;
    benchmarker_fallback.set_times(RUN).set_display(false);
    benchmarker_bench.set_times(RUN).set_display(false);

    benchmarker_fallback.set_param(mode);
    benchmarker_bench.set_param(mode);

    TensorLayout dst_layout;
    auto opr = handle_bench->create_operator<Elemwise>();
    opr->param() = mode;
    opr->deduce_layout(layouts, dst_layout);

    float computations = dst_layout.total_nr_elems() *
                         (std::max<size_t>(shapes.size(), 2) - 1);
    throughput += dst_layout.span().dist_byte();
    computations *= (1e3 / (1024.0 * 1024));
    throughput *= (1e3 / (1024.0 * 1024));

    layouts.emplace_back(dst_layout);
    auto fallback_time = benchmarker_fallback.execl(layouts) / RUN;
    auto bench_time = benchmarker_bench.execl(layouts) / RUN;

    float fallback_flops = computations / fallback_time;
    float bench_flops = computations / bench_time;
    float fallback_thr = throughput / fallback_time;
    float bench_thr = throughput / bench_time;

    printf("%s = %s (type: %s, mode: %s) cpu=%fMFLOPS %fMB/s, bench=%fMFLOPS "
           "%fMB/s "
           "computations: %fx, throughput: %fx\n",
           src_strs.c_str(), dst_layout.to_string().c_str(), type.name(),
           mode_str, fallback_flops, fallback_thr, bench_flops, bench_thr,
           bench_flops / fallback_flops, bench_thr / fallback_thr);
}
}  // namespace

#define INT_RUN(shape, mode)                                              \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Int8{}, handle());  \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Int16{}, handle()); \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Int32{}, handle());

#define FLOAT_RUN(shape, mode)                                              \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Float32{}, handle()); \
    run_elemwise_benchmark(shape, mode, #mode, dtype::Float16{}, handle());

#define BENCHMARK_CASES(shape) \
    INT_BENCHMARK_CASES(shape) \
    FLOAT_BENCHMARK_CASES(shape)

TEST_F(ARM_COMMON, BENCHMARK_UNARY) {
#define INT_BENCHMARK_CASES(shape) \
    INT_RUN(shape, Mode::RELU);    \
    INT_RUN(shape, Mode::ABS);

#define FLOAT_BENCHMARK_CASES(shape) \
    FLOAT_RUN(shape, Mode::RELU);    \
    FLOAT_RUN(shape, Mode::ABS);     \
    FLOAT_RUN(shape, Mode::SIGMOID); \
    FLOAT_RUN(shape, Mode::EXP);     \
    FLOAT_RUN(shape, Mode::TANH);    \
    FLOAT_RUN(shape, Mode::FAST_TANH);

    using Mode = param::Elemwise::Mode;
    BENCHMARK_CASES({{10000}});
    BENCHMARK_CASES({{50000}});

#undef INT_BENCHMARK_CASES
#undef FLOAT_BENCHMARK_CASES
}

TEST_F(ARM_COMMON, BENCHMARK_BINARY) {
#define INT_BENCHMARK_CASES(shape) \
    INT_RUN(shape, Mode::MIN);     \
    INT_RUN(shape, Mode::MAX);     \
    INT_RUN(shape, Mode::ADD);     \
    INT_RUN(shape, Mode::SUB);     \
    INT_RUN(shape, Mode::MUL);     \
    INT_RUN(shape, Mode::RMULH);   \
    INT_RUN(shape, Mode::FUSE_ADD_RELU);

#define FLOAT_BENCHMARK_CASES(shape)  \
    FLOAT_RUN(shape, Mode::MIN);      \
    FLOAT_RUN(shape, Mode::MAX);      \
    FLOAT_RUN(shape, Mode::ADD);      \
    FLOAT_RUN(shape, Mode::SUB);      \
    FLOAT_RUN(shape, Mode::MUL);      \
    FLOAT_RUN(shape, Mode::POW);      \
    FLOAT_RUN(shape, Mode::TRUE_DIV); \
    FLOAT_RUN(shape, Mode::FUSE_ADD_RELU);

    using Mode = param::Elemwise::Mode;
    TensorShapeArray shapes = {{1, 112, 28, 28}, {1, 112, 28, 28}};
    BENCHMARK_CASES(shapes);
    shapes = {{1, 16, 1, 1}, {1, 16, 112, 112}};
    BENCHMARK_CASES(shapes);
    shapes = {{1, 448, 7, 7}, {1, 448, 7, 7}};
    BENCHMARK_CASES(shapes);

#undef INT_BENCHMARK_CASES
#undef FLOAT_BENCHMARK_CASES
}

TEST_F(ARM_COMMON, BENCHMARK_TERNARY_FMA3) {
#define INT_BENCHMARK_CASES(shape) INT_RUN(shape, Mode::FUSE_MUL_ADD3);

#define FLOAT_BENCHMARK_CASES(shape) FLOAT_RUN(shape, Mode::FUSE_MUL_ADD3);

    using Mode = param::Elemwise::Mode;
    TensorShapeArray shapes = {{30, 40, 70}, {30, 40, 70}, {30, 40, 70}};
    BENCHMARK_CASES(shapes);
    shapes = {{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}};
    BENCHMARK_CASES(shapes);
    shapes = {{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}};
    BENCHMARK_CASES(shapes);

#undef INT_BENCHMARK_CASES
#undef FLOAT_BENCHMARK_CASES
}

#undef BENCHMARK_CASES
#undef INT_RUN
#undef FLOAT_RUN

#endif

// vim: syntax=cpp.doxygen
