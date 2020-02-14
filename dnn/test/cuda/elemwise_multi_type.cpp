/**
 * \file dnn/test/cuda/elemwise_multi_type.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/elemwise_multi_type.h"
#include "megdnn/oprs/nn_int.h"
#include "test/common/checker.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

#undef cuda_check
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace test;

namespace {
template <typename tag>
class CUDA_ELEMWISE_MULTI_TYPE : public CUDA {};
TYPED_TEST_CASE(CUDA_ELEMWISE_MULTI_TYPE, elemwise_multi_type::test_types);
}  // anonymous namespace

TYPED_TEST(CUDA_ELEMWISE_MULTI_TYPE, run) {
    elemwise_multi_type::run_test<TypeParam>(this->handle_cuda());
}


using Mode = ElemwiseMultiType::Param::Mode;
static void run_test(int arity, Checker<ElemwiseMultiType>& checker, Mode mode) {
    for (auto type : std::vector<std::pair<DType, DType>>{
                 {dtype::QuantizedS8(1.4f), dtype::QuantizedS8(1.7f)},
                 {dtype::QuantizedS8(1.4f), dtype::QuantizedS32(0.1f)},
                 {dtype::QuantizedS32(0.1f), dtype::QuantizedS8(0.4f)}
         }) {
        if (type.first.enumv() == DTypeEnum::QuantizedS32 ||
            type.second.enumv() == DTypeEnum::QuantizedS32) {
            if (mode != Mode::QRELU && mode != Mode::QH_SWISH &&
                mode != Mode::QSIGMOID && mode != Mode::QTANH &&
                mode != Mode::QFAST_TANH && mode != Mode::QADD &&
                mode != Mode::QFUSE_ADD_RELU &&
                mode != Mode::QFUSE_ADD_SIGMOID &&
                mode != Mode::QFUSE_ADD_TANH &&
                mode != Mode::QFUSE_ADD_H_SWISH) {
                return;
            }
        }
        checker.set_param(mode);
        UniformIntRNG rng_int8{-127, 127};
        UniformIntRNG rng_uint8{0, 225};
        UniformIntRNG rng_low{-4, 4};
        UniformIntRNG rng_sigmoid_tanh{-2, 2};
        UniformIntRNG rng_int32{INT16_MIN >> 1, INT16_MAX >> 1};

        auto set_rng = [&](DType dtype, size_t i) {
            if (dtype.enumv() == DTypeEnum::QuantizedS8) {
                checker.set_rng(i, &rng_int8);
            } else if (dtype.enumv() == DTypeEnum::Quantized8Asymm) {
                checker.set_rng(i, &rng_uint8);
            } else {
                megdnn_assert(dtype.enumv() == DTypeEnum::QuantizedS32);
                checker.set_rng(i, &rng_int32);
            }
            if (mode == Mode::QEXP || mode == Mode::QPOW ||
                mode == Mode::QTRUE_DIV || mode == Mode::QLOG_SUM_EXP) {
                checker.set_rng(i, &rng_low);
            }
            checker.set_dtype(i, dtype);
        };
        //! As some mode may cause compute error
        checker.set_epsilon(1 + 1e-3);

        auto src_type = type.first;
        auto dst_type = type.second;
        for (int i = 0; i < arity; i++) {
            set_rng(src_type, i);
        }
        set_rng(dst_type, arity);

        if (arity == 1) {
            checker.execs({{3, 4, 5, 6}, {}})
                    .execs({{1, 4, 5, 1}, {}})
                    .execs({{1, 1, 5, 1}, {}})
                    .execs({{3}, {}})
                    .execs({{9}, {}})
                    .execs({{17}, {}});
        } else if (arity == 2){
            checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {}})
                    .execs({{3, 4, 5, 6}, {1, 1, 1, 1}, {}})
                    .execs({{1, 4, 5, 1}, {4, 1, 1, 5}, {}})
                    .execs({{1, 1, 5, 1}, {4, 1, 5, 5}, {}});
        } else {
            megdnn_assert(0);
        }
    }
}

TEST_F(CUDA, ELEMWISE_QUANTIZED_MODE_UNARY) {
    Checker<ElemwiseMultiType> checker(handle_cuda());
    for (auto mode :
         {Mode::QRELU,    Mode::QABS,    Mode::QACOS,   Mode::QASIN,
          Mode::QCEIL,    Mode::QCOS,    Mode::QEXP,    Mode::QEXPM1,
          Mode::QFLOOR,   Mode::QLOG,    Mode::QLOG1P,  Mode::QNEGATE,
          Mode::QSIGMOID, Mode::QSIN,    Mode::QTANH,   Mode::QFAST_TANH,
          Mode::QROUND,   Mode::QERF,    Mode::QERFINV, Mode::QERFC,
          Mode::QERFCINV, Mode::QH_SWISH}) {
        run_test(1, checker, mode);
    }
}

TEST_F(CUDA, ELEMWISE_QUANTIZED_MODE_BINARY) {
    using Mode = ElemwiseMultiType::Param::Mode;

    Checker<ElemwiseMultiType> checker(handle_cuda());
    for (auto mode : {Mode::QABS_GRAD,
                      Mode::QADD,
                      Mode::QFLOOR_DIV,
                      Mode::QMAX,
                      Mode::QMIN,
                      Mode::QMOD,
                      Mode::QMUL,
                      Mode::QPOW,
                      Mode::QSUB,
                      Mode::QSWITCH_GT0,
                      Mode::QTRUE_DIV,
                      Mode::QLOG_SUM_EXP,

                      Mode::QLT,
                      Mode::QLEQ,
                      Mode::QEQ,

                      Mode::QFUSE_ADD_RELU,
                      Mode::QFUSE_ADD_SIGMOID,
                      Mode::QFUSE_ADD_TANH,
                      Mode::QFAST_TANH_GRAD,
                      Mode::QATAN2,
                      Mode::QH_SWISH_GRAD,
                      Mode::QFUSE_ADD_H_SWISH}) {
        run_test(2, checker, mode);
    }
}

TEST_F(CUDA, ELEMWISE_QUANTIZED_MODE_TENARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle_cuda());

    for (auto mode : {Mode::QFUSE_MUL_ADD3, Mode::QCOND_LEQ_MOV}) {
        printf("Testing mode: %d\n", (int)mode);
        UniformIntRNG rng_int8{-127, 127};
        UniformIntRNG rng_uint8{0, 225};
        checker.set_param({mode})
                .set_rng(0, &rng_int8)
                .set_rng(1, &rng_int8)
                .set_rng(2, &rng_int8)
                .set_dtype(0, dtype::QuantizedS8(1.2f))
                .set_dtype(1, dtype::QuantizedS8(1.6f))
                .set_dtype(2, dtype::QuantizedS8(1.8f))
                .set_dtype(3, dtype::QuantizedS8(1.4f))
                .execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {1, 1, 1, 1}, {}})
                .execs({{1, 4, 1, 1}, {3, 4, 5, 6}, {1, 4, 1, 1}, {}})
                .execs({{3}, {3}, {3}, {}})
                .execs({{9}, {9}, {9}, {}})
                .execs({{17}, {17}, {17}, {}})
                .execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {3, 4, 5, 6}, {}});

    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_ELEMWISE_QUANTIZED_MODE_UNARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    CUBenchmarker<ElemwiseMultiType> bencher(handle_cuda());
    UniformIntRNG rng{-128, 127};

    for (auto mode :
         {Mode::QRELU,    Mode::QABS,    Mode::QACOS,   Mode::QASIN,
          Mode::QCEIL,    Mode::QCOS,    Mode::QEXP,    Mode::QEXPM1,
          Mode::QFLOOR,   Mode::QLOG,    Mode::QLOG1P,  Mode::QNEGATE,
          Mode::QSIGMOID, Mode::QSIN,    Mode::QTANH,   Mode::QFAST_TANH,
          Mode::QROUND,   Mode::QERF,    Mode::QERFINV, Mode::QERFC,
          Mode::QERFCINV, Mode::QH_SWISH}) {
        printf("Benchmark mode: %d\n", (int)mode);
        bencher.set_param({mode})
                .set_rng(0, &rng)
                .set_dtype(0, dtype::QuantizedS8(0.1f))
                .set_dtype(1, dtype::QuantizedS8(0.2f));
        size_t nr_times = 50;
        bencher.set_times(nr_times);
        auto run_bench = [&](size_t N, size_t C, size_t H, size_t W) {
            printf("(NxCxHxW)=(%zux%zux%zux%zu)\n", N, C, H, W);
            auto time = bencher.execs({{N, C / 4, H, W, 4}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * N * C * H * W) / (time * 1e6));
            time = bencher.execs({{N, C / 4, H, W, 4}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * N * C * H * W) / (time * 1e6));

            time = bencher.execs({{N, C / 32, H, W, 32}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * N * C * H * W) / (time * 1e6));

            time = bencher.execs({{N, C / 32, H, W, 32}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * N * C * H * W) / (time * 1e6));
            time = bencher.execs({{N * C * H * W + 1}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * (N * C * H * W + 1)) / (time * 1e6));
            time = bencher.execs({{N * C * H * W}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * N * C * H * W) / (time * 1e6));

        };
        run_bench(256, 256, 56, 56);
        run_bench(64, 256, 56, 56);
        run_bench(256, 128, 28, 28);
        run_bench(64, 128, 28, 28);
    }
}

TEST_F(CUDA, BENCHMARK_ELEMWISE_QUANTIZED_MODE_BINARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    CUBenchmarker<ElemwiseMultiType> bencher(handle_cuda());
    UniformIntRNG rng{-128, 127};

    for (auto mode : {Mode::QABS_GRAD,
                      Mode::QADD,
                      Mode::QFLOOR_DIV,
                      Mode::QMAX,
                      Mode::QMIN,
                      Mode::QMOD,
                      Mode::QMUL,
                      Mode::QPOW,
                      Mode::QSIGMOID_GRAD,
                      Mode::QSUB,
                      Mode::QSWITCH_GT0,
                      Mode::QTANH_GRAD,
                      Mode::QTRUE_DIV,
                      Mode::QLOG_SUM_EXP,

                      Mode::QLT,
                      Mode::QLEQ,
                      Mode::QEQ,

                      Mode::QFUSE_ADD_RELU,
                      Mode::QFUSE_ADD_SIGMOID,
                      Mode::QFUSE_ADD_TANH,
                      Mode::QFAST_TANH_GRAD,
                      Mode::QATAN2,
                      Mode::QH_SWISH_GRAD,
                      Mode::QFUSE_ADD_H_SWISH}) {
        printf("Benchmark mode: %d\n", (int)mode);
        bencher.set_param({mode})
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_dtype(0, dtype::QuantizedS8(0.1f))
                .set_dtype(1, dtype::QuantizedS8(0.2f))
                .set_dtype(2, dtype::QuantizedS8(0.01f));
        size_t nr_times = 50;
        bencher.set_times(nr_times);
        auto run_bench = [&](size_t N, size_t C, size_t H, size_t W) {
            printf("(NxCxHxW)=(%zux%zux%zux%zu)\n", N, C, H, W);
            auto time =
                    bencher.execs(
                            {{N, C / 4, H, W, 4}, {N, C / 4, H, W, 4}, {}}) /
                    nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (3.0 * N * C * H * W) / (time * 1e6));
            time = bencher.execs(
                           {{N, C / 4, H, W, 4}, {1, C / 4, 1, 1, 4}, {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (C + 2.0 * N * C * H * W) / (time * 1e6));

            time = bencher.execs(
                           {{N, C / 32, H, W, 32}, {N, C / 32, H, W, 32}, {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (3.0 * N * C * H * W) / (time * 1e6));

            time = bencher.execs(
                           {{N, C / 32, H, W, 32}, {1, C / 32, 1, 1, 32}, {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (C + 2.0 * N * C * H * W) / (time * 1e6));
            time = bencher.execs(
                           {{N * C * H * W + 1}, {N * C * H * W + 1}, {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (3.0 * (N * C * H * W + 1)) / (time * 1e6));
            time = bencher.execs({{N * C * H * W}, {1}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * N * C * H * W + 1) / (time * 1e6));

        };
        run_bench(256, 256, 56, 56);
        run_bench(64, 256, 56, 56);
        run_bench(256, 128, 28, 28);
        run_bench(64, 128, 28, 28);
    }
}

TEST_F(CUDA, BENCHMARK_ELEMWISE_QUANTIZED_MODE_TENARY) {
    using Mode = ElemwiseMultiType::Param::Mode;
    CUBenchmarker<ElemwiseMultiType> bencher(handle_cuda());
    UniformIntRNG rng{-128, 127};

    for (auto mode : {Mode::QFUSE_MUL_ADD3, Mode::QCOND_LEQ_MOV}) {
        printf("Benchmark mode: %d\n", (int)mode);
        bencher.set_param({mode})
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_dtype(0, dtype::QuantizedS8(0.1f))
                .set_dtype(1, dtype::QuantizedS8(0.2f))
                .set_dtype(2, dtype::QuantizedS8(0.01f))
                .set_dtype(3, dtype::QuantizedS8(0.01f));
        size_t nr_times = 50;
        bencher.set_times(nr_times);
        auto run_bench = [&](size_t N, size_t C, size_t H, size_t W) {
            printf("(NxCxHxW)=(%zux%zux%zux%zu)\n", N, C, H, W);
            auto time = bencher.execs({{N, C / 4, H, W, 4},
                                       {N, C / 4, H, W, 4},
                                       {N, C / 4, H, W, 4},
                                       {}}) /
                        nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (4.0 * N * C * H * W) / (time * 1e6));
            time = bencher.execs({{N, C / 4, H, W, 4},
                                  {1, C / 4, 1, 1, 4},
                                  {1, C / 4, 1, 1, 4},
                                  {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2 * C + 2.0 * N * C * H * W) / (time * 1e6));

            time = bencher.execs({{N, C / 32, H, W, 32},
                                  {N, C / 32, H, W, 32},
                                  {N, C / 32, H, W, 32},
                                  {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (4.0 * N * C * H * W) / (time * 1e6));

            time = bencher.execs({{N, C / 32, H, W, 32},
                                  {1, C / 32, 1, 1, 32},
                                  {1, C / 32, 1, 1, 32},
                                  {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2 * C + 2.0 * N * C * H * W) / (time * 1e6));
            time = bencher.execs({{N * C * H * W + 1},
                                  {N * C * H * W + 1},
                                  {N * C * H * W + 1},
                                  {}}) /
                   nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (4.0 * (N * C * H * W + 1)) / (time * 1e6));
            time = bencher.execs({{N * C * H * W}, {1}, {1}, {}}) / nr_times;
            printf("time = %.2f, bandwidth = %.2f GB/s\n", time,
                   (2.0 * N * C * H * W + 1) / (time * 1e6));

        };
        run_bench(256, 256, 56, 56);
        run_bench(64, 256, 56, 56);
        run_bench(256, 128, 28, 28);
        run_bench(64, 128, 28, 28);
    }
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
