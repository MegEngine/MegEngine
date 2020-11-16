/**
 * \file dnn/test/cuda/batch_conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/nn.h"

#include "src/common/utils.h"
#include "src/cuda/cudnn_with_check.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

namespace megdnn {
namespace test {
namespace {
struct TestArg {
    param::BatchConvBias param;
    TensorShape src, filter, bias;
    TestArg(param::BatchConvBias param, TensorShape src, TensorShape filter,
            TensorShape bias)
            : param{param}, src{src}, filter{filter}, bias{bias} {}
};

std::vector<TestArg> get_int8_nchw4_args(size_t kernel_size = 1) {
    std::vector<TestArg> args;
    using NLMode = param::BatchConvBias::NonlineMode;

    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU, NLMode::H_SWISH}) {
        for (size_t b : {1, 2}) {
            for (size_t ic : {4, 8, 16}) {
                for (size_t oc : {4, 44, 84, 132}) {
                    for (size_t h : {8, 16}) {
                        for (size_t w : {4, 8}) {
                            for (int p :
                                 {0, static_cast<int>(kernel_size / 2)}) {
                                for (size_t s : {1, 2}) {
                                    size_t f = kernel_size;
                                    param::BatchConvBias param;
                                    param.nonlineMode = nlmode;
                                    param.format =
                                            param::BatchConvBias::Format::NCHW4;
                                    param.sparse =
                                            param::BatchConvBias::Sparse::DENSE;
                                    param.pad_h = param.pad_w = p;
                                    param.stride_h = param.stride_w = s;

                                    args.emplace_back(
                                            param,
                                            TensorShape{b, ic / 4, h, w, 4},
                                            TensorShape{b, oc, ic / 4, f, f, 4},
                                            TensorShape{1, oc / 4, 1, 1, 4});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return args;
}

std::vector<TestArg> get_int8_nchw4_args_gemm() {
    std::vector<TestArg> args;
    using NLMode = param::BatchConvBias::NonlineMode;

    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU, NLMode::H_SWISH}) {
        for (size_t b : {1, 2}) {
            for (size_t ic : {4, 8, 16}) {
                for (size_t oc : {32, 64, 128}) {
                    for (size_t h : {8, 16}) {
                        for (size_t w : {4, 8}) {
                            size_t s = 1;
                            size_t p = 0;
                            size_t f = 1;
                            param::BatchConvBias param;
                            param.nonlineMode = nlmode;
                            param.format = param::BatchConvBias::Format::NCHW4;
                            param.sparse = param::BatchConvBias::Sparse::DENSE;
                            param.pad_h = param.pad_w = p;
                            param.stride_h = param.stride_w = s;

                            args.emplace_back(
                                    param, TensorShape{b, ic / 4, h, w, 4},
                                    TensorShape{b, oc, ic / 4, f, f, 4},
                                    TensorShape{1, oc / 4, 1, 1, 4});
                        }
                    }
                }
            }
        }
    }
    return args;
}

std::vector<TestArg> get_int8_nchw4_args_gemm_check_bounds() {
    std::vector<TestArg> args;
    using NLMode = param::BatchConvBias::NonlineMode;

    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU, NLMode::H_SWISH}) {
        for (size_t b : {1, 2}) {
            for (size_t ic : {4, 8, 16}) {
                for (size_t oc : {4, 40, 80}) {
                    for (size_t h : {7, 15}) {
                        for (size_t w : {3, 7}) {
                            size_t s = 1;
                            size_t p = 0;
                            size_t f = 1;
                            param::BatchConvBias param;
                            param.nonlineMode = nlmode;
                            param.format = param::BatchConvBias::Format::NCHW4;
                            param.sparse = param::BatchConvBias::Sparse::DENSE;
                            param.pad_h = param.pad_w = p;
                            param.stride_h = param.stride_w = s;

                            args.emplace_back(
                                    param, TensorShape{b, ic / 4, h, w, 4},
                                    TensorShape{b, oc, ic / 4, f, f, 4},
                                    TensorShape{1, oc / 4, 1, 1, 4});
                        }
                    }
                }
            }
        }
    }
    return args;
}

void check_batch_conv_bias(DType src_dtype, DType filter_dtype,
                           DType bias_dtype, DType dst_dtype, Handle* handle,
                           const char* algo, const std::vector<TestArg>& args) {
    megdnn_assert(src_dtype.enumv() == filter_dtype.enumv());
    Checker<BatchConvBiasForward> checker(handle);
    if (algo) {
        checker.set_before_exec_callback(
                AlgoChecker<BatchConvBiasForward>(algo));
    }
    std::unique_ptr<RNG> rng;
    std::unique_ptr<RNG> bias_rng;
    std::unique_ptr<RNG> const_rng;
    // TODO: check range of rng
    if (src_dtype.enumv() == DTypeEnum::QuantizedS8) {
        rng = std::make_unique<UniformIntRNG>(-3, 3);
        const_rng = std::make_unique<UniformIntRNG>(1, 1);
        megdnn_assert(bias_dtype.enumv() == DTypeEnum::QuantizedS32);
        bias_rng = std::make_unique<UniformIntRNG>(-50, 50);
        checker.set_epsilon(1 + 1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-1);
    } else if (src_dtype.enumv() == DTypeEnum::Float16) {
        rng = std::make_unique<NormalRNG>(2.f);
        megdnn_assert(bias_dtype.enumv() == DTypeEnum::Float16);
        bias_rng = std::make_unique<NormalRNG>(2.f);
        checker.set_epsilon(1e-2);
    } else if (src_dtype.enumv() == DTypeEnum::Float32) {
        rng = std::make_unique<NormalRNG>(2.f);
        megdnn_assert(bias_dtype.enumv() == DTypeEnum::Float32);
        bias_rng = std::make_unique<NormalRNG>(2.f);
    }

    megdnn_assert(rng != nullptr && bias_rng != nullptr);
    checker.set_rng(0, rng.get())
            .set_rng(1, rng.get())
            .set_rng(2, rng.get())
            .set_rng(3, rng.get());
    for (auto&& arg : args) {
        checker.set_dtype(0, src_dtype)
                .set_dtype(1, filter_dtype)
                .set_dtype(2, bias_dtype)
                .set_dtype(4, dst_dtype)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
struct BenchArgs {
    size_t n, ci, hi, wi, co, f, s;
};

std::vector<BenchArgs> get_facerec_bench_args(size_t batch = 64) {
    std::vector<BenchArgs> args;
    args.emplace_back(BenchArgs{1, 4096, 64, 64, 4096, 1, 1});
    args.emplace_back(BenchArgs{batch, 128, 24, 24, 128, 1, 1});
    args.emplace_back(BenchArgs{batch, 256, 12, 12, 256, 1, 1});
    args.emplace_back(BenchArgs{batch, 512, 6, 6, 512, 1, 1});
    args.emplace_back(BenchArgs{batch, 1024, 4, 2, 1024, 1, 1});
    args.emplace_back(BenchArgs{batch, 108, 32, 32, 192, 1, 1});
    args.emplace_back(BenchArgs{batch, 192, 16, 16, 384, 1, 1});
    args.emplace_back(BenchArgs{batch, 384, 8, 8, 640, 1, 1});
    args.emplace_back(BenchArgs{batch, 108, 32, 32, 192, 1, 2});
    args.emplace_back(BenchArgs{batch, 192, 16, 16, 192, 1, 1});
    args.emplace_back(BenchArgs{batch, 192, 16, 16, 384, 1, 2});
    args.emplace_back(BenchArgs{batch, 384, 8, 8, 384, 1, 1});
    args.emplace_back(BenchArgs{batch, 384, 8, 8, 640, 1, 2});
    args.emplace_back(BenchArgs{batch, 640, 4, 4, 640, 1, 1});

    return args;
}

void benchmark_target_algo(Handle* handle, const std::vector<BenchArgs>& args,
                           DType src_dtype, DType filter_dtype,
                           DType bias_dtype, DType dst_dtype,
                           const char* algo = nullptr,
                           param::BatchConvBias::Format format =
                                   param::BatchConvBias::Format::NCHW4) {
    megdnn_assert(src_dtype.enumv() == filter_dtype.enumv());
    megdnn_assert(format == param::BatchConvBias::Format::NCHW4);
    CUBenchmarker<BatchConvBiasForward> benchmarker(handle);
    CUBenchmarker<ConvBiasForward> benchmarker_cudnn(handle);
    CUBenchmarker<BatchedMatrixMul> benchmarker_matmul(handle);
    size_t RUNS = 1000;
    benchmarker.set_display(false).set_times(RUNS);
    benchmarker_cudnn.set_display(false).set_times(RUNS);
    benchmarker_matmul.set_display(false).set_times(RUNS);

    std::unique_ptr<OprProxy<BatchConvBiasForward>> proxy{
            new OprProxy<BatchConvBiasForward>{true}};

    if (algo) {
        benchmarker.set_before_exec_callback(
                AlgoChecker<BatchConvBiasForward>(algo));
    } else {
        benchmarker.set_proxy(proxy);
    }

#define V1(x) #x
#define V(x) V1(x)
#define CUDNN_VERSION_STRING \
    "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)
    benchmarker_cudnn.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_"
                    "GEMM" CUDNN_VERSION_STRING));
    benchmarker_matmul.set_before_exec_callback(
            AlgoChecker<BatchedMatrixMul>("BRUTE_FORCE-CUBLAS"));

    benchmarker.set_dtype(0, src_dtype)
            .set_dtype(1, filter_dtype)
            .set_dtype(2, bias_dtype)
            .set_dtype(3, dst_dtype)
            .set_dtype(4, dst_dtype);
    benchmarker_cudnn.set_dtype(0, src_dtype)
            .set_dtype(1, filter_dtype)
            .set_dtype(2, bias_dtype)
            .set_dtype(3, dst_dtype)
            .set_dtype(4, dst_dtype);
    benchmarker_matmul.set_dtype(0, src_dtype)
            .set_dtype(1, filter_dtype)
            .set_dtype(2, bias_dtype);

    using Param = ConvBias::Param;
    using Format = Param::Format;
    if (format == Format::NCHW4) {
        for (auto&& arg : args) {
            ConvBias::Param param;
            param.pad_h = param.pad_w = arg.f / 2;
            param.stride_h = param.stride_w = arg.s;
            param.format = Format::NCHW4;

            BatchConvBias::Param bparam;
            bparam.pad_h = bparam.pad_w = arg.f / 2;
            bparam.stride_h = bparam.stride_w = arg.s;
            bparam.format = Format::NCHW4;

            size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
            size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

            benchmarker.set_param(bparam);
            if (!algo) {
                benchmarker.proxy()->target_algo_info.reset();
            }
            auto time_in_ms =
                    benchmarker.execs(
                            {{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                             {arg.n, arg.co, arg.ci / 4, arg.f, arg.f, 4},
                             {1, arg.co / 4, 1, 1, 4},
                             {},
                             {}}) /
                    RUNS;
            benchmarker_cudnn.set_param(param);
            auto time_in_ms_cudnn =
                    benchmarker_cudnn.execs(
                            {{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                             {arg.co, arg.ci / 4, arg.f, arg.f, 4},
                             {1, arg.co / 4, 1, 1, 4},
                             {},
                             {}}) /
                    RUNS;
            auto time_in_ms_matmul =
                    benchmarker_matmul.execs(
                            {{arg.n, arg.co, arg.ci * arg.f * arg.f},
                             {arg.n, arg.ci * arg.f * arg.f, ho * wo},
                             {}}) /
                    RUNS;

            float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f *
                        arg.f / (1e12);
            TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                    filter{arg.co, arg.ci, arg.f, arg.f};
            printf("src=%s, filter=%s, time(algo=%s)=%.2f %.2fTops, "
                   "time(cudnn)=%.2f %.2fTops, time(batched_matmul)=%.2f "
                   "%.2fTops, "
                   "perf(algo=%s)/perf(cudnn)=%.2f\n, "
                   "perf(algo=%s)/perf(batched_matmul)=%.2f\n",
                   src.to_string().c_str(), filter.to_string().c_str(), algo,
                   time_in_ms, (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
                   (flo / (time_in_ms_cudnn * 1e-3)), time_in_ms_matmul,
                   (flo / (time_in_ms_matmul * 1e-3)), algo,
                   time_in_ms_cudnn / time_in_ms, algo,
                   time_in_ms_matmul / time_in_ms);
        }
    }
}

#endif
}  // namespace

TEST_F(CUDA, BATCH_CONV_BIAS_QS8) {
    require_compute_capability(6, 1);
    Checker<BatchConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(AlgoChecker<BatchConvBiasForward>(
            "BATCH_CONV_BIAS_INT8_NCHW4_IMPLICIT_GEMM_PRECOMP_DOTPROD"));
    UniformIntRNG const_rng{1, 1};
    UniformIntRNG rng{-5, 5};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.1f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::BatchConvBias param;
    param.pad_h = 2, param.pad_w = 1;
    param.stride_h = 1, param.stride_w = 2;
    param.format = param::BatchConvBias::Format::NCHW4;
    checker.set_param(param).execs({{32, 4, 24, 24, 4},
                                    {32, 32, 4, 1, 1, 4},
                                    {1, 8, 1, 1, 4},
                                    {},
                                    {}});
}

TEST_F(CUDA, BATCH_CONV_BIAS_QS8_GEMM) {
    require_compute_capability(6, 1);
    check_batch_conv_bias(dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
                          dtype::QuantizedS32{1.2f * 1.3f},
                          dtype::QuantizedS8{1.1f}, handle_cuda(),
                          "BATCH_CONV_BIAS_INT8_NCHW4_GEMM_DOTPROD",
                          get_int8_nchw4_args_gemm());
}

TEST_F(CUDA, BATCH_CONV_BIAS_QS8_GEMM_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    check_batch_conv_bias(dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
                          dtype::QuantizedS32{1.2f * 1.3f},
                          dtype::QuantizedS8{1.1f}, handle_cuda(),
                          "BATCH_CONV_BIAS_INT8_NCHW4_GEMM_DOTPROD",
                          get_int8_nchw4_args_gemm_check_bounds());
}

TEST_F(CUDA, BATCH_CONV_BIAS_QS8_IMPLICIT_GEMM) {
    require_compute_capability(6, 1);
    check_batch_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "BATCH_CONV_BIAS_INT8_NCHW4_IMPLICIT_GEMM_PRECOMP_DOTPROD",
            get_int8_nchw4_args(1));
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_BATCH_CONV_BIAS_QS8) {
    require_compute_capability(6, 1);
    benchmark_target_algo(handle_cuda(), get_facerec_bench_args(128),
                          dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
                          dtype::QuantizedS32{1.2f * 1.3f},
                          dtype::QuantizedS8{1.0f}, nullptr,
                          param::ConvBias::Format::NCHW4);
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
