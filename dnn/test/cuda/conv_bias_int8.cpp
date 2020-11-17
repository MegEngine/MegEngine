/**
 * \file dnn/test/cuda/conv_bias_int8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/oprs/nn.h"

#include "src/common/utils.h"
#include "src/cuda/cudnn_with_check.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

#define V1(x) #x
#define V(x) V1(x)

namespace megdnn {
namespace test {
namespace {

#if MEGDNN_WITH_BENCHMARK
struct BenchArgs {
    size_t n, ci, hi, wi, co, f, s;
};

std::vector<BenchArgs> get_resnet50_bench_args(size_t batch = 64) {
    std::vector<BenchArgs> args;
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 256, 1, 1});
    args.emplace_back(BenchArgs{batch, 256, 56, 56, 32, 3, 1});
    args.emplace_back(BenchArgs{batch, 256, 56, 56, 32, 3, 2});
    args.emplace_back(BenchArgs{batch, 4, 256, 256, 32, 7, 2});

    args.emplace_back(BenchArgs{batch, 256, 56, 56, 64, 1, 1});
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 64, 1, 1});
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 64, 3, 1});
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 64, 3, 2});
    args.emplace_back(BenchArgs{batch, 256, 56, 56, 64, 3, 2});

    args.emplace_back(BenchArgs{batch, 256, 56, 56, 512, 1, 2});
    args.emplace_back(BenchArgs{batch, 256, 56, 56, 128, 1, 2});
    args.emplace_back(BenchArgs{batch, 512, 28, 28, 128, 1, 1});
    args.emplace_back(BenchArgs{batch, 128, 28, 28, 128, 3, 1});
    args.emplace_back(BenchArgs{batch, 128, 28, 28, 512, 1, 1});

    args.emplace_back(BenchArgs{batch, 512, 28, 28, 1024, 1, 2});
    args.emplace_back(BenchArgs{batch, 512, 28, 28, 256, 1, 2});
    args.emplace_back(BenchArgs{batch, 1024, 14, 14, 256, 1, 1});
    args.emplace_back(BenchArgs{batch, 256, 14, 14, 256, 3, 1});
    args.emplace_back(BenchArgs{batch, 256, 14, 14, 1024, 1, 1});
    args.emplace_back(BenchArgs{batch, 256, 14, 14, 1024, 1, 2});

    args.emplace_back(BenchArgs{batch, 1024, 14, 14, 2048, 1, 2});
    args.emplace_back(BenchArgs{batch, 1024, 14, 14, 512, 1, 2});
    args.emplace_back(BenchArgs{batch, 2048, 7, 7, 512, 1, 1});
    args.emplace_back(BenchArgs{batch, 512, 7, 7, 512, 3, 1});
    args.emplace_back(BenchArgs{batch, 512, 7, 7, 2048, 1, 1});
    return args;
}

std::vector<BenchArgs> get_detection_bench_args(size_t batch = 16) {
    std::vector<BenchArgs> args;
    args.emplace_back(BenchArgs{batch, 4, 736, 1280, 8, 3, 2});
    args.emplace_back(BenchArgs{batch, 32, 184, 320, 16, 3, 1});
    args.emplace_back(BenchArgs{batch, 16, 184, 320, 32, 3, 1});
    args.emplace_back(BenchArgs{batch, 8, 184, 320, 16, 3, 1});
    args.emplace_back(BenchArgs{batch, 8, 184, 320, 32, 3, 1});
    args.emplace_back(BenchArgs{batch, 64, 92, 160, 32, 3, 1});
    args.emplace_back(BenchArgs{batch, 32, 184, 320, 64, 3, 2});
    args.emplace_back(BenchArgs{batch, 32, 184, 320, 32, 3, 2});
    args.emplace_back(BenchArgs{batch, 32, 92, 160, 64, 3, 1});
    args.emplace_back(BenchArgs{batch, 64, 92, 160, 8, 3, 1});
    args.emplace_back(BenchArgs{batch, 64, 92, 160, 128, 3, 2});
    args.emplace_back(BenchArgs{batch, 128, 46, 80, 32, 3, 1});
    args.emplace_back(BenchArgs{batch, 128, 46, 80, 256, 3, 2});
    args.emplace_back(BenchArgs{batch, 128, 46, 80, 8, 3, 1});
    args.emplace_back(BenchArgs{batch, 64, 92, 160, 32, 3, 2});
    args.emplace_back(BenchArgs{batch, 32, 46, 80, 128, 3, 1});
    args.emplace_back(BenchArgs{batch, 8, 46, 80, 32, 3, 1});
    args.emplace_back(BenchArgs{batch, 64, 23, 40, 256, 3, 1});
    args.emplace_back(BenchArgs{batch, 256, 23, 40, 64, 3, 1});
    args.emplace_back(BenchArgs{batch, 128, 46, 80, 64, 3, 2});
    args.emplace_back(BenchArgs{batch, 256, 23, 40, 8, 3, 1});
    args.emplace_back(BenchArgs{batch, 8, 23, 40, 32, 3, 2});
    args.emplace_back(BenchArgs{batch, 8, 12, 20, 8, 3, 1});
    args.emplace_back(BenchArgs{batch, 8, 12, 20, 8, 3, 2});
    args.emplace_back(BenchArgs{batch, 8, 6, 10, 8, 3, 1});
    return args;
}

std::vector<BenchArgs> get_det_first_bench_args(size_t batch = 16) {
    std::vector<BenchArgs> args;
    args.emplace_back(BenchArgs{batch, 4, 736, 1280, 16, 3, 2});
    args.emplace_back(BenchArgs{batch, 16, 384, 640, 16, 3, 1});
    return args;
}

void benchmark_target_algo(
        Handle* handle, const std::vector<BenchArgs>& args, DType src_dtype,
        DType filter_dtype, DType bias_dtype, DType dst_dtype,
        const char* algo = nullptr,
        param::ConvBias::Format format = param::ConvBias::Format::NCHW4) {
    megdnn_assert(src_dtype.enumv() == filter_dtype.enumv());
    CUBenchmarker<ConvBiasForward> benchmarker(handle);
    CUBenchmarker<ConvBiasForward> benchmarker_cudnn(handle);
    size_t RUNS = 1000;
    benchmarker.set_display(false).set_times(RUNS);
    benchmarker_cudnn.set_display(false).set_times(RUNS);

#define CUDNN_VERSION_STRING \
    "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)
    benchmarker_cudnn.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "DEFAULT:CUDNN:ConvBiasActivation:CUDNN_CONVOLUTION_FWD_"
                    "ALGO_IMPLICIT_PRECOMP_"
                    "GEMM" CUDNN_VERSION_STRING));

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

    using Param = ConvBias::Param;
    using Format = Param::Format;
    // helper function to change format
    auto get_tensor_shape = [](TensorShape shape,
                               Format format) -> TensorShape {
        TensorShape ret;
        if (format == Format::NCHW4) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype::Int8()}
                            .reshape({shape[0], shape[1] / 4, 4, shape[2],
                                      shape[3]})
                            .dimshuffle({0, 1, 3, 4, 2}));
        } else if (format == Format::CHWN4) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype::Int8()}
                            .reshape({shape[0], shape[1] / 4, 4, shape[2],
                                      shape[3]})
                            .dimshuffle({1, 3, 4, 0, 2}));
        }
        return ret;
    };

    for (auto&& arg : args) {
        Param param;
        param.pad_h = param.pad_w = arg.f / 2;
        param.stride_h = param.stride_w = arg.s;
        param.format = format;

        size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
        size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

        benchmarker.set_param(param);
        if (!algo) {
            benchmarker.proxy()->target_algo_info.reset();
        }
        TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                filter{arg.co, arg.ci, arg.f, arg.f}, bias{1, arg.co, 1, 1},
                z{arg.n, arg.co, ho, wo}, dst = z;
        float time_in_ms = 0.f;
        if (algo) {
            time_in_ms =
                    algo_benchmark<ConvBiasForward, OprProxy<ConvBiasForward>,
                                   CUTimer>(benchmarker,
                                            {get_tensor_shape(src, format),
                                             get_tensor_shape(filter, format),
                                             get_tensor_shape(bias, format),
                                             {},
                                             {}},
                                            algo) /
                    RUNS;
        } else {
            time_in_ms = benchmarker.execs({get_tensor_shape(src, format),
                                            get_tensor_shape(filter, format),
                                            get_tensor_shape(bias, format),
                                            {},
                                            {}}) /
                         RUNS;
        }
        Format format_cudnn = Format::NCHW4;
        param.format = format_cudnn;
        benchmarker_cudnn.set_param(param);
        auto time_in_ms_cudnn =
                benchmarker_cudnn.execs({get_tensor_shape(src, format_cudnn),
                                         get_tensor_shape(filter, format_cudnn),
                                         get_tensor_shape(bias, format_cudnn),
                                         {},
                                         {}}) /
                RUNS;
        float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f * arg.f /
                    (1e12);
        printf("src=%s, filter=%s, dst=%s, time(algo=%s)=%.2f %.2fTops, "
               "time(cudnn)=%.2f %.2fTops, "
               "perf(algo=%s)/perf(cudnn)=%.2f\n",
               src.to_string().c_str(), filter.to_string().c_str(),
               dst.to_string().c_str(), algo, time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
               (flo / (time_in_ms_cudnn * 1e-3)), algo,
               time_in_ms_cudnn / time_in_ms);
        printf("bench with z tensor\n");
        if (algo) {
            time_in_ms =
                    algo_benchmark<ConvBiasForward, OprProxy<ConvBiasForward>,
                                   CUTimer>(benchmarker,
                                            {get_tensor_shape(src, format),
                                             get_tensor_shape(filter, format),
                                             get_tensor_shape(bias, format),
                                             get_tensor_shape(z, format),
                                             {}},
                                            algo) /
                    RUNS;
        } else {
            time_in_ms = benchmarker.execs({get_tensor_shape(src, format),
                                            get_tensor_shape(filter, format),
                                            get_tensor_shape(bias, format),
                                            get_tensor_shape(z, format),
                                            {}}) /
                         RUNS;
        }
        time_in_ms_cudnn =
                benchmarker_cudnn.execs({get_tensor_shape(src, format_cudnn),
                                         get_tensor_shape(filter, format_cudnn),
                                         get_tensor_shape(bias, format_cudnn),
                                         get_tensor_shape(z, format_cudnn),
                                         {}}) /
                RUNS;
        printf("src=%s, filter=%s, dst=%s, time(algo=%s)=%.2f %.2fTops, "
               "time(cudnn)=%.2f %.2fTops, "
               "perf(algo=%s)/perf(cudnn)=%.2f\n",
               src.to_string().c_str(), filter.to_string().c_str(),
               dst.to_string().c_str(), algo, time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
               (flo / (time_in_ms_cudnn * 1e-3)), algo,
               time_in_ms_cudnn / time_in_ms);
    }
}

void benchmark_target_algo_with_cudnn_tsc(
        Handle* handle, const std::vector<BenchArgs>& args, DType src_dtype,
        DType filter_dtype, DType bias_dtype, DType dst_dtype,
        const char* algo = nullptr,
        param::ConvBias::Format format = param::ConvBias::Format::NCHW4) {
    megdnn_assert(src_dtype.enumv() == filter_dtype.enumv());
    CUBenchmarker<ConvBiasForward> benchmarker(handle);
    CUBenchmarker<ConvBiasForward> benchmarker_cudnn(handle);
    size_t RUNS = 1000;
    benchmarker.set_display(false).set_times(RUNS);
    benchmarker_cudnn.set_display(false).set_times(RUNS);

    std::unique_ptr<OprProxy<ConvBiasForward>> proxy{
            new OprProxy<ConvBiasForward>{true}};

    if (!algo) {
        benchmarker.set_proxy(proxy);
    }

    benchmarker_cudnn.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "DEFAULT:CUDNN:ConvBiasActivation:CUDNN_CONVOLUTION_FWD_"
                    "ALGO_IMPLICIT_PRECOMP_"
                    "GEMM" CUDNN_VERSION_STRING));
#undef CUDNN_VERSION_STRING

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

    using Param = ConvBias::Param;
    using Format = Param::Format;
    // helper function to change format
    auto get_tensor_shape = [](TensorShape shape,
                               Format format) -> TensorShape {
        TensorShape ret;
        if (format == Format::NCHW4) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype::Int8()}
                            .reshape({shape[0], shape[1] / 4, 4, shape[2],
                                      shape[3]})
                            .dimshuffle({0, 1, 3, 4, 2}));
        } else if (format == Format::NCHW32) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype::Int8()}
                            .reshape({shape[0], shape[1] / 32, 32, shape[2],
                                      shape[3]})
                            .dimshuffle({0, 1, 3, 4, 2}));
        } else if (format == Format::CHWN4) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype::Int8()}
                            .reshape({shape[0], shape[1] / 4, 4, shape[2],
                                      shape[3]})
                            .dimshuffle({1, 3, 4, 0, 2}));
        }
        return ret;
    };

    for (auto&& arg : args) {
        Param param;
        param.pad_h = param.pad_w = arg.f / 2;
        param.stride_h = param.stride_w = arg.s;
        param.format = format;

        size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
        size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

        benchmarker.set_param(param);
        if (!algo) {
            benchmarker.proxy()->target_algo_info.reset();
        }
        TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                filter{arg.co, arg.ci, arg.f, arg.f}, bias{1, arg.co, 1, 1},
                z{arg.n, arg.co, ho, wo}, dst = z;
        // skip testcase which cannot enable nchw32 tensorcore
        if (format == Format::NCHW32 && (arg.co % 32 != 0 || arg.ci % 32 != 0))
            continue;
        // skip testcase which cannot enable nchw4/chwn4 tensorcore
        if ((format == Format::CHWN4 || format == Format::NCHW4) &&
            (arg.ci % 16 != 0))
            continue;
        Format format_cudnn = arg.ci % 32 == 0 && arg.co % 32 == 0
                                      ? Format::NCHW32
                                      : Format::NCHW4;
        param.format = format_cudnn;
        benchmarker_cudnn.set_param(param);

        float time_in_ms = 0.f;
        if (algo) {
            time_in_ms =
                    algo_benchmark<ConvBiasForward, OprProxy<ConvBiasForward>,
                                   CUTimer>(benchmarker,
                                            {get_tensor_shape(src, format),
                                             get_tensor_shape(filter, format),
                                             get_tensor_shape(bias, format),
                                             {},
                                             {}},
                                            algo) /
                    RUNS;
        } else {
            time_in_ms = benchmarker.execs({get_tensor_shape(src, format),
                                            get_tensor_shape(filter, format),
                                            get_tensor_shape(bias, format),
                                            {},
                                            {}}) /
                         RUNS;
        }
        float time_in_ms_cudnn =
                benchmarker_cudnn.execs({get_tensor_shape(src, format_cudnn),
                                         get_tensor_shape(filter, format_cudnn),
                                         get_tensor_shape(bias, format_cudnn),
                                         {},
                                         {}}) /
                RUNS;

        float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f * arg.f /
                    (1e12);
        printf("src=%s, filter=%s, dst=%s, time(algo=%s)=%.2f %.2fTops, "
               "time(cudnn)=%.2f %.2fTops, "
               "perf(algo=%s)/perf(cudnn)=%.2f\n",
               src.to_string().c_str(), filter.to_string().c_str(),
               dst.to_string().c_str(), algo, time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
               (flo / (time_in_ms_cudnn * 1e-3)), algo,
               time_in_ms_cudnn / time_in_ms);
        printf("bench with z tensor\n");
        if (algo) {
            time_in_ms =
                    algo_benchmark<ConvBiasForward, OprProxy<ConvBiasForward>,
                                   CUTimer>(benchmarker,
                                            {get_tensor_shape(src, format),
                                             get_tensor_shape(filter, format),
                                             get_tensor_shape(bias, format),
                                             get_tensor_shape(z, format),
                                             {}},
                                            algo) /
                    RUNS;
        } else {
            time_in_ms = benchmarker.execs({get_tensor_shape(src, format),
                                            get_tensor_shape(filter, format),
                                            get_tensor_shape(bias, format),
                                            get_tensor_shape(z, format),
                                            {}}) /
                         RUNS;
        }
        time_in_ms_cudnn =
                benchmarker_cudnn.execs({get_tensor_shape(src, format_cudnn),
                                         get_tensor_shape(filter, format_cudnn),
                                         get_tensor_shape(bias, format_cudnn),
                                         get_tensor_shape(z, format_cudnn),
                                         {}}) /
                RUNS;
        printf("src=%s, filter=%s, dst=%s, time(algo=%s)=%.2f %.2fTops, "
               "time(cudnn)=%.2f %.2fTops, "
               "perf(algo=%s)/perf(cudnn)=%.2f\n",
               src.to_string().c_str(), filter.to_string().c_str(),
               dst.to_string().c_str(), algo, time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
               (flo / (time_in_ms_cudnn * 1e-3)), algo,
               time_in_ms_cudnn / time_in_ms);
    }
}
#endif
}  // namespace

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_1x1) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4, conv_bias::get_int8_nchw4_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_3x3) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_5x5) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4, conv_bias::get_int8_nchw4_args(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_7x7) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4, conv_bias::get_int8_nchw4_args(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_WITH_Z) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW4;
    checker.set_param(param).execs({{32, 4, 12, 12, 4},
                                    {16, 4, 3, 3, 4},
                                    {1, 4, 1, 1, 4},
                                    {32, 4, 12, 12, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_STRIDE2_WITH_Z) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 2;
    param.format = param::ConvBias::Format::NCHW4;
    checker.set_param(param).execs({{32, 4, 12, 12, 4},
                                    {16, 4, 3, 3, 4},
                                    {1, 4, 1, 1, 4},
                                    {32, 4, 6, 6, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_1x1) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_3x3) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_5x5) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_CHECK_BOUNDS_7x7) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_WITH_Z) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.1f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::CHWN4;
    checker.set_param(param).execs({{4, 12, 12, 32, 4},
                                    {4, 3, 3, 16, 4},
                                    {4, 1, 1, 1, 4},
                                    {4, 12, 12, 32, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_HSWISH) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(4, dtype::QuantizedS8{0.001f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::CHWN4;
    param.nonlineMode = param::ConvBias::NonlineMode::H_SWISH;
    checker.set_param(param).execs(
            {{4, 12, 12, 32, 4}, {4, 3, 3, 16, 4}, {4, 1, 1, 1, 4}, {}, {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_1x1) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_3x3) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_5x5) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_7x7) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_SMALL_CHANNEL_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_small_channel_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_1x1_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args_check_bounds(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_5x5_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args_check_bounds(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL_7x7_CHECK_BOUNDS) {
    require_compute_capability(6, 1);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_small_channel_args_check_bounds(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_1x1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_3x3) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_5x5) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_7x7) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_tensorcore_args(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_CHECK_BOUNDS_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_CHECK_BOUNDS_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma8x32x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_CHECK_BOUNDS_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma32x8x16",
            param::ConvBias::Format::NCHW4,
            conv_bias::get_int8_nchw4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_tensorcore_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_CHECK_BOUNDS_1x1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_CHECK_BOUNDS_5x5) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_CHECK_BOUNDS_7x7) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(), "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_NCHW4_TENSORCORE_WITH_Z) {
    require_compute_capability(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::NCHW4;
    checker.set_param(param).execs({{64, 8, 12, 12, 4},
                                    {64, 8, 3, 3, 4},
                                    {1, 16, 1, 1, 4},
                                    {64, 16, 12, 12, 4},
                                    {}});
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_TENSORCORE_WITH_Z) {
    require_compute_capability(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16"));
    UniformIntRNG rng{-3, 3};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1 + 1e-3)
            .set_max_avg_error(1e-1)
            .set_max_avg_biased_error(1e-1);
    param::ConvBias param;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.format = param::ConvBias::Format::CHWN4;
    checker.set_param(param).execs({{8, 12, 12, 64, 4},
                                    {8, 3, 3, 64, 4},
                                    {16, 1, 1, 1, 4},
                                    {16, 12, 12, 64, 4},
                                    {}});
}

TEST_F(CUDA,
       CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_CHECK_BOUNDS_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA,
       CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_CHECK_BOUNDS_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA,
       CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_CHECK_BOUNDS_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_check_bounds(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma16x16x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma8x32x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_REFORMAT_FILTER_TENSORCORE_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_REORDER_FILTER_mma32x8x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_ALGO_0) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma8x32x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.3f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma32x8x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(3));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_1x1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4, conv_bias::get_int8_chwn4_args(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_5x5) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_7x7) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma16x16x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(7));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_5x5_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_5x5_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(5));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_1x1_ALGO_1) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma32x8x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(1));
}

TEST_F(CUDA, CONV_BIAS_INT8_CHWN4_UNROLL_WIDTH_TENSORCORE_1x1_ALGO_2) {
    require_compute_capability(7, 5);
    conv_bias::check_conv_bias(
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.1f},
            handle_cuda(),
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_UNROLL_WIDTH_mma8x32x16",
            param::ConvBias::Format::CHWN4,
            conv_bias::get_int8_chwn4_args_small_batch(1));
}


TEST_F(CUDA, CUTLASS_WEIGHT_PREPROCESS) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward, OprWeightPreprocessProxy<ConvBiasForward>> checker(
            handle_cuda());
    auto check = [&checker](const std::string& algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo.c_str()));
        UniformIntRNG rng{-16, 16};
        UniformIntRNG bias_rng{-50, 50};
        UniformIntRNG const_rng{1, 1};
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng)
                .set_dtype(0, dtype::QuantizedS8{1.2f})
                .set_dtype(1, dtype::QuantizedS8{1.3f})
                .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
                .set_dtype(3, dtype::QuantizedS8{1.3f})
                .set_dtype(4, dtype::QuantizedS8{1.0f})
                .set_epsilon(1 + 1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 2;
        param.format = param::ConvBias::Format::NCHW4;
        checker.set_param(param).execs({{16, 4, 14, 14, 4},
                                        {16, 4, 3, 3, 4},
                                        {1, 4, 1, 1, 4},
                                        {},
                                        {}});
    };
    check("INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_128X32X32_64X32X32");
    check("INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_16X64X8_16X64X8");
}

#if CUDA_VERSION >= 10020
/// \note: we only check several cases and block sizes in megdnn_test, the
/// full testcases are written in cutlass repository
TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NCHW32_IMMA) {
    require_compute_capability_eq(7, 5);
    Checker<ConvBiasForward> checker(handle_cuda());
    auto check = [&checker](const std::string& algo) {
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo.c_str()));
        UniformIntRNG rng{-8, 8};
        UniformIntRNG bias_rng{-50, 50};
        UniformIntRNG const_rng{1, 1};
        // use scale that are all integers to avoid rouding error
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &bias_rng)
                .set_rng(3, &rng)
                .set_dtype(0, dtype::QuantizedS8{6.0f})
                .set_dtype(1, dtype::QuantizedS8{1.0f})
                .set_dtype(2, dtype::QuantizedS32{6.0f})
                .set_dtype(3, dtype::QuantizedS8{1.0f})
                .set_dtype(4, dtype::QuantizedS8{6.0f})
                .set_epsilon(1e-3);
        param::ConvBias param;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        param.format = param::ConvBias::Format::NCHW32;
        checker.set_param(param).execs({{16, 16, 7, 7, 32},
                                        {512, 16, 3, 3, 32},
                                        {1, 16, 1, 1, 32},
                                        {},
                                        {}});
        param.nonlineMode = param::ConvBias::NonlineMode::RELU;
        checker.set_param(param).execs({{16, 16, 7, 7, 32},
                                        {512, 16, 1, 1, 32},
                                        {1, 16, 1, 1, 32},
                                        {},
                                        {}});
        param.nonlineMode = param::ConvBias::NonlineMode::H_SWISH;
        checker.set_param(param).execs({{16, 16, 7, 7, 32},
                                        {512, 16, 3, 3, 32},
                                        {1, 16, 1, 1, 32},
                                        {},
                                        {}});
        // use non integer scale
        param.nonlineMode = param::ConvBias::NonlineMode::H_SWISH;
        checker.set_dtype(0, dtype::QuantizedS8{1.1f})
                .set_dtype(1, dtype::QuantizedS8{1.2f})
                .set_dtype(2, dtype::QuantizedS32{1.1f * 1.2f})
                .set_dtype(3, dtype::QuantizedS8{1.1f})
                .set_dtype(4, dtype::QuantizedS8{6.0f})
                .set_epsilon(1 + 1e-3)
                .set_max_avg_error(1e-1)
                .set_max_avg_biased_error(1e-1)
                .execs({{16, 16, 7, 7, 32},
                        {512, 16, 3, 3, 32},
                        {1, 16, 1, 1, 32},
                        {16, 16, 7, 7, 32},
                        {}});
    };
    std::string algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NCHW32_IMMA_IMPLICIT_GEMM_256X128X64_64X64X64",
            ConvBias::DirectParam{});
    check(algo);
    algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "INT8_NCHW32_IMMA_IMPLICIT_GEMM_32X64X64_32X16X64",
            ConvBias::DirectParam{});
    check(algo);
}
#endif

TEST_F(CUDA, CUTLASS_CONV_BIAS_INT8_NCHW4_NCHW) {
    require_compute_capability(6, 1);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-3, 3};
    UniformFloatRNG float_rng{-50, 50};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW4_NCHW;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM"));
    checker.set_dtype(0, dtype::QuantizedS8(1.9980618f))
            .set_dtype(1, dtype::QuantizedS8(1.9980927f))
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &float_rng)
            .set_rng(3, &float_rng)
            .set_param(param);

    auto opr = handle_cuda()->create_operator<ConvBias>();

    auto run = [&](const TensorShapeArray& shapes) {
        opr->param() = param;
        TensorLayout dst_layout;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, {}, {}, dst_layout);
        checker.execs({shapes[0], shapes[1], shapes[2], dst_layout, {}});
    };

    run({{16, 4, 23, 40, 4}, {20, 4, 3, 3, 4}, {1, 20, 1, 1}});
    run({{16, 4, 92, 160, 4}, {24, 4, 3, 3, 4}, {1, 24, 1, 1}});
    run({{16, 4, 92, 160, 4}, {20, 4, 3, 3, 4}, {1, 20, 1, 1}});
    run({{16, 4, 92, 160, 4}, {16, 4, 3, 3, 4}, {1, 16, 1, 1}});
    run({{16, 4, 92, 160, 4}, {8, 4, 3, 3, 4}, {1, 8, 1, 1}});
    run({{16, 4, 46, 80, 4}, {4, 4, 3, 3, 4}, {1, 4, 1, 1}});
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_resnet50_bench_args(), dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_NCHW4) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_resnet50_bench_args(), dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_TENSORCORE) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_CHWN4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_TENSORCORE_ALL_ALGO) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f}, nullptr,
            param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_DET_ALL_ALGO) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_detection_bench_args(), dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, nullptr, param::ConvBias::Format::CHWN4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_NCHW4_TENSORCORE) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_NCHW4_IMMA_IMPLICIT_GEMM_mma16x16x16",
            param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_CONV_BIAS_INT8_CHWN4_SMALL_CHANNEL) {
    require_compute_capability(6, 1);
    std::vector<BenchArgs> args;
    args.push_back(BenchArgs{64, 4, 224, 224, 64, 7, 2});
    benchmark_target_algo(
            handle_cuda(), args, dtype::QuantizedS8{1.2f},
            dtype::QuantizedS8{1.3f}, dtype::QuantizedS32{1.2f * 1.3f},
            dtype::QuantizedS8{1.0f}, "INT8_CHWN4_DOTPROD_IMPLICIT_GEMM",
            param::ConvBias::Format::CHWN4);
}


#if CUDA_VERSION >= 10020
TEST_F(CUDA, BENCHMARK_CUTLASS_CONV_BIAS_INT8_NCHW32) {
    require_compute_capability(7, 5);
    benchmark_target_algo_with_cudnn_tsc(
            handle_cuda(), get_resnet50_bench_args(256),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "DIRECT:INT8_NCHW32_IMMA_IMPLICIT_GEMM",
            param::ConvBias::Format::NCHW32);
}
#endif

TEST_F(CUDA, BENCHMARK_CUTLASS_CONV_BIAS_INT8_NCHW4) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_resnet50_bench_args(64),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM", param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_SASS_CONV_BIAS_INT8_NCHW4_DET_FIRST) {
    require_compute_capability(6, 1);
    std::string algo = ConvBias::algo_name<ConvBias::DirectParam>(
            "SASS_INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_128X32_64",
            ConvBias::DirectParam{});
    benchmark_target_algo(handle_cuda(), get_det_first_bench_args(16),
                          dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
                          dtype::QuantizedS32{1.2f * 1.3f},
                          dtype::QuantizedS8{1.0f}, algo.c_str(),
                          param::ConvBias::Format::NCHW4);
}

TEST_F(CUDA, BENCHMARK_CUTLASS_CONV_BIAS_INT8_NCHW4_DET_FIRST) {
    require_compute_capability(6, 1);
    benchmark_target_algo(
            handle_cuda(), get_det_first_bench_args(16),
            dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f},
            dtype::QuantizedS32{1.2f * 1.3f}, dtype::QuantizedS8{1.0f},
            "INT8_NCHW4_DOTPROD_IMPLICIT_GEMM_16", param::ConvBias::Format::NCHW4);
}

#endif
}  // namespace test
}  // namespace megdnn

#undef V1
#undef V

// vim: syntax=cpp.doxygen
