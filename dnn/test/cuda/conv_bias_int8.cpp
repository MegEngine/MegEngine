/**
 * \file dnn/test/cuda/conv_bias_int8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
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
#if MEGDNN_WITH_BENCHMARK
namespace {
struct BenchArgs {
    size_t n, ci, hi, wi, co, f, s;
};

std::vector<BenchArgs> get_resnet50_bench_args(size_t batch = 64) {
    std::vector<BenchArgs> args;
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 256, 1, 1});
    args.emplace_back(BenchArgs{batch, 256, 56, 56, 64, 1, 1});
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 64, 1, 1});
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 64, 3, 1});
    args.emplace_back(BenchArgs{batch, 64, 56, 56, 256, 1, 1});

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

    if (algo) {
        benchmarker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo));
    }

#define V1(x) #x
#define V(x) V1(x)
#define CUDNN_VERSION_STRING \
    "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)
    benchmarker_cudnn.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_"
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
    if (format == Format::NCHW4) {
        for (auto&& arg : args) {
            Param param;
            param.pad_h = param.pad_w = arg.f / 2;
            param.stride_h = param.stride_w = arg.s;
            param.format = Format::NCHW4;

            size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
            size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

            benchmarker.set_param(param);
            auto time_in_ms =
                    benchmarker.execs({{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                                       {arg.co, arg.ci / 4, arg.f, arg.f, 4},
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
            float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f *
                        arg.f / (1e12);
            TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                    filter{arg.co, arg.ci, arg.f, arg.f};
            printf("src=%s, filter=%s, time(algo=%s)=%.2f %.2fTops, "
                   "time(cudnn)=%.2f %.2fTops, "
                   "perf(algo=%s)/perf(cudnn)=%.2f\n",
                   src.to_string().c_str(), filter.to_string().c_str(), algo,
                   time_in_ms, (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
                   (flo / (time_in_ms_cudnn * 1e-3)), algo,
                   time_in_ms_cudnn / time_in_ms);
        }
    } else if (format == Format::CHWN4) {
        for (auto&& arg : args) {
            Param param;
            param.pad_h = param.pad_w = arg.f / 2;
            param.stride_h = param.stride_w = arg.s;
            param.format = Format::CHWN4;

            size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
            size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

            benchmarker.set_param(param);
            auto time_in_ms =
                    benchmarker.execs({{arg.ci / 4, arg.hi, arg.wi, arg.n, 4},
                                       {arg.ci / 4, arg.f, arg.f, arg.co, 4},
                                       {arg.co / 4, 1, 1, 1, 4},
                                       {},
                                       {}}) /
                    RUNS;
            param.format = Format::NCHW4;
            benchmarker_cudnn.set_param(param);
            auto time_in_ms_cudnn =
                    benchmarker_cudnn.execs(
                            {{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                             {arg.co, arg.ci / 4, arg.f, arg.f, 4},
                             {1, arg.co / 4, 1, 1, 4},
                             {},
                             {}}) /
                    RUNS;
            float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f *
                        arg.f / (1e12);
            TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                    filter{arg.co, arg.ci, arg.f, arg.f};
            printf("src=%s, filter=%s, time(algo=%s)=%.2f %.2fTops, "
                   "time(cudnn)=%.2f %.2fTops, "
                   "perf(algo=%s)/perf(cudnn)=%.2f\n",
                   src.to_string().c_str(), filter.to_string().c_str(), algo,
                   time_in_ms, (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
                   (flo / (time_in_ms_cudnn * 1e-3)), algo,
                   time_in_ms_cudnn / time_in_ms);
        }
        printf("bench with z tensor\n");
        for (auto&& arg : args) {
            Param param;
            param.pad_h = param.pad_w = arg.f / 2;
            param.stride_h = param.stride_w = arg.s;
            param.format = Format::CHWN4;

            size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
            size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

            benchmarker.set_param(param);
            auto time_in_ms =
                    benchmarker.execs({{arg.ci / 4, arg.hi, arg.wi, arg.n, 4},
                                       {arg.ci / 4, arg.f, arg.f, arg.co, 4},
                                       {arg.co / 4, 1, 1, 1, 4},
                                       {arg.co / 4, ho, wo, arg.n, 4},
                                       {}}) /
                    RUNS;
            param.format = Format::NCHW4;
            benchmarker_cudnn.set_param(param);
            auto time_in_ms_cudnn =
                    benchmarker_cudnn.execs(
                            {{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                             {arg.co, arg.ci / 4, arg.f, arg.f, 4},
                             {1, arg.co / 4, 1, 1, 4},
                             {arg.n, arg.co / 4, ho, wo, 4},
                             {}}) /
                    RUNS;
            float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f *
                        arg.f / (1e12);
            TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                    filter{arg.co, arg.ci, arg.f, arg.f};
            printf("src=%s, filter=%s, time(algo=%s)=%.2f %.2fTops, "
                   "time(cudnn)=%.2f %.2fTops, "
                   "perf(algo=%s)/perf(cudnn)=%.2f\n",
                   src.to_string().c_str(), filter.to_string().c_str(), algo,
                   time_in_ms, (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
                   (flo / (time_in_ms_cudnn * 1e-3)), algo,
                   time_in_ms_cudnn / time_in_ms);
        }
 
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

    if (algo) {
        benchmarker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(algo));
    } else {
        benchmarker.set_proxy(proxy);    
    }

    benchmarker_cudnn.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_"
                    "GEMM" CUDNN_VERSION_STRING));
#undef V1
#undef V
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
    if (format == Format::NCHW4) {
        for (auto&& arg : args) {
            Param param;
            param.pad_h = param.pad_w = arg.f / 2;
            param.stride_h = param.stride_w = arg.s;
            param.format = Format::NCHW4;

            size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
            size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

            benchmarker.set_param(param);
            if (!algo) {
                benchmarker.proxy()->target_algo = nullptr;
            }
            auto time_in_ms =
                    benchmarker.execs({{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                                       {arg.co, arg.ci / 4, arg.f, arg.f, 4},
                                       {1, arg.co / 4, 1, 1, 4},
                                       {},
                                       {}}) /
                    RUNS;
            param.format = Format::NCHW32;
            benchmarker_cudnn.set_param(param);
            auto time_in_ms_cudnn =
                    benchmarker_cudnn.execs(
                            {{arg.n, arg.ci / 32, arg.hi, arg.wi, 32},
                             {arg.co, arg.ci / 32, arg.f, arg.f, 32},
                             {1, arg.co / 32, 1, 1, 32},
                             {},
                             {}}) /
                    RUNS;
            float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f *
                        arg.f / (1e12);
            TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                    filter{arg.co, arg.ci, arg.f, arg.f};
            printf("src=%s, filter=%s, time(algo=%s)=%.2f %.2fTops, "
                   "time(cudnn)=%.2f %.2fTops, "
                   "perf(algo=%s)/perf(cudnn)=%.2f\n",
                   src.to_string().c_str(), filter.to_string().c_str(), algo,
                   time_in_ms, (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
                   (flo / (time_in_ms_cudnn * 1e-3)), algo,
                   time_in_ms_cudnn / time_in_ms);
        }
    } else if (format == Format::CHWN4) {
        for (auto&& arg : args) {
            Param param;
            param.pad_h = param.pad_w = arg.f / 2;
            param.stride_h = param.stride_w = arg.s;
            param.format = Format::CHWN4;

            size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
            size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

            benchmarker.set_param(param);
            if (!algo) {
                benchmarker.proxy()->target_algo = nullptr;
            }
            auto time_in_ms =
                    benchmarker.execs({{arg.ci / 4, arg.hi, arg.wi, arg.n, 4},
                                       {arg.ci / 4, arg.f, arg.f, arg.co, 4},
                                       {arg.co / 4, 1, 1, 1, 4},
                                       {},
                                       {}}) /
                    RUNS;
            float time_in_ms_cudnn = 0.f;
            if (arg.ci % 32 == 0 && arg.co % 32 == 0) {
                param.format = Format::NCHW32;
                benchmarker_cudnn.set_param(param);
                time_in_ms_cudnn =
                        benchmarker_cudnn.execs(
                                {{arg.n, arg.ci / 32, arg.hi, arg.wi, 32},
                                 {arg.co, arg.ci / 32, arg.f, arg.f, 32},
                                 {1, arg.co / 32, 1, 1, 32},
                                 {},
                                 {}}) /
                        RUNS;
            } else {
                param.format = Format::NCHW4;
                benchmarker_cudnn.set_param(param);
                time_in_ms_cudnn =
                        benchmarker_cudnn.execs(
                                {{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                                 {arg.co, arg.ci / 4, arg.f, arg.f, 4},
                                 {1, arg.co / 4, 1, 1, 4},
                                 {},
                                 {}}) /
                        RUNS;
            }
            float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f *
                        arg.f / (1e12);
            TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                    filter{arg.co, arg.ci, arg.f, arg.f};
            printf("src=%s, filter=%s, time(algo=%s)=%.2f %.2fTops, "
                   "time(cudnn)=%.2f %.2fTops, "
                   "perf(algo=%s)/perf(cudnn)=%.2f\n",
                   src.to_string().c_str(), filter.to_string().c_str(), algo,
                   time_in_ms, (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
                   (flo / (time_in_ms_cudnn * 1e-3)), algo,
                   time_in_ms_cudnn / time_in_ms);
        }
        printf("bench with z tensor\n");
        for (auto&& arg : args) {
            Param param;
            param.pad_h = param.pad_w = arg.f / 2;
            param.stride_h = param.stride_w = arg.s;
            param.format = Format::CHWN4;

            size_t ho = infer_conv_shape(arg.hi, arg.f, arg.s, arg.f / 2);
            size_t wo = infer_conv_shape(arg.wi, arg.f, arg.s, arg.f / 2);

            benchmarker.set_param(param);
            if (!algo) {
                benchmarker.proxy()->target_algo = nullptr;
            }
            auto time_in_ms =
                    benchmarker.execs({{arg.ci / 4, arg.hi, arg.wi, arg.n, 4},
                                       {arg.ci / 4, arg.f, arg.f, arg.co, 4},
                                       {arg.co / 4, 1, 1, 1, 4},
                                       {arg.co / 4, ho, wo, arg.n, 4},
                                       {}}) /
                    RUNS;
            float time_in_ms_cudnn = 0.f;
            if (arg.ci % 32 == 0 && arg.co % 32 == 0) {
                param.format = Format::NCHW32;
                benchmarker_cudnn.set_param(param);
                time_in_ms_cudnn =
                        benchmarker_cudnn.execs(
                                {{arg.n, arg.ci / 32, arg.hi, arg.wi, 32},
                                 {arg.co, arg.ci / 32, arg.f, arg.f, 32},
                                 {1, arg.co / 32, 1, 1, 32},
                                 {arg.n, arg.co / 32, ho, wo, 32},
                                 {}}) /
                        RUNS;
            } else {
                param.format = Format::NCHW4;
                benchmarker_cudnn.set_param(param);
                time_in_ms_cudnn =
                        benchmarker_cudnn.execs(
                                {{arg.n, arg.ci / 4, arg.hi, arg.wi, 4},
                                 {arg.co, arg.ci / 4, arg.f, arg.f, 4},
                                 {1, arg.co / 4, 1, 1, 4},
                                 {arg.n, arg.co / 4, ho, wo, 4},
                                 {}}) /
                        RUNS;
            }
            float flo = 2.0 * arg.n * arg.co * ho * wo * arg.ci * arg.f *
                        arg.f / (1e12);
            TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                    filter{arg.co, arg.ci, arg.f, arg.f};
            printf("src=%s, filter=%s, time(algo=%s)=%.2f %.2fTops, "
                   "time(cudnn)=%.2f %.2fTops, "
                   "perf(algo=%s)/perf(cudnn)=%.2f\n",
                   src.to_string().c_str(), filter.to_string().c_str(), algo,
                   time_in_ms, (flo / (time_in_ms * 1e-3)), time_in_ms_cudnn,
                   (flo / (time_in_ms_cudnn * 1e-3)), algo,
                   time_in_ms_cudnn / time_in_ms);
        }
 
    }
}

}  // namespace
#endif

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
    checker.set_param(param).execs({{4, 12, 12, 32, 4},
                                    {4, 3, 3, 16, 4},
                                    {4, 1, 1, 1, 4},
                                    {},
                                    {}});
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
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
