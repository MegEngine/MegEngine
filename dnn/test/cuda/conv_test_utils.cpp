/**
 * \file dnn/test/cuda/conv_test_utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/conv_test_utils.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

#define V1(x) #x
#define V(x) V1(x)

namespace megdnn {
namespace test {
namespace conv {

#if MEGDNN_WITH_BENCHMARK

std::vector<BenchArgs> get_resnet50_bench_args(size_t batch) {
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

std::vector<BenchArgs> get_detection_bench_args(size_t batch) {
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

std::vector<BenchArgs> get_det_first_bench_args(size_t batch) {
    std::vector<BenchArgs> args;
    args.emplace_back(BenchArgs{batch, 4, 736, 1280, 16, 3, 2});
    args.emplace_back(BenchArgs{batch, 16, 384, 640, 16, 3, 1});
    args.emplace_back(BenchArgs{batch, 16, 384, 640, 32, 3, 2});
    args.emplace_back(BenchArgs{batch, 32, 184, 320, 32, 3, 1});
    args.emplace_back(BenchArgs{batch, 32, 184, 320, 32, 1, 1});
    return args;
}

void benchmark_target_algo(Handle* handle, const std::vector<BenchArgs>& args,
                           DType src_dtype, DType filter_dtype,
                           DType bias_dtype, DType dst_dtype, const char* algo,
                           param::ConvBias::Format format) {
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
            benchmarker.proxy()->target_execution_policy.algo.reset();
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
        DType filter_dtype, DType bias_dtype, DType dst_dtype, const char* algo,
        param::ConvBias::Format format, bool with_cudnn,
        const char* change_cudnn_algo,
        param::ConvBias::Format change_cudnn_format,
        DType change_cudnn_src_dtype, DType change_cudnn_filter_dtype,
        DType change_cudnn_bias_dtype, DType change_cudnn_dst_dtype) {
    megdnn_assert((src_dtype.enumv() == filter_dtype.enumv()) ||
                  (src_dtype.enumv() == DTypeEnum::Quantized4Asymm &&
                   filter_dtype.enumv() == DTypeEnum::QuantizedS4));
    CUBenchmarker<ConvBiasForward> benchmarker(handle);
    CUBenchmarker<ConvBiasForward> benchmarker_cudnn(handle);
    size_t RUNS = 200;
    benchmarker.set_display(false).set_times(RUNS);
    benchmarker.set_dtype(0, src_dtype)
            .set_dtype(1, filter_dtype)
            .set_dtype(2, bias_dtype)
            .set_dtype(3, dst_dtype)
            .set_dtype(4, dst_dtype);

    benchmarker_cudnn.set_display(false).set_times(RUNS);

    std::unique_ptr<OprProxy<ConvBiasForward>> proxy{
            new OprProxy<ConvBiasForward>{true}};

    if (!algo) {
        benchmarker.set_proxy(proxy);
    }
    if (change_cudnn_algo) {
        benchmarker_cudnn.set_dtype(0, change_cudnn_src_dtype)
                .set_dtype(1, change_cudnn_filter_dtype)
                .set_dtype(2, change_cudnn_bias_dtype)
                .set_dtype(3, change_cudnn_dst_dtype)
                .set_dtype(4, change_cudnn_dst_dtype);
    } else {
        benchmarker_cudnn.set_dtype(0, src_dtype)
                .set_dtype(1, filter_dtype)
                .set_dtype(2, bias_dtype)
                .set_dtype(3, dst_dtype)
                .set_dtype(4, dst_dtype);
        benchmarker_cudnn.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
                        "DEFAULT:CUDNN:ConvBiasActivation:CUDNN_CONVOLUTION_"
                        "FWD_"
                        "ALGO_IMPLICIT_PRECOMP_GEMM" CUDNN_VERSION_STRING));
    }
#undef CUDNN_VERSION_STRING

    using Param = ConvBias::Param;
    using Format = Param::Format;
    // helper function to change format
    auto get_tensor_shape = [](TensorShape shape, DType dtype,
                               Format format) -> TensorShape {
        TensorShape ret;
        if (format == Format::NCHW4) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype}
                            .reshape({shape[0], shape[1] / 4, 4, shape[2],
                                      shape[3]})
                            .dimshuffle({0, 1, 3, 4, 2}));
        } else if (format == Format::NCHW32) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype}
                            .reshape({shape[0], shape[1] / 32, 32, shape[2],
                                      shape[3]})
                            .dimshuffle({0, 1, 3, 4, 2}));
        } else if (format == Format::NCHW64) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype}
                            .reshape({shape[0], shape[1] / 64, 64, shape[2],
                                      shape[3]})
                            .dimshuffle({0, 1, 3, 4, 2}));
        } else if (format == Format::CHWN4) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype}
                            .reshape({shape[0], shape[1] / 4, 4, shape[2],
                                      shape[3]})
                            .dimshuffle({1, 3, 4, 0, 2}));
        } else if (format == Format::NHWC) {
            ret = static_cast<TensorShape>(
                    TensorLayout{shape, dtype}.dimshuffle({0, 2, 3, 1}));
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
            benchmarker.proxy()->target_execution_policy.algo.reset();
        }
        TensorShape src{arg.n, arg.ci, arg.hi, arg.wi},
                filter{arg.co, arg.ci, arg.f, arg.f}, bias{1, arg.co, 1, 1},
                z{arg.n, arg.co, ho, wo}, dst = z;
        // skip testcase which cannot enable nchw32 tensorcore
        if (format == Format::NCHW32 && (arg.co % 32 != 0 || arg.ci % 32 != 0))
            continue;
        // skip testcase which cannot enable nchw32 tensorcore
        if (format == Format::NCHW64 && (arg.co % 64 != 0 || arg.ci % 64 != 0))
            continue;
        // skip testcase which cannot enable nchw4/chwn4 tensorcore
        if ((format == Format::CHWN4 || format == Format::NCHW4) &&
            (arg.ci % 16 != 0))
            continue;
        // skip testcase which cannot enable nhwc tensorcore
        if ((format == Format::NHWC) && (arg.ci % 4 != 0 || arg.co % 4 != 0))
            continue;
        Format format_cudnn = arg.ci % 32 == 0 && arg.co % 32 == 0
                                      ? Format::NCHW32
                                      : Format::NCHW4;
        if (change_cudnn_algo) {
            format_cudnn = change_cudnn_format;
        }

        param.format = format_cudnn;
        benchmarker_cudnn.set_param(param);

        float time_in_ms = 0.f;
        if (algo) {
            time_in_ms =
                    algo_benchmark<ConvBiasForward, OprProxy<ConvBiasForward>,
                                   CUTimer>(
                            benchmarker,
                            {get_tensor_shape(src, src_dtype, format),
                             get_tensor_shape(filter, filter_dtype, format),
                             get_tensor_shape(bias, bias_dtype, format),
                             {},
                             {}},
                            algo) /
                    RUNS;
        } else {
            time_in_ms =
                    benchmarker.execs(
                            {get_tensor_shape(src, src_dtype, format),
                             get_tensor_shape(filter, filter_dtype, format),
                             get_tensor_shape(bias, bias_dtype, format),
                             {},
                             {}}) /
                    RUNS;
        }
        float time_in_ms_cudnn = 0;
        if (with_cudnn) {
            if (change_cudnn_algo) {
                time_in_ms_cudnn =
                        algo_benchmark<ConvBiasForward,
                                       OprProxy<ConvBiasForward>, CUTimer>(
                                benchmarker_cudnn,
                                {get_tensor_shape(src, src_dtype, format_cudnn),
                                 get_tensor_shape(filter, filter_dtype,
                                                  format_cudnn),
                                 get_tensor_shape(bias, bias_dtype,
                                                  format_cudnn),
                                 {},
                                 {}},
                                change_cudnn_algo) /
                        RUNS;
            } else {
                time_in_ms_cudnn =
                        benchmarker_cudnn.execs(
                                {get_tensor_shape(src, src_dtype, format_cudnn),
                                 get_tensor_shape(filter, filter_dtype,
                                                  format_cudnn),
                                 get_tensor_shape(bias, bias_dtype,
                                                  format_cudnn),
                                 {},
                                 {}}) /
                        RUNS;
            }
        }

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
                                   CUTimer>(
                            benchmarker,
                            {get_tensor_shape(src, src_dtype, format),
                             get_tensor_shape(filter, filter_dtype, format),
                             get_tensor_shape(bias, bias_dtype, format),
                             get_tensor_shape(z, src_dtype, format),
                             {}},
                            algo) /
                    RUNS;
        } else {
            time_in_ms =
                    benchmarker.execs(
                            {get_tensor_shape(src, src_dtype, format),
                             get_tensor_shape(filter, filter_dtype, format),
                             get_tensor_shape(bias, bias_dtype, format),
                             get_tensor_shape(z, src_dtype, format),
                             {}}) /
                    RUNS;
        }
        time_in_ms_cudnn = 0;
        if (with_cudnn) {
            if (change_cudnn_algo) {
                time_in_ms_cudnn =
                        algo_benchmark<ConvBiasForward,
                                       OprProxy<ConvBiasForward>, CUTimer>(
                                benchmarker_cudnn,
                                {get_tensor_shape(src, src_dtype, format_cudnn),
                                 get_tensor_shape(filter, filter_dtype,
                                                  format_cudnn),
                                 get_tensor_shape(bias, bias_dtype,
                                                  format_cudnn),
                                 get_tensor_shape(z, src_dtype, format_cudnn),
                                 {}},
                                change_cudnn_algo) /
                        RUNS;
            } else {
                time_in_ms_cudnn =
                        benchmarker_cudnn.execs(
                                {get_tensor_shape(src, src_dtype, format_cudnn),
                                 get_tensor_shape(filter, filter_dtype,
                                                  format_cudnn),
                                 get_tensor_shape(bias, bias_dtype,
                                                  format_cudnn),
                                 get_tensor_shape(z, src_dtype, format_cudnn),
                                 {}}) /
                        RUNS;
            }
        }
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
}  // namespace conv
}  // namespace test
}  // namespace megdnn
#undef V1
#undef V
