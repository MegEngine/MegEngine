/**
 * \file dnn/test/arm_common/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"
#include "test/common/timer.h"

using namespace megdnn;
using namespace test;

using Param = param::Convolution;

#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON, CONVOLUTION_BACKWARD_DATA_INT8_INT8_INT32) {
    Checker<ConvolutionBackwardData> checker(handle());
    using Param = ConvolutionBackwardData::Param;
    Param param;
    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t ph, size_t pw,
                   size_t group = 1) {
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = stride;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow}, dtype::Int8()};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Int8()};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Int8()};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        if(stride == 1 ){
            checker.set_before_exec_callback(AlgoChecker<
                                             ConvolutionBackwardData>(
                    "AARCH32_I8x8x32_DECONV_STRIDE1"));
        } else {
            checker.set_before_exec_callback(AlgoChecker<
                                             ConvolutionBackwardData>(
                    "AARCH32_I8x8x32_DECONV_STRIDE2"));
        }
        checker.set_param(param)
                .set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32());
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    // clang-format off
    for (size_t f : {2, 3, 5, 7})
    for (size_t ih = 1; ih < f+1; ++ih)
    for (size_t iw = 1; iw < 8*f+1; ++iw)
    for (size_t s : {1, 2})
    for (size_t ph : {f/2, f-1})
    for (size_t pw : {f / 2, f - 1})
    if (f >= ph + 1 && f >= pw + 1 && (ih - 1) * s + f > 2 * ph &&
        (iw - 1) * s + f > 2 * pw) {
        run(2, 3, ih, iw, 2, f, f, s, ph, pw, 1);
    }
    // clang-format on
}

TEST_F(ARM_COMMON, CONVOLUTION_BACKWARD_DATA_QUINT8) {
    Checker<ConvolutionBackwardData> checker(handle());
    using Param = ConvolutionBackwardData::Param;
    Param param;
    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t ph, size_t pw,
                   size_t group = 1) {
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = param.stride_w = stride;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow}, dtype::Quantized8Asymm(1.3f, (uint8_t)129)};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Quantized8Asymm(1.2f, (uint8_t)127)};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Quantized8Asymm(1.2f, (uint8_t)127)};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        NormalRNG rng(128.f);

        if(stride == 1 ){
            checker.set_before_exec_callback(
                    AlgoChecker<ConvolutionBackwardData>(
                            "ARM_COMMON_QUINT8_DIRECT_"
                            "DECONV_STRIDE1"));
        } else {
            checker.set_before_exec_callback(
                    AlgoChecker<ConvolutionBackwardData>(
                            "ARM_COMMON_QUINT8_DIRECT_"
                            "DECONV_STRIDE2"));
        }
        checker.set_param(param)
            .set_dtype(0, dtype::Quantized8Asymm(1.2f, (uint8_t)127))
            .set_dtype(1, dtype::Quantized8Asymm(1.3f, (uint8_t)129))
            .set_dtype(2, {});
        checker.set_rng(0, &rng).set_rng(1, &rng);
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    // clang-format off
    for (size_t f : {2, 3, 5, 7})
    for (size_t ih = 1; ih < f+1; ++ih)
    for (size_t iw = 1; iw < 8*f+1; ++iw)
    for (size_t s : {1, 2})
    for (size_t ph : {f/2, f-1})
    for (size_t pw : {f/2, f-1})
    if (f >= ph + 1 && f >= pw + 1 && (ih - 1) * s + f > 2 * ph &&
        (iw - 1) * s + f > 2 * pw) {
        run(2, 2, ih, iw, 2, f, f, s, ph, pw, 1);
    }
    // clang-format on
}
#endif

#if MEGDNN_WITH_BENCHMARK
#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_STRIDE1_I8x8x32_WITHDOTPROD) {
    using namespace convolution;
    using Param = param::Convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7}) {
        for (size_t ic : {1, 8, 16, 32, 64}) {
            for (size_t oc : {1, 8, 16, 32, 64}) {
                    run(oc, ic, 56, 56, kernel, 1);
                    run(oc, ic, 128, 128, kernel, 1);
                    run(oc, ic, 256, 256, kernel, 1);
                }
            }
    }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_before_exec_callback(
            AlgoChecker<Convolution>("CONVOLUTION_DEFAULT_ARMDOTS8STRD1"));
    benchmark.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark.set_display(false);
    benchmark.set_times(RUN);

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int = benchmark.set_param(arg.param).exec(
                                {arg.src, arg.filter, {}}) /
                        RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                          RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);
    }
}
TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_STRIDE2_I8x8x32_WITHDOTPROD) {
    using namespace convolution;
    using Param = param::Convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7}) {
        for (size_t ic : {1, 8, 16, 32, 64}) {
            for (size_t oc : {1, 8, 16, 32, 64}) {
                run(oc, ic, 56, 56, kernel, 2);
                run(oc, ic, 128, 128, kernel, 2);
                run(oc, ic, 256, 256, kernel, 2);
            }
        }
    }

    constexpr size_t RUN = 10;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_before_exec_callback(
            AlgoChecker<Convolution>("CONVOLUTION_DEFAULT_ARMDOTS8STRD2"));
    benchmark.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark.set_display(false);
    benchmark.set_times(RUN);

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int =
                benchmark.set_param(arg.param).exec({arg.src, arg.filter, {}}) /
                RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                          RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_STRIDE1_QUINT8_WITHDOTPROD) {
    using namespace convolution;
    using Param = param::Convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7}) {
        for (size_t ic : {1, 8, 16, 32, 64}) {
            for (size_t oc : {1, 8, 16, 32, 64}) {
                    run(oc, ic, 56, 56, kernel, 1);
                    run(oc, ic, 128, 128, kernel, 1);
                    run(oc, ic, 256, 256, kernel, 1);
                }
            }
    }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_dtype(0, dtype::Quantized8Asymm(1.2f, (uint8_t)129))
             .set_dtype(1, dtype::Quantized8Asymm(1.3f, (uint8_t)127))
             .set_dtype(2, {});

    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_ARMDOTU8STRD1"));

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
      //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int = benchmark.set_param(arg.param).exec(
                                {arg.src, arg.filter, {}}) /
                        RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                        RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);

    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_STRIDE2_QUINT8_WITHDOTPROD) {
    using namespace convolution;
    using Param = param::Convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7}) {
        for (size_t ic : {1, 8, 16, 32, 64}) {
            for (size_t oc : {1, 8, 16, 32, 64}) {
                    run(oc, ic, 56, 56, kernel, 2);
                    run(oc, ic, 128, 128, kernel, 2);
                    run(oc, ic, 256, 256, kernel, 2);
                }
            }
    }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_dtype(0, dtype::Quantized8Asymm(1.2f, (uint8_t)129))
             .set_dtype(1, dtype::Quantized8Asymm(1.3f, (uint8_t)127))
             .set_dtype(2, {});

    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_ARMDOTU8STRD2"));

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int = benchmark.set_param(arg.param).exec(
                                {arg.src, arg.filter, {}}) /
                        RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                          RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_BACKWARD_DATA_INT8_INT8_INT32) {
    using Param = ConvolutionBackwardData::Param;

    auto run = [&](const TensorLayoutArray& tensors, Param param) {
        Benchmarker<ConvolutionBackwardData> benchmarker(handle());
        size_t RUN = 50;
        auto time = benchmarker.set_display(false)
                            .set_dtype(0, dtype::Int8{})
                            .set_dtype(1, dtype::Int8{})
                            .set_dtype(2, dtype::Int32{})
                            .set_times(RUN)
                            .set_param(param)
                            .exec(tensors);

        size_t OC = tensors[0][0];
        size_t FH = tensors[0][2];
        size_t FW = tensors[0][3];
        float computations = tensors[2].total_nr_elems() * OC * FH * FW * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        printf("time = %f \n perf= %f gops\n", time, computations * RUN / time);
    };

    auto profile = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                       size_t fh, size_t fw, size_t s) {
        Param param;
        param.stride_h = param.stride_w = s;
        printf("oc: %zd ic: %zd w: %zd h: %zd kernel_size: %zd sreide: %zd\n",
               oc, ic, ow, oh, fh, s);

        TensorLayout diff = TensorLayout{{n, oc, oh, ow}, dtype::Int8()};
        TensorLayout filter = TensorLayout{{oc, ic, fh, fw}, dtype::Int8()};
        TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        run(TensorLayoutArray{filter, diff, grad}, param);
    };

    profile(1, 3, 120, 120, 2, 3, 3, 1);
    profile(1, 3, 60, 60, 2, 3, 3, 2);
    profile(1, 3, 224, 224, 2, 5, 5, 1);
    profile(1, 3, 112, 112, 2, 5, 5, 2);
    profile(1, 3, 224, 224, 2, 7, 7, 1);
    profile(1, 3, 112, 112, 2, 7, 7, 2);
}
#endif

TEST_F(ARM_COMMON, BENCHMARK_CHANWISE_CONVOLUTION) {
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<Convolution> benchmarker_naive(handle_naive.get()),
                benchmarker_float(handle()), benchmarker_int(handle());
        benchmarker_int.set_dtype(0, dtype::Int8());
        benchmarker_int.set_dtype(1, dtype::Int8());
        benchmarker_int.set_dtype(2, dtype::Int16());
        size_t RUN = 10;
        auto tfloat = benchmarker_float.set_display(false)
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        auto tnaive = benchmarker_naive.set_display(false)
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        auto iparam = param;
        auto tint = benchmarker_int.set_display(false)
                            .set_times(RUN)
                            .set_param(iparam)
                            .exec(shapes);
        float int_float_ratio = static_cast<float>(tfloat) / tint;
        printf("naive=%.3fms float=%.3fms int=%.3fms, int/float=%.3f\n",
               tnaive / RUN, tfloat / RUN, tint / RUN, int_float_ratio);
        EXPECT_GE(int_float_ratio, 1.5);
    };
    Param param;
    param.mode = Param::Mode::CROSS_CORRELATION;
    param.sparse = Param::Sparse::GROUP;
    run({{2, 12, 200, 100}, {12, 2, 1, 5, 5}, {}}, param);
    run({{10, 24, 28, 28}, {24, 1, 1, 3, 3}, {}}, param);
    param.stride_h = 2;
    param.stride_w = 2;
    param.pad_h = 1;
    param.pad_w = 1;
    run({{2, 12, 200, 100}, {12, 2, 1, 5, 5}, {}}, param);
    run({{10, 24, 28, 28}, {24, 1, 1, 3, 3}, {}}, param);
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_INT8X8X32_STRD1_WITHOUT_DOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::Convolution param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    // compare to float direct conv here,
    // but float direct conv don't support 7x7.
    for (size_t kernel : {2, 3, 5})
        for (size_t ic : {1, 8, 16, 32, 64})
            for (size_t oc : {1, 8, 16, 32, 64})
                for (size_t p : {0, 1, 2, 3}) {
                    run(oc, ic, 56, 56, kernel, p);
                    run(oc, ic, 128, 128, kernel, p);
                    run(oc, ic, 256, 256, kernel, p);
                }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_S8STRD1"));

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);
    benchmark_float.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_F32STRD1"));

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int =
                benchmark.set_param(arg.param).exec({arg.src, arg.filter, {}}) /
                RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                          RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_INT8X8X32_STRD2_WITHOUT_DOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::Convolution param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32, 64})
            for (size_t oc : {1, 8, 16, 32, 64})
                for (size_t p : {0, 1, 2, 3}) {
                    run(oc, ic, 56, 56, kernel, p);
                    run(oc, ic, 128, 128, kernel, p);
                    run(oc, ic, 256, 256, kernel, p);
                }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_S8STRD2"));

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);
#if MEGDNN_AARCH64
    benchmark_float.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_ARMV8F32STRD2"));
#else
    benchmark_float.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_F32STRD2"));
#endif

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int =
                benchmark.set_param(arg.param).exec({arg.src, arg.filter, {}}) /
                RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                          RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);
    }
}

TEST_F(ARM_COMMON,
       BENCHMARK_CONVOLUTION_INT8X8X32_STRD1_WITHOUT_DOTPROD_TO_MATMUL) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::Convolution param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t p : {0, 1, 2})
            for (size_t ic : {1, 3, 4, 8, 12, 16, 32, 48, 64})
                for (size_t oc : {1, 3, 4, 8, 12, 16, 32, 48, 64})
                    for (size_t size : {56, 128, 256}) {
                        run(oc, ic, size, size, kernel, p);
                    }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark_conv(handle());
    benchmark_conv.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark_conv.set_display(false);
    benchmark_conv.set_times(RUN);
    benchmark_conv.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_S8STRD1"));

    Benchmarker<Convolution> benchmark_matmul(handle());
    benchmark_matmul.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark_matmul.set_display(false);
    benchmark_matmul.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_conv = benchmark_conv.set_param(arg.param).exec(
                                 {arg.src, arg.filter, {}}) /
                         RUN;
        auto used_matmul = benchmark_matmul.set_param(arg.param).exec(
                                   {arg.src, arg.filter, {}}) /
                           RUN;

        printf("%s %s: conv: %f ms %f Gflops matmul: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_conv, computations / used_conv, used_matmul,
               computations / used_matmul, used_matmul / used_conv);
    }
}

TEST_F(ARM_COMMON,
       BENCHMARK_CONVOLUTION_INT8X8X32_STRD2_WITHOUT_DOTPROD_TO_MATMUL) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::Convolution param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t p : {0, 1, 2})
            for (size_t ic : {1, 3, 4, 8, 12, 16, 32, 48, 64})
                for (size_t oc : {1, 3, 4, 8, 12, 16, 32, 48, 64})
                    for (size_t size : {56, 128, 256}) {
                        run(oc, ic, size, size, kernel, p);
                    }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark_conv(handle());
    benchmark_conv.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark_conv.set_display(false);
    benchmark_conv.set_times(RUN);
    benchmark_conv.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_S8STRD2"));

    Benchmarker<Convolution> benchmark_matmul(handle());
    benchmark_matmul.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32());
    benchmark_matmul.set_display(false);
    benchmark_matmul.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_conv = benchmark_conv.set_param(arg.param).exec(
                                 {arg.src, arg.filter, {}}) /
                         RUN;
        auto used_matmul = benchmark_matmul.set_param(arg.param).exec(
                                   {arg.src, arg.filter, {}}) /
                           RUN;

        printf("%s %s: conv: %f ms %f Gflops matmul: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_conv, computations / used_conv, used_matmul,
               computations / used_matmul, used_matmul / used_conv);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_QUINT8X8X32_STRD1_WITHOUT_DOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::Convolution param;
        param.stride_h = 1;
        param.stride_w = 1;
        param.pad_h = p;
        param.pad_w = p;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    // compare to float direct conv here,
    // but float direct conv don't support 7x7.
    for (size_t kernel : {2, 3, 5})
        for (size_t ic : {1, 8, 16, 32, 64})
            for (size_t oc : {1, 8, 16, 32, 64})
                for (size_t p : {0, 1, 2, 3}) {
                    run(oc, ic, 56, 56, kernel, p);
                    run(oc, ic, 128, 128, kernel, p);
                    run(oc, ic, 256, 256, kernel, p);
                }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_dtype(0, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(120)))
            .set_dtype(1, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.01f));
    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_QU8STRD1"));

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);
    benchmark_float.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_F32STRD1"));

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int =
                benchmark.set_param(arg.param).exec({arg.src, arg.filter, {}}) /
                RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                          RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_QUINT8X8X32_STRD2_WITHOUT_DOTPROD) {
    // have to remove preferred restrict in usable func before run the benchmark
    using namespace convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t p) {
        if (w + 2 * p < kernel || h + 2 * p < kernel)
            return;
        param::Convolution param;
        param.stride_h = 2;
        param.stride_w = 2;
        param.pad_h = p;
        param.pad_w = p;

        args.emplace_back(param, TensorShape{1, ic, h, w},
                          TensorShape{oc, ic, kernel, kernel});

    };

    for (size_t kernel : {2, 3, 5, 7})
        for (size_t ic : {1, 8, 16, 32, 64})
            for (size_t oc : {1, 8, 16, 32, 64})
                for (size_t p : {0, 1, 2, 3}) {
                    run(oc, ic, 56, 56, kernel, p);
                    run(oc, ic, 128, 128, kernel, p);
                    run(oc, ic, 256, 256, kernel, p);
                }

    constexpr size_t RUN = 50;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_dtype(0, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(120)))
            .set_dtype(1, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(120)))
            .set_dtype(2, dtype::QuantizedS32(0.01f));
    benchmark.set_display(false);
    benchmark.set_times(RUN);
    benchmark.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_QU8STRD2"));

    Benchmarker<Convolution> benchmark_float(handle());
    benchmark_float.set_display(false);
    benchmark_float.set_times(RUN);
#if MEGDNN_AARCH64
    benchmark_float.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_ARMV8F32STRD2"));
#else
    benchmark_float.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_F32STRD2"));
#endif

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * arg.filter[1] *
                             arg.filter[2] * arg.filter[3] * 2.0 /
                             (1024 * 1024 * 1024) * 1e3;

        auto used_int =
                benchmark.set_param(arg.param).exec({arg.src, arg.filter, {}}) /
                RUN;
        auto used_float = benchmark_float.set_param(arg.param).exec(
                                  {arg.src, arg.filter, {}}) /
                          RUN;

        printf("%s %s: int: %f ms %f Gflops float: %f ms %f GFlops speedup: "
               "%f\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               used_int, computations / used_int, used_float,
               computations / used_float, used_float / used_int);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_INT8_INT8_INT16) {
    using Param = param::Convolution;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        TensorLayoutArray layouts;
        layouts.emplace_back(shapes[0], dtype::Int8());
        layouts.emplace_back(shapes[1], dtype::Int8());
        layouts.emplace_back(shapes[2], dtype::Int16());
        Benchmarker<Convolution> benchmarker_cpu(handle()),
                benchmarker_float(handle());
        benchmarker_cpu.set_dtype(0, dtype::Int8());
        benchmarker_cpu.set_dtype(1, dtype::Int8());
        benchmarker_cpu.set_dtype(2, dtype::Int16());
        auto iparam = param;
        size_t RUN = 10;
        auto t2 = benchmarker_cpu.set_display(false)
                          .set_times(RUN)
                          .set_param(iparam)
                          .execl(layouts);
        auto t4 = benchmarker_float.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec(shapes);
        auto speedup = t4 / t2;
        std::cout << "src=" << shapes[0].to_string()
                  << " filter=" << shapes[1].to_string()
                  << " stride=" << param.stride_h << " float=" << t4 << "ms"
                  << " int=" << t2 << "ms"
                  << " speedup=" << speedup << std::endl;
        ASSERT_GE(speedup, 1);
    };
    /*
    for (size_t s: {1, 2})
    for (size_t k: {3})
    for (size_t c: {16})
    for (size_t h = 20; h <= 60; ++h)
    {
        Param param;
        param.stride_h = param.stride_w = s;
        run({{1, c, h, h}, {c, c, k, k}, {}}, param);
    }

    for (size_t s: {1})
    for (size_t k: {1})
    for (size_t c: {16})
    for (size_t h = 16; h <= 1024; h*=2)
    {
        Param param;
        param.stride_h = param.stride_w = s;
        run({{1, c, h, h}, {c, c, k, k}, {}}, param);
    }
    */
    for (size_t s : {1}) {
        Param param;
        param.stride_h = param.stride_w = s;

        run({{2, 3, 480, 270}, {12, 3, 1, 1}, {}}, param);
        run({{2, 12, 240, 135}, {48, 12, 1, 1}, {}}, param);
        run({{2, 16, 240, 135}, {4, 16, 1, 1}, {}}, param);
        run({{2, 4, 240, 135}, {16, 4, 1, 1}, {}}, param);
        run({{2, 16, 240, 135}, {8, 16, 1, 1}, {}}, param);
        run({{2, 8, 120, 68}, {32, 8, 1, 1}, {}}, param);
        run({{2, 32, 120, 68}, {8, 32, 1, 1}, {}}, param);
        run({{2, 64, 60, 34}, {16, 64, 1, 1}, {}}, param);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_INT8_INT8_INT32) {
    using Param = param::Convolution;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        TensorLayoutArray layouts;
        layouts.emplace_back(shapes[0], dtype::Int8());
        layouts.emplace_back(shapes[1], dtype::Int8());
        layouts.emplace_back(shapes[2], dtype::Int32());
        Benchmarker<Convolution> benchmarker_cpu(handle()),
                benchmarker_float(handle());
        benchmarker_cpu.set_dtype(0, dtype::Int8());
        benchmarker_cpu.set_dtype(1, dtype::Int8());
        benchmarker_cpu.set_dtype(2, dtype::Int32());
        auto iparam = param;
        size_t RUN = 10;
        auto t2 = benchmarker_cpu.set_display(false)
                          .set_times(RUN)
                          .set_param(iparam)
                          .execl(layouts);
        auto t4 = benchmarker_float.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .exec(shapes);
        auto speedup = t4 / t2;
        std::cout << "src=" << shapes[0].to_string()
                  << " filter=" << shapes[1].to_string()
                  << " stride=" << param.stride_h << " float=" << t4 << "ms"
                  << " int=" << t2 << "ms"
                  << " speedup=" << speedup << std::endl;
        ASSERT_GE(speedup, 1);
    };
    for (size_t s : {1, 2})
        for (size_t k : {3})
            for (size_t c : {16})
                for (size_t h = 20; h <= 60; ++h) {
                    Param param;
                    param.stride_h = param.stride_w = s;
                    run({{1, c, h, h}, {c, c, k, k}, {}}, param);
                }

    for (size_t s : {1})
        for (size_t k : {1})
            for (size_t c : {16})
                for (size_t h = 16; h <= 1024; h *= 2) {
                    Param param;
                    param.stride_h = param.stride_w = s;
                    run({{1, c, h, h}, {c, c, k, k}, {}}, param);
                }
    for (size_t s : {1}) {
        Param param;
        param.stride_h = param.stride_w = s;

        run({{2, 3, 480, 270}, {12, 3, 1, 1}, {}}, param);
        run({{2, 12, 240, 135}, {48, 12, 1, 1}, {}}, param);
        run({{2, 16, 240, 135}, {4, 16, 1, 1}, {}}, param);
        run({{2, 4, 240, 135}, {16, 4, 1, 1}, {}}, param);
        run({{2, 16, 240, 135}, {8, 16, 1, 1}, {}}, param);
        run({{2, 8, 120, 68}, {32, 8, 1, 1}, {}}, param);
        run({{2, 32, 120, 68}, {8, 32, 1, 1}, {}}, param);
        run({{2, 64, 60, 34}, {16, 64, 1, 1}, {}}, param);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_DIRECT) {
    using Param = param::Convolution;
    Benchmarker<Convolution> benchmarker_float(handle());
    Benchmarker<Convolution> benchmarker_half(handle());
    const size_t RUNS = 10;
    benchmarker_float.set_display(false)
            .set_times(RUNS)
            .set_dtype(0, dtype::Float32{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Float32{})
            .set_before_exec_callback(
                    AlgoChecker<Convolution>("CONVOLUTION_DEFAULT_F32DIRECT"));
    benchmarker_half.set_display(false)
            .set_times(RUNS)
            .set_dtype(0, dtype::Float16{})
            .set_dtype(1, dtype::Float16{})
            .set_dtype(2, dtype::Float16{})
            .set_before_exec_callback(
                    AlgoChecker<Convolution>("CONVOLUTION_DEFAULT_F16DIRECT"));

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto tfloat = benchmarker_float.set_param(param).exec(shapes) / RUNS;
        auto thalf = benchmarker_half.set_param(param).exec(shapes) / RUNS;

        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float computations = dst_layout.total_nr_elems() * shapes[1][1] *
                             shapes[1][2] * shapes[1][3] * 2.0 /
                             (1024 * 1024 * 1024);
        printf("run:%s %s float: %f ms %f Gflops VS half: %f ms %f Gflops "
               "speepup: %f\n",
               shapes[0].to_string().c_str(), shapes[1].to_string().c_str(),
               tfloat, computations / tfloat * 1e3, thalf,
               computations / thalf * 1e3, tfloat / thalf);
    };

    auto profile = [&](size_t n, size_t oc, size_t ic, size_t w, size_t h,
                       size_t kernel, size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;

        run({{n, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);

    };

    for (size_t kernel : {1, 2, 3, 4, 5, 6, 7}) {
        for (size_t ic : {12}) {
            for (size_t oc : {4}) {
                for (size_t size : {17, 28, 32, 34, 64, 112, 256}) {
                    profile(1, oc, ic, size, size, kernel, 1);
                }
            }
        }
    }
    for (auto k : {1, 2, 3, 4, 5, 6, 7}) {
        profile(2, 12, 3, 480, 270, k, 1);
        profile(2, 48, 12, 240, 135, k, 1);
        profile(2, 4, 16, 240, 135, k, 1);
        profile(2, 16, 4, 240, 135, k, 1);
        profile(2, 8, 16, 240, 135, k, 1);
        profile(2, 32, 8, 240, 135, k, 1);
        profile(2, 8, 32, 120, 68, k, 1);
        profile(2, 16, 64, 60, 34, k, 1);
    }
}

TEST_F(ARM_COMMON, BENCHMARK_CONVOLUTION_STRIDE1) {
    using Param = param::Convolution;
    auto run_fp32 = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 50;
        auto tfloat =
                benchmarker_float.set_display(false)
                        .set_dtype(0, dtype::Float32())
                        .set_dtype(1, dtype::Float32())
                        .set_dtype(2, dtype::Float32())
                        .set_before_exec_callback(AlgoChecker<Convolution>(
                                "CONVOLUTION_DEFAULT_F32STRD1"))
                        .set_times(RUN)
                        .set_param(param)
                        .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, dst_layout);
        printf("fp32 flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    auto run_fp16 = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 50;
        auto tfloat =
                benchmarker_float.set_display(false)
                        .set_dtype(0, dtype::Float16())
                        .set_dtype(1, dtype::Float16())
                        .set_dtype(2, dtype::Float16())
                        .set_before_exec_callback(AlgoChecker<Convolution>(
                                "CONVOLUTION_DEFAULT_F16STRD1"))
                        .set_times(RUN)
                        .set_param(param)
                        .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float16()},
                           {shapes[1], dtype::Float16()}, dst_layout);
        printf("fp16 flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };
#endif
    auto profile = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                       size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, w, h, stride, kernel);

        run_fp32({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        run_fp16({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);
#endif

    };

    for (size_t kernel : {2, 3, 5}) {
        for (size_t ic : {3, 6, 12, 24}) {
            for (size_t oc : {3, 6, 12, 24}) {
                for (size_t size : {4, 7, 8, 14, 16, 17, 28, 32, 34, 64, 112}) {
                    profile(oc, ic, size, size, kernel, 1);
                }
            }
        }
    }
}
#endif

// vim: syntax=cpp.doxygen
