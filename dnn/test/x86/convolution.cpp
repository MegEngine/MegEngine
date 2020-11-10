/**
 * \file dnn/test/x86/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/x86/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace {
#if MEGDNN_X86_WITH_MKL_DNN
struct ConvArg {
    size_t batch_size, fh, sh, ph, ic, ih, iw, oc, groups;
};

std::vector<ConvArg> get_dense_conv_args() {
    std::vector<ConvArg> args;
    for (size_t batch_size : {1}) {
        for (size_t fh : {3, 5, 7}) {
            for (size_t sh : {1, 2}) {
                for (size_t ph : std::vector<size_t>{0, fh / 2}) {
                    for (size_t oc : {3, 4}) {
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  15, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  14, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  13, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  12, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  11, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  10, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  9, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  8, oc, 1});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 4, 7,
                                                  8, oc, 1});

                    }  // end oc
                }      // end ph
            }          // end sh
        }              // end fh
    }                  // end batch_size
    return args;
}

std::vector<ConvArg> get_group_conv_args() {
    std::vector<ConvArg> args;
    for (size_t batch_size : {1}) {
        for (size_t fh : {3, 5, 7}) {
            for (size_t sh : {1, 2}) {
                for (size_t ph : std::vector<size_t>{0, fh / 2}) {
                    for (size_t oc : {3}) {
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  15, oc, 2});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  14, oc, 2});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  13, oc, 2});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  12, oc, 2});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  11, oc, 2});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  10, oc, 2});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  9, oc, 2});
                        args.emplace_back(ConvArg{batch_size, fh, sh, ph, 2, 7,
                                                  8, oc, 2});
                    }  // end oc
                }      // end ph
            }          // end sh
        }              // end fh
    }                  // end batch_size
    args.emplace_back(ConvArg{2, 1, 1, 0, 6, 18, 18, 9, 3});
    return args;
}
#endif

}  // namespace

namespace megdnn {
namespace test {

TEST_F(X86, DEFAULT_CONV_DIRECT_STRIDE1) {
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

    for (size_t kernel : {1, 2, 3, 4, 5, 6, 7})
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        run(oc, ic, size, size, kernel, p);

    Checker<ConvolutionForward> checker(handle());
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_X86_CONV_BIAS_DIRECT_STRIDE1_LARGE_GROUP"));
    checker.set_epsilon(1);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    for (auto&& arg : args) {
        checker.set_param(arg.param).exec({arg.src, arg.filter, {}});
    }
}

TEST_F(X86, DEFAULT_CONV_DIRECT_STRIDE2) {
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
        for (size_t ic : {1, 4, 8, 16})
            for (size_t oc : {1, 4, 8})
                for (size_t p : {0, 2})
                    for (size_t size : {20, 21, 24})
                        run(oc, ic, size, size, kernel, p);

    Checker<ConvolutionForward> checker(handle());
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_X86_CONV_BIAS_DIRECT_STRIDE2_LARGE_GROUP"));
    checker.set_epsilon(1);
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    for (auto&& arg : args) {
        checker.set_param(arg.param).exec({arg.src, arg.filter, {}});
    }
}

#if MEGDNN_X86_WITH_MKL_DNN
TEST_F(X86, CONVOLUTION_FORWARD_INT8) {
    Checker<ConvolutionForward> checker(handle());
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("CONVOLUTION_DEFAULT_MKLDNN_INT8"));
    param::Convolution param;
    param.sparse = param::Convolution::Sparse::GROUP;
    UniformIntRNG rng{-128, 127};
    std::vector<ConvArg> args = get_group_conv_args();
    for (auto&& arg : args) {
        param.stride_h = param.stride_w = arg.sh;
        param.pad_h = param.pad_w = arg.ph;
        checker.set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_param(param)
                .execs({{arg.batch_size, arg.ic * arg.groups, arg.ih, arg.iw},
                        {arg.groups, arg.oc, arg.ic, arg.fh, arg.fh},
                        {}});
    }
    args = get_dense_conv_args();
    param.sparse = param::Convolution::Sparse::DENSE;
    for (auto&& arg : args) {
        param.stride_h = param.stride_w = arg.sh;
        param.pad_h = param.pad_w = arg.ph;
        checker.set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_param(param)
                .execs({{arg.batch_size, arg.ic, arg.ih, arg.iw},
                        {arg.oc, arg.ic, arg.fh, arg.fh},
                        {}});
    }
}

TEST_F(X86, CONVOLUTION_FORWARD_MATMUL_INT8) {
    std::vector<ConvArg> args = get_dense_conv_args();
    Checker<ConvolutionForward> checker(handle());
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_MKLDNN_MATMUL_INT8"));
    param::Convolution param;
    param.sparse = param::Convolution::Sparse::DENSE;
    UniformIntRNG rng{-128, 127};
    for (auto&& arg : args) {
        param.stride_h = param.stride_w = arg.sh;
        param.pad_h = param.pad_w = arg.ph;
        checker.set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_param(param)
                .execs({{arg.batch_size, arg.ic, arg.ih, arg.iw},
                        {arg.oc, arg.ic, arg.fh, arg.fh},
                        {}});
    }
}

static void x86_correctness_fp32_mkldnn_run(Checker<Convolution>& checker,
                                            UniformIntRNG& rng, Handle* handle,
                                            size_t n, size_t stride,
                                            size_t kernel, size_t oc, size_t ic,
                                            size_t h, size_t w, size_t group) {
    auto oc_per_group = oc / group;
    auto ic_per_group = ic / group;
    bool ok_group = oc_per_group % 8 == 0 && oc_per_group > 0 &&
                    (ic_per_group % 8 == 0 || ic_per_group == 3) &&
                    ic_per_group > 0;
    bool ok_depthwise = oc == ic && oc == group;
    if (!(ok_group || ok_depthwise)) {
        return;
    }
    size_t pad = kernel / 2;
    size_t kernel_h = kernel;
    size_t kernel_w = kernel;
    param::Convolution param;
    param.format = param::Convolution::Format::NCHW88;
    param.stride_h = stride;
    param.stride_w = stride;
    param.pad_h = pad;
    param.pad_w = pad;
    auto src_tensor_shape = TensorShape{n, ic / 8, h, w, 8};
    if (ic == 3) {
        src_tensor_shape = TensorShape{n, ic, h, w};
    }

    auto weight_tensor_shape =
            TensorShape{oc / 8, ic / 8, kernel_h, kernel_w, 8, 8};
    if (ic == 3) {
        weight_tensor_shape = TensorShape{oc / 8, kernel_h, kernel_w, ic, 8};
    }

    if (group == 1) {
        param.sparse = param::Convolution::Sparse::DENSE;
    } else if (group > 1 && ic / group == 1 && oc / group == 1) {
        param.sparse = param::Convolution::Sparse::GROUP;
        weight_tensor_shape =
                TensorShape{group / 8, 1, 1, kernel_h, kernel_w, 8};
    } else if (group > 1 && oc / group % 8 == 0 && oc / group > 0 &&
               ic / group % 8 == 0 && ic / group > 0) {
        param.sparse = param::Convolution::Sparse::GROUP;
        weight_tensor_shape = TensorShape{
                group, oc / group / 8, ic / group / 8, kernel_h, kernel_w, 8,
                8};
    }
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_epsilon(1e-3)
            .set_param(param)
            .execs({src_tensor_shape, weight_tensor_shape, {}});
}

static void x86_correctness_fp32_mkldnn(Handle* handle) {
    Checker<Convolution> checker(handle);
    UniformIntRNG rng{-127, 127};
    checker.set_before_exec_callback(AlgoChecker<ConvolutionForward>(
            "CONVOLUTION_DEFAULT_MKLDNN_CONV_FP32"));
    for (size_t n : {1, 2})
        for (size_t stride : {1, 2})
            for (size_t kernel : {3, 5, 7})
                for (size_t oc : {8, 16})
                    for (size_t ic : {3, 8, 16})
                        for (size_t h : {22, 33})
                            for (size_t w : {22, 33}) {
                                for (size_t group = 1;
                                     group <= std::min(oc, ic); ++group) {
                                    x86_correctness_fp32_mkldnn_run(
                                            checker, rng, handle, n, stride,
                                            kernel, oc, ic, h, w, group);
                                }
                            }
}

TEST_F(X86, CONVOLUTION_DIRECT_MKLDNN_C8) {
    x86_correctness_fp32_mkldnn(handle());
}
#endif

#if MEGDNN_WITH_BENCHMARK
TEST_F(X86, BENCHMARK_CONVOLUTION_I8x8x16) {
    using namespace convolution;
    using Param = param::Convolution;

    std::vector<TestArg> args;
    auto run = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                   size_t stride, size_t group = 1) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        if (group > 1) {
            param.sparse = param::Convolution::Sparse::GROUP;
            args.emplace_back(
                    param, TensorShape{1, ic, h, w},
                    TensorShape{group, oc / group, ic / group, kernel, kernel});
        } else {
            param.sparse = param::Convolution::Sparse::DENSE;
            args.emplace_back(param, TensorShape{1, ic, h, w},
                              TensorShape{oc, ic, kernel, kernel});
        }
    };

    run(48, 96, 15, 15, 1, 1);
    run(64, 64, 60, 60, 3, 1);
    run(64, 64, 60, 60, 3, 1, 64);

    constexpr size_t RUN = 30;
    Benchmarker<Convolution> benchmark(handle());
    benchmark.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int16());
    benchmark.set_before_exec_callback(AlgoChecker<Convolution>(".*"));
    benchmark.set_display(false);
    benchmark.set_times(RUN);

    for (auto&& arg : args) {
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = arg.param;
        opr->deduce_layout({arg.src, dtype::Float32()},
                           {arg.filter, dtype::Float32()}, dst_layout);
        //! dst.nr_elems * IC * FH * FW * 2
        float icpg = arg.filter.ndim == 4 ? arg.filter[1] : arg.filter[2];
        float filter = arg.filter.ndim == 4 ? arg.filter[2] : arg.filter[3];
        float computations = dst_layout.total_nr_elems() * icpg * filter *
                             filter * 2.0 / (1024 * 1024 * 1024) * 1e3;

        auto used_int =
                benchmark.set_param(arg.param).exec({arg.src, arg.filter, {}}) /
                RUN;

        printf("%s %s: int: %f ms %f Gflops \n", arg.src.to_string().c_str(),
               arg.filter.to_string().c_str(), used_int,
               computations / used_int);
    }
}
#if MEGDNN_X86_WITH_MKL_DNN
TEST_F(X86, BENCHMARK_CONVOLUTION_I8x8x32_MKLDNN) {
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
#endif
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
