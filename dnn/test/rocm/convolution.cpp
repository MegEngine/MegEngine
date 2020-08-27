/**
 * \file dnn/test/rocm/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/convolution.h"
#include "hcc_detail/hcc_defs_prologue.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/rocm/benchmarker.h"
#include "test/rocm/fixture.h"

#include "src/common/utils.h"
#include "src/rocm/utils.h"

namespace megdnn {
namespace test {
namespace convolution {
std::vector<TestArg> get_args_0() {
    std::vector<TestArg> args, tmp_args;
#define ADD_ARGS(NAME)            \
    tmp_args = get_args_##NAME(); \
    args.insert(args.end(), tmp_args.begin(), tmp_args.end());
    ADD_ARGS(common)
    ADD_ARGS(padding)
    ADD_ARGS(large_channel)
    ADD_ARGS(1x1)
    ADD_ARGS(large_filter)
    ADD_ARGS(exhaustive_search)
    ADD_ARGS(4x4)
    ADD_ARGS(large_channels)
    ADD_ARGS(x86_direct_case_2)
    ADD_ARGS(cudnn_5_1_failures)
    ADD_ARGS(x86_winograd_algorithm)
    ADD_ARGS(BRAIN_481)
#undef ADD_ARGS

    return args;
}

std::vector<TestArg> get_args_1() {
    return get_args_fallback_templated_impl();
}

std::vector<TestArg> get_args_2() {
    return get_args_fallback_non_templated_impl();
}

std::vector<TestArg> get_group_conv_args() {
    std::vector<TestArg> args;
    for (size_t batch_size : {2}) {
        for (size_t ih : {23}) {
            for (size_t iw : {ih + 1}) {
                for (size_t icpg : {2, 4, 8}) {
                    for (size_t ocpg : {4, 8}) {
                        for (size_t fh : {3, 5, 7}) {
                            for (size_t fw : {fh, fh + 1}) {
                                for (size_t ph : {0_z, size_t{fw / 2}}) {
                                    for (size_t sh : {1, 2}) {
                                        for (size_t dh : {1, 2}) {
                                            param::Convolution param;
                                            size_t groups = 2;
                                            param.sparse = param::Convolution::
                                                    Sparse::GROUP;

                                            param.mode = param::Convolution::
                                                    Mode::CROSS_CORRELATION;
                                            param.stride_h = param.stride_w =
                                                    sh;
                                            param.pad_h = param.pad_w = ph;
                                            param.dilate_h = param.dilate_w =
                                                    dh;
                                            args.emplace_back(
                                                    param,
                                                    TensorShape{batch_size,
                                                                icpg * groups,
                                                                ih, iw},
                                                    TensorShape{groups, ocpg,
                                                                icpg, fh, fw});
                                        }
                                    }
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
} // namespace convolution

TEST_F(ROCM, CONV_GROUP) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_group_conv_args();
    Checker<ConvolutionForward> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(ROCM, CONV_CHANNWISE) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_chanwise_args();
    Checker<ConvolutionForward> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

TEST_F(ROCM, CONVOLUTION_FORWARD_0) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_args_0();
    Checker<ConvolutionForward> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        float scale =
                1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
#if !MEGDNN_DISABLE_FLOAT16
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
#endif
    }
}

TEST_F(ROCM, CONVOLUTION_FORWARD_1) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_args_1();
    Checker<ConvolutionForward> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        float scale =
                1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
#if !MEGDNN_DISABLE_FLOAT16
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
#endif
    }
}

TEST_F(ROCM, CONVOLUTION_FORWARD_2) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_args_2();
    Checker<ConvolutionForward> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        float scale =
                1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
#if !MEGDNN_DISABLE_FLOAT16
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
#endif
    }
}

TEST_F(ROCM, CONVOLUTION_1X1_FORWARD) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_1x1_args();
    Checker<ConvolutionForward> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale =
                1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ROCM, CONVOLUTION_1X1_FORWARD_ALL_ALGOS) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    using namespace convolution;
    OprProxy<ConvolutionForward> proxy{true};
    proxy.warmup_times = 1;
    proxy.exec_times = 10;
    Benchmarker<ConvolutionForward> checker(handle_rocm());
    checker.set_times(1);

    auto get_computation = [&](TestArg arg) -> float {
        megdnn_assert(arg.param.format == param::Convolution::Format::NCHW);
        size_t N = arg.src[0], IC = arg.src[1], IH = arg.src[2],
               IW = arg.src[3], OC = arg.filter[0], FH = arg.filter[2],
               FW = arg.filter[3], SH = arg.param.stride_h,
               SW = arg.param.stride_w, PH = arg.param.pad_h,
               PW = arg.param.pad_w;

        size_t OH = infer_conv_shape(IH, FH, SH, PH);
        size_t OW = infer_conv_shape(IW, FW, SW, PW);
        float flops = 2.0 * N * OC * OH * OW * IC * FH * FW;
        return flops;
    };

    std::vector<TestArg> args = get_1x1_args();
    NormalRNG default_rng;
    for (auto&& arg : args) {
        float scale =
                1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.set_proxy(proxy)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_param(arg.param);
        float time_in_ms = checker.execs({arg.src, arg.filter, {}});
        float flops = get_computation(arg);
        printf("inp=%s,flt=%s,flops=%.2fGflo,time = %.2f ms, perf = %.2f "
               "GFLOPS\n",
               arg.src.to_string().c_str(), arg.filter.to_string().c_str(),
               flops / 1e9, time_in_ms, flops / (1e6 * time_in_ms));
    }
}
#endif

TEST_F(ROCM, CONVOLUTION_BACKWARD_DATA_0) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_args_0();
    Checker<ConvolutionBackwardData> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        float scale =
                1.0f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_rocm()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
#if !MEGDNN_DISABLE_FLOAT16
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
#endif
    }
}

TEST_F(ROCM, CONVOLUTION_BACKWARD_DATA_1) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_args_1();
    Checker<ConvolutionBackwardData> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        float scale =
                1.0f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_rocm()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
#if !MEGDNN_DISABLE_FLOAT16
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
#endif
    }
}

TEST_F(ROCM, CONVOLUTION_BACKWARD_DATA_2) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;

    std::vector<TestArg> args = get_args_2();
    Checker<ConvolutionBackwardData> checker(handle_rocm());
    NormalRNG default_rng;
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        float scale =
                1.0f / sqrt(arg.filter[0] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_rocm()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
#if !MEGDNN_DISABLE_FLOAT16
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{filter, dst, src});
#endif
    }
}

TEST_F(ROCM, DISABLED_CONVOLUTION_BACKWARD_FILTER) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), false);
    using namespace convolution;
    std::vector<TestArg> args = get_args();
    Checker<ConvolutionBackwardFilter> checker(handle_rocm());
    NormalRNG default_rng;
    bool f16_checked = false;
    MEGDNN_MARK_USED_VAR(f16_checked);
    for (auto&& arg : args) {
        using Mode = param::Convolution::Mode;
        //! non xcorr not supported for miopen
        if (arg.param.mode == Mode::CONVOLUTION) {
            continue;
        }
        auto src = TensorLayout(arg.src, dtype::Float32());
        auto filter = TensorLayout(arg.filter, dtype::Float32());
        TensorLayout dst;
        {
            auto opr = handle_rocm()->create_operator<Convolution>();
            opr->param() = arg.param;
            opr->deduce_layout(src, filter, dst);
        }
        float scale = 1.0f / sqrt(dst[2] * dst[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        src.dtype = dst.dtype = filter.dtype = dtype::Float32();
        checker.set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});

#if !MEGDNN_DISABLE_FLOAT16
//! FIXME: MIOpen convolution backward weights for FP16 with bugs
#if 0
        // reduce on large f16 array may introduce significant error
        if (dst.total_nr_elems() >= 1000 && f16_checked)
            continue;

        f16_checked = true;
        src.dtype = dst.dtype = filter.dtype = dtype::Float16();
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_epsilon(1e-1)
                .set_param(arg.param)
                .exec(TensorLayoutArray{src, dst, filter});
#endif
#endif
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ROCM, CONV_FWD_BENCHMARK) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker = ROCMBenchmarker<ConvolutionForward>(handle_rocm(),
                                                           handle_naive(false));
    auto run = [&](size_t N, size_t OC, size_t IC, size_t IH, size_t IW,
                   size_t SH = 1, size_t SW = 1, size_t FH = 1, size_t FW = 1,
                   size_t PH = 0, size_t PW = 0,
                   DType dtype = dtype::Float32()) {
        benchmarker.set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(2, dtype);
        benchmarker.set_display(true);
        ConvolutionForward::Param param;
        param.stride_h = SH;
        param.stride_w = SW;
        param.pad_h = PH;
        param.pad_w = PW;
        benchmarker.set_param(param);
        size_t OH = (IH - FH + 2 * PH) / SH + 1;
        size_t OW = (IW - FW + 2 * PW) / SW + 1;
        // warm up find best algo
        benchmarker.execs({{N, IC, IH, IW}, {OC, IC, FH, FW}, {N, OC, OH, OW}});
        // do actual benchmark
        auto time_ms = benchmarker.execs(
                {{N, IC, IH, IW}, {OC, IC, FH, FW}, {N, OC, OH, OW}});
        auto flo = (double)N * OC * IC * OH * OW * FH * FW * 2;
        auto flops = flo / (time_ms * 1e9);
        printf("%.3fG FLO, flops %.3fTFLOPS\n", flo / 1e9, flops);
    };
    run(32, 24, 16, 224, 224, 2, 2, 7, 7, 3, 3);
    run(32, 128, 32, 112, 112, 1, 1, 3, 3, 1, 1);
    run(32, 128, 128, 56, 56, 1, 1, 3, 3, 1, 1);
    run(32, 128, 256, 28, 28, 1, 1, 3, 3, 1, 1);
    run(32, 256, 256, 28, 28, 1, 1, 1, 1, 0, 0);
    run(32, 256, 256, 28, 28, 2, 2, 3, 3, 1, 1);
    run(32, 256, 256, 14, 14, 1, 1, 3, 3, 1, 1);
    run(32, 512, 512, 7, 7, 1, 1, 3, 3, 1, 1);
#if !MEGDNN_DISABLE_FLOAT16
    run(32, 256, 256, 56, 56, 1, 1, 1, 1, 0, 0, dtype::Float16());
#endif
}
#endif

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
