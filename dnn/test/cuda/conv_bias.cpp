/**
 * \file dnn/test/cuda/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/cuda/handle.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/utils.h"

using namespace megdnn;
using namespace test;
using namespace conv_bias;

namespace {
#if CUDA_VERSION >= 10000
void test_conv_bias_forward_wmma_int4_nchw8(Handle* handle_cuda, size_t fh) {
    require_compute_capability(7, 5);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda);

    UniformIntRNG int_rng{0, 8};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW8;

    using NonlineMode = ConvBias::Param::NonlineMode;
    for (NonlineMode mode : {NonlineMode::RELU}) {
        for (size_t batch : {1}) {
            for (size_t ic : {128, 32}) {
                for (size_t oc : {32}) {
                    for (int ph : {static_cast<int>(fh / 2), 0}) {
                        for (size_t ih : {8, 9, 13, 15, 16}) {
                            for (size_t iw : {8, 16, 24, 32, 40}) {
                                param.nonlineMode = mode;
                                param.stride_h = param.stride_w = 1;
                                param.pad_h = param.pad_w = ph;
                                checker.set_dtype(0,
                                                  dtype::Quantized4Asymm(
                                                          1.3f, (uint8_t)(1)))
                                        .set_dtype(1,
                                                   dtype::Quantized4Asymm(
                                                           1.3f, (uint8_t)(2)))
                                        .set_dtype(2, dtype::QuantizedS32(1.3f *
                                                                          1.3f))
                                        .set_dtype(4, dtype::QuantizedS32(1.3f *
                                                                          1.3f))
                                        .set_rng(0, &int_rng)
                                        .set_rng(1, &int_rng)
                                        .set_rng(2, &int_rng)
                                        .set_param(param);
                                if (!ph)
                                    iw += 2 * (fh / 2);
                                size_t oh = infer_conv_shape(ih, fh, 1, ph);
                                size_t ow = infer_conv_shape(iw, fh, 1, ph);
                                if (ow % 8 != 0)
                                    continue;
                                checker.execs({{batch, ic / 8, ih, iw, 8},
                                               {oc, ic / 8, fh, fh, 8},
                                               {1, oc / 8, 1, 1, 8},
                                               {},
                                               {}});
                                checker.execs({{batch, ic / 8, ih, iw, 8},
                                               {oc, ic / 8, fh, fh, 8},
                                               {batch, oc / 8, oh, ow, 8},
                                               {},
                                               {}});
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif
}  // namespace

#if CUDNN_VERSION >= 7400
TEST_F(CUDA, CONV_BIAS_FORWARD_F32) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args();
    Checker<ConvBiasForward> checker(handle_cuda());

    NormalRNG default_rng;
    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_QS8) {
    require_compute_capability(6, 1);

    UniformIntRNG int_rng{-50, 50};
    Checker<ConvBiasForward> checker(handle_cuda());
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NHWC;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
    {
        auto src_shape = TensorShape{20, 224, 224, 4};
        auto filter_shape = TensorShape{24, 1, 1, 4};
        auto bias_shape = TensorShape{1, 1, 1, 24};
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(60.25f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape, {}, {}});
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(40.25f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape, {}, {}});
    }
    {
        auto src_shape = TensorShape{20, 224, 224, 4};
        auto filter_shape = TensorShape{24, 1, 1, 4};
        auto bias_shape = TensorShape{1, 1, 1, 24};
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(60.25f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape, {}, {}});
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(40.25f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape, {}, {}});
    }

    {
        param.sparse = ConvBias::Param::Sparse::GROUP;
        auto src_shape = TensorShape{20, 224, 224, 16};
        auto filter_shape = TensorShape{4, 4, 1, 1, 4};
        auto bias_shape = TensorShape{1, 1, 1, 16};
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(60.25f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape, {}, {}});
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(40.25f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng)
                .set_param(param)
                .execs({src_shape, filter_shape, bias_shape, {}, {}});
    }
}

TEST_F(CUDA, CONV_BIAS_NCHW_QS8) {
    //! not support NonlineMode::SIGMOID and NonlineMode::H_SWISH
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-128, 127};
    using NonlineMode = ConvBias::Param::NonlineMode;

    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW;

    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(3, dtype::QuantizedS8(0.25f))
            .set_dtype(4, dtype::QuantizedS8(0.25f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &int_rng)
            .set_rng(3, &int_rng);

    for (NonlineMode mode : {NonlineMode::RELU,
                             NonlineMode::IDENTITY, NonlineMode::H_SWISH}) {
        for (size_t g : {1, 2}) {
           for (size_t b : {2}) {
               for (size_t ic : {6, 16}) {
                   for (size_t oc : {4}) {
                       for (size_t fh : {1, 3}) {
                           for (int ph : {static_cast<int>(fh / 2)}) {
                               for (int sh : {1, 2}) {
                                    size_t ih = 16, iw = 16;
                                    param.nonlineMode = mode;
                                    param.stride_h = param.stride_w = sh;
                                    param.pad_h = param.pad_w = ph;
                                    param.sparse =
                                        ConvBias::Param::Sparse::DENSE;
                                    checker.set_param(param)
                                            .execs({{b, ic / 2, ih, iw},
                                                    {oc, ic / 2, fh, fh},
                                                    {1, oc, 1, 1},
                                                    {},
                                                    {}});
                                    param.sparse =
                                        ConvBias::Param::Sparse::GROUP;
                                    checker.set_param(param)
                                            .execs({{b, ic, ih, iw},
                                                    {g, oc/g, ic/g, fh, fh},
                                                    {1, oc, 1, 1},
                                                    {},
                                                    {}});
                               }
                           }
                       }
                   }
               }
           }
        }
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CONV_BIAS_NCHW4_INT8) {
    require_compute_capability(6, 1);
    Benchmarker<ConvBiasForward> bencher(handle_cuda());
    bencher.set_display(false);
    ConvBias::Param param_nchw;
    param_nchw.format = ConvBias::Param::Format::NCHW;
    ConvBias::Param param_nchw4;
    param_nchw4.format = ConvBias::Param::Format::NCHW4;

    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};

    param_nchw.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
    auto run_bench = [&](size_t b, size_t ci, size_t hi, size_t wi,
                         size_t co, size_t fh, size_t fw, size_t sh,
                         size_t sw, size_t nr_times) {
        param_nchw.pad_h = fh / 2;
        param_nchw.pad_w = fw / 2;
        param_nchw.stride_h = sh;
        param_nchw.stride_w = sw;
        param_nchw4.pad_h = fh / 2;
        param_nchw4.pad_w = fh / 2;
        param_nchw4.stride_h = sh;
        param_nchw4.stride_w = sw;
        bencher.set_times(nr_times)
                .set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .set_dtype(4, dtype::QuantizedS8(0.35f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng);
        bencher.set_param(param_nchw);
        size_t ho = infer_conv_shape(hi, fh, sh, param_nchw.pad_h);
        size_t wo = infer_conv_shape(wi, fw, sw, param_nchw.pad_w);
        TensorShape inp{b, ci, hi, wi}, kern{co, ci, fh, fw},
                    out{b, co, ho, wo};
        auto time_in_ms = bencher.execs(
                {inp, kern, {1, co, 1, 1}, {}, out}) / nr_times;
        auto ops_nchw = 2.0 * b * co * ho * wo * ci * fh * fw /
                    (time_in_ms * 1e-3) * 1e-12;
        printf("inp=%s, kern=%s, out=%s, time: %.2fms, perf: %.2f Tops "
               "(NCHW)\n",
                inp.to_string().c_str(), kern.to_string().c_str(),
                out.to_string().c_str(), time_in_ms, ops_nchw);
        bencher.set_param(param_nchw4);
        decltype(ops_nchw) ops_nchw4;
        {
            TensorShape inp{b, ci / 4, hi, wi, 4},
                kern{co, ci / 4, fh, fw, 4}, out{b, co / 4, ho, wo, 4};
            auto time_in_ms = bencher.execs(
                    {inp, kern, {1, co / 4, 1, 1, 4}, {}, out}) / nr_times;
            ops_nchw4 = 2.0 * b * co * ho * wo * ci * fh * fw /
                        (time_in_ms * 1e-3) * 1e-12;
            printf("inp=%s, kern=%s, out=%s, time: %.2fms, perf: %.2f Tops "
                   "(NCHW4)\n",
                    inp.to_string().c_str(), kern.to_string().c_str(),
                    out.to_string().c_str(), time_in_ms, ops_nchw4);
        }
        printf("speedup: %.2fx\n", ops_nchw4 / ops_nchw);
    };
    // resnet-50
    // bottleneck-1
    // proj
    run_bench(1, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(1, 64, 56, 56, 64, 1, 1, 1, 1, 1000);
    run_bench(1, 64, 56, 56, 64, 3, 3, 1, 1, 1000);
    run_bench(1, 64, 56, 56, 256, 1, 1, 1, 1, 1000);

    // bottleneck-2
    // proj
    run_bench(1, 256, 56, 56, 512, 1, 1, 2, 2, 1000);
    run_bench(1, 256, 56, 56, 128, 1, 1, 2, 2, 1000);
    run_bench(1, 128, 28, 28, 128, 3, 3, 1, 1, 1000);
    run_bench(1, 128, 28, 28, 512, 1, 1, 1, 1, 1000);

    // bottleneck-3
    // proj
    run_bench(1, 512, 28, 28, 1024, 1, 1, 2, 2, 1000);
    run_bench(1, 512, 28, 28, 256, 1, 1, 2, 2, 1000);
    run_bench(1, 256, 14, 14, 256, 3, 3, 1, 1, 1000);
    run_bench(1, 256, 14, 14, 1024, 1, 1, 1, 1, 1000);

    // bottleneck-4
    // proj
    run_bench(1, 1024, 14, 14, 2048, 1, 1, 2, 2, 1000);
    run_bench(1, 1024, 14, 14, 512, 1, 1, 2, 2, 1000);
    run_bench(1, 512, 7, 7, 512, 3, 3, 1, 1, 1000);
    run_bench(1, 512, 7, 7, 2048, 1, 1, 1, 1, 1000);

    run_bench(32, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(32, 64, 56, 56, 64, 1, 1, 1, 1, 1000);
    run_bench(32, 64, 56, 56, 64, 3, 3, 1, 1, 1000);
    run_bench(32, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(32, 256, 56, 56, 512, 1, 1, 2, 2, 1000);
    run_bench(32, 256, 56, 56, 128, 1, 1, 2, 2, 1000);
    run_bench(32, 128, 28, 28, 128, 3, 3, 1, 1, 1000);
    run_bench(32, 128, 28, 28, 512, 1, 1, 1, 1, 1000);
    run_bench(32, 512, 28, 28, 1024, 1, 1, 2, 2, 1000);
    run_bench(32, 512, 28, 28, 256, 1, 1, 2, 2, 1000);
    run_bench(32, 256, 14, 14, 256, 3, 3, 1, 1, 1000);
    run_bench(32, 256, 14, 14, 1024, 1, 1, 1, 1, 1000);
    run_bench(32, 1024, 14, 14, 2048, 1, 1, 2, 2, 1000);
    run_bench(32, 1024, 14, 14, 512, 1, 1, 2, 2, 1000);
    run_bench(32, 512, 7, 7, 512, 3, 3, 1, 1, 1000);
    run_bench(32, 512, 7, 7, 2048, 1, 1, 1, 1, 1000);

    run_bench(256, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(256, 64, 56, 56, 64, 1, 1, 1, 1, 1000);
    run_bench(256, 64, 56, 56, 64, 3, 3, 1, 1, 1000);
    run_bench(256, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(256, 256, 56, 56, 512, 1, 1, 2, 2, 1000);
    run_bench(256, 256, 56, 56, 128, 1, 1, 2, 2, 1000);
    run_bench(256, 128, 28, 28, 128, 3, 3, 1, 1, 1000);
    run_bench(256, 128, 28, 28, 512, 1, 1, 1, 1, 1000);
    run_bench(256, 512, 28, 28, 1024, 1, 1, 2, 2, 1000);
    run_bench(256, 512, 28, 28, 256, 1, 1, 2, 2, 1000);
    run_bench(256, 256, 14, 14, 256, 3, 3, 1, 1, 1000);
    run_bench(256, 256, 14, 14, 1024, 1, 1, 1, 1, 1000);
    run_bench(256, 1024, 14, 14, 2048, 1, 1, 2, 2, 1000);
    run_bench(256, 1024, 14, 14, 512, 1, 1, 2, 2, 1000);
    run_bench(256, 512, 7, 7, 512, 3, 3, 1, 1, 1000);
    run_bench(256, 512, 7, 7, 2048, 1, 1, 1, 1, 1000);
}
#endif

TEST_F(CUDA, CONV_BIAS_FORWARD_NCHW4) {
    require_compute_capability(6, 1);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-5, 5};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW4;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;

    checker.set_dtype(0, dtype::QuantizedS8(0.5f))
            .set_dtype(1, dtype::QuantizedS8(0.5f))
            .set_dtype(2, dtype::QuantizedS32(0.25f))
            .set_dtype(3, dtype::QuantizedS8(0.13f))
            .set_dtype(4, dtype::QuantizedS8(0.35f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &int_rng)
            .set_rng(3, &int_rng)
            .set_param(param);

    auto opr = handle_cuda()->create_operator<ConvBias>();

    auto run = [&](const TensorShapeArray& shapes) {
        opr->param() = param;
        TensorLayout dst_layout;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, {}, {}, dst_layout);
        checker.execs({shapes[0], shapes[1], shapes[2], dst_layout, {}});
    };

    run({{1, 4, 4, 4, 4}, {4, 4, 3, 3, 4}, {1, 1, 1, 1, 4}});
    run({{20, 1, 24, 24, 4}, {24, 1, 2, 2, 4}, {1, 6, 1, 1, 4}});
    run({{20, 2, 24, 24, 4}, {24, 2, 3, 3, 4}, {1, 6, 1, 1, 4}});

    param.sparse = ConvBias::Param::Sparse::GROUP;
    checker.set_param(param);
    run({{1, 4, 24, 24, 4}, {4, 4, 1, 1, 1, 4}, {1, 4, 1, 1, 4}});
    run({{20, 8, 24, 24, 4}, {4, 24, 2, 2, 2, 4}, {1, 24, 1, 1, 4}});
    run({{1, 3, 24, 24, 4}, {3, 8, 1, 3, 3, 4}, {1, 6, 1, 1, 4}});

    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 2;
    checker.set_param(param);
    run({{10, 16, 28, 28, 4}, {8, 8, 2, 3, 3, 4}, {1, 16, 1, 1, 4}});

    // case which cudnn not supported
    param.sparse = ConvBias::Param::Sparse::DENSE;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    checker.set_param(param);
    checker.exec({{1, 4, 2, 2, 4}, {16, 4, 3, 3, 4}, {1, 4, 1, 1, 4}, {}, {}});
}

TEST_F(CUDA, CONV_BIAS_FORWARD_NCHW4_NCHW) {
    require_compute_capability(6, 1);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-3, 3};
    UniformFloatRNG float_rng{-50, 50};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW4_NCHW;
    param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;

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

    run({{1, 4, 4, 4, 4}, {4, 4, 3, 3, 4}, {1, 4, 1, 1}});
    run({{20, 1, 24, 24, 4}, {24, 1, 2, 2, 4}, {1, 24, 1, 1}});
    run({{20, 2, 24, 24, 4}, {24, 2, 3, 3, 4}, {1, 24, 1, 1}});

    param.sparse = ConvBias::Param::Sparse::GROUP;
    param.nonlineMode = ConvBias::Param::NonlineMode::RELU;
    checker.set_param(param);
    run({{1, 4, 24, 24, 4}, {4, 4, 1, 1, 1, 4}, {1, 16, 1, 1}});
    run({{20, 8, 24, 24, 4}, {4, 24, 2, 2, 2, 4}, {1, 96, 1, 1}});
    run({{1, 3, 24, 24, 4}, {3, 8, 1, 3, 3, 4}, {1, 24, 1, 1}});

    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 2;
    checker.set_param(param);
    run({{10, 16, 28, 28, 4}, {8, 8, 2, 3, 3, 4}, {1, 64, 1, 1}});

    // case which cudnn not supported
    param.sparse = ConvBias::Param::Sparse::DENSE;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    param.nonlineMode = ConvBias::Param::NonlineMode::H_SWISH;
    checker.set_param(param);
    checker.exec({{1, 4, 2, 2, 4}, {16, 4, 3, 3, 4}, {1, 16, 1, 1}, {}, {}});
}

#endif

TEST_F(CUDA, CONV_BIAS_FORWARD_CHANWISE) {
    Checker<ConvBiasForward> checker(handle_cuda());
    std::vector<TestArg> args = get_chanwise_args();
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBias::DirectParam>("CHANNEL_WISE",
                                                              {})
                    .c_str()));

    for (auto dtype : std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
        checker.set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_dtype(4, dtype);
        if (dtype.enumv() == DTypeEnum::Float16)
            checker.set_epsilon(2e-2);
        for (auto&& arg : args) {
            checker.set_param(arg.param).execs(
                    {arg.src, arg.filter, arg.bias, {}, {}});
        }
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_CHANWISE_SMALL) {
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBias::DirectParam>(
                    "CHANNEL_WISE_SMALL", {})
                    .c_str()));
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.sparse = ConvBias::Param::Sparse::GROUP;

    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
        cur_param.nonlineMode = nlmode;
        for (auto dtype : std::vector<DType> {
                 dtype::Float32(),
#if CUDA_VERSION >= 9000
                         dtype::Float16()
#endif
             }) {
            checker.set_dtype(0, dtype)
                    .set_dtype(1, dtype)
                    .set_dtype(2, dtype)
                    .set_dtype(3, dtype)
                    .set_dtype(4, dtype);
            if (dtype.enumv() == DTypeEnum::Float16)
                checker.set_epsilon(2e-2);

            for (uint32_t s : {1}) {
                for (uint32_t f : {1, 3, 5, 7}) {
                    cur_param.pad_h = cur_param.pad_w = f / 2;
                    cur_param.stride_h = cur_param.stride_w = s;
                    checker.set_param(cur_param).execs({{2, 3, 16, 16},
                                                        {3, 1, 1, f, f},
                                                        {1, 3, 1, 1},
                                                        {},
                                                        {}});
                }
            }

            cur_param.pad_h = cur_param.pad_w = 1;
            cur_param.stride_h = cur_param.stride_w = 1;
            checker.set_param(cur_param)
                    .execs({{2, 3, 3, 16},
                            {3, 1, 1, 3, 3},
                            {1, 3, 1, 1},
                            {},
                            {}})
                    .execs({{2, 3, 8, 3},
                            {3, 1, 1, 3, 3},
                            {1, 3, 1, 1},
                            {},
                            {}});
        }
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_CHANWISE_8x8x32) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBias::DirectParam>(
                    "CHANNEL_WISE_8X8X32", {})
                    .c_str()));
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.sparse = ConvBias::Param::Sparse::GROUP;
    cur_param.format = ConvBias::Param::Format::NHWC;

    UniformIntRNG rng(-4, 4);
    checker.set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_dtype(4, dtype::Int32{})
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);
    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
        cur_param.nonlineMode = nlmode;
        for (uint32_t s : {1, 2}) {
            for (uint32_t f : {1, 3, 5, 7}) {
                for (uint32_t g : {4, 8}) {
                    cur_param.pad_h = cur_param.pad_w = f / 2;
                    cur_param.stride_h = cur_param.stride_w = s;
                    checker.set_param(cur_param).execs({{2, 9, 16, g},
                                                        {g, 1, f, f, 1},
                                                        {1, 1, 1, g},
                                                        {},
                                                        {}});
                }
            }
        }
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_CUDNN_CONVOLUTION) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args();
    Checker<ConvBiasForward> checker(handle_cuda());

    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBias::DefaultParam>(
                    "CUDNN:Convolution", {})
                    .c_str()));

    NormalRNG default_rng;
    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_rng(0, &default_rng)
                .set_rng(1, &default_rng)
                .set_rng(2, &default_rng)
                .set_epsilon(1e-3)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_INPLACE_MATMUL) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args();
    Checker<ConvBiasForward> checker(handle_cuda());

    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBias::MatmulParam>("INPLACE_MATMUL",
                                                              {})
                    .c_str()));
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.sparse = ConvBias::Param::Sparse::DENSE;
    NormalRNG default_rng;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &default_rng)
            .set_rng(1, &default_rng)
            .set_rng(2, &default_rng)
            .set_epsilon(1e-3);

    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
        cur_param.nonlineMode = nlmode;
        for (uint32_t s : {1}) {
            for (uint32_t f : {1, 3, 5, 7}) {
                cur_param.pad_h = cur_param.pad_w = f / 2;
                cur_param.stride_h = cur_param.stride_w = s;
                checker.set_param(cur_param).execs(
                        {{2, 4, 16, 16}, {4, 4, f, f}, {1, 4, 1, 1}, {}, {}});
            }
        }

        cur_param.pad_h = cur_param.pad_w = 1;
        cur_param.stride_h = cur_param.stride_w = 1;
        checker.set_param(cur_param)
                .execs({{2, 3, 3, 16}, {5, 3, 3, 3}, {1, 5, 1, 1}, {}, {}})
                .execs({{2, 2, 8, 3}, {3, 2, 3, 3}, {1, 3, 1, 1}, {}, {}});
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_MATMUL) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args();
    Checker<ConvBiasForward> checker(handle_cuda());

    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>("MATMUL",
                                                                     {})
                    .c_str()));
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.sparse = ConvBias::Param::Sparse::DENSE;
    NormalRNG default_rng;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &default_rng)
            .set_rng(1, &default_rng)
            .set_rng(2, &default_rng)
            .set_epsilon(1e-3);

    for (auto nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
        cur_param.nonlineMode = nlmode;
        for (uint32_t s : {1}) {
            for (uint32_t f : {1, 3, 5, 7}) {
                cur_param.pad_h = cur_param.pad_w = f / 2;
                cur_param.stride_h = cur_param.stride_w = s;
                checker.set_param(cur_param).execs(
                        {{2, 4, 16, 16}, {4, 4, f, f}, {1, 4, 1, 1}, {}, {}});
            }
        }

        cur_param.pad_h = cur_param.pad_w = 0;
        cur_param.stride_h = cur_param.stride_w = 1;
        checker.set_param(cur_param)
                .execs({{2, 3, 3, 16}, {5, 3, 3, 3}, {1, 5, 1, 1}, {}, {}})
                .execs({{2, 2, 8, 3}, {3, 2, 3, 3}, {1, 3, 1, 1}, {}, {}});
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_MATMUL_8x8x32) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "MATMUL8X8X32", {})
                    .c_str()));
    param::ConvBias cur_param;
    using NLMode = param::ConvBias::NonlineMode;
    cur_param.mode = param::ConvBias::Mode::CROSS_CORRELATION;
    cur_param.sparse = ConvBias::Param::Sparse::DENSE;
    cur_param.format = param::ConvBias::Format::NHWC;

    UniformIntRNG rng{-100, 100};
    UniformIntRNG bias_rng{-1000, 1000};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &bias_rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.0f})
            .set_epsilon(1);

    for (auto nlmode : {NLMode::IDENTITY, NLMode::RELU}) {
        cur_param.nonlineMode = nlmode;
        for (uint32_t s : {1}) {
            for (uint32_t f : {1, 3, 5, 7}) {
                cur_param.pad_h = cur_param.pad_w = f / 2;
                cur_param.stride_h = cur_param.stride_w = s;
                checker.set_param(cur_param).execs(
                        {{2, 16, 16, 4}, {4, f, f, 4}, {1, 1, 1, 4}, {}, {}});
            }
        }

        cur_param.pad_h = cur_param.pad_w = 0;
        cur_param.stride_h = cur_param.stride_w = 1;
        checker.set_param(cur_param)
                .execs({{2, 3, 16, 3}, {5, 3, 3, 3}, {1, 1, 1, 5}, {}, {}})
                .execs({{2, 8, 3, 2}, {3, 3, 3, 2}, {1, 1, 1, 3}, {}, {}});
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_MATMUL_NCHW4) {
    require_compute_capability(6, 1);
    Checker<ConvBiasForward> checker(handle_cuda());
    checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBias>(
            ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                    "MATMUL8X8X32", {})
                    .c_str()));

    UniformIntRNG int_rng{-127, 127};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW4;
    using NLMode = ConvBias::Param::NonlineMode;

    checker.set_dtype(0, dtype::QuantizedS8(0.5f))
            .set_dtype(1, dtype::QuantizedS8(0.5f))
            .set_dtype(2, dtype::QuantizedS32(0.25f))
            .set_dtype(4, dtype::QuantizedS8(0.35f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &int_rng);

    param.sparse = Convolution::Param::Sparse::DENSE;
    param.nonlineMode = NLMode::IDENTITY;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    checker.set_param(param);
    checker.exec(
            {{8, 4, 10, 10, 4}, {16, 4, 3, 3, 4}, {1, 4, 1, 1, 4}, {}, {}});
    checker.exec({{1, 4, 2, 2, 4}, {16, 4, 3, 3, 4}, {1, 4, 1, 1, 4}, {}, {}});
    checker.exec(
            {{8, 64, 12, 12, 4}, {256, 64, 3, 3, 4}, {1, 64, 1, 1, 4}, {}, {}});
}

TEST_F(CUDA, CONV_BIAS_FORWARD_MATMUL_1x1) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_args_1x1();
    Checker<ConvBiasForward> checker(handle_cuda());

    NormalRNG default_rng;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &default_rng)
            .set_rng(1, &default_rng)
            .set_rng(2, &default_rng)
            .set_epsilon(1e-3);
    for (auto&& arg : args) {
        checker.set_param(arg.param);
        checker.set_before_exec_callback(
                conv_bias::ConvBiasAlgoChecker<ConvBias>(
                        ConvBiasForward::algo_name<
                                ConvBiasForward::MatmulParam>("MATMUL1X1", {})
                                .c_str()));
        checker.execs({arg.src, arg.filter, arg.bias, {}, {}});
        checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<
                                         ConvBias>(
                ConvBiasForward::algo_name<ConvBiasForward::MatmulParam>(
                        "BATCHEDMATMUL", {})
                        .c_str()));
        checker.execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(CUDA, CONV_BIAS_FORWARD_GROUP) {
    using NLMode = ConvBias::Param::NonlineMode;
    bool is_int_available = false;
    if (megdnn::test::check_compute_capability(6, 1)) {
        is_int_available = true;
    } else {
        is_int_available = false;
    }

    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t FH,
                   size_t FW, size_t OC, size_t PH, size_t PW, size_t SH,
                   size_t SW, size_t DH, size_t DW, size_t group, NLMode mode) {
        {
            // float case
            Checker<ConvBiasForward> checker(handle_cuda());
            checker.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<
                                             ConvBias>(
                    ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                            "CUDA:GROUP_CONV", {})
                            .c_str()));
            ConvBias::Param param;
            param.sparse = ConvBias::Param::Sparse::GROUP;
            param.nonlineMode = mode;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_h = DH;
            param.dilate_w = DW;
            auto ICg = IC / group;
            auto OCg = OC / group;
            checker.set_param(param).exec({{N, IC, IH, IW},
                                           {group, OCg, ICg, FH, FW},
                                           {1, OCg * group, 1, 1},
                                           {},
                                           {}});
        }
        if (is_int_available) {
            // int 8x8x32 case
            Checker<ConvBiasForward> checker(handle_cuda());
            ConvBias::Param param;
            param.sparse = Convolution::Param::Sparse::GROUP;
            param.format = Convolution::Param::Format::NHWC;
            param.nonlineMode = NLMode::IDENTITY;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_h = DH;
            param.dilate_w = DW;
            auto ICg = IC / group;
            auto OCg = OC / group;
            UniformIntRNG rng(-4, 4);
            checker.set_param(param)
                    .set_dtype(0, dtype::QuantizedS8(0.5f))
                    .set_dtype(1, dtype::QuantizedS8(0.5f))
                    .set_dtype(2, dtype::QuantizedS32(0.25f))
                    .set_dtype(3, dtype::QuantizedS8(0.13f))
                    .set_dtype(4, dtype::QuantizedS8(0.35f))
                    .set_rng(0, &rng)
                    .set_rng(1, &rng)
                    .set_rng(2, &rng)
                    .exec({{N, IH, IW, IC},
                           {group, OCg, FH, FW, ICg},
                           {1, 1, 1, OCg * group},
                           {},
                           {}});
        }
    };

    for (NLMode nlmode :
         {NLMode::IDENTITY, NLMode::RELU, NLMode::SIGMOID, NLMode::H_SWISH}) {
        // normal case
        run(2, 64, 7, 7, 3, 3, 32, 0, 0, 1, 1, 1, 1, 2, nlmode);
        // padded case
        run(2, 32, 7, 7, 3, 3, 64, 1, 1, 1, 1, 1, 1, 4, nlmode);
        // strided case
        run(2, 32, 7, 7, 3, 3, 64, 0, 0, 2, 2, 1, 1, 8, nlmode);
        // dilated case
        run(2, 32, 7, 7, 3, 3, 64, 0, 0, 1, 1, 2, 2, 8, nlmode);
    }
}

#if CUDA_VERSION >= 10000
TEST_F(CUDA, CONV_BIAS_FORWARD_NCHW8_PART_1) {
    test_conv_bias_forward_wmma_int4_nchw8(handle_cuda(), 3);
}

TEST_F(CUDA, CONV_BIAS_FORWARD_NCHW8_PART_2) {
    test_conv_bias_forward_wmma_int4_nchw8(handle_cuda(), 5);
}

TEST_F(CUDA, CONV_BIAS_FORWARD_NCHW8_PART_3) {
    test_conv_bias_forward_wmma_int4_nchw8(handle_cuda(), 7);
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CONV_BIAS_QUANTIZED4x4x32) {
    require_compute_capability(7, 5);
    Benchmarker<ConvBiasForward> bencher(handle_cuda());

    UniformIntRNG int_rng{0, 8};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW8;
    param.stride_h = param.stride_w = 1;

    using NonlineMode = ConvBias::Param::NonlineMode;
    param.nonlineMode = NonlineMode::RELU;
    auto run_bench = [&](size_t batch, size_t ci, size_t hi, size_t wi,
                         size_t co, size_t fh, size_t fw, size_t nr_times) {
        param.pad_h = fh / 2;
        param.pad_w = fw / 2;
        bencher.set_param(param)
                .set_dtype(0, dtype::Quantized4Asymm(1.3f, (uint8_t)(1)))
                .set_dtype(1, dtype::Quantized4Asymm(1.3f, (uint8_t)(2)))
                .set_dtype(2, dtype::QuantizedS32(1.3f * 1.3f))
                .set_dtype(4, dtype::QuantizedS32(1.3f * 1.3f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng);
        bencher.set_times(nr_times);
        size_t ho = infer_conv_shape(hi, fh, 1, param.pad_h);
        size_t wo = infer_conv_shape(wi, fw, 1, param.pad_w);
        TensorShape inp{batch, ci / 8, hi, wi, 8}, kern{co, ci / 8, fh, fw, 8},
                out{batch, co / 8, ho, wo, 8};
        auto time_in_ms =
                bencher.execs({inp, kern, {1, co / 8, 1, 1, 8}, {}, out}) /
                nr_times;
        auto ops = 2.0 * batch * co * ho * wo * ci * fh * fw /
                   (time_in_ms * 1e-3) * 1e-12;
        printf("inp=%s, kern=%s, out=%s, time: %.2fms, perf: %.2f Tops\n",
               inp.to_string().c_str(), kern.to_string().c_str(),
               out.to_string().c_str(), time_in_ms, ops);
    };
    run_bench(256, 256, 16, 16, 256, 3, 3, 1000);

    run_bench(1, 32, 224, 224, 64, 7, 7, 1000);
    run_bench(1, 8192, 64, 64, 4096, 3, 3, 1000);
    run_bench(1, 256, 64, 64, 256, 3, 3, 1000);
    run_bench(1, 64, 128, 128, 64, 3, 3, 1000);
    run_bench(1, 512, 32, 32, 512, 3, 3, 1000);
    run_bench(1, 1024, 16, 16, 1024, 3, 3, 1000);

    run_bench(1, 64, 56, 56, 64, 3, 3, 1000);
    run_bench(1, 128, 32, 32, 128, 3, 3, 1000);
    run_bench(1, 256, 16, 16, 256, 3, 3, 1000);
    run_bench(1, 512, 8, 8, 512, 3, 3, 1000);

    run_bench(32, 32, 224, 224, 64, 7, 7, 1000);
    run_bench(32, 64, 56, 56, 64, 3, 3, 1000);
    run_bench(32, 128, 32, 32, 128, 3, 3, 1000);
    run_bench(32, 256, 16, 16, 256, 3, 3, 1000);
    run_bench(32, 512, 8, 8, 512, 3, 3, 1000);

    run_bench(256, 32, 224, 224, 64, 7, 7, 1000);
    run_bench(256, 64, 56, 56, 64, 3, 3, 1000);
    run_bench(256, 128, 32, 32, 128, 3, 3, 1000);
    run_bench(256, 256, 16, 16, 256, 3, 3, 1000);
    run_bench(256, 512, 8, 8, 512, 3, 3, 1000);
}
#endif
#endif

TEST_F(CUDA, CONV_BIAS_FORWARD_DILATED) {
    require_compute_capability(6, 0);
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t FH,
                   size_t FW, size_t OC, size_t PH, size_t PW, size_t SH,
                   size_t SW, size_t DH, size_t DW) {
        {
            // float case
            Checker<ConvBiasForward> checker(handle_cuda());
            ConvBias::Param param;
            param.sparse = ConvBias::Param::Sparse::DENSE;
            param.pad_h = PH;
            param.pad_w = PW;
            param.stride_h = SH;
            param.stride_w = SW;
            param.dilate_h = DH;
            param.dilate_w = DW;
            param.nonlineMode = ConvBias::Param::NonlineMode::IDENTITY;
            checker.set_param(param).exec(
                    {{N, IC, IH, IW}, {OC, IC, FH, FW}, {1, OC, 1, 1}, {}, {}});
        }
    };

    // dilated case
    run(2, 8, 7, 7, 3, 3, 4, 0, 0, 1, 1, 2, 2);
}

#if CUDNN_VERSION >= 7500
TEST_F(CUDA, CONV_BIAS_FORWARD_TENSORCORE_INT8) {
    require_compute_capability(7, 5);
    using namespace conv_bias;
    Checker<ConvBiasForward> checker(handle_cuda());
    auto opr = handle_cuda()->create_operator<ConvBias>();
    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW32;

    using NonlineMode = ConvBias::Param::NonlineMode;
    for (NonlineMode mode : {NonlineMode::IDENTITY, NonlineMode::RELU}) {
        for (size_t batch : {2}) {
            for (size_t ic : {64, 32}) {
                for (size_t oc : {32}) {
                    for (size_t fh : {3, 5, 7}) {
                        for (int ph : {static_cast<int>(fh / 2), 0}) {
                            for (int sh : {1, 2}) {
                                for (size_t ih : {9, 11, 12, 13, 16}) {
                                    for (size_t iw : {8, 27, 32, 40}) {
                                        param.nonlineMode = mode;
                                        param.stride_h = param.stride_w = sh;
                                        param.pad_h = param.pad_w = ph;

                                        opr->param() = param;
                                        TensorLayout dst_layout;
                                        opr->deduce_layout(
                                                {{batch, ic / 32, ih, iw, 32},
                                                 dtype::Float32()},
                                                {{oc, ic / 32, fh, fh, 32},
                                                 dtype::Float32()},
                                                {}, {}, dst_layout);

                                        checker.set_dtype(0, dtype::QuantizedS8(
                                                                     1.3f))
                                                .set_dtype(1,
                                                           dtype::QuantizedS8(
                                                                   1.3f))
                                                .set_dtype(2,
                                                           dtype::QuantizedS32(
                                                                   1.3f * 1.3f))
                                                .set_dtype(3,
                                                           dtype::QuantizedS8(
                                                                   1.7f))

                                                .set_dtype(4,
                                                           dtype::QuantizedS8(
                                                                   1.2f * 1.2f))
                                                .set_rng(0, &int_rng)
                                                .set_rng(1, &int_rng)
                                                .set_rng(2, &int_rng)
                                                .set_rng(3, &int_rng)
                                                .set_epsilon(1 + 1e-3)
                                                .set_param(param)
                                                .execs({{batch, ic / 32, ih, iw,
                                                         32},
                                                        {oc, ic / 32, fh, fh,
                                                         32},
                                                        {1, oc / 32, 1, 1, 32},
                                                        dst_layout,
                                                        {}});
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

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CONV_BIAS_FORWARD_TENSORCORE_INT8) {
    require_compute_capability(7, 5);
    Benchmarker<ConvBiasForward> bencher(handle_cuda());
    bencher.set_display(false);

    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW32;
    ConvBias::Param param_without_tensorcore;
    param_without_tensorcore.format = ConvBias::Param::Format::NCHW4;

    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};

    using NonlineMode = ConvBias::Param::NonlineMode;
    param.nonlineMode = NonlineMode::IDENTITY;
    auto run_bench = [&](size_t batch, size_t ci, size_t hi, size_t wi,
                         size_t co, size_t fh, size_t fw, size_t sh, size_t sw,
                         size_t nr_times) {
        param.pad_h = fh / 2;
        param.pad_w = fw / 2;
        param.stride_h = sh;
        param.stride_w = sw;

        param_without_tensorcore.pad_h = fh / 2;
        param_without_tensorcore.pad_w = fw / 2;
        param_without_tensorcore.stride_h = sh;
        param_without_tensorcore.stride_w = sw;
        bencher.set_param(param)
                .set_dtype(0, dtype::QuantizedS8(1.3f))
                .set_dtype(1, dtype::QuantizedS8(1.3f))
                .set_dtype(2, dtype::QuantizedS32(1.3f * 1.3f))
                .set_dtype(4, dtype::QuantizedS8(1.2f))
                .set_rng(0, &int_rng)
                .set_rng(1, &int_rng)
                .set_rng(2, &int_rng);
        bencher.set_times(nr_times);
        size_t ho = infer_conv_shape(hi, fh, sh, param.pad_h);
        size_t wo = infer_conv_shape(wi, fw, sw, param.pad_w);
        TensorShape inp{batch, ci / 32, hi, wi, 32},
                kern{co, ci / 32, fh, fw, 32}, out{batch, co / 32, ho, wo, 32};
        auto time_in_ms =
                bencher.execs({inp, kern, {1, co / 32, 1, 1, 32}, {}, out}) /
                nr_times;
        auto ops = 2.0 * batch * co * ho * wo * ci * fh * fw /
                   (time_in_ms * 1e-3) * 1e-12;
        printf("inp=%s, kern=%s, out=%s, time: %.2fms, perf: %.2f Tops "
               "(TensorCore)",
               inp.to_string().c_str(), kern.to_string().c_str(),
               out.to_string().c_str(), time_in_ms, ops);
        decltype(ops) ops_without_tensorcore;
        bencher.set_param(param_without_tensorcore);
        {
            TensorShape inp{batch, ci / 4, hi, wi, 4},
                    kern{co, ci / 4, fh, fw, 4}, out{batch, co / 4, ho, wo, 4};
            auto time_in_ms =
                    bencher.execs({inp, kern, {1, co / 4, 1, 1, 4}, {}, out}) /
                    nr_times;
            ops_without_tensorcore = 2.0 * batch * co * ho * wo * ci * fh * fw /
                                     (time_in_ms * 1e-3) * 1e-12;
            printf(", time: %.2fms perf: %.2f Tops (without TensorCore) ",
                   time_in_ms, ops_without_tensorcore);
        }
        printf("speedup: %.2fx\n", ops / ops_without_tensorcore);
    };

    // resnet-50
    // bottleneck-1
    // proj
    run_bench(1, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(1, 64, 56, 56, 64, 1, 1, 1, 1, 1000);
    run_bench(1, 64, 56, 56, 64, 3, 3, 1, 1, 1000);
    run_bench(1, 64, 56, 56, 256, 1, 1, 1, 1, 1000);

    // bottleneck-2
    // proj
    run_bench(1, 256, 56, 56, 512, 1, 1, 2, 2, 1000);
    run_bench(1, 256, 56, 56, 128, 1, 1, 2, 2, 1000);
    run_bench(1, 128, 28, 28, 128, 3, 3, 1, 1, 1000);
    run_bench(1, 128, 28, 28, 512, 1, 1, 1, 1, 1000);

    // bottleneck-3
    // proj
    run_bench(1, 512, 28, 28, 1024, 1, 1, 2, 2, 1000);
    run_bench(1, 512, 28, 28, 256, 1, 1, 2, 2, 1000);
    run_bench(1, 256, 14, 14, 256, 3, 3, 1, 1, 1000);
    run_bench(1, 256, 14, 14, 1024, 1, 1, 1, 1, 1000);

    // bottleneck-4
    // proj
    run_bench(1, 1024, 14, 14, 2048, 1, 1, 2, 2, 1000);
    run_bench(1, 1024, 14, 14, 512, 1, 1, 2, 2, 1000);
    run_bench(1, 512, 7, 7, 512, 3, 3, 1, 1, 1000);
    run_bench(1, 512, 7, 7, 2048, 1, 1, 1, 1, 1000);

    run_bench(32, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(32, 64, 56, 56, 64, 1, 1, 1, 1, 1000);
    run_bench(32, 64, 56, 56, 64, 3, 3, 1, 1, 1000);
    run_bench(32, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(32, 256, 56, 56, 512, 1, 1, 2, 2, 1000);
    run_bench(32, 256, 56, 56, 128, 1, 1, 2, 2, 1000);
    run_bench(32, 128, 28, 28, 128, 3, 3, 1, 1, 1000);
    run_bench(32, 128, 28, 28, 512, 1, 1, 1, 1, 1000);
    run_bench(32, 512, 28, 28, 1024, 1, 1, 2, 2, 1000);
    run_bench(32, 512, 28, 28, 256, 1, 1, 2, 2, 1000);
    run_bench(32, 256, 14, 14, 256, 3, 3, 1, 1, 1000);
    run_bench(32, 256, 14, 14, 1024, 1, 1, 1, 1, 1000);
    run_bench(32, 1024, 14, 14, 2048, 1, 1, 2, 2, 1000);
    run_bench(32, 1024, 14, 14, 512, 1, 1, 2, 2, 1000);
    run_bench(32, 512, 7, 7, 512, 3, 3, 1, 1, 1000);
    run_bench(32, 512, 7, 7, 2048, 1, 1, 1, 1, 1000);

    run_bench(256, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(256, 64, 56, 56, 64, 1, 1, 1, 1, 1000);
    run_bench(256, 64, 56, 56, 64, 3, 3, 1, 1, 1000);
    run_bench(256, 64, 56, 56, 256, 1, 1, 1, 1, 1000);
    run_bench(256, 256, 56, 56, 512, 1, 1, 2, 2, 1000);
    run_bench(256, 256, 56, 56, 128, 1, 1, 2, 2, 1000);
    run_bench(256, 128, 28, 28, 128, 3, 3, 1, 1, 1000);
    run_bench(256, 128, 28, 28, 512, 1, 1, 1, 1, 1000);
    run_bench(256, 512, 28, 28, 1024, 1, 1, 2, 2, 1000);
    run_bench(256, 512, 28, 28, 256, 1, 1, 2, 2, 1000);
    run_bench(256, 256, 14, 14, 256, 3, 3, 1, 1, 1000);
    run_bench(256, 256, 14, 14, 1024, 1, 1, 1, 1, 1000);
    run_bench(256, 1024, 14, 14, 2048, 1, 1, 2, 2, 1000);
    run_bench(256, 1024, 14, 14, 512, 1, 1, 2, 2, 1000);
    run_bench(256, 512, 7, 7, 512, 3, 3, 1, 1, 1000);
    run_bench(256, 512, 7, 7, 2048, 1, 1, 1, 1, 1000);
}
#endif
#endif

// vim: syntax=cpp.doxygen
