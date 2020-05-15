/**
 * \file dnn/test/naive/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "test/common/benchmarker.h"
#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/common/convolution.h"
#include "test/common/extra_impl_helper.h"

using namespace megdnn;
using namespace test;

#if MEGDNN_WITH_BENCHMARK

TEST_F(NAIVE, BENCHMARK_CONVOLUTION_BACKWARD_DATA) {
    using Param = ConvolutionBackwardData::Param;
    auto run = [&](const TensorLayoutArray& tensors, Param param) {
        Benchmarker<ConvolutionBackwardData> benchmarker_naive(handle());
        size_t RUN = 500;
        auto tfloat = benchmarker_naive.set_display(false)
                              .set_dtype(0, dtype::Float32{})
                              .set_dtype(1, dtype::Float32{})
                              .set_times(RUN)
                              .set_param(param)
                              .exec(tensors);
        size_t IC = tensors[0][1];
        size_t FH = tensors[0][2];
        size_t FW = tensors[0][3];
        printf("fp32 flops: %.3f mflops\n",
               (IC * tensors[1].total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };

    auto profile = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                       size_t fh, size_t fw, size_t stride = 1,
                       size_t padding = 0) {
        Param param;
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, ow, oh, stride, fh);

        TensorLayout diff = TensorLayout{{n, oc, oh, ow}, dtype::Float32()};
        TensorLayout filter = TensorLayout{{oc, ic, fh, fw}, dtype::Float32()};
        TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        run(TensorLayoutArray{filter, diff, grad}, param);
    };

    profile(1, 1, 2, 2, 1, 3, 3);
    profile(1, 1, 4, 4, 1, 3, 3);
    profile(1, 1, 8, 8, 1, 3, 3);
    profile(1, 1, 16, 16, 1, 3, 3);
    profile(1, 1, 32, 32, 1, 3, 3);
    profile(1, 1, 64, 64, 1, 3, 3);
    profile(1, 1, 128, 128, 1, 3, 3);
}

#endif

TEST_F(NAIVE, CONVOLUTION_QUANTIZED8x8x32) {
    Checker<Convolution> checker(handle(), /* check_dispatch */false);
    Convolution::Param param;
    param.format = Convolution::Param::Format::NCHW;

    checker.set_param(param).exect(
        Testcase{
          TensorValue({1, 1, 4, 4}, dtype::Quantized8Asymm(0.1f, (uint8_t)128),
                      {90, 136, 85, 204,
                       48, 9, 226, 25,
                       118, 109, 87, 132,
                       104, 163, 25, 90}),
          TensorValue({3, 1, 3, 3}, dtype::Quantized8Asymm(0.2f, (uint8_t)124),
                      {153, 170, 102,
                       103, 23,  213,
                       116, 195, 191,

                       44,  50,  247,
                       172, 42,  32,
                       233, 163, 247,

                       120, 241, 209,
                       83,  201, 115,
                       32,  140, 147}),
          {}},
        Testcase{
          {},
          {},
          TensorValue({1, 3, 2, 2}, dtype::QuantizedS32(0.1f * 0.2f),
                      {18617, -22475,
                       -15694, -1920,

                       -12813, 4440,
                       18190, -13195,

                       -9659, 15933,
                       -5558, -4969})});
}


TEST_F(NAIVE, DECONVOLUTION_QUANTIZED8x8x32) {
    Checker<ConvolutionBackwardData> checker(handle(), /* check_dispatch */false);
    ConvolutionBackwardData::Param param;
    param.format = ConvolutionBackwardData::Param::Format::NCHW;

    checker.set_param(param).exect(
        Testcase{
          TensorValue({1, 3, 3, 3}, dtype::Quantized8Asymm(0.0084f, (uint8_t)135),
                      {131, 155, 190,
                       255,  43, 155,
                        97, 238, 127,

                       157,  72, 161,
                       157,   0,  69,
                       204, 167, 180,

                       108,  47, 203,
                       179, 136,  83,
                       143, 182, 105}),
          TensorValue({1, 1, 4, 4}, dtype::Quantized8Asymm(0.1f, (uint8_t)157),
                      {126,  49,  99,   0,
                       173,  19, 129,  19,
                       161, 180,  32, 255,
                       203, 120, 208,  96}),
          {}},
        Testcase{
          {},
          {},
          TensorValue({1, 3, 6, 6}, dtype::QuantizedS32(0.1f * 0.0084f),
                      {   124,   -188,  -3633,  -6472,  -6330,  -8635,
                        -3784,  -9236,    588, -23262,   8984, -10730,
                         3082, -17133,   2164, -17515,  -8486,   3886,
                         -312,  10352, -28728,  26413, -23921,   -291,
                         5368,  -9134,  17531, -29535,  17726,  -2004,
                        -1748,   6144,  -6117,   7867,  -6691,    488,

                         -682,   -423,   4722,  -2608,   8383,  -4082,
                         -330,  -2235,  23844,   6644,  32989,   6774,
                        -1699, -13386,   4010,   2932,   3420,   4591,
                         2204, -12756,  -7098,  -4632,  -5487, -14264,
                         1288,  -5309,  -4628,  -1988,   2380,   8436,
                         3174,  -1081,   4405,  -4242,    343,  -2745,

                          837,   5644,   8962,   1999,   9872, -10676,
                        -1796,  -2465,  12940,  -4544,  13099,  -1220,
                          348,  -9350,  -5189,  10252, -21445,  18550,
                         -938,  -2385,  -7868,   -646,   9788,  -5104,
                         2056,  -1210,   -224,  -6490,   5643,    232,
                          368,   1866,  -2711,   3019,  -4397,   1830})});
}

TEST_F(NAIVE, CONVOLUTION_WITH_NCHW4) {
    Checker<Convolution> checker(handle());
    Convolution::Param param;
    param.format = Convolution::Param::Format::NCHW4;
    auto convert_true_format = [](const TensorLayout& layout) {
        if (layout.ndim == 4)
            return layout
                    .reshape(
                            {layout[0], layout[1] / 4, layout[2], layout[3], 4})
                    .dimshuffle({0, 1, 4, 2, 3});
        else
            return layout
                    .reshape({layout[0], layout[1], layout[2] / 4, layout[3],
                              layout[4], 4})
                    .dimshuffle({0, 1, 2, 5, 3, 4});
    };

    auto extra_impl = [&, this](const TensorNDArray& tensors) {
        auto conv = handle()->create_operator<Convolution>();
        conv->param() = param;
        conv->param().format = Convolution::Param::Format::NCHW;

        TensorNDArray nchw_tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto layout = tensors[i].layout;
            if (layout.ndim == 5) {
                layout = layout.reshape({layout[0], layout[1] * layout[4],
                        layout[2], layout[3]});
            } else {
                megdnn_assert(layout.ndim == 6 && 
                        param.sparse == Convolution::Param::Sparse::GROUP);
                layout = layout.reshape(
                        {layout[0], layout[1], layout[2] * layout[5],
                        layout[3], layout[4]});
            }
            nchw_tensors.emplace_back(
                    malloc(layout.span().dist_byte()), layout);
        }

        TensorNDArray nchw4_tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto layout = convert_true_format(nchw_tensors[i].layout);
            nchw4_tensors.emplace_back(tensors[i].raw_ptr, std::move(layout));
        }

        auto workspace_size = conv->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout,
                nullptr);
        dt_byte* workspace_ptr = static_cast<dt_byte*>(malloc(workspace_size));
        Workspace workspace{workspace_ptr, workspace_size};

        auto relayout = handle()->create_operator<RelayoutForward>();
        relayout->exec(nchw4_tensors[0], nchw_tensors[0]);
        relayout->exec(nchw4_tensors[1], nchw_tensors[1]);

        conv->exec(nchw_tensors[0], nchw_tensors[1], nchw_tensors[2], nullptr,
                   workspace);

        relayout->exec(nchw_tensors[2], nchw4_tensors[2]);

        free(workspace_ptr);
        for (auto&& tensor : nchw_tensors) {
            free(tensor.raw_ptr);
        }
    };

    UniformIntRNG rng{0, 4};
    ConstValue filter_rng{1};
    checker.set_extra_opr_impl(extra_impl)
            .set_rng(0, &filter_rng)
            .set_rng(1, &filter_rng);
    checker.set_param(param)
            .execs({{1, 2, 2, 2, 4}, {4, 2, 1, 1, 4}, {}})
            .execs({{20, 3, 30, 30, 4}, {4, 3, 1, 1, 4}, {}})
            .execs({{20, 2, 30, 30, 4}, {4, 2, 3, 3, 4}, {}});

    param.sparse = Convolution::Param::Sparse::GROUP;
    checker.set_param(param)
            .execs({{20, 15, 30, 30, 4}, {5, 4, 3, 3, 3, 4}, {}})
            .execs({{20, 25, 30, 30, 4}, {5, 4, 5, 1, 1, 4}, {}})
            .execs({{20, 27, 30, 30, 4}, {3, 4, 9, 1, 1, 4}, {}});
}

TEST_F(NAIVE, CONVOLUTION_BFLOAT16) {
    Checker<Convolution> checker(handle(), false);
    using Param = Convolution::Param;
    Param param;
    param.sparse = param::Convolution::Sparse::DENSE;
    Param impl_param = param;

    auto run = [&](size_t n, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t fh, size_t fw) {
        float scale = 1.0f / sqrt(ic * fh * fw);
        UniformFloatRNG rng(scale, 2 * scale);
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        impl_param.pad_h = impl_param.pad_w = 1;
        impl_param.stride_h = impl_param.stride_w = 1;
        auto extra_impl =
                    extra_impl_helper<Convolution>(handle(), impl_param);
        for (auto cmode :
             std::vector<Param::ComputeMode>{Param::ComputeMode::DEFAULT,
                                             Param::ComputeMode::FLOAT32}) {
            param.compute_mode = cmode;
            checker.set_param(param)
                    .set_dtype(0, dtype::BFloat16())
                    .set_dtype(1, dtype::BFloat16())
                    // Use inferred output dtype.
                    .set_dtype(2, {})
                    .set_rng(0, &rng)
                    .set_rng(1, &rng)
                    .set_extra_opr_impl(extra_impl)
                    .set_epsilon(1e-1)
                    .execs({{n, ic, ih, iw}, {oc, ic, fh, fw}, {}});
        }
    };

    run(1, 1, 20, 20, 5, 3, 3);
    run(1, 2, 8, 7, 11, 3, 1);
}

TEST_F(NAIVE, CONVOLUTION_BACKWARD_DATA_BFLOAT16) {
    Checker<ConvolutionBackwardData> checker(handle(), false);
    using Param = ConvolutionBackwardData::Param;

    Param param, impl_param;
    param.sparse = Param::Sparse::DENSE;
    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t padding,
                   const Param::ComputeMode& cmode =
                           Param::ComputeMode::DEFAULT) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = 1;
        param.compute_mode = cmode;

        TensorLayout diff =
                TensorLayout{{n, oc, oh, ow}, dtype::BFloat16()};
        TensorLayout grad;
        TensorLayout filter;
        filter = {{oc, ic, fh, fw}, dtype::BFloat16()};
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        impl_param = param;
        impl_param.compute_mode = Param::ComputeMode::DEFAULT;
        auto extra_impl = extra_impl_helper<ConvolutionBackwardData>(
                handle(), impl_param);
        checker.set_param(param)
                .set_extra_opr_impl(extra_impl)
                .set_epsilon(1e-1)
                .set_dtype(0, dtype::BFloat16())
                .set_dtype(1, dtype::BFloat16());
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    run(4, 3, 10, 13, 5, 1, 1, 1, 0);
    run(2, 1, 24, 43, 11, 3, 3, 2, 1, Param::ComputeMode::FLOAT32);
}

TEST_F(NAIVE, CONVOLUTION_BACKWARD_FILTER_BFLOAT16) {
    using namespace convolution;
    Checker<ConvolutionBackwardFilter> checker(handle(), false);
    using Param = ConvolutionBackwardFilter::Param;
    Param param;
    Param impl_param = param;

    auto run = [&](size_t n, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t fh, size_t fw,
                   const Param::ComputeMode& cmode =
                           Param::ComputeMode::DEFAULT) {
        auto src = TensorLayout({n, ic, ih, iw}, dtype::BFloat16());
        auto filter = TensorLayout({oc, ic, fh, fw}, dtype::BFloat16());
        TensorLayout dst;
        {
            auto opr = handle()->create_operator<Convolution>();
            opr->param() = param;
            opr->deduce_layout(src, filter, dst);
        }
        float scale = 1.0f / sqrt(dst[2] * dst[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        src.dtype = dst.dtype = filter.dtype = dtype::BFloat16();
        param.compute_mode = cmode;
        impl_param = param;
        impl_param.compute_mode = Param::ComputeMode::DEFAULT;
        auto extra_impl = extra_impl_helper<ConvolutionBackwardFilter>(
                handle(), impl_param);
        checker.set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_dtype(0, dtype::BFloat16())
                .set_dtype(1, dtype::BFloat16())
                .set_epsilon(1e-1)
                .set_extra_opr_impl(extra_impl)
                .set_param(param)
                .exec(TensorLayoutArray{src, dst, filter});
    };

    run(1, 2, 8, 7, 11, 3, 1);
    run(1, 1, 20, 20, 5, 3, 3, Param::ComputeMode::FLOAT32);
}

// vim: syntax=cpp.doxygen
