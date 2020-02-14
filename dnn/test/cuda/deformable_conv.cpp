/**
 * \file dnn/test/cuda/deformable_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/nn.h"
#include "src/cuda/utils.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

namespace {
void calc_output_shape(const size_t& ih, const size_t& iw, const size_t& fh,
                       const size_t& fw, const size_t& ph, const size_t& pw,
                       const size_t& sh, const size_t& sw, const size_t& dh,
                       const size_t& dw, size_t& oh, size_t& ow) {
    auto kh = 1 + (fh - 1) * dh;
    auto kw = 1 + (fw - 1) * dw;

    int deduced_oh = ((int)ih + ph * 2 - kh) / sh + 1;
    int deduced_ow = ((int)iw + pw * 2 - kw) / sw + 1;
    oh = deduced_oh, ow = deduced_ow;
}
}  // namespace

TEST_F(CUDA, DEFORMABLE_CONV_FWD) {
    Checker<DeformableConv> checker(handle_cuda());
    Convolution::Param param;

    UniformFloatRNG im_rng{-10, 10};
    UniformFloatRNG filter_rng{-1, 1};
    UniformFloatRNG offset_rng{-2, 2};
    UniformFloatRNG mask_rng{-1, 1};

    checker.set_epsilon(0.01)
            .set_rng(0, &im_rng)
            .set_rng(1, &filter_rng)
            .set_rng(2, &offset_rng)
            .set_rng(3, &mask_rng);

    auto run_test = [&](size_t ih, size_t iw, size_t fh, size_t fw, size_t ph,
                        size_t pw, size_t sh, size_t sw, size_t dh, size_t dw,
                        size_t ic, size_t oc, size_t batch, size_t group,
                        size_t deformable_group) {
        size_t oh, ow;
        calc_output_shape(ih, iw, fh, fw, ph, pw, sh, sw, dh, dw, oh, ow);
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilate_h = dh;
        param.dilate_w = dw;
        param.format = DeformableConv::Param::Format::NCHW;
        param.mode = DeformableConv::Param::Mode::CROSS_CORRELATION;
        if (group > 1) {
            param.sparse = DeformableConv::Param::Sparse::GROUP;
            checker.set_param(param).execs(
                    {{batch, ic, ih, iw},
                     {group, oc / group, ic / group, fh, fw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow},
                     {batch, oc, oh, ow}});
        } else {
            param.sparse = DeformableConv::Param::Sparse::DENSE;
            checker.set_param(param).execs(
                    {{batch, ic, ih, iw},
                     {oc, ic, fh, fw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow},
                     {batch, oc, oh, ow}});
        }
    };

    for (auto batch : std::vector<int>{1, 3})
        for (auto hw : std::vector<int>{16, 20})
            for (auto fhw : std::vector<int>{3, 5, 7})
                for (auto phw : std::vector<int>{2, 5})
                    for (auto shw : std::vector<int>{1, 3})
                        for (auto g : std::vector<int>{1, 2})
                            for (auto icpg : std::vector<int>{1, 3})
                                for (auto ocpg : std::vector<int>{1, 3}) {
                                    auto dhw = shw;
                                    run_test(hw, hw, fhw, fhw, phw, phw, shw,
                                             shw, dhw, dhw, g * icpg, g * ocpg,
                                             batch, g, g);
                                }
}

TEST_F(CUDA, DEFORMABLE_CONV_BWD_FILTER) {
    Checker<DeformableConvBackwardFilter> checker(handle_cuda());
    Convolution::Param param;

    UniformFloatRNG im_rng{-10, 10};
    UniformFloatRNG offset_rng{-2, 2};
    UniformFloatRNG mask_rng{-1, 1};
    UniformFloatRNG out_grad_rng{-1, 1};

    checker.set_epsilon(0.01)
            .set_rng(0, &im_rng)
            .set_rng(1, &offset_rng)
            .set_rng(2, &mask_rng)
            .set_rng(3, &out_grad_rng);

    auto run_test = [&](size_t ih, size_t iw, size_t fh, size_t fw, size_t ph,
                        size_t pw, size_t sh, size_t sw, size_t dh, size_t dw,
                        size_t ic, size_t oc, size_t batch, size_t group,
                        size_t deformable_group) {
        size_t oh, ow;
        calc_output_shape(ih, iw, fh, fw, ph, pw, sh, sw, dh, dw, oh, ow);
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilate_h = dh;
        param.dilate_w = dw;
        param.format = DeformableConv::Param::Format::NCHW;
        param.mode = DeformableConv::Param::Mode::CROSS_CORRELATION;
        if (group > 1) {
            param.sparse = DeformableConv::Param::Sparse::GROUP;
            checker.set_param(param).execs(
                    {{batch, ic, ih, iw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow},
                     {batch, oc, oh, ow},
                     {group, oc / group, ic / group, fh, fw}});
        } else {
            param.sparse = DeformableConv::Param::Sparse::DENSE;
            checker.set_param(param).execs(
                    {{batch, ic, ih, iw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow},
                     {batch, oc, oh, ow},
                     {oc, ic, fh, fw}});
        }
    };

    for (auto batch : std::vector<int>{1, 2})
        for (auto hw : std::vector<int>{16, 20})
            for (auto fhw : std::vector<int>{3, 5, 7})
                for (auto phw : std::vector<int>{2, 5})
                    for (auto shw : std::vector<int>{1, 3})
                        for (auto g : std::vector<int>{1, 2})
                            for (auto icpg : std::vector<int>{1, 5})
                                for (auto ocpg : std::vector<int>{1, 5}) {
                                    auto dhw = shw;
                                    run_test(hw, hw, fhw, fhw, phw, phw, shw,
                                             shw, dhw, dhw, g * icpg, g * ocpg,
                                             batch, g, g);
                                }
}

TEST_F(CUDA, DEFORMABLE_CONV_BWD_DATA) {
    Checker<DeformableConvBackwardData> checker(handle_cuda());
    Convolution::Param param;

    UniformFloatRNG im_rng{0, 255};
    UniformFloatRNG filter_rng{-1, 1};
    UniformFloatRNG offset_rng{-2, 2};
    UniformFloatRNG mask_rng{0, 1};
    UniformFloatRNG out_grad_rng{0, 2};

    checker.set_epsilon(0.1f)
            .set_rng(0, &im_rng)
            .set_rng(1, &filter_rng)
            .set_rng(2, &offset_rng)
            .set_rng(3, &mask_rng)
            .set_rng(4, &out_grad_rng);

    auto run_test = [&](size_t ih, size_t iw, size_t fh, size_t fw, size_t ph,
                        size_t pw, size_t sh, size_t sw, size_t dh, size_t dw,
                        size_t ic, size_t oc, size_t batch, size_t group,
                        size_t deformable_group) {
        size_t oh, ow;
        calc_output_shape(ih, iw, fh, fw, ph, pw, sh, sw, dh, dw, oh, ow);
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilate_h = dh;
        param.dilate_w = dw;
        param.format = DeformableConv::Param::Format::NCHW;
        param.mode = DeformableConv::Param::Mode::CROSS_CORRELATION;
        if (group > 1) {
            param.sparse = DeformableConv::Param::Sparse::GROUP;
            checker.set_param(param).execs(
                    {{batch, ic, ih, iw},
                     {group, oc / group, ic / group, fh, fw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow},
                     {batch, oc, oh, ow},
                     {batch, ic, ih, iw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow}});
        } else {
            param.sparse = DeformableConv::Param::Sparse::DENSE;
            checker.set_param(param).execs(
                    {{batch, ic, ih, iw},
                     {oc, ic, fh, fw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow},
                     {batch, oc, oh, ow},
                     {batch, ic, ih, iw},
                     {batch, 2 * deformable_group * fh * fw, oh, ow},
                     {batch, deformable_group * fh * fw, oh, ow}});
        }
    };

    for (auto batch : std::vector<int>{1, 3})
        for (auto hw : std::vector<int>{16, 20})
            for (auto fhw : std::vector<int>{3, 5, 7})
                for (auto phw : std::vector<int>{2, 5})
                    for (auto shw : std::vector<int>{1, 3})
                        for (auto g : std::vector<int>{1, 2})
                            for (auto icpg : std::vector<int>{1, 3})
                                for (auto ocpg : std::vector<int>{1, 3}) {
                                    auto dhw = shw;
                                    run_test(hw, hw, fhw, fhw, phw, phw, shw,
                                             shw, dhw, dhw, g * icpg, g * ocpg,
                                             batch, g, g);
                                }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_DEFORMABLE_CONV_FORWARD) {
    CUBenchmarker<DeformableConvForward> bencher(handle_cuda());
    bencher.set_display(true);

    Convolution::Param param;

    UniformFloatRNG im_rng{-10, 10};
    UniformFloatRNG filter_rng{-10, 10};
    UniformFloatRNG offset_rng{-10, 10};
    UniformFloatRNG mask_rng{-10, 10};
    UniformFloatRNG out_grad_rng{-10, 10};

    auto run_bench = [&](size_t batch, size_t ic, size_t oc, size_t ih,
                         size_t iw, size_t fh, size_t fw, size_t ph, size_t pw,
                         size_t sh, size_t sw, size_t dh, size_t dw,
                         size_t group, size_t deformable_group,
                         size_t nr_times) {
        size_t oh, ow;

        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilate_h = dh;
        param.dilate_w = dw;

        calc_output_shape(ih, iw, fh, fw, ph, pw, sh, sw, dh, dw, oh, ow);

        param.format = DeformableConv::Param::Format::NCHW;
        param.sparse = DeformableConv::Param::Sparse::DENSE;

        bencher.set_param(param)
                .set_rng(0, &im_rng)
                .set_rng(1, &im_rng)
                .set_rng(2, &offset_rng)
                .set_rng(3, &mask_rng);
        bencher.set_times(nr_times);

        TensorShape im{batch, ic, ih, iw}, filter{oc, ic, fh, fw},
                offset{batch, 2 * deformable_group * fh * fw, oh, ow},
                mask{batch, deformable_group * fh * fw, oh, ow};
        auto time_in_ms =
                bencher.execs({im, filter, offset, mask, {}}) / nr_times;
        auto ops = 2.0 * group * (oc / group) * (oh * ow * batch) *
                   (ic / group) * fh * fw / (time_in_ms * 1e-3) * 1e-12;
        printf("deformable conv forward performance: %fTops\n", ops);
    };
    run_bench(64, 64, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 100);
}

TEST_F(CUDA, BENCHMARK_DEFORMABLE_CONV_BWD_FILTER) {
    CUBenchmarker<DeformableConvBackwardFilter> bencher(handle_cuda());
    bencher.set_display(true);

    Convolution::Param param;

    UniformFloatRNG im_rng{-10, 10};
    UniformFloatRNG filter_rng{-10, 10};
    UniformFloatRNG offset_rng{-10, 10};
    UniformFloatRNG mask_rng{-10, 10};
    UniformFloatRNG out_grad_rng{-10, 10};

    auto run_bench = [&](size_t batch, size_t icpg, size_t ocpg, size_t ih,
                         size_t iw, size_t fh, size_t fw, size_t ph, size_t pw,
                         size_t sh, size_t sw, size_t dh, size_t dw,
                         size_t group, size_t deformable_group,
                         size_t nr_times) {
        size_t oh, ow;
        size_t ic = icpg * group, oc = ocpg * group;

        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilate_h = dh;
        param.dilate_w = dw;

        calc_output_shape(ih, iw, fh, fw, ph, pw, sh, sw, dh, dw, oh, ow);

        param.format = DeformableConv::Param::Format::NCHW;
        param.sparse = DeformableConv::Param::Sparse::DENSE;

        bencher.set_param(param)
                .set_rng(0, &im_rng)
                .set_rng(1, &im_rng)
                .set_rng(2, &offset_rng)
                .set_rng(3, &mask_rng);
        bencher.set_times(nr_times);

        TensorShape im{batch, ic, ih, iw}, filter{ic, ic, fh, fw},
                offset{batch, 2 * deformable_group * fh * fw, oh, ow},
                mask{batch, deformable_group * fh * fw, oh, ow},
                out_grad{batch, oc, oh, ow}, filter_grad{oc, ic, fh, fw};
        auto time_in_ms =
                bencher.execs({im, offset, mask, out_grad, filter_grad}) /
                nr_times;
        auto ops = 2.0 * group * (oc / group) * (oh * ow * batch) *
                   (ic / group) * fh * fw / (time_in_ms * 1e-3) * 1e-12;
        printf("deformable conv bwd filter performance: %fTops\n", ops);
    };
    run_bench(64, 64, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 100);
    //    run_bench(16, 64, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 100);
}

TEST_F(CUDA, BENCHMARK_DEFORMABLE_CONV_BWD_DATA) {
    CUBenchmarker<DeformableConvBackwardData> bencher(handle_cuda());
    bencher.set_display(true);

    Convolution::Param param;

    UniformFloatRNG im_rng{-10, 10};
    UniformFloatRNG filter_rng{-10, 10};
    UniformFloatRNG offset_rng{-10, 10};
    UniformFloatRNG mask_rng{-10, 10};
    UniformFloatRNG out_grad_rng{-10, 10};

    auto run_bench = [&](size_t batch, size_t ic, size_t oc, size_t ih,
                         size_t iw, size_t fh, size_t fw, size_t ph, size_t pw,
                         size_t sh, size_t sw, size_t dh, size_t dw,
                         size_t group, size_t deformable_group,
                         size_t nr_times) {

        size_t oh, ow;
        param.pad_h = ph;
        param.pad_w = pw;
        param.stride_h = sh;
        param.stride_w = sw;
        param.dilate_h = dh;
        param.dilate_w = dw;

        calc_output_shape(ih, iw, fh, fw, ph, pw, sh, sw, dh, dw, oh, ow);

        param.format = DeformableConv::Param::Format::NCHW;
        param.sparse = DeformableConv::Param::Sparse::DENSE;

        bencher.set_param(param)
                .set_rng(0, &im_rng)
                .set_rng(1, &im_rng)
                .set_rng(2, &offset_rng)
                .set_rng(3, &mask_rng);
        bencher.set_times(nr_times);

        TensorShape im{batch, ic, ih, iw}, filter{oc, ic, fh, fw},
                offset{batch, 2 * deformable_group * fh * fw, oh, ow},
                mask{batch, deformable_group * fh * fw, oh, ow},
                out_grad{batch, oc, oh, ow}, im_grad{batch, ic, ih, iw},
                offset_grad{batch, 2 * deformable_group * fh * fw, oh, ow},
                mask_grad{batch, deformable_group * fh * fw, oh, ow};
        auto time_in_ms = bencher.execs({im, filter, offset, mask, out_grad,
                                         im_grad, offset_grad, mask_grad}) /
                          nr_times;
        auto ops = 2.0 * group * (oc / group) * oh * ow * batch * (ic / group) *
                   fh * fw / (time_in_ms * 1e-3) * 1e-12;
        printf("deformable conv bwd data performance: %fTops\n", ops);
    };
    run_bench(64, 64, 256, 56, 56, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 100);
}
#endif
// vim: syntax=cpp.doxygen
