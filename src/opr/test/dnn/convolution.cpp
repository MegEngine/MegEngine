/**
 * \file src/opr/test/dnn/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node_env.h"
#include "./legacy_checker.h"

#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/opr/tensor_manip.h"
#include "megdnn/oprs/base.h"

#include <gmock/gmock.h>

#include <cmath>
#include <random>

using namespace mgb;

namespace {

using Param = opr::Convolution::Param;
using Param3D = opr::Convolution3D::Param;
using Mode = Param::Mode;

Mode modes_to_check[] = {Mode::CONVOLUTION, Mode::CROSS_CORRELATION};

void conv_bwd_flt_brute(const std::vector<std::shared_ptr<HostTensorND>>& inps,
                  std::shared_ptr<HostTensorND>& out,
                  const opr::ConvolutionBackwardFilter::Param& param) {
    auto &&src = *inps[0], &&diff = *inps[1], &&filter = *inps[2];
    size_t N = src.shape(0), IH = src.shape(2), IW = src.shape(3),
           OC = filter.shape(0), IC = filter.shape(1), FH = filter.shape(2),
           FW = filter.shape(3), OH = diff.shape(2), OW = diff.shape(3);
    out = std::make_shared<HostTensorND>(CompNode::load("xpu0"),
                                         TensorShape{OC, IC, FH, FW});
    auto&& grad = *out;
    auto sptr = src.ptr<float>(), dptr = diff.ptr<float>(),
         gptr = grad.ptr<float>();
    memset(gptr, 0, sizeof(float) * grad.shape().total_nr_elems());
    auto valid = [&](size_t ih, size_t iw) { return ih < IH && iw < IW; };
    for (size_t n = 0; n < N; ++n)
        for (size_t oc = 0; oc < OC; ++oc)
            for (size_t ic = 0; ic < IC; ++ic) {
                for (size_t oh = 0; oh < OH; ++oh)
                    for (size_t ow = 0; ow < OW; ++ow) {
                        for (size_t fh = 0; fh < FH; ++fh)
                            for (size_t fw = 0; fw < FW; ++fw) {
                                size_t ih = oh * param.stride_h + fh -
                                            param.pad_h,
                                       iw = ow * param.stride_w + fw -
                                            param.pad_w;
                                auto src_data =
                                        valid(ih, iw)
                                                ? sptr[(n * IC + ic) * IH * IW +
                                                       ih * IW + iw]
                                                : 0;
                                gptr[(oc * IC + ic) * FH * FW + fh * FW + fw] +=
                                        dptr[(n * OC + oc) * OH * OW + oh * OW +
                                             ow] *
                                        src_data;
                            }
                    }
            }
}

void local_share_brute(const std::vector<std::shared_ptr<HostTensorND>>& inps,
                       std::shared_ptr<HostTensorND>& out,
                       const opr::LocalShare::Param& param) {
    auto in = inps[0], filter = inps[1];
    mgb_assert(in->shape().ndim == 4);
    mgb_assert(filter->shape().ndim == 6);
    int batch_size = in->shape()[0], ci = in->shape()[1], hi = in->shape()[2],
        wi = in->shape()[3];
    int fh = filter->shape()[3], fw = filter->shape()[4];
    int ph = param.pad_h, pw = param.pad_w;
    int sh = param.stride_h, sw = param.stride_w;
    int dh = param.dilate_h, dw = param.dilate_w;
    int sgh = filter->shape()[0], sgw = filter->shape()[1];
    mgb_assert(dh == 1 && dw == 1);
    mgb_assert(static_cast<uint32_t>(sgh) == param.spatial_groups_h &&
               static_cast<uint32_t>(sgw) == param.spatial_groups_w);

    int ho = (hi + 2 * ph - fh) / sh + 1;
    int wo = (wi + 2 * pw - fw) / sw + 1;
    mgb_assert(ho % sgh == 0 && wo % sgw == 0);
    int grp_ho = ho / sgh, grp_wo = wo / sgw;
    int co = filter->shape()[5];

    size_t u_batch = batch_size, u_co = co, u_ho = ho, u_wo = wo;
    out = std::make_shared<HostTensorND>(
            CompNode::load("xpu0"), TensorShape{u_batch, u_co, u_ho, u_wo});
    mgb_assert(param.mode == Param::Mode::CROSS_CORRELATION);
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < co; ++oc) {
            for (int oh = 0; oh < ho; ++oh) {
                for (int ow = 0; ow < wo; ++ow) {
                    size_t u_n = n, u_oc = oc, u_oh = oh, u_ow = ow;
                    float& dval = out->ptr<float>({u_n, u_oc, u_oh, u_ow})[0];
                    dval = 0;
                    int grp_oh_idx = oh / grp_ho;
                    int grp_ow_idx = ow / grp_wo;
                    for (int ic = 0; ic < ci; ++ic) {
                        for (int kh = 0; kh < fh; ++kh) {
                            for (int kw = 0; kw < fw; ++kw) {
                                int ih = oh * sh - ph + kh;
                                int iw = ow * sw - pw + kw;
                                float sval = 0.f;
                                float fval = 0.f;
                                if (ih >= 0 && ih < hi && iw >= 0 && iw < wi) {
                                    sval = in->ptr<float>(
                                            {static_cast<size_t>(n),
                                             static_cast<size_t>(ic),
                                             static_cast<size_t>(ih),
                                             static_cast<size_t>(iw)})[0];
                                }
                                fval = filter->ptr<float>(
                                        {static_cast<size_t>(grp_oh_idx),
                                         static_cast<size_t>(grp_ow_idx),
                                         static_cast<size_t>(ic),
                                         static_cast<size_t>(kh),
                                         static_cast<size_t>(kw),
                                         static_cast<size_t>(oc)})[0];
                                dval += fval * sval;
                            }
                        }
                    }
                }
            }
        }
    }
}

void convolution_brute(const std::vector<std::shared_ptr<HostTensorND>> &in_tensor,
        std::shared_ptr<HostTensorND> &out_tensor,
        const opr::Convolution::Param &param)
{
    mgb_assert(in_tensor.size() == 2);
    auto in = in_tensor[0], filter = in_tensor[1];
    mgb_assert(in->shape().ndim == 4);
    mgb_assert(filter->shape().ndim == 4);

    int batch_size = in->shape().shape[0];
    int ic = in->shape().shape[1];
    int ih = in->shape().shape[2];
    int iw = in->shape().shape[3];

    int fh = filter->shape().shape[2];
    int fw = filter->shape().shape[3];

    int ph = param.pad_h;
    int pw = param.pad_w;

    int sh = param.stride_h;
    int sw = param.stride_w;

    int dh = param.dilate_h;
    int dw = param.dilate_w;

    mgb_assert(ih + 2*ph >= (fh - 1) * dh + 1);
    mgb_assert(iw + 2*pw >= (fw - 1) * dw + 1);
    int oh = (ih + 2*ph - ((fh - 1) * dh + 1)) / sh + 1;
    int ow = (iw + 2*pw - ((fw - 1) * dw + 1)) / sw + 1;
    mgb_assert(static_cast<size_t>(ic) == filter->shape().shape[1]);
    int oc = filter->shape().shape[0];

    out_tensor = std::make_shared<HostTensorND>(CompNode::load("xpu0"),
            TensorShape{
            static_cast<size_t>(batch_size),
            static_cast<size_t>(oc),
            static_cast<size_t>(oh),
            static_cast<size_t>(ow)});

    int pn, poc, poh, pow, pih, piw, pic, pfh, pfw;
    for (pn = 0; pn < batch_size; ++pn)
    for (poc = 0; poc < oc; ++poc)
    for (poh = 0, pih = -ph; poh < oh; ++poh, pih += sh)
    for (pow = 0, piw = -pw; pow < ow; ++pow, piw += sw)
    {
        float &target = out_tensor->ptr<float>({
                static_cast<size_t>(pn),
                static_cast<size_t>(poc),
                static_cast<size_t>(poh),
                static_cast<size_t>(pow)})[0];
        target = 0;
        for (pic = 0; pic < ic; ++pic)
        for (pfh = 0; pfh < fh; ++pfh)
        for (pfw = 0; pfw < fw; ++pfw)
        {
            int prih, priw;
            float img_data, filter_data;
            if (param.mode == Param::Mode::CONVOLUTION) {
                prih = pih + (fh - pfh - 1) * dh;
                priw = piw + (fw - pfw - 1) * dw;
            } else {
                mgb_assert(param.mode == Param::Mode::CROSS_CORRELATION);
                prih = pih + pfh * dh;
                priw = piw + pfw * dw;
            }
            if (prih >= 0 && prih < ih &&
                    priw >= 0 && priw < iw) {
                img_data = in_tensor[0]->ptr<float>({
                        static_cast<size_t>(pn),
                        static_cast<size_t>(pic),
                        static_cast<size_t>(prih),
                        static_cast<size_t>(priw)})[0];
            } else {
                img_data = 0;
            }
            filter_data = filter->ptr<float>({
                    static_cast<size_t>(poc),
                    static_cast<size_t>(pic),
                    static_cast<size_t>(pfh),
                    static_cast<size_t>(pfw)})[0];
            target += img_data * filter_data;
        }
    }
}



opr::Convolution::Param convert_to_conv_param(
        const opr::ConvBiasForward::Param& param) {
    return opr::Convolution::Param{
            param.mode,     param.pad_h,    param.pad_w,
            param.stride_h, param.stride_w, param.dilate_h,
            param.dilate_w, param.sparse,   param.format};
};
#if MGB_CUDA
opr::Convolution::Param convert_to_conv_param(
        const opr::BatchConvBiasForward::Param& param) {
    return opr::Convolution::Param{
            param.mode,     param.pad_h,    param.pad_w,
            param.stride_h, param.stride_w, param.dilate_h,
            param.dilate_w, param.sparse,   param.format};
};
#endif

TEST(TestOprDNN, ConvolutionForward) {
    uint32_t ih = 10, ic = 16, oc = 32, ph = 0, sh = 1, fh = 2;
    for (auto mode: modes_to_check) {
        uint32_t iw = ih + 1, fw = fh + 1, pw = ph + 1, sw = sh + 1;
        Param param{mode, ph, pw, sh, sw};
        size_t batch_size = 32;
        // !!! DEPRECATED. use AutoOprChecker instead.
        opr::test::ForwardChecker<opr::Convolution, 2> forward_checker({
                {batch_size, ic, ih, iw},
                {oc, ic, fh, fw}},
                convolution_brute, param);
        forward_checker.run();
    }
}

TEST(TestOprDNN, ConvolutionBackward) {
    uint32_t ih = 10, ic = 16, oc = 32, ph = 0, sh = 1, fh = 2;
    for (auto mode: modes_to_check) {
        uint32_t iw = 11, fw = 4, pw = 1, sw = 3;
        Param param{mode, ph, pw, sh, sw};
        size_t batch_size = 32;
        // !!! DEPRECATED. use AutoOprChecker instead.
        opr::test::BackwardChecker<opr::Convolution, 2> backward_checker({
                {batch_size, ic, ih, iw},
                {oc, ic, fh, fw}}, param, 1e-2, 1);
        backward_checker.run();
    }
}

TEST(TestOprDNN, ConvBiasExePolicy) {
    using Param = opr::ConvBias::Param;
    Param param;
    using Policy = opr::ConvBias::ExecutionPolicy;
    using S = Policy::Strategy;

    auto cn = CompNode::load("cpux");

#if MGB_ENABLE_FASTRUN
    for (auto strategy: {S::PROFILE, S::HEURISTIC, S::PROFILE_REPRODUCIBLE, S::PROFILE_HEURISTIC}) {
#else
    for (auto strategy: {S:HEURISTIC, S::PROFILE_HEURISTIC}) {
#endif

        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen;

        auto mkvar = [&](const char* name, const TensorShape& shp,
                         const DType& dtype) {
            return opr::TypeCvt::make(
                    opr::Host2DeviceCopy::make(*graph, gen(shp), cn).rename(name),
                    dtype);
        };

        auto x = mkvar("x", {20, 50, 50, 16}, dtype::QuantizedS8(2.5f));
        auto w = mkvar("w", {24, 3, 3, 16}, dtype::QuantizedS8(2.5f));
        auto bias = mkvar("bias", {1, 1, 1, 24}, dtype::QuantizedS32(6.25f));

        param.nonlineMode = Param::NonlineMode::RELU;
        param.format = Param::Format::NHWC;

        Policy policy;
        policy.strategy = strategy;

        auto conv_bias = opr::ConvBias::make(
                x, w, bias, param, policy,
                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(conv_bias, host_y)});
        func->execute();
    }
}

TEST(TestOprDNN, ConvBiasExePolicy_Quantized8Asym) {
    using Param = opr::ConvBias::Param;
    Param param;
    using Policy = opr::ConvBias::ExecutionPolicy;
    using S = Policy::Strategy;

    auto cn = CompNode::load("cpux");

    for (auto strategy: {S::PROFILE, S::PROFILE_REPRODUCIBLE}) {

        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen;

        auto mkvar = [&](const char* name, const TensorShape& shp,
                         const DType& dtype) {
            return opr::TypeCvt::make(
                    opr::Host2DeviceCopy::make(*graph, gen(shp), cn).rename(name),
                    dtype);
        };

        auto x = mkvar("x", {20, 50, 50, 16}, dtype::Quantized8Asymm(2.5f, static_cast<uint8_t>(0)));
        auto w = mkvar("w", {24, 3, 3, 16}, dtype::Quantized8Asymm(2.5f, static_cast<uint8_t>(0)));
        auto bias = mkvar("bias", {1, 1, 1, 24}, dtype::QuantizedS32(6.25f));

        param.nonlineMode = Param::NonlineMode::RELU;
        param.format = Param::Format::NHWC;

        Policy policy;
        policy.strategy = strategy;

        auto conv_bias = opr::ConvBias::make(
                x, w, bias, param, policy,
                OperatorNodeConfig{dtype::Quantized8Asymm(2.5f, static_cast<uint8_t>(0))});
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(conv_bias, host_y)});
        func->execute();
    }
}

TEST(TestOprDNN, ConvolutionExePolicy) {
    Param param{Mode::CONVOLUTION};
    using Policy = opr::Convolution::ExecutionPolicy;
    using S = Policy::Strategy;

    int nr_get = 0;
    auto on_get = [&nr_get](const std::string&, const void*, size_t,
                            const void*, size_t) { ++nr_get; };
    PersistentCacheHook cache_hook{on_get};

#if MGB_ENABLE_FASTRUN
    for (auto strategy: {S::PROFILE, S::HEURISTIC, S::PROFILE_REPRODUCIBLE, S::PROFILE_HEURISTIC}) {
#else
    for (auto strategy: {S:HEURISTIC, S::PROFILE_HEURISTIC}) {
#endif
        using Checker = AutoOprChecker<2, 1>;

        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            Policy policy;
            policy.strategy = strategy;
            auto out =
                    opr::Convolution::make(inputs[0], inputs[1], param, policy);
            return {out};
        };

        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            std::shared_ptr<HostTensorND> sh_out;
            convolution_brute({inp.begin(), inp.end()}, sh_out, param);
            dest[0] = *sh_out;
        };

        Checker::RunOptions opt;
        opt.numdiff_eps = 1;
        nr_get = 0;
        Checker(make_graph, fwd)
                .run({TensorShape{3, 2, 10, 6}, {4, 2, 3, 2}}, opt)
                .run({TensorShape{6, 3, 8, 13}, {2, 3, 2, 13}}, opt)
                .run({TensorShape{1, 1, 10, 10}, {2, 1, 3, 3}}, opt);
        if (strategy == S::HEURISTIC) {
            ASSERT_EQ(0, nr_get);
        } else {
            ASSERT_LT(0, nr_get);
        }
    }
}

TEST(TestOprDNN, Deconvolution) {
    // dilated grouped deconv
    using Checker = AutoOprChecker<2, 1>;

    Param param{Mode::CROSS_CORRELATION, 0, 1, 1, 2};
    param.dilate_h = 2;
    param.sparse = Param::Sparse::GROUP;
    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        return {opr::ConvolutionBackwardData::make_deconv(
                inputs[0], inputs[1], param)};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&filter = *inp[1];
        size_t N = data.shape(0), IH = data.shape(2), IW = data.shape(3);
        size_t GROUP = filter.shape(0), ICPG = filter.shape(1),
               OCPG = filter.shape(2), FH = filter.shape(3),
               FW = filter.shape(4);
        auto get_shp = [](size_t inp, size_t filter, size_t stride, size_t pad,
                          size_t dilate) {
            return (inp - 1) * stride + (filter - 1) * dilate + 1 - pad * 2;
        };
        auto &&out = dest[0];
        size_t OH = get_shp(IH, FH, param.stride_h, param.pad_h,
                            param.dilate_h),
               OW = get_shp(IW, FW, param.stride_w, param.pad_w,
                            param.dilate_w);
        out.resize({N, OCPG * GROUP, OH, OW});
        auto fptr = filter.ptr<float>(), dptr = data.ptr<float>(),
             optr = out.ptr<float>();
        memset(optr, 0, sizeof(float) * out.shape().total_nr_elems());
        auto ol = out.layout(), fl = filter.layout();

#define FOR2(a, A, b, B)           \
    for (size_t a = 0; a < A; ++a) \
        for (size_t b = 0; b < B; ++b)
#define FOR3(a, A, b, B, c, C) \
    FOR2(a, A, b, B)           \
    for (size_t c = 0; c < C; ++c)

        FOR3(n, N, group, GROUP, icg, ICPG)
        FOR2(ih, IH, iw, IW) {
            float scale = *(dptr++);

            FOR3(ocg, OCPG, fh, FH, fw, FW) {
                auto oc_tot = group * OCPG + ocg;
                int oh = int(ih * param.stride_h + fh * param.dilate_h) -
                         int(param.pad_h),
                    ow = int(iw * param.stride_w + fw * param.dilate_w) -
                         int(param.pad_w);
                if (oh >= 0 && ow >= 0 && oh < static_cast<int>(OH) &&
                    ow < static_cast<int>(OW)) {
                    auto out_off = n * ol.stride[0] + oc_tot * ol.stride[1] +
                                   oh * ol.stride[2] + ow,
                         flt_off = group * fl.stride[0] + icg * fl.stride[1] +
                                   ocg * fl.stride[2] + fh * fl.stride[3] + fw;
                    optr[out_off] += scale * fptr[flt_off];
                }
            }
        }
#undef FOR3
#undef FOR2
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    Checker(make_graph, fwd).
        run({TensorShape{2, 4, 6, 8}, {1, 4, 5, 3, 2}}, opt).
        run({TensorShape{3, 2, 1, 1}, {2, 1, 1, 4, 3}}, opt).
        run({TensorShape{4, 6, 7, 2}, {2, 3, 4, 8, 13}}, opt);
}

TEST(TestOprDNN, ConvolutionBackwardFilter) {
    using Checker = AutoOprChecker<3, 1>;

    constexpr size_t PH = 0, PW = 1, SH = 1, SW = 2;

    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        Param param{Mode::CROSS_CORRELATION, PH, PW, SH, SW};
        return {opr::ConvolutionBackwardFilter::make(
                inputs[0], inputs[1], inputs[2], param)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        std::shared_ptr<HostTensorND> out;
        conv_bwd_flt_brute({inp[0], inp[1], inp[2]}, out,
                           Param{Mode::CROSS_CORRELATION, PH, PW, SH, SW});
        dest[0] = *out;
    };

#define get_shp(N, P, S, F) ((N + 2 * P - F) / S + 1)
#define inp_tensor(N, IC, OC, IH, IW, FH, FW) \
    { TensorShape{N, IC, IH, IW}, \
      {N, OC, get_shp(IH, PH, SH, FH), get_shp(IW, PW, SW, FW)}, \
      {OC, IC, FH, FW} }
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    Checker(make_graph, fwd).
        run(inp_tensor(2, 3, 4, 9, 8, 4, 3), opt).
        run(inp_tensor(1, 5, 3, 7, 9, 3, 4), opt).
        run(inp_tensor(3, 4, 4, 9, 9, 3, 3), opt);
#undef inp_tensor
#undef get_shp
}

TEST(TestOprDNN, DilatedConvolution) {
    using Checker = AutoOprChecker<2, 1>;

    opr::ConvolutionForward::Param param;
    param.pad_h = 5;
    param.pad_w = 2;
    param.stride_w = 2;
    param.dilate_h = 2;

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
        return {opr::Convolution::make(inputs[0], inputs[1], param)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()->create_operator<
            megdnn::Convolution>();
        opr->param() = param;
        TensorLayout dest_layout;
        opr->deduce_layout(inp[0]->layout(), inp[1]->layout(), dest_layout);
        std::vector<dt_byte> workspace(opr->get_workspace_in_bytes(
                    inp[0]->layout(), inp[1]->layout(), dest_layout, nullptr));
        dest[0].dtype(dtype::Float32()).
            comp_node(inp[0]->comp_node()).resize(dest_layout);
        opr->exec(inp[0]->as_megdnn(), inp[1]->as_megdnn(), dest[0].as_megdnn(),
                  nullptr, {workspace.data(), workspace.size()});
    };
    Checker::RunOptions option;
    option.numdiff_eps = 0.1;

    Checker(make_graph, fwd).
        run({TensorShape{2, 3, 8, 7}, TensorShape{4, 3, 2, 2}}, option).
        run({TensorShape{2, 3, 8, 7}, TensorShape{4, 3, 3, 2}}, option).
        run({TensorShape{2, 3, 8, 9}, TensorShape{4, 3, 3, 2}}, option);
}

TEST(TestOprDNN, GroupConv) {
    using Checker = AutoOprChecker<2, 1>;
    opr::Convolution::Param param;
    param.pad_h = 1;
    param.pad_w = 2;
    param.stride_h = 2;
    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        auto p1 = param;
        p1.sparse = opr::Convolution::Param::Sparse::GROUP;
        return {opr::Convolution::make(inputs[0], inputs[1], p1)};
    };

    auto cn = CompNode::load("xpux");
    auto inp0 = std::make_shared<HostTensorND>(cn, dtype::Float32()),
         inp1 = std::make_shared<HostTensorND>(cn, dtype::Float32());
    HostTensorND out_raw;
    auto graph_raw = ComputingGraph::make();
    auto func_raw = graph_raw->compile({
            make_callback_copy(
                    opr::Convolution::make(
                        opr::Host2DeviceCopy::make(*graph_raw, inp0),
                        opr::Host2DeviceCopy::make(*graph_raw, inp1),
                        param),
                    out_raw)});
    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto &&out = dest[0];
        auto sl = inp[0]->layout(),
             fl = inp[1]->layout().remove_axis(0);
        TensorLayout ol;
        auto group = inp[1]->layout()[0];
        sl.shape[1] /= group;
        for (size_t i = 0; i < group; ++ i) {
            inp0->copy_from(inp[0]->sub(SubTensorSpec::make_from_offset_elem(
                            sl, i * sl[1] * sl[2] * sl[3])));
            inp1->copy_from(inp[1]->sub(SubTensorSpec::make_from_offset_elem(
                            fl, i * fl.total_nr_elems())));
            func_raw->execute();
            if (!i) {
                auto oshp = out_raw.shape();
                oshp[1] *= group;
                out.resize(oshp);
                ol = out.layout();
                ol[1] /= group;
            }
            out.sub(SubTensorSpec::make_from_offset_elem(
                        ol, i * ol[1] * ol[2] * ol[3])).copy_from_fixlayout(
                        out_raw);
        }
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    opt.outputs_max_err = 5e-5;
    Checker checker{make_graph, fwd};
    auto run = [&](const TensorShape &ishp,
            size_t fh, size_t fw, size_t oc, size_t group) {
        size_t ic = ishp[1];
        TensorShape flt{group, oc/group, ic/group, fh, fw};
        checker.run({ishp, flt}, opt);
    };
    run({1, 2, 1, 1}, 1, 1, 2, 2);
    run({3, 9, 5, 4}, 1, 2, 6, 3);
    run({3, 6, 8, 9}, 3, 1, 4, 2);
    run({2, 5, 3, 6}, 2, 3, 5, 1);
    run({2, 6, 3, 6}, 2, 3, 6, 6);
}

TEST(TestOprDNN, MaskConvolution) {
    using Checker = AutoOprChecker<3, 1>;
    opr::Convolution::Param param;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::MaskConvolution::make(inputs[0], inputs[1], inputs[2],
                                           param)};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        std::shared_ptr<HostTensorND> sh_out;
        convolution_brute({inp[0], inp[1]}, sh_out, param);
        dest[0] = *sh_out;
        size_t N = dest[0].shape()[0];
        size_t OC = dest[0].shape()[1];
        size_t OH = dest[0].shape()[2];
        size_t OW = dest[0].shape()[3];
        auto mask_ptr = inp[2]->ptr<int8_t>();
        auto dest_ptr = dest[0].ptr<float>();
        for (size_t i = 0; i < N * OC; ++i) {
            for (size_t mask_idx = 0; mask_idx < OH * OW; ++mask_idx) {
                if (mask_ptr[mask_idx] == 0) {
                    dest_ptr[i * OH * OW + mask_idx] = 0;
                }
            }
        }
    };

    auto gen_mask = [](HostTensorND& dest) {
        HostTensorGenerator<dtype::Int8, RandomDistribution::UNIFORM>
                mask_generator{0, 1};
        dest = *mask_generator(dest.shape(), dest.comp_node());
    };

    auto run_with_param = [&](size_t SH = 1, size_t SW = 1, size_t PH = 0,
                              size_t PW = 0) {
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        checker.set_output_allow_grad(0, false);
        checker.set_input_dtype(2, dtype::Int8());
        checker.set_input_generator(2, gen_mask);
        auto run = [&](size_t N, size_t IC, size_t OC, size_t IH, size_t IW,
                       size_t FH, size_t FW) {
            size_t OH = (IH + 2 * PH - FH) / SH + 1;
            size_t OW = (IW + 2 * PW - FW) / SW + 1;
            checker.run(
                    {TensorShape{N, IC, IH, IW}, {OC, IC, FH, FW}, {OH, OW}},
                    opt);
        };
        run(1, 1, 1, 5, 5, 3, 3);
        run(2, 3, 4, 5, 5, 3, 3);
        run(3, 3, 4, 224, 223, 3, 3);
        run(3, 3, 4, 224, 223, 2, 2);
    };

    run_with_param();
    run_with_param(2, 2, 3, 3);
    run_with_param(3, 2, 1, 2);
    run_with_param(2, 3, 2, 2);
}

TEST(TestOprDNN, MaskPropagate) {
    using Checker = AutoOprChecker<3, 1>;
    opr::MaskPropagate::Param mask_param;
    opr::Convolution::Param conv_param;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto inp_mask = inputs[2];
        auto out_mask = opr::MaskPropagate::make(inp_mask, mask_param);
        return {opr::MaskConvolution::make(inputs[0], inputs[1], out_mask,
                                           conv_param)};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto& src = *inp[0];
        auto& mask = *inp[2];
        auto src_ptr = inp[0]->ptr<float>();
        auto mask_ptr = inp[2]->ptr<int>();
        mgb_assert(src.shape()[2] == mask.shape()[0] &&
                   src.shape()[3] == mask.shape()[1]);
        for (size_t i = 0; i < src.shape()[0] * src.shape()[1]; ++i) {
            for (size_t mask_idx = 0;
                 mask_idx < src.shape()[2] * src.shape()[3]; ++mask_idx) {
                if (mask_ptr[mask_idx] == 0) {
                    src_ptr[i * src.layout().stride[1] + mask_idx] = 0;
                }
            }
        }
        std::shared_ptr<HostTensorND> sh_out;
        convolution_brute({inp[0], inp[1]}, sh_out, conv_param);
        dest[0] = *sh_out;
    };

    auto gen_mask = [](HostTensorND& dest) {
        HostTensorGenerator<dtype::Int32, RandomDistribution::UNIFORM>
                mask_generator{0, 1};
        dest = *mask_generator(dest.shape(), dest.comp_node());
    };

    auto run_with_param = [&](size_t FH, size_t FW, size_t SH = 1,
                              size_t SW = 1, size_t PH = 0, size_t PW = 0,
                              size_t DH = 1, size_t DW = 1) {
        conv_param.pad_h = PH;
        conv_param.pad_w = PW;
        conv_param.stride_h = SH;
        conv_param.stride_w = SW;
        conv_param.dilate_h = DH;
        conv_param.dilate_w = DW;

        mask_param.pad_h = PH;
        mask_param.pad_w = PW;
        mask_param.stride_h = SH;
        mask_param.stride_w = SW;
        mask_param.kernel_h = FH;
        mask_param.kernel_w = FW;
        mask_param.dilate_h = DH;
        mask_param.dilate_w = DW;

        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        checker.set_output_allow_grad(0, false);
        checker.set_input_dtype(2, dtype::Int32());
        checker.set_input_generator(2, gen_mask);
        auto run = [&](size_t N, size_t IC, size_t OC, size_t IH, size_t IW) {
            checker.run(
                    {TensorShape{N, IC, IH, IW}, {OC, IC, FH, FW}, {IH, IW}},
                    opt);
        };
        run(1, 1, 1, 5, 5);
        run(2, 3, 4, 5, 5);
        run(3, 3, 4, 224, 223);
        run(3, 3, 4, 224, 223);
    };

    run_with_param(3, 3, 1, 1, 0, 0, 2, 2);
    run_with_param(3, 3, 2, 2, 3, 3);
    run_with_param(4, 2, 3, 2, 1, 2);
    run_with_param(2, 4, 2, 3, 2, 2);
    run_with_param(4, 2, 3, 2, 1, 2, 2, 2);
    run_with_param(2, 4, 2, 3, 2, 2, 2, 1);
}
void convolution3d_brute(const std::vector<std::shared_ptr<HostTensorND>> &in_tensor,
        std::shared_ptr<HostTensorND> &out_tensor,
        const opr::Convolution3D::Param &param)
{
    mgb_assert(in_tensor.size() == 2);
    auto in = in_tensor[0], filter = in_tensor[1];
    mgb_assert(in->shape().ndim == 5);
    mgb_assert(filter->shape().ndim == 5);

    int batch_size = in->shape().shape[0];
    int ic = in->shape().shape[1];
    int id = in->shape().shape[2];
    int ih = in->shape().shape[3];
    int iw = in->shape().shape[4];

    int fd = filter->shape().shape[2];
    int fh = filter->shape().shape[3];
    int fw = filter->shape().shape[4];

    int pd = param.pad_d;
    int ph = param.pad_h;
    int pw = param.pad_w;

    int sd = param.stride_d;
    int sh = param.stride_h;
    int sw = param.stride_w;

    int dd = param.dilate_d;
    int dh = param.dilate_h;
    int dw = param.dilate_w;

    mgb_assert(id + 2*pd >= (fd - 1) * dd + 1);
    mgb_assert(ih + 2*ph >= (fh - 1) * dh + 1);
    mgb_assert(iw + 2*pw >= (fw - 1) * dw + 1);
    int od = (id + 2*pd - ((fd - 1) * dd + 1)) / sd + 1;
    int oh = (ih + 2*ph - ((fh - 1) * dh + 1)) / sh + 1;
    int ow = (iw + 2*pw - ((fw - 1) * dw + 1)) / sw + 1;
    mgb_assert(static_cast<size_t>(ic) == filter->shape().shape[1]);
    int oc = filter->shape().shape[0];

    out_tensor = std::make_shared<HostTensorND>(CompNode::load("xpu0"),
            TensorShape{
            static_cast<size_t>(batch_size),
            static_cast<size_t>(oc),
            static_cast<size_t>(od),
            static_cast<size_t>(oh),
            static_cast<size_t>(ow)});

    int pn, poc, pod, poh, pow,
            pic, pid, pih, piw,
                 pfd, pfh, pfw;
    for (pn = 0; pn < batch_size; ++pn)
    for (poc = 0; poc < oc; ++poc)
    for (pod = 0, pid = -pd; pod < od; ++pod, pid += sd)
    for (poh = 0, pih = -ph; poh < oh; ++poh, pih += sh)
    for (pow = 0, piw = -pw; pow < ow; ++pow, piw += sw)
    {
        float &target = out_tensor->ptr<float>({
                static_cast<size_t>(pn),
                static_cast<size_t>(poc),
                static_cast<size_t>(pod),
                static_cast<size_t>(poh),
                static_cast<size_t>(pow)})[0];
        target = 0;
        for (pic = 0; pic < ic; ++pic)
        for (pfd = 0; pfd < fd; ++pfd)
        for (pfh = 0; pfh < fh; ++pfh)
        for (pfw = 0; pfw < fw; ++pfw)
        {
            int prid, prih, priw;
            float img_data, filter_data;
            if (param.mode == opr::Convolution3D::Param::Mode::CONVOLUTION) {
                prid = pid + (fd - pfd - 1) * dd;
                prih = pih + (fh - pfh - 1) * dh;
                priw = piw + (fw - pfw - 1) * dw;
            } else {
                mgb_assert(param.mode == opr::Convolution3D::Param::Mode::CROSS_CORRELATION);
                prid = pid + pfd * dd;
                prih = pih + pfh * dh;
                priw = piw + pfw * dw;
            }
            if (prid >= 0 && prid < id &&
                prih >= 0 && prih < ih &&
                priw >= 0 && priw < iw) {
                img_data = in_tensor[0]->ptr<float>({
                        static_cast<size_t>(pn),
                        static_cast<size_t>(pic),
                        static_cast<size_t>(prid),
                        static_cast<size_t>(prih),
                        static_cast<size_t>(priw)})[0];
            } else {
                img_data = 0;
            }
            filter_data = filter->ptr<float>({
                    static_cast<size_t>(poc),
                    static_cast<size_t>(pic),
                    static_cast<size_t>(pfd),
                    static_cast<size_t>(pfh),
                    static_cast<size_t>(pfw)})[0];
            target += img_data * filter_data;
        }
    }
}
TEST(TestOprDNN, Convolution3DForward) {
    for (uint32_t batch_size : {8})
    for (uint32_t id : {12})
    for (uint32_t fd : {1, 3})
    for (uint32_t ic : {4})
    for (uint32_t oc : {ic})
    for (uint32_t pd : {0, 2})
    for (uint32_t sd : {1, 3})
    for (uint32_t dd : {1, 3})
    for (bool xcorr : {0, 1}) {
        uint32_t ih = id + 1, fh = fd, ph = pd + 1, sh = sd + 1;
        uint32_t iw = ih + 1, fw = fh, pw = ph + 1, sw = sh + 1;
        Param3D param{xcorr ? Param3D::Mode::CROSS_CORRELATION :
            Param3D::Mode::CONVOLUTION , pd, ph, pw,
                sd, sh, sw, dd, dd, dd};
        // !!! DEPRECATED. use AutoOprChecker instead.
        opr::test::ForwardChecker<opr::Convolution3D, 2> forward_checker({
                {batch_size, ic, id, ih, iw},
                {oc, ic, fd, fh, fw}},
                convolution3d_brute, param);
        forward_checker.run();
    }
}

TEST(TestOprDNN, Convolution3DBackward) {
    for (uint32_t batch_size : {8})
    for (uint32_t id : {12})
    for (uint32_t fd : {1, 3})
    for (uint32_t ic : {4})
    for (uint32_t oc : {ic})
    for (uint32_t pd : {0, 2})
    for (uint32_t sd : {1, 3})
    for (uint32_t dd : {1, 3})
    for (bool xcorr : {0, 1}) {
        uint32_t ih = id + 1, fh = fd, ph = pd + 1, sh = sd + 1;
        uint32_t iw = ih + 1, fw = fh, pw = ph + 1, sw = sh + 1;
        Param3D param{xcorr ? Param3D::Mode::CROSS_CORRELATION :
            Param3D::Mode::CONVOLUTION,
                pd, ph, pw, sd, sh, sw, dd, dd, dd};
        // !!! DEPRECATED. use AutoOprChecker instead.
        opr::test::BackwardChecker<opr::Convolution3D, 2> backward_checker(
                {{batch_size, ic, id, ih, iw},
                {oc, ic, fd, fh, fw}}, param, 1e-2, 1);
        backward_checker.run();
    }
}

TEST(TestOprDNN, GroupConv3D) {
    using Checker = AutoOprChecker<2, 1>;
    opr::Convolution3D::Param param;
    param.pad_d = 0;
    param.pad_h = 1;
    param.pad_w = 0;
    param.stride_d = 1;
    param.stride_h = 2;
    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        auto p1 = param;
        p1.sparse = opr::Convolution3D::Param::Sparse::GROUP;
        return {opr::Convolution3D::make(inputs[0], inputs[1], p1)};
    };

    auto cn = CompNode::load("xpux");
    auto inp0 = std::make_shared<HostTensorND>(cn, dtype::Float32()),
         inp1 = std::make_shared<HostTensorND>(cn, dtype::Float32());
    HostTensorND out_raw;
    auto graph_raw = ComputingGraph::make();
    auto func_raw = graph_raw->compile({
            make_callback_copy(
                    opr::Convolution3D::make(
                        opr::Host2DeviceCopy::make(*graph_raw, inp0),
                        opr::Host2DeviceCopy::make(*graph_raw, inp1),
                        param),
                    out_raw)});
    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto &&out = dest[0];
        auto sl = inp[0]->layout(),
             fl = inp[1]->layout().remove_axis(0);
        TensorLayout ol;
        auto group = inp[1]->layout()[0];
        sl.shape[1] /= group;
        for (size_t i = 0; i < group; ++ i) {
            inp0->copy_from(inp[0]->sub(SubTensorSpec::make_from_offset_elem(
                            sl, i * sl[1] * sl[2] * sl[3] * sl[4])));
            inp1->copy_from(inp[1]->sub(SubTensorSpec::make_from_offset_elem(
                            fl, i * fl.total_nr_elems())));
            func_raw->execute();
            if (!i) {
                auto oshp = out_raw.shape();
                oshp[1] *= group;
                out.resize(oshp);
                ol = out.layout();
                ol[1] /= group;
            }
            out.sub(SubTensorSpec::make_from_offset_elem(
                ol, i * ol[1] * ol[2] * ol[3] * ol[4])).
                copy_from_fixlayout(out_raw);
        }
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    opt.outputs_max_err = 5e-5;
    Checker checker{make_graph, fwd};
    auto run = [&](const TensorShape &ishp,
            size_t fd, size_t fh, size_t fw, size_t oc, size_t group) {
        size_t ic = ishp[1];
        TensorShape flt{group, oc/group, ic/group, fd, fh, fw};
        checker.
            run({ishp, flt}, opt);
    };
    run({1, 2, 1, 1, 1}, 1, 1, 1, 2, 2);
    run({3, 9, 5, 4, 3}, 1, 2, 3, 6, 3);
    run({2, 1, 3, 6, 9}, 2, 3, 3, 5, 1);
    run({2, 1, 3, 6, 9}, 2, 3, 3, 5, 1);
}

TEST(TestOprDNN, Deconvolution3D) {
    using Checker = AutoOprChecker<2, 1>;
    Param3D param{Param3D::Mode::CROSS_CORRELATION, 0, 1, 1, 1, 2, 2};
    param.sparse = Param3D::Sparse::GROUP;
    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        return {opr::Convolution3DBackwardData::make_deconv(
                inputs[0], inputs[1], param)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto &&data = *inp[0], &&filter = *inp[1];
        size_t N = data.shape(0),
               ID = data.shape(2), IH = data.shape(3), IW = data.shape(4),
               GROUP = filter.shape(0),
               ICPG = filter.shape(1), OCPG = filter.shape(2),
               FD = filter.shape(3), FH = filter.shape(4), FW = filter.shape(5);
        auto &&out = dest[0];
        auto get_shp = [](
                size_t inp, size_t filter, size_t stride, size_t pad,
                size_t dilate) {
            return (inp - 1) * stride + (filter - 1) * dilate + 1 - pad * 2;
        };
        size_t OD = get_shp(ID, FD,
                        param.stride_d, param.pad_d, param.dilate_d),
               OH = get_shp(IH, FH,
                        param.stride_h, param.pad_h, param.dilate_h),
               OW = get_shp(IW, FW,
                        param.stride_w, param.pad_w, param.dilate_w);
        out.resize({N, OCPG * GROUP, OD, OH, OW});
        auto fptr = filter.ptr<float>(),
             dptr = data.ptr<float>(),
             optr = out.ptr<float>();
        memset(optr, 0, sizeof(float) * out.shape().total_nr_elems());
        auto ol = out.layout(), fl = filter.layout();
#define FOR2(a, A, b, B) \
        for (size_t a = 0; a < A; ++ a) \
        for (size_t b = 0; b < B; ++ b)
#define FOR3(a, A, b, B, c, C) \
        FOR2(a, A, b, B) \
        for (size_t c = 0; c < C; ++ c)
#define FOR4(a, A, b, B, c, C, d, D) \
        FOR3(a, A, b, B, c, C) \
        for (size_t d = 0; d < D; ++ d)
        FOR3(n, N, group, GROUP, icg, ICPG)
        FOR3(id, ID, ih, IH, iw, IW) {
            float scale = *(dptr ++);
            FOR4(ocg, OCPG, fd, FD, fh, FH, fw, FW) {
                auto oc_tot = group * OCPG + ocg;
                int od = int(id * param.stride_d +
                            fd * param.dilate_d) - int(param.pad_d),
                    oh = int(ih * param.stride_h +
                            fh * param.dilate_h) - int(param.pad_h),
                    ow = int(iw * param.stride_w +
                            fw * param.dilate_w) - int(param.pad_w);
                if (od >= 0 && oh >= 0 && ow >= 0 &&
                        od < static_cast<int>(OD) &&
                        oh < static_cast<int>(OH) &&
                        ow < static_cast<int>(OW)) {
                    auto out_off = n * ol.stride[0] + oc_tot * ol.stride[1] +
                                   od * ol.stride[2] + oh * ol.stride[3] + ow,
                         flt_off = group * fl.stride[0] + icg * fl.stride[1] +
                             ocg * fl.stride[2] + fd * fl.stride[3] +
                             fh * fl.stride[4] + fw;
                    optr[out_off] += scale * fptr[flt_off];
                }
            }
        }
#undef FOR4
#undef FOR3
#undef FOR2
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    Checker(make_graph, fwd).
        run({TensorShape{2, 4, 3, 3, 2}, {1, 4, 5, 3, 2, 2}}, opt).
        run({TensorShape{3, 2, 1, 1, 1}, {2, 1, 1, 4, 3, 3}}, opt).
        run({TensorShape{4, 6, 2, 2, 2}, {2, 3, 4, 6, 5, 4}}, opt);
}

TEST(TestOprDNN, Convolution3DExePolicy) {
    Param3D param{Param3D::Mode::CONVOLUTION};
    using Policy = opr::Convolution3D::ExecutionPolicy;
    using S = Policy::Strategy;

#if MGB_ENABLE_FASTRUN
    for (auto strategy: {S::PROFILE, S::HEURISTIC, S::PROFILE_REPRODUCIBLE, S::PROFILE_HEURISTIC}) {
#else
    for (auto strategy: {S:HEURISTIC, S::PROFILE_HEURISTIC}) {
#endif

        using Checker = AutoOprChecker<2, 1>;

        auto make_graph = [&](const Checker::SymInpArray &inputs) ->
                Checker::SymOutArray {
            Policy policy;
            policy.strategy = strategy;
            auto out = opr::Convolution3D::make(
                    inputs[0], inputs[1], param, policy);
            return {out};
        };

        auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
            std::shared_ptr<HostTensorND> sh_out;
            convolution3d_brute({inp.begin(), inp.end()}, sh_out, param);
            dest[0] = *sh_out;
        };

        Checker::RunOptions opt;
        opt.numdiff_eps = 1;
        Checker(make_graph, fwd).
            run({TensorShape{3, 2, 3, 4, 1}, {4, 2, 2, 2, 1}}, opt).
            run({TensorShape{3, 3, 2, 6, 2}, {2, 3, 1, 4, 1}}, opt).
            run({TensorShape{1, 1, 4, 4, 4}, {2, 1, 3, 3, 3}}, opt);
    }
}

TEST(TestOprDNN, ConvBiasForward) {
    using Checker2 = AutoOprChecker<2, 1>;
    using Checker3 = AutoOprChecker<3, 1>;
    opr::ConvBiasForward::Param param;
    auto make_graph2 =
            [&](const Checker2::SymInpArray& inputs) -> Checker2::SymOutArray {
        return {opr::ConvBiasForward::make(inputs[0], inputs[1], param)};
    };

    auto make_graph3 =
            [&](const Checker3::SymInpArray& inputs) -> Checker3::SymOutArray {
        return {opr::ConvBiasForward::make(inputs[0], inputs[1], inputs[2],
                                           param)};
    };

    auto fwd2 = [&](Checker2::NumOutArray& dest, Checker2::NumInpArray inp) {
        std::shared_ptr<HostTensorND> sh_out;
        convolution_brute({inp[0], inp[1]}, sh_out,
                          convert_to_conv_param(param));
        dest[0] = *sh_out;
    };


    auto fwd3 = [&](Checker3::NumOutArray& dest, Checker3::NumInpArray inp) {
        std::shared_ptr<HostTensorND> sh_out;
        convolution_brute({inp[0], inp[1]}, sh_out,
                          convert_to_conv_param(param));
        dest[0] = *sh_out;
        size_t N = dest[0].shape()[0];
        size_t OC = dest[0].shape()[1];
        size_t OH = dest[0].shape()[2];
        size_t OW = dest[0].shape()[3];
        auto dest_ptr = dest[0].ptr<float>();
        for (size_t i = 0; i < N; i++) {
            auto bias_ptr = inp[2]->ptr<float>();
            for (size_t c = 0; c < OC; c++) {
                for (size_t hw = 0; hw < OH * OW; hw++) {
                    *(dest_ptr++) += *(bias_ptr);
                }
                bias_ptr++;
            }
        }
    };

    auto run_with_param = [&](size_t SH = 1, size_t SW = 1, size_t PH = 0,
                              size_t PW = 0) {
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        Checker2 checker2{make_graph2, fwd2};
        Checker2::RunOptions opt2;
        checker2.set_output_allow_grad(0, false);
        Checker3 checker3{make_graph3, fwd3};
        Checker3::RunOptions opt3;
        checker3.set_output_allow_grad(0, false);

        auto run = [&](size_t N, size_t IC, size_t OC, size_t IH, size_t IW,
                       size_t FH, size_t FW) {
            auto opr = megdnn_naive_handle()
                               ->create_operator<megdnn::ConvolutionForward>();
            opr->param() = convert_to_conv_param(param);
            TensorLayout dest_layout;
            opr->deduce_layout({{N, IC, IH, IW}, dtype::Float32()},
                               {{OC, IC, FH, FW}, dtype::Float32()},
                               dest_layout);
            checker2.run({TensorShape{N, IC, IH, IW}, {OC, IC, FH, FW}}, opt2);

            checker3.run({TensorShape{N, IC, IH, IW},
                          {OC, IC, FH, FW},
                          {1, OC, 1, 1}},
                         opt3);
        };
        run(1, 1, 1, 5, 5, 1, 1);
        run(1, 1, 1, 5, 5, 3, 3);
        run(2, 3, 4, 5, 5, 3, 3);
        run(3, 3, 4, 224, 223, 3, 3);
        run(3, 3, 4, 224, 223, 2, 2);
    };
    run_with_param();
    run_with_param(2, 2, 3, 3);
    run_with_param(3, 2, 1, 2);
    run_with_param(2, 3, 2, 2);
}

TEST(TestOprDNN, ConvBiasForwardWithZ) {
    REQUIRE_GPU(1);
    using Checker4 = AutoOprChecker<4, 1>;
    opr::ConvBiasForward::Param param;

    auto make_graph4 =
            [&](const Checker4::SymInpArray& inputs) -> Checker4::SymOutArray {
        return {opr::ConvBiasForward::make(inputs[0], inputs[1], inputs[2],
                                           inputs[3], param)};
    };

    auto fwd4 = [&](Checker4::NumOutArray& dest, Checker4::NumInpArray inp) {
        std::shared_ptr<HostTensorND> sh_out;
        convolution_brute({inp[0], inp[1]}, sh_out,
                          convert_to_conv_param(param));
        dest[0] = *sh_out;
        size_t N = dest[0].shape()[0];
        size_t OC = dest[0].shape()[1];
        size_t OH = dest[0].shape()[2];
        size_t OW = dest[0].shape()[3];
        auto dest_ptr = dest[0].ptr<float>();
        float* z_ptr = inp[3]->ptr<float>();

        for (size_t i = 0; i < N; i++) {
            auto bias_ptr = inp[2]->ptr<float>();
            for (size_t c = 0; c < OC; c++) {
                for (size_t hw = 0; hw < OH * OW; hw++) {
                    *(dest_ptr++) += *(bias_ptr) + *(z_ptr++);
                }
                bias_ptr++;
            }
        }
    };

    auto run_with_param = [&](size_t SH = 1, size_t SW = 1, size_t PH = 0,
                              size_t PW = 0) {
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        Checker4 checker4{make_graph4, fwd4};
        Checker4::RunOptions opt4;
        checker4.set_output_allow_grad(0, false);

        auto run = [&](size_t N, size_t IC, size_t OC, size_t IH, size_t IW,
                       size_t FH, size_t FW) {
            auto opr = megdnn_naive_handle()
                               ->create_operator<megdnn::ConvolutionForward>();
            opr->param() = convert_to_conv_param(param);
            TensorLayout dest_layout;
            opr->deduce_layout({{N, IC, IH, IW}, dtype::Float32()},
                               {{OC, IC, FH, FW}, dtype::Float32()},
                               dest_layout);
            checker4.run({TensorShape{N, IC, IH, IW},
                          {OC, IC, FH, FW},
                          {1, OC, 1, 1},
                          {N, OC, dest_layout[2], dest_layout[3]}},
                         opt4);
        };
        run(1, 1, 1, 5, 5, 3, 3);
        run(2, 3, 4, 5, 5, 3, 3);
        run(3, 3, 4, 224, 223, 3, 3);
        run(3, 3, 4, 224, 223, 2, 2);
    };
    run_with_param();
    run_with_param(2, 2, 3, 3);
    run_with_param(3, 2, 1, 2);
    run_with_param(2, 3, 2, 2);
}

TEST(TestOprDNN, ConvBiasINT8x8xX_NCHW4) {
    using Checker = AutoOprChecker<3, 1>;
    using Param = opr::ConvBias::Param;
    opr::ConvBiasForward::Param param;

    auto make_quantized = [&](SymbolVar x, const DType& dtype) {
        return opr::TypeCvt::make(x, dtype);
    };
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto conv_param = convert_to_conv_param(param);
        auto y = opr::Convolution::make(
                make_quantized(inputs[0], dtype::QuantizedS8(0.3f)),
                make_quantized(inputs[1], dtype::QuantizedS8(0.1f)), conv_param);
        y = y + make_quantized(inputs[2], dtype::QuantizedS32(0.03f));
        if (param.nonlineMode == Param::NonlineMode::RELU)
            y = opr::Elemwise::make(
                    {y}, {opr::Elemwise::Mode::RELU});
        y = opr::TypeCvt::make(y, dtype::QuantizedS8(0.5f));
        return {opr::TypeCvt::make(y, dtype::Float32())};
    };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto graph = ComputingGraph::make();
        Checker::SymInpArray inputs;
        for (size_t i = 0; i < inp.size(); ++i) {
            inputs[i] = opr::Host2DeviceCopy::make(
                    *graph, inp[i]);
        }

        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity();
        auto y = gopt::optimize_for_inference({make_graph(inputs)[0]},
                                              options)[0];
        auto func = graph->compile({make_callback_copy(y, dest[0])});
        func->execute();
        func->wait();
    };

    auto run_with_param = [&](size_t SH = 1, size_t SW = 1, size_t PH = 0,
                              size_t PW = 0, size_t group = 1) {
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        param.format = Param::Format::NCHW4;
        if (group != 1)
            param.sparse = Param::Sparse::GROUP;
        Checker checker{make_graph, fwd, CompNode::load("cpu0")};
        Checker::RunOptions opt;
        checker.set_output_allow_grad(0, false);
        auto run = [&](size_t N, size_t IC, size_t OC, size_t IH, size_t IW,
                       size_t FH, size_t FW) {

            mgb_assert(IC % 4 == 0 && OC % 4 == 0);
            checker.run({TensorShape{N, group * IC / 4, IH, IW, 4},
                         {group, OC, IC / 4, FH, FW, 4},
                         {1, group * OC / 4, 1, 1, 4}},
                        opt);
        };
        run(1, 8, 8, 56, 56, 3, 3);
        run(1, 8, 8, 56, 56, 3, 3);
        run(1, 8, 8, 56, 56, 3, 3);
    };
    run_with_param(1, 1, 1, 1, 8);
    run_with_param();
    run_with_param(2, 2, 3, 3);
    run_with_param(3, 2, 1, 2);
    run_with_param(2, 3, 2, 2);
}


TEST(TestOprDNN, ConvolutionDTypeInference) {
    Param param;
    param.mode = Mode::CONVOLUTION;

    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    HostTensorND inp_host{
            cn, {1, 3, 7, 7}, dtype::Quantized8Asymm(0.233f, (uint8_t)123)};
    HostTensorND filt_host{
            cn, {8, 3, 1, 1}, dtype::Quantized8Asymm(0.874f, (uint8_t)234)};
    auto inp = opr::ImmutableTensor::make(*graph, inp_host);
    auto filt = opr::ImmutableTensor::make(*graph, filt_host);
    auto opr = opr::Convolution::make(inp, filt, param);
    ASSERT_EQ(opr.dtype().enumv(), DTypeEnum::QuantizedS32);
    // This has to be EQ instead of NEAR
    EXPECT_EQ(opr.dtype().param<dtype::QuantizedS32>().scale, 0.233f * 0.874f);

    inp_host = {cn, {1, 3, 7, 7}, dtype::QuantizedS8(0.1234f)};
    filt_host = {cn, {8, 3, 1, 1}, dtype::QuantizedS8(0.2345f)};
    inp = opr::ImmutableTensor::make(*graph, inp_host);
    filt = opr::ImmutableTensor::make(*graph, filt_host);
    opr = opr::Convolution::make(inp, filt, param);
    ASSERT_EQ(opr.dtype().enumv(), DTypeEnum::QuantizedS32);
    EXPECT_EQ(opr.dtype().param<dtype::QuantizedS32>().scale,
              0.1234f * 0.2345f);

    inp_host = {cn, {1, 3, 7, 7}, dtype::Int8()};
    filt_host = {cn, {8, 3, 1, 1}, dtype::Int8()};
    inp = opr::ImmutableTensor::make(*graph, inp_host);
    filt = opr::ImmutableTensor::make(*graph, filt_host);
    opr = opr::Convolution::make(inp, filt, param);
    ASSERT_EQ(opr.dtype().enumv(), DTypeEnum::Int32);
}

TEST(TestOprDNN, ConvBiasINT8x8xXDTypeInference) {
    float inp_scale = 1.926f;
    float filt_scale = 0.817f;
    float bias_scale = inp_scale * filt_scale;
    opr::ConvBias::Param param;
    param.mode = Mode::CONVOLUTION;

    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    HostTensorND inp_host{cn, {1, 3, 7, 7}, dtype::QuantizedS8(inp_scale)};
    HostTensorND filt_host{cn, {8, 3, 1, 1}, dtype::QuantizedS8(filt_scale)};
    DType output_dtype = dtype::QuantizedS8(bias_scale);
    HostTensorND bias_host{cn, {1, 3, 7, 7}, dtype::QuantizedS32(bias_scale)};
    auto inp = opr::ImmutableTensor::make(*graph, inp_host);
    auto filt = opr::ImmutableTensor::make(*graph, filt_host);
    auto bias = opr::ImmutableTensor::make(*graph, filt_host);
    auto opr = opr::ConvBiasForward::make(inp, filt, bias, param,
            {}, OperatorNodeConfig{output_dtype});
    ASSERT_EQ(opr.dtype().enumv(), DTypeEnum::QuantizedS8);
    EXPECT_EQ(opr.dtype().param<dtype::QuantizedS8>().scale, bias_scale);
}

TEST(TestOprDNN, ConvBiasINT8x8xXSerialization) {
    using namespace serialization;

    float inp_scale = 1.926f;
    float filt_scale = 0.817f;
    float bias_scale = inp_scale * filt_scale;
    DType output_dtype = dtype::QuantizedS8(bias_scale);

    auto fname = output_file("ConvBiasINT8x8xXTest");
    auto dump = [&]() {
        opr::ConvBias::Param param;
        param.mode = Mode::CONVOLUTION;

        auto cn = CompNode::load("cpu0");
        auto graph = ComputingGraph::make();
        HostTensorND inp_host{cn, {1, 3, 7, 7}, dtype::QuantizedS8(inp_scale)};
        HostTensorND filt_host{
                cn, {8, 3, 1, 1}, dtype::QuantizedS8(filt_scale)};
        HostTensorND bias_host{
                cn, {1, 3, 7, 7}, dtype::QuantizedS32(bias_scale)};
        auto inp = opr::ImmutableTensor::make(*graph, inp_host);
        auto filt = opr::ImmutableTensor::make(*graph, filt_host);
        auto bias = opr::ImmutableTensor::make(*graph, filt_host);
        auto opr = opr::ConvBiasForward::make(inp, filt, bias, param,
                {},
                                              OperatorNodeConfig{output_dtype});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({opr});
        ASSERT_EQ(rst.outputs.size(), 1u);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
        EXPECT_EQ(rst.output_var_list[0].dtype(), output_dtype);
    };

    dump();
    load();
}

TEST(TestOprDNN, LocalShareForward) {
    REQUIRE_GPU(1);
    using Checker = AutoOprChecker<2, 1>;
    using Param = opr::LocalShare::Param;
    Param param;
    param.mode = Param::Mode::CROSS_CORRELATION;
    param.sparse = Param::Sparse::DENSE;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::LocalShare::make(inputs[0], inputs[1], param)};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        mgb_assert(inp.size() == 2);
        mgb_assert(dest.size() == 1);
        std::shared_ptr<HostTensorND> out;
        local_share_brute({inp[0], inp[1]}, out, param);
        dest[0] = *out;
    };

    auto run_with_param = [&](size_t fh = 3, size_t fw = 3, size_t sh = 1,
                              size_t sw = 1, size_t sgh = 3, size_t sgw = 3) {
        size_t ph = fh / 2, pw = fw / 2;
        param.pad_h = ph, param.pad_w = pw;
        param.stride_h = sh, param.stride_w = sw, param.spatial_groups_h = sgh,
        param.spatial_groups_w = sgw;
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        checker.set_output_allow_grad(0, false);
        checker.set_input_dtype(0, dtype::Float32());
        checker.set_input_dtype(1, dtype::Float32());
        auto run = [&](size_t n, size_t ci, size_t co, size_t hi, size_t wi) {
            size_t ho = (hi + 2 * ph - fh) / sh + 1;
            size_t wo = (wi + 2 * pw - fw) / sw + 1;
            if (ho % sgh != 0 || wo % sgw != 0)
                return;
            checker.run({TensorShape{n, ci, hi, wi},
                         TensorShape{sgh, sgw, ci, fh, fw, co}},
                        opt);
        };
        run(32, 2, 7, 24, 24);
        run(16, 2, 7, 24, 24);
        run(32, 2, 8, 12, 12);
        run(16, 2, 9, 6, 6);
    };
    run_with_param(1, 1, 1, 1, 3, 3);
    run_with_param(3, 3, 1, 1, 2, 2);
    run_with_param(5, 5, 1, 1, 2, 2);
    run_with_param(7, 7, 1, 1, 2, 2);
    run_with_param(1, 1, 2, 2, 3, 3);
    run_with_param(3, 3, 2, 2, 2, 2);
    run_with_param(5, 5, 1, 1, 2, 2);
    run_with_param(7, 7, 1, 1, 2, 2);
}

TEST(TestOprDNN, LocalShareForwardGrad) {
    REQUIRE_GPU(1);
    using Checker = AutoOprChecker<2, 1>;
    using Param = opr::LocalShare::Param;
    Param param;
    param.mode = Param::Mode::CROSS_CORRELATION;
    param.sparse = Param::Sparse::DENSE;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::LocalShare::make(inputs[0], inputs[1], param)};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        mgb_assert(inp.size() == 2);
        mgb_assert(dest.size() == 1);
        std::shared_ptr<HostTensorND> out;
        local_share_brute({inp[0], inp[1]}, out, param);
        dest[0] = *out;
    };

    auto run_with_param = [&](size_t fh = 3, size_t fw = 3, size_t sh = 1,
                              size_t sw = 1, size_t sgh = 3, size_t sgw = 3) {
        size_t ph = fh / 2, pw = fw / 2;
        param.pad_h = ph, param.pad_w = pw;
        param.stride_h = sh, param.stride_w = sw, param.spatial_groups_h = sgh,
        param.spatial_groups_w = sgw;
        Checker checker{make_graph, fwd};
        Checker::RunOptions opt;
        checker.set_output_allow_grad(0, true);
        opt.numdiff_max_err = 1e-1;
        checker.set_input_dtype(0, dtype::Float32());
        checker.set_input_dtype(1, dtype::Float32());
        auto run = [&](size_t n, size_t ci, size_t co, size_t hi, size_t wi) {
            size_t ho = (hi + 2 * ph - fh) / sh + 1;
            size_t wo = (wi + 2 * pw - fw) / sw + 1;
            if (ho % sgh != 0 || wo % sgw != 0)
                return;
            checker.run({TensorShape{n, ci, hi, wi},
                         TensorShape{sgh, sgw, ci, fh, fw, co}},
                        opt);
        };
        run(4, 2, 8, 24, 24);
        run(8, 2, 4, 6, 6);
        run(16, 4, 8, 12, 12);
        run(4, 4, 8, 12, 12);
    };
    run_with_param(1, 1, 1, 1, 3, 3);
    run_with_param(1, 1, 2, 2, 3, 3);
    run_with_param(3, 3, 2, 2, 2, 2);
}

TEST(TestOprDNN, LocalShareForwardExecPolicy) {
    REQUIRE_GPU(1);
    using Checker = AutoOprChecker<2, 1>;
    using Policy = opr::LocalShare::ExecutionPolicy;
    using S = Policy::Strategy;
    using Param = opr::LocalShare::Param;
    Param param;
    param.mode = Param::Mode::CROSS_CORRELATION;
    param.sparse = Param::Sparse::DENSE;

    int nr_get = 0;
    auto on_get = [&nr_get](const std::string&, const void*, size_t,
                            const void*, size_t) { ++nr_get; };
    PersistentCacheHook cache_hook{on_get};

#if MGB_ENABLE_FASTRUN
    for (auto strategy: {S::PROFILE, S::HEURISTIC, S::PROFILE_REPRODUCIBLE, S::PROFILE_HEURISTIC}) {
#else
    for (auto strategy: {S:HEURISTIC, S::PROFILE_HEURISTIC}) {
#endif
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            Policy policy;
            policy.strategy = strategy;
            return {opr::LocalShare::make(inputs[0], inputs[1], param, policy)};
        };

        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            mgb_assert(inp.size() == 2);
            mgb_assert(dest.size() == 1);
            std::shared_ptr<HostTensorND> out;
            local_share_brute({inp[0], inp[1]}, out, param);
            dest[0] = *out;
        };

        auto run_with_param = [&](size_t fh = 3, size_t fw = 3, size_t sh = 1,
                                  size_t sw = 1, size_t sgh = 3,
                                  size_t sgw = 3) {
            size_t ph = fh / 2, pw = fw / 2;
            param.pad_h = ph, param.pad_w = pw;
            param.stride_h = sh, param.stride_w = sw,
            param.spatial_groups_h = sgh, param.spatial_groups_w = sgw;
            Checker checker{make_graph, fwd};
            Checker::RunOptions opt;
            checker.set_output_allow_grad(0, false);
            checker.set_input_dtype(0, dtype::Float32());
            checker.set_input_dtype(1, dtype::Float32());
            nr_get = 0;
            opt.outputs_max_err = 1e-3;
            auto run = [&](size_t n, size_t ci, size_t co, size_t hi,
                           size_t wi) {
                size_t ho = (hi + 2 * ph - fh) / sh + 1;
                size_t wo = (wi + 2 * pw - fw) / sw + 1;
                if (ho % sgh != 0 || wo % sgw != 0)
                    return;
                checker.run({TensorShape{n, ci, hi, wi},
                             TensorShape{sgh, sgw, ci, fh, fw, co}},
                            opt);
            };
            run(32, 4, 8, 24, 24);
            run(32, 4, 8, 12, 12);
            run(16, 4, 8, 12, 12);
            run(32, 4, 8, 6, 6);
            if (strategy == S::HEURISTIC) {
                ASSERT_EQ(0, nr_get);
            } else {
                ASSERT_LT(0, nr_get);
            }
        };
        run_with_param(1, 1, 1, 1, 3, 3);
        run_with_param(3, 3, 1, 1, 2, 2);
        run_with_param(5, 5, 1, 1, 2, 2);
        run_with_param(7, 7, 1, 1, 2, 2);
        run_with_param(1, 1, 2, 2, 3, 3);
        run_with_param(3, 3, 2, 2, 2, 2);
        run_with_param(5, 5, 1, 1, 2, 2);
        run_with_param(7, 7, 1, 1, 2, 2);
    }
}

TEST(TestOprDNN, LocalShareSerialization) {
    using namespace serialization;

    auto fname = output_file("LocalShareForwardTest");
    auto dump = [&]() {
        opr::LocalShare::Param param;
        param.mode = Mode::CROSS_CORRELATION;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = 0;
        param.spatial_groups_h = param.spatial_groups_w = 3;

        auto cn = CompNode::load("cpu0");
        auto graph = ComputingGraph::make();
        HostTensorND inp_host{cn, {32, 4, 24, 24}, dtype::Float32()};
        HostTensorND filt_host{
                cn, {3, 3, 4, 1, 1, 8}, dtype::Float32()};
        auto inp = opr::ImmutableTensor::make(*graph, inp_host);
        auto filt = opr::ImmutableTensor::make(*graph, filt_host);
        auto opr = opr::LocalShareForward::make(inp, filt, param,
                {});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({opr});
        ASSERT_EQ(rst.outputs.size(), 1u);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
    };

    dump();
    load();
}

TEST(TestOprDNN, DeformableConvForward) {
    REQUIRE_GPU(1);
    using Checker = AutoOprChecker<4, 1>;
    using Policy = opr::DeformableConvForward::ExecutionPolicy;
    using S = Policy::Strategy;
    using Param = opr::DeformableConvForward::Param;
    Param param;

#if MGB_ENABLE_FASTRUN
    for (auto strategy : {S::PROFILE, S::HEURISTIC, S::PROFILE_REPRODUCIBLE,
                          S::PROFILE_HEURISTIC}) {
#else
    for (auto strategy : {S : HEURISTIC, S::PROFILE_HEURISTIC}) {
#endif
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            Policy policy;
            policy.strategy = strategy;
            return {opr::DeformableConvForward::make(
                    inputs[0], inputs[1], inputs[2], inputs[3], param, policy)};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto opr =
                    megdnn_naive_handle()
                            ->create_operator<megdnn::DeformableConvForward>();
            opr->param() = param;
            TensorLayout dest_layout;
            opr->deduce_layout(inp[0]->layout(), inp[1]->layout(),
                               inp[2]->layout(), inp[3]->layout(), dest_layout);
            std::vector<dt_byte> workspace(opr->get_workspace_in_bytes(
                    inp[0]->layout(), inp[1]->layout(), inp[2]->layout(),
                    inp[3]->layout(), dest_layout));
            dest[0].dtype(dtype::Float32())
                    .comp_node(inp[0]->comp_node())
                    .resize(dest_layout);
            opr->exec(inp[0]->as_megdnn(), inp[1]->as_megdnn(),
                      inp[2]->as_megdnn(), inp[3]->as_megdnn(),
                      dest[0].as_megdnn(),
                      {workspace.data(), workspace.size()});
        };
        auto run_with_param = [&](size_t fh, size_t fw, size_t sh, size_t sw,
                                  size_t dh, size_t dw, size_t group,
                                  size_t deformable_group) {
            Checker checker{make_graph, fwd};
            size_t ph = fh / 2, pw = fw / 2;
            param.pad_h = ph, param.pad_w = pw;
            param.stride_h = sh, param.stride_w = sw;
            param.dilate_h = dh, param.dilate_w = dw;

            param.format = Param::Format::NCHW;
            param.mode = Param::Mode::CROSS_CORRELATION;
            param.sparse = Param::Sparse::DENSE;
            if (group > 1)
                param.sparse = Param::Sparse::GROUP;
            Checker::RunOptions opt;
	    float DELTA = 1e-3;
            opt.numdiff_eps = DELTA;
            opt.numdiff_max_err = 1e-1;
            auto gen_off = [DELTA](HostTensorND& off, float l = -2.f, float h = 2.f) {
                RNGxorshf rng{next_rand_seed()};
                auto elems = off.shape().total_nr_elems();
                auto ptr = off.ptr<float>();
                auto rand_real = [](RNGxorshf& rng, float lo, float hi) {
                    std::uniform_real_distribution<float> dist(lo, hi);
                    return dist(rng);
                };
                for (size_t i = 0; i < elems; ++i) {
                    do {
                        float val = rand_real(rng, l, h);
                        if (abs(floor(val + 2 * DELTA) - floor(val)) <= 1e-6f &&
                            abs(floor(val - 2 * DELTA) - floor(val)) <= 1e-6f) {
                            ptr[i] = val;
                            break;
                        }
                    } while (true);
                }
            };
            //! generate offset to avoid value near integer
	    /// because bilinear function is not derivable over there
	    checker.set_input_generator(2, gen_off);
            checker.set_input_dtype(0, dtype::Float32());
            checker.set_input_dtype(1, dtype::Float32());
            checker.set_input_dtype(2, dtype::Float32());
            checker.set_input_dtype(3, dtype::Float32());
            auto run = [&](size_t n, size_t ih, size_t iw, size_t icpg,
                           size_t ocpg) {
                size_t oh = (ih + 2 * ph - fh) / sh + 1;
                size_t ow = (iw + 2 * pw - fw) / sw + 1;
                checker.run({TensorShape{n, group * icpg, ih, iw},
                             (param.sparse == Param::Sparse::GROUP)
                                     ? TensorShape{group, ocpg, icpg, fh, fw}
                                     : TensorShape{group * ocpg, group * icpg,
                                                   fh, fw},
                             {n, 2 * deformable_group * fh * fw, oh, ow},
                             {n, deformable_group * fh * fw, oh, ow}},
                            opt);
            };
            run(1, 3, 3, 2, 1);
            run(2, 3, 3, 2, 2);
            run(1, 5, 5, 2, 1);
        };
        // run_with_param(1, 1, 1, 1, 1, 1, 1, 1);
        run_with_param(3, 3, 1, 1, 1, 1, 2, 2);
        // run_with_param(5, 5, 1, 1, 1, 1, 2, 2);
    }
}

TEST(TestOprDNN, DeformableConvSerialization) {
    using namespace serialization;

    auto fname = output_file("DeformableConvTest");
    auto dump = [&]() {
        using Param = opr::DeformableConvForward::Param;
        Param param;
        size_t n = 16, ocpg = 2, icpg = 4;
	size_t ih = 24, iw = 24, fh = 3, fw = 3, ph = 2, pw = 2, sh = 1, sw = 1, dh = 1, dw = 1;
        size_t group = 1, deformable_group =1;

        size_t oh = (ih + 2 * ph - fh) / sh + 1;
        size_t ow = (iw + 2 * pw - fw) / sw + 1;

	param.pad_h = ph, param.pad_w = pw;
        param.stride_h = sh, param.stride_w = sw;
        param.dilate_h = dh, param.dilate_w = dw;

        param.format = Param::Format::NCHW;
        param.mode = Param::Mode::CROSS_CORRELATION;
        param.sparse = Param::Sparse::DENSE;

        auto cn = CompNode::load("cpu0");
        auto graph = ComputingGraph::make();
        HostTensorND inp_host{cn, {n, group * icpg, ih, iw}, dtype::Float32()};
        HostTensorND filt_host{
                cn, {group * ocpg, group * icpg, fh, fw}, dtype::Float32()};
        HostTensorND offset_host{
                cn, {n, 2 * deformable_group * fh * fw, oh, ow}, dtype::Float32()};
        HostTensorND mask_host{
                cn, {n, deformable_group * fh * fw, oh, ow}, dtype::Float32()};
        auto inp = opr::ImmutableTensor::make(*graph, inp_host);
        auto filt = opr::ImmutableTensor::make(*graph, filt_host);
        auto offset = opr::ImmutableTensor::make(*graph, offset_host);
        auto mask = opr::ImmutableTensor::make(*graph, mask_host);
        auto opr = opr::DeformableConvForward::make(inp, filt, offset, mask,
                                                    param, {}, {});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({opr});
        ASSERT_EQ(rst.outputs.size(), 1u);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
    };

    dump();
    load();
}

#if MGB_CUDA
TEST(TestOprDNN, BatchConvBiasForward) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY(6, 1);

    using Checker = AutoOprChecker<3, 1>;
    using Policy = opr::BatchConvBiasForward::ExecutionPolicy;
    using S = Policy::Strategy;
    using Param = opr::BatchConvBiasForward::Param;
    Param param;
    param.format = Param::Format::NCHW4;
    param.mode = Param::Mode::CROSS_CORRELATION;
    param.sparse = Param::Sparse::DENSE;

#if MGB_ENABLE_FASTRUN
    for (auto strategy : {S::PROFILE, S::HEURISTIC, S::PROFILE_REPRODUCIBLE,
                          S::PROFILE_HEURISTIC}) {
#else
    for (auto strategy : {S : HEURISTIC, S::PROFILE_HEURISTIC}) {
#endif

        auto make_quantized = [&](SymbolVar x, const DType& dtype) {
            return opr::TypeCvt::make(x, dtype);
        };
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            Policy policy;
            policy.strategy = strategy;
            auto conv_bias = opr::BatchConvBiasForward::make(
                    make_quantized(inputs[0], dtype::QuantizedS8{1.1f}),
                    make_quantized(inputs[1], dtype::QuantizedS8{1.2f}),
                    make_quantized(inputs[2], dtype::QuantizedS32{1.1f * 1.2f}),
                    param, policy,
                    OperatorNodeConfig{dtype::QuantizedS8{1.3f}});
            return {opr::TypeCvt::make(conv_bias, dtype::Float32())};
        };
        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            mgb_assert(inp.size() == 3);
            mgb_assert(dest.size() == 1);
            auto graph = ComputingGraph::make();
            Checker::SymInpArray inputs;
            for (size_t i = 0; i < inp.size(); ++i) {
                inputs[i] = opr::Host2DeviceCopy::make(*graph, inp[i]);
            }
            auto src = make_quantized(inputs[0], dtype::QuantizedS8{1.1f}),
                 filter = make_quantized(inputs[1], dtype::QuantizedS8{1.2f}),
                 bias = make_quantized(inputs[2],
                                       dtype::QuantizedS32{1.1f * 1.2f});
            {
                auto xshp = opr::GetVarShape::make(src);

                auto cv = [&src](int v) { return src.make_scalar(v); };
                auto sub = [&xshp, &cv](int idx) {
                    return opr::IndexAt::make(xshp, {{0, cv(idx)}});
                };
                auto tshp = opr::Concat::make(
                        {cv(1), sub(0) * sub(1), sub(2), sub(3), sub(4)}, 0);
                src = opr::Reshape::make(src, tshp);
            }
            auto conv_param = convert_to_conv_param(param);
            conv_param.sparse = opr::BatchConvBias::Param::Sparse::GROUP;
            auto y = opr::Convolution::make(src, filter, conv_param);
            {
                auto fshp = opr::GetVarShape::make(filter);
                auto batch =
                        opr::IndexAt::make(fshp, {{0, filter.make_scalar(0)}});

                auto xshp = opr::GetVarShape::make(y);

                auto cv = [&y](int v) { return y.make_scalar(v); };
                auto sub = [&xshp, &cv](int idx) {
                    return opr::IndexAt::make(xshp, {{0, cv(idx)}});
                };
                auto tshp = opr::Concat::make(
                        {batch, sub(1) / batch, sub(2), sub(3), sub(4)}, 0);
                y = opr::Reshape::make(y, tshp);
            }
            y = y + bias;
            y = opr::TypeCvt::make(y, dtype::QuantizedS8{1.3f});
            y = opr::TypeCvt::make(y, dtype::Float32());
            auto func = graph->compile({make_callback_copy(y, dest[0])});
            func->execute();
            func->wait();
        };

        auto run_with_param = [&](size_t sh = 1, size_t sw = 1) {
            size_t fh = 1;
            size_t fw = 1;
            size_t ph = fh / 2, pw = fw / 2;
            param.pad_h = ph, param.pad_w = pw;
            param.stride_h = sh, param.stride_w = sw;
            Checker checker{make_graph, fwd, cn};
            Checker::RunOptions opt;
            checker.set_output_allow_grad(0, false);
            checker.set_input_dtype(0, dtype::Float32());
            checker.set_input_dtype(1, dtype::Float32());
            checker.set_input_dtype(2, dtype::Float32());
            auto run = [&](size_t n, size_t ci, size_t co, size_t hi,
                           size_t wi) {
                checker.run({TensorShape{n, ci / 4, hi, wi, 4},
                             TensorShape{n, co, ci / 4, fh, fw, 4},
                             TensorShape{1, co / 4, 1, 1, 4}},

                            opt);
            };
            run(32, 16, 32, 24, 24);
            run(16, 16, 32, 24, 24);
            run(32, 16, 64, 12, 12);
            run(16, 16, 64, 6, 6);
        };
        run_with_param(1, 1);
        run_with_param(2, 2);
    }
}
#endif

TEST(TestOprDNN, BatchConvBiasSerialization) {
    using namespace serialization;

    auto fname = output_file("BatchConvBiasForwardTest");
    auto dump = [&]() {
        opr::BatchConvBias::Param param;
        param.mode = Mode::CROSS_CORRELATION;
        param.format = opr::BatchConvBias::Param::Format::NCHW4;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = 0;

        auto cn = CompNode::load("cpu0");
        auto graph = ComputingGraph::make();
        HostTensorND inp_host{cn, {32, 1, 24, 24, 4}, dtype::QuantizedS8{1.1f}};
        HostTensorND filt_host{cn, {32, 8, 1, 1, 1, 4}, dtype::QuantizedS8{1.2f}};
        auto inp = opr::ImmutableTensor::make(*graph, inp_host);
        auto filt = opr::ImmutableTensor::make(*graph, filt_host);
        auto opr = opr::BatchConvBiasForward::make(
                inp, filt, param, {},
                OperatorNodeConfig{dtype::QuantizedS8{1.3f}});
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({opr});
        ASSERT_EQ(rst.outputs.size(), 1u);
    };

    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
    };

    dump();
    load();
}

TEST(TestOprDNN, HeuristicReproducible) {
    using Policy = opr::ConvolutionBackwardFilter::ExecutionPolicy;
    using S = Policy::Strategy;

    using Checker = AutoOprChecker<3, 1>;

    constexpr size_t PH = 1, PW = 1, SH = 1, SW = 1;

    for (auto strategy : {S::HEURISTIC, S::HEURISTIC_REPRODUCIBLE}) {
        VarNode* bwd_flt;
        auto make_graph = [&](const Checker::SymInpArray& inputs)
                -> Checker::SymOutArray {
            Param param{Mode::CROSS_CORRELATION, PH, PW, SH, SW};
            Policy policy;
            policy.strategy = strategy;
            auto out = opr::ConvolutionBackwardFilter::make(
                    inputs[0], inputs[1], inputs[2], param, policy);
            bwd_flt = out.node();
            return {out};
        };

        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            std::shared_ptr<HostTensorND> out;
            conv_bwd_flt_brute({inp[0], inp[1], inp[2]}, out,
                               Param{Mode::CROSS_CORRELATION, PH, PW, SH, SW});
            dest[0] = *out;
        };

#define get_shp(N, P, S, F) ((N + 2 * P - F) / S + 1)
#define inp_tensor(N, IC, OC, IH, IW, FH, FW)                                \
    {                                                                        \
        TensorShape{N, IC, IH, IW},                                          \
                {N, OC, get_shp(IH, PH, SH, FH), get_shp(IW, PW, SW, FW)}, { \
            OC, IC, FH, FW                                                   \
        }                                                                    \
    }
        Checker::RunOptions opt;
        opt.numdiff_eps = 1;
        opt.outputs_max_err = 1e-3;
        std::string algo_name0, algo_name1;
        {
            Checker checker(make_graph, fwd);
            checker.run(inp_tensor(2, 3, 4, 9, 8, 3, 3), opt)
                    .run(inp_tensor(1, 5, 3, 7, 9, 3, 3), opt)
                    .run(inp_tensor(3, 4, 4, 9, 9, 3, 3), opt);

            auto algo = static_cast<megdnn::ConvolutionBackwardFilter*>(
                                static_cast<opr::ConvolutionBackwardFilter*>(
                                        bwd_flt->owner_opr())
                                        ->megdnn_opr())
                                ->execution_policy()
                                .algo;
            if (strategy == S::HEURISTIC_REPRODUCIBLE) {
                EXPECT_TRUE(algo.is_reproducible);
            }
            algo_name0 = algo.name.c_str();
        }
        {
            Checker checker(make_graph, fwd);
            checker.run(inp_tensor(2, 3, 4, 9, 8, 3, 3), opt)
                    .run(inp_tensor(1, 5, 3, 7, 9, 3, 3), opt)
                    .run(inp_tensor(3, 4, 4, 9, 9, 3, 3), opt);
            auto algo = static_cast<megdnn::ConvolutionBackwardFilter*>(
                                static_cast<opr::ConvolutionBackwardFilter*>(
                                        bwd_flt->owner_opr())
                                        ->megdnn_opr())
                                ->execution_policy()
                                .algo;
            algo_name1 = algo.name.c_str();
        }
        EXPECT_TRUE(algo_name0 == algo_name1);
    }
#undef inp_tensor
#undef get_shp
}

#if MGB_CUDA
TEST(TestOprDNN, ConvolutionMultiCompNode) {
    REQUIRE_GPU(1);
    auto cn0 = CompNode::load("gpu0:0"), cn1 = CompNode::load("gpu0:1");
    cn0.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn0).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
    auto mkvar = [&gen](const char* name, const TensorShape& shp,
                     const DType& dtype,
                     std::shared_ptr<ComputingGraph> graph,
                     const CompNode& cn) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&gen](const char* name, const TensorShape& shp,
                      const DType& dtype,
                      std::shared_ptr<ComputingGraph> graph,
                      const CompNode& cn) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto graph0 = ComputingGraph::make();
    graph0->options().graph_opt_level = 0;
    auto graph1 = ComputingGraph::make();
    graph1->options().graph_opt_level = 0;
    auto make_func = [&gen, &mkvar, &mkcvar](
                             std::shared_ptr<ComputingGraph> graph,
                             const CompNode& cn) {
        using Policy = opr::ConvBias::ExecutionPolicy;
        using S = Policy::Strategy;
        auto x = mkvar("x", {64, 32, 28, 28, 4}, dtype::QuantizedS8(2.5f),
                       graph, cn),
             w1 = mkcvar("w1", {256, 32, 5, 5, 4}, dtype::QuantizedS8(2.5f),
                         graph, cn),
             b1 = mkcvar("b1", {1, 64, 1, 1, 4}, dtype::QuantizedS32(6.25f),
                         graph, cn),
             w2 = mkcvar("w2", {256, 64, 3, 3, 4}, dtype::QuantizedS8(2.5f),
                         graph, cn),
             b2 = mkcvar("b2", {1, 64, 1, 1, 4}, dtype::QuantizedS32(6.25f),
                         graph, cn);
        opr::ConvBias::Param param;
        param.format = opr::ConvBias::Param::Format::NCHW4;
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = 2;
        Policy policy;
        policy.strategy = S::PROFILE;

        auto y = opr::ConvBias::make(
                x, w1, b1, param, policy,
                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = 1;
        y = opr::ConvBias::make(y, w2, b2, param, policy,
                                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        return y;
    };
    auto y0 = make_func(graph0, cn0);
    auto y1 = make_func(graph1, cn1);
    HostTensorND host_y0, host_y1;
    auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
    auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});

    auto worker = [&func0, &func1](int wid) {
        static const int iter_num = 1000;
        if (wid == 0) {
            for (int i = 0; i < iter_num; ++i)
                func0->execute();
        } else {
            for (int i = 0; i < iter_num; ++i)
                func1->execute();
        }
    };
    std::thread worker0(worker, 0);
    std::thread worker1(worker, 1);
    worker0.join();
    worker1.join();
}

#endif

}  // anonymous namespace

#ifndef _WIN32

namespace mgb {
namespace opr {
namespace testing {

class ConvolutionTestingPeer {
    opr::ConvolutionForward& m_conv_opr;
public:
    explicit ConvolutionTestingPeer(cg::OperatorNodeBase* opr)
            : m_conv_opr(opr->cast_final_safe<opr::ConvolutionForward>()) {}
    void set_megdnn_opr(
            std::unique_ptr<megdnn::ConvolutionForward> megdnn_opr) {
        m_conv_opr.set_megdnn_opr(std::move(megdnn_opr));
    }
};

}  // namespace testing
}  // namespace opr
}  // namespace mgb

namespace {

using megdnn::TensorND;
using megdnn::Workspace;
using opr::testing::ConvolutionTestingPeer;

class MockConvolutionForward : public megdnn::ConvolutionForward {
    const char* m_algorithm_set_name;
public:
    MockConvolutionForward(megdnn::ConvolutionForward* orig,
                           const char* algo_set_name)
            : megdnn::ConvolutionForward(orig->handle()),
              m_algorithm_set_name(algo_set_name) {}

    MOCK_METHOD5(exec, void(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                            _megdnn_tensor_out dst,
                            const PreprocessedFilter* preprocessed_filter,
                            _megdnn_workspace workspace));
    MOCK_METHOD5(exec_preprocess,
                 void(const TensorLayout& src_layout, _megdnn_tensor_in filter,
                      const TensorLayout& dst_layout,
                      PreprocessedFilter* preprocessed_filter,
                      _megdnn_workspace workspace));
    MOCK_METHOD4(get_workspace_in_bytes,
                 size_t(const TensorLayout& src, const TensorLayout& filter,
                        const TensorLayout& dst,
                        const PreprocessedFilter* preprocessed_filter));
    MOCK_METHOD3(deduce_preprocessed_filter_layout,
                 SmallVector<TensorLayout>(const TensorLayout& src,
                                           const TensorLayout& filter,
                                           const TensorLayout& dst));
    MOCK_METHOD3(get_preprocess_workspace_in_bytes,
                 size_t(const TensorLayout& src, const TensorLayout& filter,
                        const TensorLayout& dst));

    MOCK_METHOD3(get_all_algorithms_info,
                 std::vector<AlgorithmInfo>(const TensorLayout& p0,
                                         const TensorLayout& p1,
                                         const TensorLayout& p2));
    MOCK_METHOD5(get_algorithm_info_heuristic,
                 AlgorithmInfo(const TensorLayout& p0, const TensorLayout& p1,
                            const TensorLayout& p2,
                            size_t workspace_limit_in_bytes,
                            bool reproducible));

    MOCK_METHOD3(get_all_algorithms,
                 std::vector<Algorithm*>(const TensorLayout& p0,
                                         const TensorLayout& p1,
                                         const TensorLayout& p2));
    MOCK_METHOD5(get_algorithm_heuristic,
                 Algorithm*(const TensorLayout& p0, const TensorLayout& p1,
                            const TensorLayout& p2,
                            size_t workspace_limit_in_bytes,
                            bool reproducible));
protected:
    const char* get_algorithm_set_name() const override {
        return m_algorithm_set_name;
    }
};

class MockAlgorithm : public megdnn::detail::Algorithm {
    const char* m_name;

public:
    MockAlgorithm(const char* name = "NotImportant") : m_name(name) {}
    bool is_reproducible() const override { return true; }
    const char* name() const override { return m_name; }
    uint32_t type() const override {
        return megdnn::detail::Algorithm::INVALID_ALGO_TYPE;
    }

    virtual ~MockAlgorithm() = default;
};

class TestWeightPreprocess : public ::testing::Test {
protected:
    CompNode comp_node;
    std::shared_ptr<ComputingGraph> graph;
    std::shared_ptr<HostTensorND> x_host;
    MockConvolutionForward* mock_conv_ptr;
    SymbolVar y;
    HostTensorND y_host;
    std::unique_ptr<cg::AsyncExecutable> func;

    MockConvolutionForward& mock_conv() { return *mock_conv_ptr; }

    void SetUp() override {
        constexpr uint32_t ih = 10, ic = 16, oc = 32, ph = 0, sh = 1, fh = 2,
                           iw = ih;
        comp_node = CompNode::load("cpux");
        graph = ComputingGraph::make();
        graph->options().graph_opt.weight_preprocess = is_weight_preprocess();
        TensorShape x_shape{1, ic, ih, iw}, w_shape{oc, ic, fh, fh};
        x_host = std::make_shared<HostTensorND>(comp_node, x_shape);
        auto x = opr::Host2DeviceCopy::make(*graph, x_host);
        auto w = opr::ImmutableTensor::make(*graph, {comp_node, w_shape});
        Param param;
        param.pad_h = param.pad_w = ph;
        param.stride_h = param.stride_w = sh;
        param.format = Param::Format::NCHW;
        y = opr::ConvolutionForward::make(x, w, param);
        auto& opr =
                y.node()->owner_opr()->cast_final<opr::ConvolutionForward>();
        auto mock = std::make_unique<MockConvolutionForward>(
                opr.megdnn_opr(), ::testing::UnitTest::GetInstance()
                                          ->current_test_info()
                                          ->name());
        mock_conv_ptr = mock.get();
        ConvolutionTestingPeer{&opr}.set_megdnn_opr(std::move(mock));
        func = graph->compile({make_callback_copy(y, y_host)});
    }

    void run() { func->execute().wait(); }

    virtual bool is_weight_preprocess() { return true; }

    void TearDown() override {
        func.reset();
        // Triggers mock check
        graph.reset();
        x_host.reset();
    }
};

TEST_F(TestWeightPreprocess, NoPreprocessNeeded) {
    using ::testing::_;
    using ::testing::Return;
    auto& mock = mock_conv();

    MockAlgorithm algo;
    EXPECT_CALL(mock, get_algorithm_heuristic(_, _, _, _, _))
            .WillRepeatedly(Return(&algo));
    EXPECT_CALL(mock, get_workspace_in_bytes(_, _, _, _))
            .WillRepeatedly(Return(0));
    EXPECT_CALL(mock, get_preprocess_workspace_in_bytes(_, _, _))
            .WillRepeatedly(Return(0));

    {
        ::testing::InSequence seq;
        // Return empty preprocess filters, indicating no need to preprocess
        EXPECT_CALL(mock, deduce_preprocessed_filter_layout(_, _, _))
                .WillRepeatedly(Return(SmallVector<TensorLayout>{}));
        EXPECT_CALL(mock, exec_preprocess(_, _, _, _, _)).Times(0);
        EXPECT_CALL(mock, exec(_, _, _, nullptr, _));
        run();
    }
}

TEST_F(TestWeightPreprocess, PreprocessCalledOnlyOnce) {
    using ::testing::_;
    using ::testing::Return;
    using ::testing::Field;
    using ::testing::Invoke;
    using ::testing::Expectation;
    using PF = MockConvolutionForward::PreprocessedFilter;

    auto& mock = mock_conv();
    MockAlgorithm algo;
    SmallVector<TensorLayout> filter_layout{{{1, 2, 3, 4}, dtype::Float32()},
                                            {{5, 6, 7, 8}, dtype::Float32()}};

    EXPECT_CALL(mock, deduce_preprocessed_filter_layout(_, _, _))
            .WillRepeatedly(Return(filter_layout));

    Expectation algo_call =
            EXPECT_CALL(mock, get_algorithm_heuristic(_, _, _, _, _))
                    .WillOnce(Return(&algo));
    Expectation ws_call = EXPECT_CALL(mock, get_workspace_in_bytes(_, _, _, _))
                                  .After(algo_call)
                                  .WillOnce(Return(0));
    Expectation pre_ws_call =
            EXPECT_CALL(mock, get_preprocess_workspace_in_bytes(_, _, _))
                    .After(algo_call)
                    .WillOnce(Return(233));
    {
        ::testing::InSequence seq;

        // exec_preprocess should be called only once, with workspace allocated
        int salt = 0;
        EXPECT_CALL(mock, exec_preprocess(_, _, _, _, _))
                .After(ws_call, pre_ws_call)
                .WillOnce(Invoke([&](const TensorLayout&, _megdnn_tensor_in,
                                     const TensorLayout&, PF* pf,
                                     _megdnn_workspace workspace) {
                    ASSERT_EQ(workspace.size, 233);
                    ASSERT_NE(pf, nullptr);
                    pf->algorithm_id = &salt;
                    ASSERT_EQ(pf->tensors.size(), 2);
                    ASSERT_TRUE(pf->tensors[0].layout.eq_shape({1, 2, 3, 4}));
                    ASSERT_TRUE(pf->tensors[1].layout.eq_shape({5, 6, 7, 8}));
                    ASSERT_NE(pf->tensors[0].raw_ptr, nullptr);
                    ASSERT_NE(pf->tensors[1].raw_ptr, nullptr);
                    pf->tensors[0].ptr<float>()[0] = 114.514f;
                    pf->tensors[1].ptr<float>()[0] = 1926.0817f;
                }));

        // Run the graph multiple times.
        for (int i = 0; i < 3; i++) {
            if (i > 0) {
                EXPECT_CALL(mock, exec_preprocess(_, _, _, _, _)).Times(0);
            }
            EXPECT_CALL(mock, exec(_, _, _, _, _))
                    .WillOnce(Invoke([&](_megdnn_tensor_in, _megdnn_tensor_in,
                                         _megdnn_tensor_out, const PF* pf,
                                         _megdnn_workspace) {
                        ASSERT_NE(pf, nullptr);
                        ASSERT_EQ(pf->algorithm_id, &salt);
                        ASSERT_EQ(pf->tensors[0].ptr<float>()[0], 114.514f);
                        ASSERT_EQ(pf->tensors[1].ptr<float>()[0], 1926.0817f);
                    }));
            run();
        }
    }
}

class TestNoWeightPreprocess : public TestWeightPreprocess {
    bool is_weight_preprocess() override { return false; }
};

TEST_F(TestNoWeightPreprocess, NoPreprocess) {
    using ::testing::_;
    using ::testing::Return;
    auto& mock = mock_conv();

    MockAlgorithm algo;
    EXPECT_CALL(mock, get_algorithm_heuristic(_, _, _, _, _))
            .WillRepeatedly(Return(&algo));
    EXPECT_CALL(mock, get_workspace_in_bytes(_, _, _, _))
            .WillRepeatedly(Return(0));
    EXPECT_CALL(mock, get_preprocess_workspace_in_bytes(_, _, _))
            .WillRepeatedly(Return(0));

    {
        ::testing::InSequence seq;
        // Return empty preprocess filters, indicating no need to preprocess
        EXPECT_CALL(mock, deduce_preprocessed_filter_layout(_, _, _)).Times(0);
        EXPECT_CALL(mock, exec_preprocess(_, _, _, _, _)).Times(0);
        EXPECT_CALL(mock, exec(_, _, _, nullptr, _));
        run();
    }
}

}  // anonymous namespace

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
