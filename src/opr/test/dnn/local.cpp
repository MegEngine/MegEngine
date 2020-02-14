/**
 * \file src/opr/test/dnn/local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./legacy_checker.h"

#include "megbrain/opr/dnn/local.h"
#include "megbrain/test/autocheck.h"

using namespace mgb;

namespace {

using Param = opr::Local::Param;
using Mode = Param::Mode;

Mode modes_to_check[] = {Mode::CONVOLUTION, Mode::CROSS_CORRELATION};

void local_brute(const std::vector<std::shared_ptr<HostTensorND>> &in_tensor,
        std::shared_ptr<HostTensorND> &out_tensor, const Param &param)
{
    ASSERT_EQ(2u, in_tensor.size());
    auto in = in_tensor[0], filter = in_tensor[1];
    ASSERT_EQ(4u, in->shape().ndim);
    ASSERT_EQ(6u, filter->shape().ndim);

    int batch_size = in->shape().shape[0];
    int ic = in->shape().shape[1];
    int ih = in->shape().shape[2];
    int iw = in->shape().shape[3];

    int fh = filter->shape().shape[3];
    int fw = filter->shape().shape[4];

    int ph = param.pad_h;
    int pw = param.pad_w;

    int sh = param.stride_h;
    int sw = param.stride_w;
    ASSERT_GE(ih + 2*ph, fh);
    ASSERT_GE(iw + 2*pw, fw);
    int oh = (ih + 2*ph - fh) / sh + 1;
    int ow = (iw + 2*pw - fw) / sw + 1;
    ASSERT_EQ(static_cast<size_t>(ic), filter->shape().shape[2]);
    int oc = filter->shape().shape[5];


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
                prih = pih + fh - pfh - 1;
                priw = piw + fw - pfw - 1;
            } else {
                mgb_assert(param.mode == Param::Mode::CROSS_CORRELATION);
                prih = pih + pfh;
                priw = piw + pfw;
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
                    static_cast<size_t>(poh),
                    static_cast<size_t>(pow),
                    static_cast<size_t>(pic),
                    static_cast<size_t>(pfh),
                    static_cast<size_t>(pfw),
                    static_cast<size_t>(poc)})[0];
            target += img_data * filter_data;
        }
    }
}

} // anonymous namespace

TEST(TestOprDNN, LocalForward) {
    uint32_t ih = 10, ic = 16, oc = 32, ph = 0, sh = 1, fh = 2;
    for (auto mode: modes_to_check) {
        uint32_t iw = ih, fw = fh, pw = ph, sw = sh;
        uint32_t oh = (ih+2*ph-fh)/sh+1, ow = (iw+2*pw-fw)/sw+1;
        Param param{mode, ph, pw, sh, sw};
        size_t batch_size = 32;
        opr::test::ForwardChecker<opr::Local, 2> forward_checker({
                {batch_size, ic, ih, iw},
                {oh, ow, ic, fh, fw, oc}},
                local_brute, param);
        forward_checker.run();
    }
}

TEST(TestOprDNN, LocalBackward)
{
    uint32_t ih = 10, ic = 16, oc = 32, ph = 0, sh = 1, fh = 2;
    uint32_t iw = ih, fw = fh, pw = ph, sw = sh;
    uint32_t oh = (ih+2*ph-fh)/sh+1, ow = (iw+2*pw-fw)/sw+1;
    Param param{Mode::CROSS_CORRELATION, ph, pw, sh, sw};
    size_t batch_size = 32;
    opr::test::BackwardChecker<opr::Local, 2> backward_checker({
            {batch_size, ic, ih, iw},
            {oh, ow, ic, fh, fw, oc}}, param, 1e-2, 1);
    backward_checker.run();
}

TEST(TestOprDNN, GroupLocal) {
    using Checker = AutoOprChecker<2, 1>;
    opr::GroupLocal::Param param;
    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        return {opr::GroupLocal::make(inputs[0], inputs[1], param)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto &&out = dest[0];
        auto inp0 = std::make_shared<HostTensorND>(),
             inp1 = std::make_shared<HostTensorND>();
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
            std::shared_ptr<HostTensorND> cur_out;
            local_brute({inp0, inp1}, cur_out, {});
            if (!i) {
                auto oshp = cur_out->shape();
                oshp[1] *= group;
                out.resize(oshp);
                ol = out.layout();
                ol[1] /= group;
            }
            out.sub(SubTensorSpec::make_from_offset_elem(
                        ol, i * ol[1] * ol[2] * ol[3])).copy_from_fixlayout(
                        *cur_out);
        }
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    opt.outputs_max_err = 5e-5;
    Checker checker{make_graph, fwd};
    auto run = [&](const TensorShape &ishp,
            size_t fs, size_t oc, size_t group) {
        size_t ic = ishp[1], ih = ishp[2], iw = ishp[3];
        TensorShape flt{group, ih-fs+1, iw-fs+1, ic/group, fs, fs, oc/group};
        checker.run({ishp, flt}, opt);
    };
    run({32, 9, 2, 2}, 1, 96, 3);
    run({32, 4, 2, 3}, 2, 32, 2);
    run({32, 3, 4, 3}, 3, 16, 1);
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

