/**
 * \file src/opr/test/dnn/lrn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./legacy_checker.h"
#include "megbrain/opr/dnn/lrn.h"

using namespace std;
using namespace mgb;

namespace {

using Param = opr::LRNForward::Param;

void lrn_brute(const vector<shared_ptr<HostTensorND>> &in_tensor,
     shared_ptr<HostTensorND> &out_tensor, const Param &param)
{
    ASSERT_EQ(1u, in_tensor.size());
    ASSERT_EQ(4u, in_tensor[0]->shape().ndim);
    size_t n = in_tensor[0]->shape().shape[0];
    size_t c = in_tensor[0]->shape().shape[1];
    size_t h = in_tensor[0]->shape().shape[2];
    size_t w = in_tensor[0]->shape().shape[3];
    int window_size = static_cast<int>(param.n);
    out_tensor = make_shared<HostTensorND>(CompNode::load("xpu0"),
            TensorShape{n, c, h, w});
    for (size_t in = 0; in < n; ++in)
    for (size_t ih = 0; ih < h; ++ih)
    for (size_t iw = 0; iw < w; ++iw)
    for (int ic = 0; ic < static_cast<int>(c); ++ic)
    {
        float ori = in_tensor[0]->ptr<float>({in, static_cast<size_t>(ic),
                ih, iw})[0];
        float &res = out_tensor->ptr<float>({in, static_cast<size_t>(ic),
                ih, iw})[0];
        int offset = (window_size - 1) / 2;
        int from = max(0, ic - offset);
        int to = min(static_cast<int>(c), ic + window_size - offset);
        float sum = 0;
        for (int jc = from; jc < to; ++jc) {
            float here = in_tensor[0]->ptr<float>({in,
                    static_cast<size_t>(jc), ih, iw})[0];
            sum += here * here;
        }
        sum *= param.alpha;
        sum += 1.0f;
        sum = exp(log(sum) * param.beta);
        res = ori / sum;
    }
}

TEST(TestOprDNN, LRNForward)
{
    for (size_t window_size = 1; window_size < 10; window_size += 2)
    for (float alpha = 100; alpha <= 100; alpha *= 2)
    for (float beta = 0.5; beta <= 0.5; beta *= 2)
    {
        Param param(window_size, 1.0f, alpha, beta);
        opr::test::ForwardChecker<opr::LRNForward, 1> forward_checker(
                {{10, 9, 8, 7}}, lrn_brute, param);
        forward_checker.run();
    }
}

TEST(TestOprDNN, LRNBackward)
{
    for (size_t window_size = 1; window_size < 10; window_size += 2)
    for (float alpha = 100; alpha <= 100; alpha *= 2)
    for (float beta = 0.5; beta <= 0.5; beta *= 2)
    {
        Param param(window_size, 1.0f, alpha, beta);
        opr::test::BackwardChecker<opr::LRNForward, 1> backward_checker(
                {{10, 9, 8, 7}}, param, 1e-1, 1e-2);
        backward_checker.run();
    }
}

} // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

