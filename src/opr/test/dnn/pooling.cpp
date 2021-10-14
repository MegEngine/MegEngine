/**
 * \file src/opr/test/dnn/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/pooling.h"
#include "./legacy_checker.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/utils/persistent_cache.h"

using namespace std;
using namespace mgb;
namespace {

using Param = opr::Pooling::Param;
using Mode = Param::Mode;

void pooling_brute(
        const vector<shared_ptr<HostTensorND>>& in_tensor,
        shared_ptr<HostTensorND>& out_tensor, const Param& param) {
    ASSERT_EQ(1u, in_tensor.size());
    ASSERT_EQ(4u, in_tensor[0]->shape().ndim);
    size_t n = in_tensor[0]->shape().shape[0];
    size_t c = in_tensor[0]->shape().shape[1];
    size_t ih = in_tensor[0]->shape().shape[2];
    size_t iw = in_tensor[0]->shape().shape[3];
    size_t oh = (ih + 2 * param.pad_h - param.window_h) / param.stride_h + 1;
    size_t ow = (iw + 2 * param.pad_w - param.window_w) / param.stride_w + 1;
    out_tensor = make_shared<HostTensorND>(
            CompNode::load("xpu0"), TensorShape{n, c, oh, ow});
    int fx, fy;
    size_t tx, ty;
    for (size_t on = 0; on < n; ++on)
        for (size_t oc = 0; oc < c; ++oc)
            for (tx = 0, fx = -param.pad_h; tx < oh; ++tx, fx += param.stride_h)
                for (ty = 0, fy = -param.pad_w; ty < ow; ++ty, fy += param.stride_w) {
                    float& cur = out_tensor->ptr<float>({on, oc, tx, ty})[0];
                    bool valid = false;
                    if (param.mode == Param::Mode::AVERAGE ||
                        param.mode == Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING) {
                        cur = 0;
                        int fx2 = fx + static_cast<int>(param.window_h);
                        int fy2 = fy + static_cast<int>(param.window_w);
                        int cnt = 0;
                        for (int rx = fx; rx < fx2; ++rx)
                            for (int ry = fy; ry < fy2; ++ry)
                                if (rx >= 0 && rx < static_cast<int>(ih) && ry >= 0 &&
                                    ry < static_cast<int>(iw)) {
                                    cur += in_tensor[0]->ptr<float>(
                                            {on, oc, static_cast<size_t>(rx),
                                             static_cast<size_t>(ry)})[0];
                                    valid = true;
                                    ++cnt;
                                }
                        if (param.mode == Param::Mode::AVERAGE) {
                            cnt = param.window_h * param.window_w;
                        }
                        cur /= static_cast<float>(cnt);
                    } else {
                        cur = -numeric_limits<float>::max();
                        ASSERT_EQ(Param::Mode::MAX, param.mode);
                        int fx2 = fx + static_cast<int>(param.window_h);
                        int fy2 = fy + static_cast<int>(param.window_w);
                        for (int rx = fx; rx < fx2; ++rx)
                            for (int ry = fy; ry < fy2; ++ry)
                                if (rx >= 0 && rx < static_cast<int>(ih) && ry >= 0 &&
                                    ry < static_cast<int>(iw)) {
                                    cur = std::max(
                                            cur,
                                            in_tensor[0]->ptr<float>(
                                                    {on, oc, static_cast<size_t>(rx),
                                                     static_cast<size_t>(ry)})[0]);
                                    valid = true;
                                }
                    }
                    mgb_assert(valid);
                }
}

TEST(TestOprDNN, PoolingForward) {
    size_t sx = 2, sy = 3, wx = 4, wy = 2, ix = 23, iy = 15, ph = 0, pw = 3;
    for (uint32_t i = 0; i < Param::MODE_NR_MEMBER; ++i) {
        Param param(static_cast<Mode>(i), ph, pw, sy, sx, wy, wx);
        opr::test::ForwardChecker<opr::Pooling, 1> forward_checker(
                {{2, 3, ix, iy}}, pooling_brute, param);
        forward_checker.run();
    }
}

TEST(TestOprDNN, PoolingBackward) {
    size_t sx = 2, sy = 3, wx = 3, wy = 2, ix = 23, iy = 15, ph = 1, pw = 1;
    for (uint32_t i = 0; i < Param::MODE_NR_MEMBER; ++i) {
        Param param(static_cast<Mode>(i), ph, pw, sy, sx, wy, wx);
        opr::test::BackwardChecker<opr::Pooling, 1> backward_checker(
                {{2, 3, ix, iy}}, param, 1e-2, 1e-2, false);
        backward_checker.run();
    }
}

TEST(TestOprDNN, PoolingExePolicy) {
    using Param = opr::Pooling::Param;
    Param param;
    using Policy = opr::Pooling::ExecutionPolicy;
    using S = Policy::Strategy;

    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();

    auto orig_impl =
            PersistentCache::set_impl(std::make_shared<InMemoryPersistentCache>());

    HostTensorND host_y, host_y_copy;
    S strategy = S::HEURISTIC | S::REPRODUCIBLE;

    auto graph = ComputingGraph::make();

    HostTensorGenerator<> gen;
    TensorShape shape = {1, 20, 24, 24};
    auto input = opr::Host2DeviceCopy::make(*graph, gen(shape, cn));

    param.mode = Param::Mode::MAX;
    param.window_h = param.window_w = 2;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 0;
    param.format = Param::Format::NCHW;

    Policy policy;
    policy.strategy = strategy;

    auto pooling = opr::PoolingForward::make(input, param, {}, policy);

    auto loss0 = opr::reduce_sum_sqr(pooling, pooling.make_scalar(1));
    auto grad = cg::grad(loss0, input, true, false);

    opr::PoolingBackward* found = nullptr;
    auto cb = [&found](cg::OperatorNodeBase* opr) {
        if (opr->same_type<opr::PoolingBackward>()) {
            found = &opr->cast_final_safe<opr::PoolingBackward>();
        }
    };
    cg::DepOprIter{cb}.add(grad.node()->owner_opr());
    found->set_execution_policy(strategy);

    auto func = graph->compile({make_callback_copy(grad, host_y)});
    func->execute().wait();

    mgb_assert(
            found->megdnn_opr()->execution_policy().algo.name.find(
                    "cudnnReproducible") != std::string::npos);
}

TEST(TestOprDNN, PoolingForwardFastrun) {
    using Param = opr::Pooling::Param;
    Param param;
    using Policy = opr::Pooling::ExecutionPolicy;
    using S = Policy::Strategy;

    auto cn = CompNode::load("xpux");
    cn.activate();

    auto orig_impl =
            PersistentCache::set_impl(std::make_shared<InMemoryPersistentCache>());

    HostTensorND host_y;
    S strategy = S::PROFILE | S::REPRODUCIBLE;

    auto graph = ComputingGraph::make();

    HostTensorGenerator<> gen;
    TensorShape shape = {1, 20, 24, 24};
    auto input = opr::Host2DeviceCopy::make(*graph, gen(shape, cn));

    param.mode = Param::Mode::MAX;
    param.window_h = param.window_w = 2;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 0;
    param.format = Param::Format::NCHW;

    Policy policy;
    policy.strategy = strategy;

    auto pooling = opr::PoolingForward::make(input, param, {}, policy);

    auto func = graph->compile({make_callback_copy(pooling, host_y)});
    func->execute().wait();
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
