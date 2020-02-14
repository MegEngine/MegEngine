/**
 * \file src/opr/test/loop/grad_trans.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/test/autocheck.h"

#include "megbrain/opr/io.h"
#include "megbrain/opr/loop.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/gopt/framework.h"

using namespace mgb;

using LoopDesc = opr::Loop::Desc;
using OutputMode = opr::Loop::Desc::OutputMode;

TEST(TestOprLoop, SetGrad) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({23});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    VarNode *xi_chg_grad = nullptr;
    auto grad_getter = [&](const opr::SetGrad &set_grad) {
        auto &&graph = *set_grad.owner_graph();
        auto trans = cg::InterGraphVarTransformer::get(graph);
        mgb_assert(trans);
        auto grad = cg::grad(cg::current_grad_target(graph),
                trans->trans(xi_chg_grad), false);
        grad = grad * 2;
        return grad.node();
    };
    auto desc_maker = [&](LoopDesc &desc) {
        auto xi = opr::SetGrad::make(desc.add_input(x), grad_getter);
        xi_chg_grad = xi.node();
        desc.add_output(xi, OutputMode::SUM);
        desc.set_loop_condition(xi.make_scalar(0));
    };
    auto y = opr::Loop::make(desc_maker)[0];
    HostTensorND host_gx, host_gx_expect;
    auto func = graph->compile({
            make_callback_copy(
                    cg::grad(opr::reduce_sum_sqr(y, y.make_scalar(1)), x),
                    host_gx),
            make_callback_copy(x * 4, host_gx_expect)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_gx_expect, host_gx);
}

TEST(TestOprLoop, SetGradGOpt) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({23});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    VarNode *grad_wrt = nullptr;
    auto grad_getter = [&](const opr::SetGrad &set_grad) {
        auto &&graph = *set_grad.owner_graph();
        auto trans = cg::InterGraphVarTransformer::get(graph);
        mgb_assert(trans);
        auto wrt = gopt::GraphOptimizer::var_replace_lookup(grad_wrt);
        wrt = trans->trans(wrt);
        auto grad = cg::grad(cg::current_grad_target(graph), wrt, false);
        return grad.node();
    };
    auto desc_maker = [&](LoopDesc &desc) {
        auto xi = opr::SetGrad::make(desc.add_input(x) * 2 + .5f, grad_getter);
        grad_wrt = (xi * 3 + 1).node();
        desc.add_output(grad_wrt, OutputMode::SUM);
        desc.set_loop_condition(xi.make_scalar(0));
    };
    auto y = opr::Loop::make(desc_maker)[0];
    HostTensorND host_gx, host_gx_expect;
    auto func = graph->compile({
            make_callback_copy(
                    cg::grad(opr::reduce_sum(y, y.make_scalar(1)), x),
                    host_gx),
            make_callback_copy(x.fill_retain_dtype(2), host_gx_expect)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_gx_expect, host_gx);
}

TEST(TestOprLoop, SetGradGOptLoopCopy) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({23});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    VarNode *grad_wrt = nullptr;
    auto grad_getter = [&](const opr::SetGrad &set_grad) {
        auto &&graph = *set_grad.owner_graph();
        auto trans = cg::InterGraphVarTransformer::get(graph);
        mgb_assert(trans);
        auto wrt = gopt::GraphOptimizer::var_replace_lookup(grad_wrt);
        wrt = trans->trans(wrt);
        auto grad = cg::grad(cg::current_grad_target(graph), wrt, false);
        return grad.node();
    };
    auto desc_maker = [&](LoopDesc &desc) {
        auto xo = x * .5f + 1;
        auto xi = opr::SetGrad::make(desc.add_input(xo) * 2 + .5f, grad_getter);
        grad_wrt = (xi * 3 + 1).node();
        desc.add_output(grad_wrt, OutputMode::SUM);
        desc.set_loop_condition(xi.make_scalar(0));
    };
    auto y = opr::Loop::make(desc_maker)[0];
    HostTensorND host_gx, host_gx_expect;
    auto func = graph->compile({
            make_callback_copy(
                    cg::grad(opr::reduce_sum(y, y.make_scalar(1)), x),
                    host_gx),
            make_callback_copy(x.fill_retain_dtype(1), host_gx_expect)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_gx_expect, host_gx);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

