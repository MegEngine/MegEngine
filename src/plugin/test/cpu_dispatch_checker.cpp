/**
 * \file src/plugin/test/cpu_dispatch_checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/opr/loop.h"
#include "megbrain/opr/utility.h"
#include "megbrain/plugin/cpu_dispatch_checker.h"
#include "megbrain/test/helper.h"

using namespace mgb;

TEST(TestCPUDispatchChecker, Simple) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    CPUDispatchChecker checker(graph.get());
    auto host_x = gen({3}, CompNode::load("cpux"));
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::CallbackInjector::make(x, [](DeviceTensorND&){}),
         z = y + 1;
    auto func = graph->compile({{z, {}}});
    func->execute();
    ASSERT_EQ(1u, checker.failed_oprs().count(y.node()->owner_opr()));
}

TEST(TestCPUDispatchChecker, Loop) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    CPUDispatchChecker checker(graph.get());
    auto host_x = gen({3}, CompNode::load("cpux"));
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    SymbolVar y;
    auto loop_cb = [&](opr::Loop::Desc &desc) {
        auto xi = desc.add_input(x);
        desc.set_loop_condition(xi.make_scalar(0));
        y = opr::CallbackInjector::make(xi, [](DeviceTensorND&){});
        desc.add_output(y + 1, opr::Loop::Desc::OutputMode::LAST);
    };
    auto z = opr::Loop::make(loop_cb)[0];
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    ASSERT_EQ(1u, checker.failed_oprs().count(y.node()->owner_opr()));
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
