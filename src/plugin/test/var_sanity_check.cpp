/**
 * \file src/plugin/test/var_sanity_check.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"

#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/plugin/var_sanity_check.h"

using namespace mgb;

TEST(TestVarSanityCheck, Simple) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1024}),
         host_y = gen({1024});
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::Host2DeviceCopy::make(*graph, host_x),
        y = opr::Host2DeviceCopy::make(*graph, host_y),
        y1 = y.reshape({1024, 1}),
        z = x + y1.reshape({1024});

    bool should_change = false;
    ComputingGraph::OutputSpec out_spec = {
        {y1, [&](DeviceTensorND &v){
            if (should_change) {
                HostTensorND hv;
                hv.copy_from(v).sync().ptr<float>()[123] ++;
                v.copy_from(hv);
            }
        }},
        {z, [&](DeviceTensorND &v){
            HostTensorND hv;
            hv.copy_from(v).sync();
            for (int i = 0; i < 1024; i ++) {
                ASSERT_EQ(host_x->ptr<float>()[i] + host_y->ptr<float>()[i],
                        hv.ptr<float>()[i]) << "failed at " << i;;
            }
        }}
    };
    auto func = graph->compile(out_spec);
    func->execute().wait();
    func = graph->compile(out_spec);
    func->execute().wait();

    should_change = true;
    func = graph->compile(out_spec);
    ASSERT_THROW(func->execute().wait(),
            VarSanityCheck::Error);
}

TEST(TestVarSanityCheck, InputModify) {
    HostTensorGenerator<> gen;
    auto host_x = gen({333});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    SymbolVar y;
    auto cb = [&y](DeviceTensorND &) {
        auto &&dv = y.node()->owner_opr()->input(0)->dev_tensor();
        HostTensorND hv;
        hv.copy_from(dv).sync().ptr<float>()[23] ++;
        dv.copy_from_fixlayout(hv).sync();
    };
    y = opr::CallbackInjector::make(x, cb);
    graph->options().seq_opt.enable_mem_plan_opt = false;
    auto func = graph->compile({{y, {}}});
    ASSERT_THROW(func->execute(), VarSanityCheck::Error);
}

TEST(TestVarSanityCheck, AddUpdateWithMultiCN) {
    HostTensorGenerator<> gen;
    auto host_x = gen({123}),
         host_delta = gen({123});
    auto comp_node0 = CompNode::load("xpu0:0");
    auto comp_node1 = CompNode::load("xpu0:1");
    auto graph = ComputingGraph::make();
    auto x = opr::SharedDeviceTensor::make(*graph, *host_x, {comp_node0});
    auto delta = opr::ImmutableTensor::make(*graph, *host_delta, {comp_node1});
    auto x_new = opr::AddUpdate::make(x, delta, {}, {comp_node1});
    auto on_exec_start = [&comp_node0](const cg::event::OprExecKernelStart& event) {
        auto &&comp_node = event.opr->output(0)->comp_node();
        if (comp_node == comp_node0) {
            auto cb = []{
                using namespace std::literals;
                std::this_thread::sleep_for(50ms);
            };
            event.env->dispatch_on_comp_node(comp_node, cb);
        }
    };
    auto handle = graph->event().register_receiver<cg::event::OprExecKernelStart>(
            on_exec_start);
    auto func = graph->compile({{x_new, {}}});
    ASSERT_NO_THROW(func->execute().wait());
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

