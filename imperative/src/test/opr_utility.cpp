/**
 * \file imperative/src/test/opr_utility.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/opr_utility.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/test/helper.h"

using namespace mgb;
using namespace opr;

TEST(TestOprUtility, InputCallback) {
    HostTensorGenerator<> gen;
    DeviceTensorND dv;
    auto hv = gen({2, 3});
    dv.copy_from(*hv).sync();
    auto graph = ComputingGraph::make();
    auto callback = [dv]() {return dv;};
    auto outputs = opr::InputCallback::make(*graph, callback, dv.comp_node(), dv.dtype(), {2, 3});

    HostTensorND hout;
    ComputingGraph::OutputSpec outspec{make_callback_copy(outputs[0], hout)};
    auto func = graph->compile(outspec);
    func->execute();
    MGB_ASSERT_TENSOR_EQ(hout, *hv);
}

TEST(TestOprUtility, OutputCallback) {
    HostTensorGenerator<> gen;
    auto hx = gen({2, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, hx);
    HostTensorND hy;
    auto callback = [&hy](DeviceTensorND dv) {hy.copy_from(dv);};
    auto dummy = opr::OutputCallback::make({callback}, x);
    auto y = opr::VirtualDep::make({x, dummy});

    ComputingGraph::OutputSpec outspec{{y, [](DeviceTensorND&){}}};
    auto func = graph->compile(outspec);
    func->execute();
    MGB_ASSERT_TENSOR_EQ(hy, *hx);
}

TEST(TestOprUtility, OutputCallbackPreferHost) {
    HostTensorGenerator<> gen;
    auto hx = gen({2, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, hx);
    x = opr::GetVarShape::make(x);
    HostTensorND hy;
    auto callback = [&hy](DeviceTensorND dv) {hy.copy_from(dv);};
    opr::OutputCallback::Param param{callback};
    param.prefer_host_value = true;
    auto dummy = opr::OutputCallback::make(param, x);
    auto y = opr::VirtualDep::make({x, dummy});

    ComputingGraph::OutputSpec outspec{{y, [](DeviceTensorND&){}}};
    auto func = graph->compile(outspec);
    func->execute();
    ASSERT_TRUE(hy.comp_node() == CompNode::default_cpu());
    ASSERT_EQ(hy.ptr<int>()[0], 2);
    ASSERT_EQ(hy.ptr<int>()[1], 3);
}

TEST(TestOprUtility, NopCallback) {
    HostTensorGenerator<> gen;
    auto hx = gen({2, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, hx);
    bool fired = false;
    auto callback = [&fired]() {fired = true;};
    auto dummy = opr::NopCallback::make(*graph, callback, x.node()->comp_node(), {x});
    auto y = opr::VirtualDep::make({x, dummy});

    ComputingGraph::OutputSpec outspec{{y, [](DeviceTensorND&){}}};
    auto func = graph->compile(outspec);
    func->execute();
    ASSERT_TRUE(fired);
}

TEST(TestOprUtility, NopCallbackMixedInput) {
    auto graph = ComputingGraph::make();
    auto x0 = opr::Host2DeviceCopy::make(*graph, HostTensorGenerator<dtype::Int32>()({2, 3}), OperatorNodeConfig(CompNode::load("xpu0")));
    auto x1 = opr::Host2DeviceCopy::make(*graph, HostTensorGenerator<dtype::Float32>()({2, 3}), OperatorNodeConfig(CompNode::load("xpu1")));

    bool fired = false;
    auto callback = [&fired]() {fired = true;};
    auto dummy = opr::NopCallback::make(*graph, callback, CompNode::load("xpux"), {x0, x1});
    auto y = opr::VirtualDep::make({x0, dummy});

    ComputingGraph::OutputSpec outspec{{y, [](DeviceTensorND&){}}};
    auto func = graph->compile(outspec);
    func->execute();
    ASSERT_TRUE(fired);
}

TEST(TestOprUtility, CallbackChain) {
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    HostTensorGenerator<dtype::Int16> gen;
    SymbolVar x, dummy;
    DeviceTensorND dev_x, dev_y;
    auto host_x = gen({2, 3});
    dev_x.copy_from(*host_x).sync();
    auto cn = dev_x.comp_node();
    auto dev_x_weakptr = std::weak_ptr<dt_byte>(dev_x.storage().raw_storage());

    {
        auto callback = [&dev_x]() {
            DeviceTensorND ret = dev_x;
            dev_x.storage({});
            return ret;
        };
        auto out = opr::InputCallback::make(*graph, callback, cn, dev_x.dtype(), {2, 3});
        x = out[0];
        dummy = out[1];
    }

    {
        x = opr::TypeCvt::make(x, dtype::Int32());
        x = opr::TypeCvt::make(x, dtype::Int16());
        auto callback = [&](DeviceTensorND y) {
            // dev_x.storage has been reset in InputCallback
            mgb_assert(!dev_x.storage().comp_node_valid());
            dev_y = y;
        };
        dummy = opr::OutputCallback::make({callback}, {x, dummy});
    }

    bool fired = false;
    {
        auto callback = [&]() {
            fired = true;
            ASSERT_FALSE(dev_x_weakptr.lock());
        };
        dummy = opr::NopCallback::make(*graph, callback, cn, {dummy});
    }

    {
        auto out = opr::VirtualDep::make({x.make_scalar(0), dummy});
        ComputingGraph::OutputSpec outspec{{out, [](DeviceTensorND&){}}};
        auto func = graph->compile(outspec);
        func->execute();
    }

    ASSERT_TRUE(fired);
    HostTensorND host_y;
    host_y.copy_from(dev_y).sync();
    MGB_ASSERT_TENSOR_EQ(host_y, *host_x);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
