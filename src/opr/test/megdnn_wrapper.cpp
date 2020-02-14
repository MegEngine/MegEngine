/**
 * \file src/opr/test/megdnn_wrapper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"

#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/utility.h"
#include "megbrain/tensor.h"

using namespace mgb;

TEST(TestOprMegDNNWrapper, Stream) {
    using Param = opr::Convolution::Param;
    Param param;
    HostTensorGenerator<> gen;
    auto host_x = gen({8, 1, 20, 20}),
         host_kern = gen({3, 1, 4, 4});
    HostTensorND host_y_expect;
    {
        // gen host_y_expect
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             kern = opr::Host2DeviceCopy::make(*graph, host_kern),
             y = opr::Convolution::make(x, kern, param);
        auto func = graph->compile({make_callback_copy(y, host_y_expect)});
        func->execute();
    }
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         kern = opr::Host2DeviceCopy::make(*graph, host_kern),
         y = opr::Convolution::make(x, kern, param);
    // change stream
    auto chg = [](SymbolVar var) {
        auto opr = var.node()->owner_opr();
        for (auto i: opr->output())
            i->comp_node(CompNode::load("xpu0:1"));
        opr->on_output_comp_node_stream_changed();
    };
    chg(x);
    chg(kern);
    chg(y);
    HostTensorND host_y;

    opr::Sleep::sleep(host_x->comp_node(), 0.5);
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y);
}

TEST(TestOprMegDNNWrapper, ShapeDep) {
    using Param = opr::Convolution::Param;
    Param param;
    HostTensorGenerator<> gen;
    auto host_x = gen({8, 1, 20, 20}),
         host_kern = gen({3, 1, 4, 4});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         kern = opr::Host2DeviceCopy::make(*graph, host_kern),
         y = opr::Convolution::make(x, kern, param),
         gk = cg::grad(opr::reduce_sum(y, y.make_scalar(1)), kern);
    using NP = cg::OperatorNodeBase::NodeProp;
    auto dt = gk.node()->owner_opr()->node_prop().dep_map().at(kern.node());
    ASSERT_EQ(NP::DepType::SHAPE, dt);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
