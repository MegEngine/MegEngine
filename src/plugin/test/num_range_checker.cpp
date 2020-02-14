/**
 * \file src/plugin/test/num_range_checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/loop.h"
#include "megbrain/plugin/num_range_checker.h"
#include "megbrain/test/helper.h"

using namespace mgb;

TEST(TestNumRangeChecker, Simple) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    NumRangeChecker checker{graph.get(), 1e30f};
    auto av = gen({3}), bv = gen({3});
    auto a = opr::Host2DeviceCopy::make(*graph, av),
         b = opr::Host2DeviceCopy::make(*graph, bv),
         c = a / b;
    auto func = graph->compile({{c, {}}});
    auto pb = bv->ptr<float>();
    pb[0] = 2; pb[1] = -1; pb[2] = 3;
    func->execute();
    pb[1] = 0;
    ASSERT_THROW(func->execute(), NumRangeChecker::Error);
}

TEST(TestNumRangeChecker, MultiDType) {
    HostTensorGenerator<dtype::Int32> gen;
    auto graph = ComputingGraph::make();
    NumRangeChecker checker{graph.get(), 1e30f};
    auto av = gen({3});
    auto a = opr::Host2DeviceCopy::make(*graph, av),
         b = a + a,
         c = opr::TypeCvt::make(b, dtype::Float32());
    auto func = graph->compile({{c, {}}});
    func->execute();
}

TEST(TestNumRangeChecker, MultiShape) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    NumRangeChecker checker{graph.get(), 1e30f};
    auto av = gen({1, 3}), bv = gen({3, 1});
    auto a = opr::Host2DeviceCopy::make(*graph, av),
         b = opr::Host2DeviceCopy::make(*graph, bv),
         c = (a + 2) / (b - 4);
    auto func = graph->compile({{c, {}}});
    auto pb = bv->ptr<float>();
    pb[0] = 2; pb[1] = -1; pb[2] = 3;
    func->execute();
    pb[2] = 4;
    ASSERT_THROW(func->execute(), NumRangeChecker::Error);
}

TEST(TestNumRangeChecker, Loop) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    NumRangeChecker checker{graph.get(), 1e30f};
    auto av = gen({3}), bv = gen({3});
    auto a = opr::Host2DeviceCopy::make(*graph, av),
         b = opr::Host2DeviceCopy::make(*graph, bv);
    auto loop_cb = [&](opr::Loop::Desc &desc) {
        auto ai = desc.add_input(a),
             bi = desc.add_input(b);
        desc.set_loop_condition(desc.get_counter_var() < 0);
        auto out = ai + bi;
        desc.add_output(out, opr::Loop::Desc::OutputMode::LAST);
        out.node()->owner_graph()->options().extra_vardeps[
            out.node()].push_back((ai / bi).node());
    };
    auto c = opr::Loop::make(loop_cb)[0];
    HostTensorND host_c;
    auto func = graph->compile({make_callback_copy(c, host_c)});
    auto pb = bv->ptr<float>();
    pb[0] = 2; pb[1] = -1; pb[2] = 3;
    func->execute();
    pb[1] = 0;
    ASSERT_THROW(func->execute(), NumRangeChecker::Error);
}

TEST(TestNumRangeChecker, MultiStreamDyn) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    NumRangeChecker checker{graph.get(), 1e30f};
    auto xv = gen({3}, cns[0]);
    auto x = opr::Host2DeviceCopy::make(*graph, xv),
         y = opr::Copy::make(x, cns[1]);
    auto func = graph->compile({{y, {}}});
    func->execute();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
