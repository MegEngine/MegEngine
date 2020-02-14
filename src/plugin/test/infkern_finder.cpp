/**
 * \file src/plugin/test/infkern_finder.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/plugin/infkern_finder.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/loop.h"
#include <thread>

#include <atomic>

using namespace mgb;

namespace {

class TestInfkernFinder: public ::testing::Test {
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x;
    std::shared_ptr<ComputingGraph> graph;
    std::unique_ptr<InfkernFinder> finder;
    std::atomic_bool should_sleep, entered;
    cg::OperatorNodeBase *expected_bad_opr = nullptr;

    protected:
        SymbolVar x;

        auto make_callback(SymbolVar x) {
            auto cb = [&](DeviceTensorND &dv) {
                dv.comp_node().sync();
                entered.store(true);
                while (should_sleep.load());
                entered.store(false);
            };
            auto cbx = opr::CallbackInjector::make(x, cb).rename("cbx"),
                 y = (cbx * 23).rename("cby");
            expected_bad_opr = cbx.node()->owner_opr();
            return y;
        }

        void SetUp() override {
            host_x = gen({1});
            graph = ComputingGraph::make();
            finder = std::make_unique<InfkernFinder>(graph.get(), true);
            should_sleep = false;
            entered = false;
            x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x");
        }

        void run(SymbolVar y, int tid) {
            y.rename("y");
            auto outfile = [&](int id) {
                return output_file(ssprintf("InfkernFinder%d%d.txt", tid, id));
            };
            HostTensorND host_y;
            auto func = graph->compile({make_callback_copy(y, host_y)});
            HostTensorND expected_y;
            expected_y.copy_from(*host_x).ptr<float>()[0] *= 23;

            func->execute().wait();
            MGB_ASSERT_TENSOR_EQ(host_y, expected_y);

            host_y.ptr<float>()[0] ++; // mark output invalid
            func->execute().wait();
            MGB_ASSERT_TENSOR_EQ(host_y, expected_y);

            ASSERT_EQ(nullptr, finder->write_to_file(outfile(0).c_str()));

            cg::OperatorNodeBase* bad_opr = nullptr;
            auto worker = [&]() {
                while(!entered.load());
                bad_opr = finder->write_to_file(outfile(1).c_str());
                should_sleep.store(false);
            };
            host_y.ptr<float>()[0] ++; // mark output invalid
            should_sleep.store(true);
            std::thread thread(worker);
            func->execute().wait();
            thread.join();
            MGB_ASSERT_TENSOR_EQ(host_y, expected_y);
            ASSERT_EQ(expected_bad_opr, bad_opr) << ssprintf(
                    "get opr: %s{%s}", bad_opr->cname(),
                    bad_opr->dyn_typeinfo()->name);
            ASSERT_FALSE(finder->get_input_values(
                        expected_bad_opr->id()).empty());
        }
};

} // anonymous namespace

TEST_F(TestInfkernFinder, Normal) {
    run(make_callback(x), 0);
}

TEST_F(TestInfkernFinder, UnusedVar) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 3});
    auto cg = ComputingGraph::make();
    cg->options().graph_opt_level = 0;
    InfkernFinder finder{cg.get(), true};
    auto x = opr::Host2DeviceCopy::make(*cg, host_x),
         tshp = x.make_scalar(3),
         xrshp = x.reshape(tshp),
         y = xrshp + 2;
    HostTensorND host_y;
    auto func = cg->compile({make_callback_copy(y, host_y)});
    func->execute().wait();
    auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
    for (size_t i = 0; i < 3; ++ i) {
        ASSERT_FLOAT_EQ(px[i] + 2, py[i]);
    }
    auto val = finder.get_input_values(y.node()->owner_opr()->id());
    if (val[0].first != xrshp.node())
        std::swap(val[0], val[1]);
    ASSERT_EQ(val[0].first, xrshp.node());
    ASSERT_EQ(val[0].second.val.ptr<float>()[0], host_x->ptr<float>()[0]);
    ASSERT_THROW(finder.get_input_values(tshp.node()->owner_opr()->id()),
            MegBrainError);
}

TEST_F(TestInfkernFinder, MultiCompile) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 3});
    auto cg = ComputingGraph::make();
    cg->options().graph_opt_level = 0;
    InfkernFinder finder{cg.get(), true};
    auto x = opr::Host2DeviceCopy::make(*cg, host_x),
         y0 = x + 2, y1 = x + 3;
    HostTensorND host_y0, host_y1;
    auto func0 = cg->compile({make_callback_copy(y0, host_y0)});
    func0->execute().wait();
    auto func1 = cg->compile({make_callback_copy(y1, host_y1)});
    func1->execute().wait();
    auto px = host_x->ptr<float>(),
         py0 = host_y0.ptr<float>(), py1 = host_y1.ptr<float>();
    for (size_t i = 0; i < 3; ++ i) {
        ASSERT_FLOAT_EQ(px[i] + 2, py0[i]);
        ASSERT_FLOAT_EQ(px[i] + 3, py1[i]);
    }
}

TEST_F(TestInfkernFinder, InSubgraph) {
    auto loop_cb = [&](opr::Loop::Desc &desc) {
        auto xi = desc.add_input(x),
             y = make_callback(xi);
        desc.set_loop_condition(desc.get_counter_var() < 0);
        desc.add_output(y, opr::Loop::Desc::OutputMode::LAST);
    };
    run(opr::Loop::make(loop_cb)[0], 1);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

