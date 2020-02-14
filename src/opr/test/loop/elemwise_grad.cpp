/**
 * \file src/opr/test/loop/elemwise_grad.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"

#include "megbrain/opr/loop.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/blas.h"

#include <cmath>

using namespace mgb;

using LoopDesc = opr::Loop::Desc;
using OutputMode = opr::Loop::Desc::OutputMode;

namespace {

class TestOprLoopElemwiseGrad: public ::testing::Test {
    protected:
        std::shared_ptr<ComputingGraph> graph;
        std::shared_ptr<HostTensorND> host_x, host_loss_p;
        SymbolVar x;
        opr::Loop::DescMaker desc_maker;

        void SetUp() override {
            constexpr size_t SIZE = 23;
            graph = ComputingGraph::make();
            HostTensorGenerator<> gen;
            host_x = gen({SIZE});
            host_loss_p = gen({SIZE});
#if 0
            for (size_t i = 0; i < SIZE; i ++) {
                host_x->ptr<float>()[i] = i + 1;
                host_loss_p->ptr<float>()[i] = 1;
            }
#endif
            x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x");
        }

        void check(thin_function<float(float)> grad_raw) {
            auto y = opr::Loop::make(desc_maker).at(0).rename("y");
            auto grad = cg::grad(
                    opr::Dot::make(y,
                        opr::Host2DeviceCopy::make(*graph, host_loss_p))
                    .rename("loss"),
                    x);
            HostTensorND host_grad;
            auto func = graph->compile({make_callback_copy(grad, host_grad)});
            func->execute();

            ASSERT_EQ(host_x->shape(), host_grad.shape());
            auto px = host_x->ptr<float>(), pg = host_grad.ptr<float>(),
                 lp = host_loss_p->ptr<float>();
            for (size_t i = 0; i < host_x->shape().total_nr_elems(); i ++) {
                MGB_ASSERT_FLOAT_EQ(grad_raw(px[i]) * lp[i], pg[i]) <<
                    ssprintf("failed at %zd: x=%.6f lp=%.6f", i, px[i], lp[i]);
            }
        }
};

}

/* y = x */
TEST_F(TestOprLoopElemwiseGrad, Identity) {
    desc_maker = [this](LoopDesc &desc) {
        auto x = desc.add_input(this->x);
        desc.set_loop_condition(desc.get_counter_var() < 0);
        desc.add_output(x, OutputMode::LAST);
    };
    check([](float){return 1.f;});
}

/* y = sum(x, 1 <= i <= N) */
TEST_F(TestOprLoopElemwiseGrad, UpdateWithSimpleSum) {
    constexpr float N = 4;
    desc_maker = [this](LoopDesc &desc) {
        auto x = desc.add_input_assignable(this->x),
             x0 = desc.add_input(this->x);
        desc.set_loop_condition(desc.get_counter_var() < N - 1);
        desc.assign(x, x + x0);
        desc.add_output(x, OutputMode::LAST);
    };
    check([](float){return N;});
}

/* y = prod(x, 1 <= i <= N) */
TEST_F(TestOprLoopElemwiseGrad, UpdateWithSimpleExp) {
    constexpr float N = 7;
    desc_maker = [this](LoopDesc &desc) {
        auto x = desc.add_input_assignable(this->x).rename("x"),
             x0 = desc.add_input(this->x).rename("x0");
        desc.set_loop_condition(desc.get_counter_var() < N - 1);
        desc.assign(x, (x * x0).rename("xmul"));
        desc.add_output(x, OutputMode::LAST);
    };
    check([](float x)->float{return N * pow(x, N - 1);});
}

/* y = prod(x, 1 <= i <= N) */
TEST_F(TestOprLoopElemwiseGrad, UpdateWithSimpleExp2) {
    constexpr float N = 8;
    desc_maker = [this](LoopDesc &desc) {
        auto x = desc.add_input_assignable(
                this->x.fill_retain_dtype(1)).rename("x"),
             x0 = desc.add_input(this->x).rename("x0"),
             xmul = (x * x0).rename("xmul");
        desc.set_loop_condition(desc.get_counter_var() < N - 1);
        desc.assign(x, xmul);
        desc.add_output(xmul, OutputMode::LAST);
    };
    check([](float x)->float{return N * pow(x, N - 1);});
}

/* y = sum(i * x^i, 1 <= i <= N) */
TEST_F(TestOprLoopElemwiseGrad, InvolveCounterVar) {
    constexpr float N = 8;
    desc_maker = [this](LoopDesc &desc) {
        auto x = desc.add_input_assignable(
                this->x.fill_retain_dtype(1)).rename("x"),
             x0 = desc.add_input(this->x).rename("x0"),
             xmul = (x * x0).rename("xmul");
        desc.set_loop_condition(desc.get_counter_var() < N - 1);
        desc.assign(x, xmul);
        desc.add_output(xmul * (desc.get_counter_var() + 1), OutputMode::SUM);
    };
    auto grad = [](float x) {
        float y = 0;
        for (int i = 1; i <= N; i ++)
            y += i * i * pow(x, i - 1);
        return y;
    };
    check(grad);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

