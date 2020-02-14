/**
 * \file src/opr/test/loop/basic.cpp
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
#include "megbrain/test/host_static_calc.h"

#include "megbrain/utils/timer.h"
#include "megbrain/opr/loop.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/tensor_manip.h"

#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/basic_arith.h"

#include <cmath>

using namespace mgb;

namespace mgb {
namespace opr {
namespace intl {

    class LoopTest {
        public:
            static bool is_static_loop_time(cg::OperatorNodeBase *opr) {
                return static_cast<bool>(opr->cast_final_safe<Loop>().
                        m_static_loop_time_infer);
            }

            static bool& check_output_recorder_sum_optimize_success() {
                return Loop::LoopImpl::
                    test_check_grad_output_recorder_sum_optimize_success();
            }

            static ThinHashMap<VarNode*, bool> var_rec_spec(
                    cg::OperatorNodeBase *opr) {
                return opr->cast_final_safe<Loop>().test_get_var_rec_spec();
            }
    };

} // namespace intl
} // namespace opr
} // namespace mgb

using LoopDesc = opr::Loop::Desc;
using OutputMode = opr::Loop::Desc::OutputMode;
using opr::intl::LoopTest;

namespace {

void test_basic_fwd_with_grad(bool dyn) {
    HostTensorGenerator<> gen;

    auto host_x = gen({23});
    auto host_loop_time = std::make_shared<HostTensorND>(
            host_x->comp_node(), dtype::Int32());
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    auto d = [dyn](SymbolVar var) -> SymbolVar {
        if (dyn)
            var = opr::MarkDynamicVar::make(var).node();
        return var;
    };

    auto desc_maker =
            [&, loop_time = opr::Host2DeviceCopy::make(*graph, host_loop_time)]
            (LoopDesc &loop_desc) {

        auto xl = loop_desc.add_input_assignable(x).rename("xl"),
             x0 = d(loop_desc.add_input(x)).rename("x0"),
             xu = (opr::pow(d(xl), xl.make_scalar(0.9f)) * x0).rename("xu");
        loop_desc.assign(xl, xu);
        loop_desc.add_output(xu, OutputMode::LAST);
        auto cnt = d(loop_desc.get_counter_var());
        loop_desc.set_loop_condition(cnt < loop_desc.add_input(loop_time) - 1);
    };

    auto y = opr::Loop::make(desc_maker, 3)[0],
         loss = opr::reduce_sum(y, y.make_scalar(1)),
         gx = cg::grad(loss, x);
    HostTensorND host_y, host_gx;
    auto func = graph->compile({
            make_callback_copy(y, host_y),
            make_callback_copy(gx, host_gx)
            });

    int& loop_time = host_loop_time->resize({1}).ptr<int>()[0];
    for (size_t sz: {12, 24}) {
        *host_x = *gen({sz});
        auto px = host_x->ptr<float>();
        for (size_t i = 0; i < sz; ++ i)
            px[i] = std::abs(px[i]);

        for (loop_time = 1; loop_time <= 50; ++ loop_time) {
            func->execute();
            ASSERT_EQ(host_x->shape(), host_y.shape());
            ASSERT_EQ(host_x->shape(), host_gx.shape());

            auto py = host_y.ptr<float>(), pgx = host_gx.ptr<float>();

            float dpow = 1;
            for (int i = 0; i < loop_time; ++ i)
                dpow = dpow * 0.9f + 1.f;
            for (size_t i = 0; i < sz; ++ i) {
                auto x = px[i], y = py[i], gx = pgx[i];
                MGB_ASSERT_FLOAT_NEAR(std::pow(x, dpow), y, 1e-5) << loop_time;
                MGB_ASSERT_FLOAT_NEAR(
                        dpow * std::pow(x, (dpow - 1.f)), gx, 1e-5)
                    << loop_time;
            }
        }
    }
}

} // anonymous namespace

TEST(TestOprLoop, APlusB) {
    HostTensorGenerator<> gen;

    auto host_x = gen({23});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto xl = loop_desc.add_input_assignable(x).rename("xl"),
             xu = xl * 2 + 1;
        loop_desc.assign(xl, xu);
        loop_desc.add_output(xu, OutputMode::LAST);
        loop_desc.set_loop_condition(loop_desc.get_counter_var() < 3);
    };

    auto y = opr::Loop::make(desc_maker, 3)[0];
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    ASSERT_EQ(host_x->shape(), host_y.shape());
    auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
    for (size_t i = 0; i < 23; ++ i) {
        auto yv = px[i];
        for (int j = 0; j < 4; ++ j) {
            yv = yv * 2 + 1;
        }
        ASSERT_EQ(yv, py[i]);
    }
}

TEST(TestOprLoop, APlusBGrad) {
    HostTensorGenerator<> gen;

    auto host_x = gen({23});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto xl = loop_desc.add_input_assignable(x).rename("xl");
        loop_desc.assign(xl, xl * loop_desc.add_input(x).rename("x"));
        loop_desc.add_output(xl, OutputMode::LAST);
        loop_desc.set_loop_condition(loop_desc.get_counter_var() < 3);
    };

    auto gx = cg::grad(opr::reduce_sum(opr::Loop::make(desc_maker, 3)[0],
                x.make_scalar(1)), x);
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->execute();
    ASSERT_EQ(host_x->shape(), host_gx.shape());
    auto px = host_x->ptr<float>(), pgx = host_gx.ptr<float>();
    for (size_t i = 0; i < 23; ++ i) {
        auto x = px[i];
        ASSERT_EQ(4 * x * x * x, pgx[i]);
    }
}

TEST(TestOprLoop, APlusBGradWithShallowCopy) {
    HostTensorGenerator<> gen;

    auto host_x = gen({23});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    auto desc_maker = [x=x*2+1](LoopDesc &loop_desc) {
        auto xl = loop_desc.add_input_assignable(x).rename("xl");
        loop_desc.assign(xl, xl * loop_desc.add_input(x).rename("x"));
        loop_desc.add_output(xl, OutputMode::LAST);
        loop_desc.set_loop_condition(loop_desc.get_counter_var() < 3);
    };

    auto gx = cg::grad(opr::reduce_sum(opr::Loop::make(desc_maker, 3)[0],
                x.make_scalar(1)), x);
    auto gx0 = gx;

    unpack_vector(
            gopt::GraphOptimizer{}.add_pass<gopt::ArithFusePass>().
            apply({{gx}}).endpoint_vars(),
            gx);
    ASSERT_NE(gx0.node(), gx.node());

    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->to_json()->writeto_fpath(
            output_file("TestOprLoop.APlusBGradWithShallowCopy"));
    func->execute();
    ASSERT_EQ(host_x->shape(), host_gx.shape());
    auto px = host_x->ptr<float>(), pgx = host_gx.ptr<float>();
    for (size_t i = 0; i < 23; ++ i) {
        auto x = px[i] * 2 + 1;
        ASSERT_EQ(4 * x * x * x * 2, pgx[i]);
    }
}

TEST(TestOprLoop, MultiReaderGrad) {
    using Checker = AutoOprChecker<1, 1>;

    auto make_graph = [](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        SymbolVar x = inputs[0];
        auto desc_maker = [x](LoopDesc &loop_desc) {
            auto x0 = loop_desc.add_input_assignable(x).rename("x0"),
                 x1 = loop_desc.add_input_assignable(x).rename("x1"),
                 x2 = loop_desc.add_input_assignable(x).rename("x2"),
                 xu = (x0 + x1 - x2) * loop_desc.add_input(x).rename("x") + 1;
            loop_desc.assign(x0, xu);
            loop_desc.assign(x1, xu);
            loop_desc.assign(x2, xu - 1);
            loop_desc.add_output(x2, OutputMode::SUM);
            loop_desc.set_loop_condition(loop_desc.get_counter_var() < 3);
        };
        return {opr::Loop::make(desc_maker)[0]};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto nr = inp[0]->shape().total_nr_elems();
        auto px = inp[0]->ptr<float>();
        auto py = dest[0].resize(inp[0]->shape()).ptr<float>();
        for (size_t i = 0; i < nr; ++ i) {
            float x = px[i], x0 =  x, x1 = x, x2 = x, y = 0;
            for (int j = 0; j <= 3; ++ j) {
                auto xu = (x0 + x1 - x2) * x + 1;
                y += x2;
                x0 = xu;
                x1 = xu;
                x2 = xu - 1;
            }
            py[i] = y;
        }
    };
    Checker{make_graph, fwd}.
        disable_multi_loss_check().
        run({TensorShape{2}}).
        run({TensorShape{3}}).
        run({TensorShape{2, 3}});
}

TEST(TestOprLoop, BasicFwdWithGrad) {
    test_basic_fwd_with_grad(false);
}

TEST(TestOprLoop, BasicFwdWithGradDyn) {
    test_basic_fwd_with_grad(true);
}

TEST(TestOprLoop, OutputCounter) {
    HostTensorGenerator<> gen;

    constexpr int LOOP_TIME = 3;
    auto host_x = gen({23});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto xl = loop_desc.add_input_assignable(x).rename("xl"),
             x0 = loop_desc.add_input(x).rename("x0"),
             xu = (x0 * xl).rename("xu");
        loop_desc.assign(xl, xu);
        loop_desc.add_output(xu, OutputMode::LAST);
        auto cnt = loop_desc.get_counter_var();
        loop_desc.add_output(cnt, OutputMode::LAST);
        loop_desc.set_loop_condition(cnt < LOOP_TIME);
    };

    auto y = opr::Loop::make(desc_maker, 3);
    HostTensorND host_y0, host_y1;
    auto func = graph->compile({
            make_callback_copy(y[0], host_y0),
            make_callback_copy(y[1], host_y1)
            });

    func->execute();
    ASSERT_EQ(host_x->shape(), host_y0.shape());
    ASSERT_EQ(TensorShape{1}, host_y1.shape());

    auto py0 = host_y0.ptr<float>(), px = host_x->ptr<float>();
    auto py1 = host_y1.ptr<int>();

    constexpr double dpow = LOOP_TIME + 2;
    for (size_t i = 0; i < host_x->shape(0); ++ i) {
        MGB_ASSERT_FLOAT_EQ(std::pow(px[i], dpow), py0[i]);
    }
    ASSERT_EQ(LOOP_TIME, py1[0]);
}

TEST(TestOprLoop, InputDedup) {
    set_rand_seed(19931102);
    constexpr int LOOP_TIME = 5;
    HostTensorGenerator<> gen;

    auto host_x = gen({23});
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto x_s0 = loop_desc.add_input_assignable(x).rename("x_s0"),
             x_s1 = loop_desc.add_input_assignable(x).rename("x_s1"),
             x0 = loop_desc.add_input(x).rename("x0"),
             x1 = loop_desc.add_input(x).rename("x1");
        ASSERT_EQ(x0.node(), x1.node());
        ASSERT_NE(x_s0.node(), x_s1.node());
        loop_desc.assign(x_s0, x_s0 + 1);
        loop_desc.assign(x_s1, x_s1 - 1);
        auto cnt = loop_desc.get_counter_var();
        auto ov = x_s0 * 1 + x_s1 * 2 + x0 * 3 + x1 * cnt;
        loop_desc.add_output(ov.rename("ov"), OutputMode::SUM);
        loop_desc.set_loop_condition(cnt < LOOP_TIME - 1);
    };
    // sum(x + k + (x - k)*2 + 3*x + k*x, 0 <= k < LOOP_TIME)

    auto y = opr::Loop::make(desc_maker)[0],
         loss = opr::Dot::make(y, y),
         gx = cg::grad(loss, x);
    HostTensorND host_y, host_gx;
    auto func = graph->compile({
            make_callback_copy(y, host_y),
            make_callback_copy(gx, host_gx)
            });

    float EQUIV_K = LOOP_TIME + 2.0 * LOOP_TIME + 3 * LOOP_TIME +
                        LOOP_TIME * (LOOP_TIME - 1) / 2,
          EQUIV_B = LOOP_TIME * (LOOP_TIME - 1.0) / 2 * (1 - 2);
    for (size_t sz: {12, 24}) {
        *host_x = *gen({sz});
        func->execute();
        ASSERT_EQ(host_x->shape(), host_y.shape());
        ASSERT_EQ(host_x->shape(), host_gx.shape());

        auto px = host_x->ptr<float>(), py = host_y.ptr<float>(),
             pgx = host_gx.ptr<float>();
        for (size_t i = 0; i < sz; ++ i) {
            auto x = px[i], y = py[i], gx = pgx[i];
            MGB_ASSERT_FLOAT_EQ(x * EQUIV_K + EQUIV_B, y);
            MGB_ASSERT_FLOAT_EQ(EQUIV_K * 2 * y, gx);
        }
    }
}

TEST(TestOprLoop, OutputDedup) {
    constexpr size_t LOOP_TIME = 5;
    HostTensorGenerator<> gen;
    auto host_x = gen({23}), host_y = gen({23});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);
    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto loop_x = loop_desc.add_input(x),
             loop_y = loop_desc.add_input(y);
        loop_desc.set_loop_condition(
                loop_desc.get_counter_var() < int(LOOP_TIME - 1));
        loop_desc.add_output(loop_x, OutputMode::SUM);
        loop_desc.add_output(loop_y, OutputMode::LAST);
        loop_desc.add_output(loop_x, OutputMode::SUM);
    };
    auto rst = opr::Loop::make(desc_maker);
    // rst: x * LOOP_TIME, y, x * LOOP_TIME
    ASSERT_EQ(3u, rst.size());
    ASSERT_EQ(rst[0].node(), rst[2].node());
    ASSERT_NE(rst[0].node(), rst[1].node());

    auto loss = opr::Dot::make(rst[0], rst[1]) + opr::Dot::make(rst[0], rst[2]),
         gx = cg::grad(loss, x),
         gy = cg::grad(loss, y);
    HostTensorND host_gx, host_gy;
    auto func = graph->compile({
            make_callback_copy(gx, host_gx),
            make_callback_copy(gy, host_gy)});
    func->execute();
    ASSERT_EQ(host_x->shape(), host_gx.shape());
    ASSERT_EQ(host_x->shape(), host_gy.shape());

    constexpr float K = LOOP_TIME;
    for (size_t i = 0; i < host_x->shape().shape[0]; ++ i) {
        auto x = host_x->ptr<float>()[i], y = host_y->ptr<float>()[i],
             gx = host_gx.ptr<float>()[i], gy = host_gy.ptr<float>()[i];
        MGB_ASSERT_FLOAT_EQ(K * x, gy);
        MGB_ASSERT_FLOAT_EQ(K * y + 2 * K * K * x, gx);
    }
}

TEST(TestOprLoop, CyclicUpdate) {
    constexpr size_t SIZE = 23;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({SIZE});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y");

    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto loop_x = loop_desc.add_input_assignable(x).rename("lx"),
             loop_y = loop_desc.add_input_assignable(y).rename("ly");
        loop_desc.assign(loop_x, loop_y);
        loop_desc.assign(loop_y, loop_x);
        loop_desc.set_loop_condition(loop_desc.get_counter_var() < 3);
        loop_desc.add_output(loop_x, OutputMode::LAST);
        loop_desc.add_output(loop_y, OutputMode::LAST);
    };

    auto rst = opr::Loop::make(desc_maker);
    ASSERT_EQ(2u, rst.size());

    HostTensorND host_r0, host_r1;

    auto func = graph->compile({
            make_callback_copy(rst[0], host_r0),
            make_callback_copy(rst[1], host_r1)});
    func->execute();

    auto px = host_x->ptr<float>(), py = host_y->ptr<float>(),
         pr0 = host_r0.ptr<float>(), pr1 = host_r1.ptr<float>();
    for (size_t i = 0; i < SIZE; i ++) {
        ASSERT_EQ(px[i], pr1[i]) <<
            ssprintf("fail at %zd: y=%.2f r0=%.2f",
                    i, py[i], pr0[i]);
        ASSERT_EQ(py[i], pr0[i]) <<
            ssprintf("fail at %zd: x=%.2f r1=%.2f",
                    i, px[i], pr1[i]);
    }
}

TEST(TestOprLoop, CyclicUpdateGradInpShapeOnly) {
    constexpr size_t SIZE = 1;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({SIZE}),
         host_loss_p0 = gen({SIZE}),
         host_loss_p1 = gen({SIZE});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y"),
         loss_p0 = opr::Host2DeviceCopy::make(*graph, host_loss_p0),
         loss_p1 = opr::Host2DeviceCopy::make(*graph, host_loss_p1);

    // grad only depends on input shape
    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto loop_x = loop_desc.add_input_assignable(x).rename("lx"),
             loop_y = loop_desc.add_input_assignable(y).rename("ly");
        loop_desc.assign(loop_x, loop_y);
        loop_desc.assign(loop_y, loop_x);
        loop_desc.set_loop_condition(loop_desc.get_counter_var() < 3);
        loop_desc.add_output(loop_x, OutputMode::LAST);
        loop_desc.add_output(loop_y, OutputMode::LAST);
    };

    auto rst = opr::Loop::make(desc_maker);
    auto loss = opr::Dot::make(rst.at(0), loss_p0) +
        opr::Dot::make(rst.at(1), loss_p1);

    HostTensorND host_r0, host_r1;

    auto func = graph->compile({
            make_callback_copy(cg::grad(loss, x), host_r0),
            make_callback_copy(cg::grad(loss, y), host_r1)});

    auto run = [&](size_t size) {
        *host_x = *gen({size});
        *host_y = *gen({size});
        *host_loss_p0 = *gen({size});
        *host_loss_p1 = *gen({size});

        func->execute();

        ASSERT_EQ(host_r0.shape(), host_y->shape());
        ASSERT_EQ(host_r1.shape(), host_x->shape());

        auto px = host_loss_p0->ptr<float>(), py = host_loss_p1->ptr<float>(),
             pr0 = host_r0.ptr<float>(), pr1 = host_r1.ptr<float>();
        for (size_t i = 0; i < size; i ++) {
            ASSERT_EQ(px[i], pr1[i]) <<
                ssprintf("fail at %zd: y=%.2f r0=%.2f",
                        i, py[i], pr0[i]);
            ASSERT_EQ(py[i], pr0[i]) <<
                ssprintf("fail at %zd: x=%.2f r1=%.2f",
                        i, px[i], pr1[i]);
        }
    };

    run(10);
    run(23);
}

TEST(TestOprLoop, CyclicUpdateGrad) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
            auto x0 = inputs[0], y0 = inputs[1];

            auto desc_maker = [&](LoopDesc &desc) {
                auto x = desc.add_input_assignable(x0).rename("x"),
                     y = desc.add_input_assignable(y0).rename("y");
                desc.assign(x, x + y);
                desc.assign(y, x * 3 + 1);
                desc.set_loop_condition(desc.get_counter_var() < 3);
                desc.add_output(y / 3, OutputMode::LAST);
            };
            // x0, y0 = x, y
            // x1, y1 = x + y, x * 3 + 1
            // x2, y2 = x * 4 + y + 1, x * 3 + y * 3 + 1
            // y3 = x2 * 3 + 1
            // out: x2 + 1 / 3

            return {opr::Loop::make(desc_maker)[0]};
        };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto oshp = inp[0]->shape();
        auto ix = inp[0]->ptr<float>(), iy = inp[1]->ptr<float>(),
             o = dest[0].resize(oshp).ptr<float>();
        for (size_t i = 0, sz = oshp.total_nr_elems(); i < sz; ++ i) {
            o[i] = ix[i] * 4 + iy[i] + 1 + 1.f / 3;
        }
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    auto mki = [](const TensorShape &s) -> Checker::ShapeInpArray {
        return {s, s};
    };
    Checker{make_graph, fwd}.
        disable_multi_loss_check().
        run(mki({3}), opt).
        run(mki({5}), opt).
        run(mki({2, 3}), opt);
}


TEST(TestOprLoop, BasicUpdate) {
    bool failed = false;

    auto static_calc_opr = opr::intl::create_megdnn_opr<
        megdnn::Elemwise>(CompNode::load("xpu0"));

    auto run = [&](bool x_dynamic) {
        ASSERT_FALSE(failed);
        failed = true;
        constexpr float EXP = 3;
        constexpr size_t SIZE0 = 4, SIZE1 = 7;
        HostTensorGenerator<> gen;
        auto host_x = gen({SIZE0, SIZE1});

        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        if (x_dynamic)
            x = opr::MarkDynamicVar::make(x);
        x.rename("x");

        float *y0_ptr = nullptr;
        auto desc_maker = [&](LoopDesc &loop_desc) {
            auto loop_x0 = loop_desc.add_input_assignable(
                    x.fill_retain_dtype(1)).rename("x0");

            loop_desc.assign(loop_x0, loop_x0 * loop_desc.add_input(x));
            loop_desc.set_loop_condition(loop_desc.get_counter_var() < EXP);
            auto cb = [&](DeviceTensorND &val) {
                y0_ptr = val.ptr<float>();
            };
            auto y = opr::CallbackInjector::make(loop_x0, cb);
            loop_desc.add_output(y, OutputMode::LAST);
            loop_desc.add_output(y + 2, OutputMode::LAST);
        };
        auto y = opr::Loop::make(desc_maker);
        ASSERT_EQ(2u, y.size());
        y[0].rename("y");
        y[1].rename("y1");

        HostTensorND host_y, host_y1, expected,
                     host_exp{host_x->comp_node(), host_x->dtype()}, host_bias;
        host_exp.resize({1}).ptr<float>()[0] = EXP;
        host_bias.copy_from(host_exp).ptr<float>()[0] = 2;

        auto func = graph->compile({
                make_callback_copy(y[0], host_y),
                make_callback_copy(y[1], host_y1)});

        for (size_t i = 0; i < 2; i ++) {
            func->execute();

            mgb::host_pow(expected, *host_x, host_exp);
            MGB_ASSERT_TENSOR_EQ(expected, host_y);
            mgb::host_add(expected, expected, host_bias);
            MGB_ASSERT_TENSOR_EQ(expected, host_y1);

            ASSERT_EQ(y0_ptr, y[0].node()->prev_dev_ptr());

            host_x->copy_from(*gen({12, 23}));
        }
        failed = false;
    };
    run(false);
    run(true);
}

TEST(TestOprLoop, BenchmarkOverhead) {
    constexpr size_t LOOP_TIME = 100;
    double time_loop = -1, time_raw = -1;
    auto zero = [&](ComputingGraph &graph) {
        return SymbolVar::make_scalar(0, graph, CompNode::load("xpu0"));
    };
    auto run_loop = [&]() {
        auto graph = ComputingGraph::make();
        auto desc_maker = [&](LoopDesc &loop_desc) {
            auto x = loop_desc.add_input_assignable(zero(*graph)),
                 xnext = x + 1;
            loop_desc.assign(x, xnext);
            loop_desc.add_output(xnext, OutputMode::LAST);
            auto cnt = loop_desc.get_counter_var();
            loop_desc.set_loop_condition(cnt < int(LOOP_TIME - 1));
        };
        auto y = opr::Loop::make(desc_maker)[0];
        HostTensorND host_y;
        auto f = graph->compile({make_callback_copy(y, host_y)});
        f->execute();
        RealTimer timer;
        f->execute();
        ASSERT_EQ(LOOP_TIME, size_t(host_y.ptr<int>()[0]));
        time_loop = timer.get_secs();
    };
    auto run_raw = [&]() {
        auto graph = ComputingGraph::make();
        auto dev_delta = std::make_shared<DeviceTensorND>();
        HostTensorND host_delta{CompNode::load("xpu0"), dtype::Float32()};
        host_delta.resize({1}).ptr<float>()[0] = 1;
        dev_delta->copy_from(host_delta);
        auto x = zero(*graph),
             delta = opr::SharedDeviceTensor::make(*graph, dev_delta);
        for (size_t i = 0; i < LOOP_TIME; ++ i)
            x = x + delta;
        HostTensorND host_x;
        auto f = graph->compile({make_callback_copy(x, host_x)});
        f->execute();

        RealTimer timer;
        f->execute();
        ASSERT_EQ(LOOP_TIME, size_t(host_x.ptr<float>()[0]));
        time_raw = timer.get_secs();
    };

    run_loop();
    run_raw();
    mgb_log("time_loop/time_raw=%.3g/%.3g=%.3g overhead_per_loop=%.3gms",
            time_loop, time_raw, time_loop / time_raw,
            (time_loop - time_raw) / LOOP_TIME * 1000);
}

TEST(TestOprLoop, RecordOutputAll) {
    using Checker = AutoOprChecker<1, 4>;
    static constexpr int LOOP_TIME = 7;
    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
            auto x = inputs[0];
            auto desc_maker = [&](LoopDesc &desc) {
                auto xl = desc.add_input_assignable(x),
                     xu = opr::pow(xl, xl.make_scalar(.7f)) * desc.add_input(x),
                     cnt = desc.get_counter_var();
                desc.assign(xl, xu);
                desc.add_output(xl, OutputMode::ALL);
                desc.add_output(xl * cnt, OutputMode::ALL);
                desc.add_output(xu, OutputMode::ALL);
                desc.add_output(xu * cnt, OutputMode::ALL);
                desc.set_loop_condition(cnt < LOOP_TIME - 1);
            };
            auto y = opr::Loop::make(desc_maker);
            return {y[0], y[1], y[2], y[3]};
        };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        float *py[4];
        size_t size = inp[0]->shape(0);
        for (int i = 0; i < 4; ++ i) {
            py[i] = dest[i].resize({LOOP_TIME, size}).ptr<float>();
        }
        auto px = inp[0]->ptr<float>();
        for (size_t i = 0; i < size; ++ i) {
            float x = px[i], epow = 0;
            for (int j = 0; j < LOOP_TIME; ++ j) {
                epow = epow * .7f + 1.f;
                auto xl = std::pow(x, epow), xu = std::pow(xl, .7f) * x;
                auto off = j * size + i;
                py[0][off] = xl;
                py[1][off] = xl * j;
                py[2][off] = xu;
                py[3][off] = xu * j;
            }
        }
    };

    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
        1e-2, 1};
    auto genx = [&](HostTensorND &dest) {
        dest = *gen(dest.shape());
    };
    Checker::RunOptions opt;
    opt.numdiff_eps = 1e-3;
    opt.numdiff_max_err = 4e-3;
    Checker{make_graph, fwd}.
        disable_multi_loss_check().
        set_input_generator(0, genx).
        run({TensorShape{2}}, opt).
        run({TensorShape{3}}, opt).
        run({TensorShape{23}}, opt);
}

TEST(TestOprLoop, RecordOutputSum) {
    using Checker = AutoOprChecker<1, 4>;
    static constexpr int LOOP_TIME = 7;
    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
            auto x = inputs[0];
            auto desc_maker = [&](LoopDesc &desc) {
                auto xl = desc.add_input_assignable(x),
                     xu = opr::pow(xl, xl.make_scalar(.7f)) * desc.add_input(x),
                     cnt = desc.get_counter_var();
                desc.assign(xl, xu);
                desc.add_output(xl, OutputMode::SUM);
                desc.add_output(xl * cnt, OutputMode::SUM);
                desc.add_output(xu, OutputMode::SUM);
                desc.add_output(xu * cnt, OutputMode::SUM);
                desc.set_loop_condition(cnt < LOOP_TIME - 1);
            };
            auto y = opr::Loop::make(desc_maker);
            return {y[0], y[1], y[2], y[3]};
        };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        float *py[4];
        size_t size = inp[0]->shape(0);
        for (int i = 0; i < 4; ++ i) {
            py[i] = dest[i].resize({size}).ptr<float>();
            memset(py[i], 0, sizeof(float) * size);
        }
        auto px = inp[0]->ptr<float>();
        for (size_t i = 0; i < size; ++ i) {
            float x = px[i], epow = 0;
            for (int j = 0; j < LOOP_TIME; ++ j) {
                epow = epow * .7f + 1.f;
                auto xl = std::pow(x, epow), xu = std::pow(xl, .7f) * x;
                py[0][i] += xl;
                py[1][i] += xl * j;
                py[2][i] += xu;
                py[3][i] += xu * j;
            }
        }
    };

    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
        1e-2, 1};
    auto genx = [&](HostTensorND &dest) {
        dest = *gen(dest.shape());
    };
    Checker::RunOptions opt;
    opt.numdiff_eps = 1e-3;
    opt.numdiff_max_err = 4e-3;
    Checker{make_graph, fwd}.
        disable_multi_loss_check().
        set_input_generator(0, genx).
        run({TensorShape{2}}, opt).
        run({TensorShape{3}}, opt).
        run({TensorShape{23}}, opt);
}

TEST(TestOprLoop, DynamicCases) {
    using Checker = AutoOprChecker<1, 4>;

    bool failed = false;
    auto run = [&](bool dyn_inp, bool dyn_cnt) {
        ASSERT_FALSE(failed);
        failed = true;

        DeviceTensorND xdev_prev;
        constexpr ptrdiff_t LOOP_TIME = 4;

        auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
            auto glb_x = inputs.at(0);
            if (dyn_inp)
                glb_x = opr::MarkDynamicVar::make(glb_x);
            glb_x.rename("glb_x");

            auto desc_maker = [glb_x, dyn_inp, dyn_cnt, &xdev_prev](
                    LoopDesc &desc) {

                auto check_xsub = [&xdev_prev](DeviceTensorND &xsub) {
                    mgb_assert(xsub.ptr<float>() >= xdev_prev.ptr<float>());
                    mgb_assert(xsub.ptr<float>() <= xdev_prev.ptr<float>() +
                            xdev_prev.layout().total_nr_elems());
                };

                using AIdx = opr::Subtensor::AxisIndexer;
                auto x = desc.add_input_assignable(glb_x).rename("x"),
                     cnt = desc.get_counter_var(),
                     xchk = opr::CallbackInjector::make(x,
                        [&](DeviceTensorND &v){
                            xdev_prev = v;
                        }),
                     xsub = opr::CallbackInjector::make(
                             opr::Subtensor::make(xchk,
                                 {AIdx::make_interval(0, cnt, cnt + 1, None)}
                                 ), check_xsub).rename("xsub"),
                     y0 = (xsub + 1).rename("y0"),
                     y0o = opr::AxisAddRemove::make(
                             y0,
                             {opr::AxisAddRemove::AxisDesc::make_remove(0)}).
                        rename("y0o"),
                     y1 = (y0 + 1).rename("y1"),
                     loop_time = x.make_scalar(int(LOOP_TIME)).rename("lt");

                mgb_assert(cg::is_static_var_shape(cnt.node()));
                if (!dyn_inp)
                    mgb_assert(cg::is_static_var_shape(x.node()));
                if (dyn_cnt) {
                    cnt = opr::MarkDynamicVar::make(cnt);
                    loop_time = opr::MarkDynamicVar::make(loop_time);
                }
                auto y2 = ((x + y0) * cnt / 3.f).rename("y2");
                auto y3 = (x + (cnt * cnt).reshape({1})).rename("y3");
                desc.assign(x, (x + y1 - cnt / 2.f).rename("xnew"));
                desc.add_output(y0o, OutputMode::ALL);
                desc.add_output(y1, OutputMode::LAST);
                desc.add_output(y2, OutputMode::SUM);
                desc.add_output(y3, OutputMode::SUM);
                desc.set_loop_condition(cnt < loop_time);
            };

            auto loop_out = opr::Loop::make(desc_maker);
            mgb_assert(dyn_inp ==
                    !cg::is_static_var_shape(loop_out.at(3).node()));
            return {loop_out[0], loop_out[1], loop_out[2], loop_out[3]};
        };

        auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
            HostTensorND x;
            x.copy_from(*inp[0]);
            auto x_shp = x.shape(), y0_shp = x.shape(), y1_shp = x.shape();
            y0_shp.shape[0] = LOOP_TIME + 1;
            y1_shp.shape[0] = 1;
            auto &&y0 = dest[0].comp_node(x.comp_node()).resize(y0_shp),
                 &&y1 = dest[1].comp_node(x.comp_node()).resize(y1_shp),
                 &&y2 = dest[2].comp_node(x.comp_node()).resize(x_shp),
                 &&y3 = dest[3].comp_node(x.comp_node()).resize(x_shp);
            memset(y2.ptr<float>(), 0, sizeof(float) * y2.layout().total_nr_elems());
            memset(y3.ptr<float>(), 0, sizeof(float) * y3.layout().total_nr_elems());
            ptrdiff_t cnt = 0;
            bool should_loop;
            do {
                // compute outputs
                auto xsub = x[{{cnt, cnt + 1}}];
                auto y0_dest = y0[{{cnt, cnt + 1}}];
                for (ptrdiff_t i = 0, it = xsub.layout().total_nr_elems();
                        i < it; ++ i) {
                    auto xv = xsub.ptr<float>()[i];
                    y0_dest.ptr<float>()[i] = xv + 1;
                    y1.ptr<float>()[i] = xv + 2;
                }

                HostTensorND tmp;
                mgb::host_add(tmp, x, y0_dest);
                mgb_assert(tmp.layout().eq_layout(y2.layout()));
                for (ptrdiff_t i = 0, it = y2.layout().total_nr_elems();
                        i < it; ++ i) {
                    y2.ptr<float>()[i] += tmp.ptr<float>()[i] * cnt / 3.0;
                    y3.ptr<float>()[i] += x.ptr<float>()[i] + cnt * cnt;
                }

                should_loop = cnt < LOOP_TIME;

                // update
                mgb::host_add(tmp, x, y1);
                mgb_assert(tmp.layout().eq_layout(x.layout()));
                for (ptrdiff_t i = 0, it = x.layout().total_nr_elems();
                        i < it; ++ i) {
                    x.ptr<float>()[i] = tmp.ptr<float>()[i] - cnt / 2.0;
                }
                ++ cnt;
            } while (should_loop);
        };

        Checker::RunOptions opt;
        // large eps because all linear
        opt.numdiff_eps = 1;
        Checker{make_graph, fwd}.
            disable_multi_loss_check().
            run({TensorShape{6, 1}}, opt).
            run({TensorShape{8, 4}}, opt).
            run({TensorShape{7, 2, 3}}, opt).
            run({TensorShape{6, 2, 1, 2}}, opt);

        failed = false;
    };

    for (int i = 0; i < 2; ++ i)
        for (int j = 0; j < 2; ++ j)
            run(i, j);
}

TEST(TestOprLoop, UnusedOutput) {
    constexpr float EXP = 3;
    constexpr size_t SIZE0 = 4, SIZE1 = 7;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE0, SIZE1});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x");

    auto desc_maker = [&](LoopDesc &desc) {
        auto x0 = desc.add_input_assignable(x.fill_retain_dtype(1)).rename("x0");

        desc.assign(x0, x0 * desc.add_input(x));
        desc.set_loop_condition(desc.get_counter_var() < EXP);
        desc.add_output(x0, OutputMode::LAST);
        desc.add_output(desc.add_input(x).rename("y1"), OutputMode::LAST);
        desc.add_output(
                opr::MarkDynamicVar::make(desc.add_input(x)).rename("y2"),
                OutputMode::LAST);
    };
    auto y = opr::Loop::make(desc_maker);
    ASSERT_EQ(3u, y.size());

    HostTensorND host_y, expected,
                 host_pow{host_x->comp_node(), host_x->dtype()};
    host_pow.resize({1}).ptr<float>()[0] = EXP;

    auto func = graph->compile({make_callback_copy(y[0], host_y)});

    for (auto &&ishp: {TensorShape{5}, TensorShape{4, 3},
            TensorShape{12, 3, 4}}) {
        *host_x = *gen(ishp);

        func->execute();
        ASSERT_NE(nullptr, y[0].node()->prev_dev_ptr());
        ASSERT_EQ(nullptr, y[1].node()->prev_dev_ptr());
        ASSERT_EQ(nullptr, y[2].node()->prev_dev_ptr());

        mgb::host_pow(expected, *host_x, host_pow);
        MGB_ASSERT_TENSOR_EQ(expected, host_y);
    }
}

TEST(TestOprLoop, UnusedOutputGrad) {
    constexpr float EXP = 5;
    constexpr size_t SIZE0 = 4, SIZE1 = 7;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE0, SIZE1}),
         host_loss_p = gen({SIZE0 * SIZE1});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x");

    bool used = false;
    auto used_cb = [&](DeviceTensorND &) {
        used = true;
    };

    auto desc_maker = [&](LoopDesc &desc) {
        auto x0 = desc.add_input_assignable(
                x.fill_retain_dtype(1)).rename("x0");
        auto y0 = desc.add_input_assignable(
                x.fill_retain_dtype(0)).rename("y0");
        auto cur_x = desc.add_input(x);

        desc.assign(x0, x0 * cur_x);
        desc.assign(y0, y0 + cur_x);
        desc.set_loop_condition(desc.get_counter_var() < EXP);

        y0 = opr::CallbackInjector::make(y0, used_cb);
        desc.add_output(y0, OutputMode::LAST);

        desc.add_output(x0, OutputMode::LAST);
    };
    auto y = opr::Loop::make(desc_maker)[1];

    auto loss = opr::Dot::make(y.flatten(),
            opr::Host2DeviceCopy::make(*graph, host_loss_p)),
         gx = cg::grad(loss, x);

    HostTensorND host_y, host_gx,
                 expected{host_x->comp_node(), host_x->dtype()};

    auto func = graph->compile({
            make_callback_copy(y, host_y),
            make_callback_copy(gx, host_gx)});

    for (auto &&ishp: {TensorShape{5}, TensorShape{4, 3},
            TensorShape{12, 3, 4}}) {
        *host_x = *gen(ishp);
        *host_loss_p = *gen({ishp.total_nr_elems()});

        func->execute();
        expected.resize(ishp);
        for (size_t i = 0, it = ishp.total_nr_elems(); i < it; ++ i) {
            expected.ptr<float>()[i] = std::pow(host_x->ptr<float>()[i], EXP);
        }
        MGB_ASSERT_TENSOR_EQ(expected, host_y);

        for (size_t i = 0, it = ishp.total_nr_elems(); i < it; ++ i) {
            expected.ptr<float>()[i] = EXP * host_loss_p->ptr<float>()[i] *
                std::pow(host_x->ptr<float>()[i], EXP - 1);
        }
        MGB_ASSERT_TENSOR_EQ(expected, host_gx);

        ASSERT_FALSE(used);
    }
}

TEST(TestOprLoop, ComputeWithoutCopyResult) {
    HostTensorGenerator<> gen;
    auto host_x = gen({23}), host_loss_p = gen({23 * 23});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto cb = [x](LoopDesc &desc){
        auto xv = desc.add_input(x);
        desc.add_output(xv, OutputMode::ALL);
        desc.set_loop_condition(desc.get_counter_var() <
                opr::GetVarShape::make(xv, 0) - 1);
    };
    auto y = opr::Loop::make(cb)[0],
         loss = opr::Dot::make(y.flatten(),
                 opr::Host2DeviceCopy::make(*graph, host_loss_p)),
         gx = cg::grad(loss, x);
    auto func = graph->compile({{gx, {}}});
    func->execute();
}

TEST(TestOprLoop, StaticLoopTimeInfer) {
    HostTensorGenerator<> gen;
    auto host_loop_time = gen({1});
    auto graph = ComputingGraph::make();
    auto host_shp = gen({2});
    auto loop_time = opr::MarkDynamicVar::make(
            opr::Host2DeviceCopy::make_no_value_infer(*graph, host_loop_time)),
         shp = opr::Host2DeviceCopy::make(*graph, host_shp);
    // actual loop time is loop_time + shp
    auto desc_maker = [&](LoopDesc &loop_desc) {
        auto x = loop_desc.add_input_assignable(loop_time.make_scalar(0)),
             xnext = x + 1;
        loop_desc.assign(x, xnext);
        loop_desc.add_output(xnext, OutputMode::LAST);
        auto cnt = loop_desc.get_counter_var(),
             dest = loop_desc.add_input(loop_time) - 1 +
                 opr::GetVarShape::make(loop_desc.add_input(shp));
        loop_desc.set_loop_condition(cnt < dest);
    };
    auto y = opr::Loop::make(desc_maker)[0];
    HostTensorND host_y;
    auto f = graph->compile({make_callback_copy(y, host_y)});

    ASSERT_TRUE(LoopTest::is_static_loop_time(y.node()->owner_opr()));

    host_loop_time->ptr<float>()[0] = 10;
    f->execute();
    ASSERT_EQ(12, host_y.ptr<int>()[0]);

    *host_shp = *gen({5});
    f->execute();
    ASSERT_EQ(15, host_y.ptr<int>()[0]);

    host_loop_time->ptr<float>()[0] = 20;
    f->execute();
    ASSERT_EQ(25, host_y.ptr<int>()[0]);
}

TEST(TestOprLoop, StaticOutputShape) {
    auto run = [](bool dyn) {
        HostTensorGenerator<> gen;
        auto host_loop_time = gen({1}),
             host_delta = gen({1}),
             host_x0 = gen({1});

        host_loop_time->ptr<float>()[0] = 1;
        auto graph = ComputingGraph::make();
        auto host_shp = gen({2});
        auto loop_time = opr::Host2DeviceCopy::make(*graph, host_loop_time),
             shp = opr::Host2DeviceCopy::make(*graph, host_shp),
             delta = opr::Host2DeviceCopy::make(*graph, host_delta),
             x0 = opr::Host2DeviceCopy::make(*graph, host_x0);

        if (dyn)
            loop_time = opr::MarkDynamicVar::make(loop_time);

        // actual loop time is loop_time + shp
        auto desc_maker = [&](LoopDesc &loop_desc) {
            auto x = loop_desc.add_input_assignable(x0),
                 xnext = x + loop_desc.add_input(delta);
            loop_desc.assign(x, xnext);
            loop_desc.add_output(x, OutputMode::ALL);
            auto cnt = loop_desc.get_counter_var(),
                 dest = loop_desc.add_input(loop_time) - 1 +
                     opr::GetVarShape::make(loop_desc.add_input(shp));
            loop_desc.set_loop_condition(cnt < dest);
        };
        auto y = opr::Loop::make(desc_maker)[0],
             loss = opr::Dot::make(y.flatten(), y.flatten()),
             gx = cg::grad(loss, x0),
             gd = cg::grad(loss, delta);

        if (!dyn) {
            ASSERT_EQ(TensorShape({3, 1}), y.node()->shape());
        }

        HostTensorND host_y, host_gx, host_gd;
        auto f = graph->compile({
                make_callback_copy(y, host_y),
                make_callback_copy(gx, host_gx),
                make_callback_copy(gd, host_gd)});

        ASSERT_TRUE(LoopTest::is_static_loop_time(y.node()->owner_opr()));

        HostTensorND y_expect{host_loop_time->comp_node(), dtype::Float32()},
                     gx_expect, gd_expect;
        gx_expect.copy_from(*host_x0);
        gd_expect.copy_from(gx_expect);

        auto run = [&](size_t sz0, size_t sz1) {
            *host_delta = *gen({1});
            *host_x0 = *gen({1});
            host_loop_time->ptr<float>()[0] = sz0;
            if (host_shp->shape(0) != sz1)
                *host_shp = *gen({sz1});

            y_expect.resize({sz0 + sz1, 1});

            float delta = host_delta->ptr<float>()[0], x0 = host_x0->ptr<float>()[0],
                  gx = 0, gd = 0;
            auto n = sz0 + sz1;
            for (size_t i = 0; i < n; ++ i) {
                auto cur = x0 + delta * i;
                y_expect.ptr<float>()[i] = cur;
                gx += cur * 2;
                gd += cur * 2 * i;
            }

            gx_expect.ptr<float>()[0] = gx;
            gd_expect.ptr<float>()[0] = gd;
            f->execute();
        };

#define RUN(sz0, sz1) \
        do { \
            run(sz0, sz1); \
            MGB_ASSERT_TENSOR_EQ(y_expect, host_y); \
            MGB_ASSERT_TENSOR_EQ(gx_expect, host_gx); \
            MGB_ASSERT_TENSOR_EQ(gd_expect, host_gd); \
        } while(0)

        RUN(1, 2);
        RUN(1, 5);
        RUN(8, 5);
#undef RUN
    };
    run(false);
    run(true);
}

TEST(TestOprLoop, CounterEdgeCases) {
    auto run = [&](
            thin_function<SymbolVar(SymbolVar)> cond, int expected_value) {
        auto graph = ComputingGraph::make();
        auto desc_maker = [&](LoopDesc &desc) {
            auto x = desc.add_input_assignable(SymbolVar::make_scalar(0, *graph,
                        CompNode::load("xpu0"))),
                 xnext = x + 1;
            desc.set_loop_condition(cond(desc.get_counter_var()));
            desc.assign(x, xnext);
            desc.add_output(xnext, OutputMode::LAST);
        };
        auto x = opr::Loop::make(desc_maker)[0];
        HostTensorND host_x;
        auto func = graph->compile({make_callback_copy(x, host_x)});
        func->execute();
        ASSERT_EQ(expected_value, host_x.ptr<int>()[0]);
    };

    run([](SymbolVar c){return c < 5;}, 6);
    run([](SymbolVar c){return c <= 5;}, 7);
    run([](SymbolVar c){return c < 5.f;}, 6);
    run([](SymbolVar c){return c < 5.2f;}, 7);
    run([](SymbolVar c){return c <= 5.f;}, 7);
    run([](SymbolVar c){return c <= 5.2f;}, 7);

    run([](SymbolVar c){return c < -1;}, 1);
    run([](SymbolVar c){return c <= -1;}, 1);
    run([](SymbolVar c){return c < .2f;}, 2);
    run([](SymbolVar c){return c <= .2f;}, 2);

    run([](SymbolVar c){return c < 0;}, 1);
    run([](SymbolVar c){return c <= 0;}, 2);
    run([](SymbolVar c){return c < 0.f;}, 1);
    run([](SymbolVar c){return c <= 0.f;}, 2);
}

TEST(TestOprLoop, OutputDType) {
    static constexpr int LOOP_TIME = 4;
    using Checker = AutoOprChecker<1, 1>;

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
        using Cvt = opr::TypeCvt;
        auto desc_maker = [xout=inputs[0]](LoopDesc &desc) {
            auto x = desc.add_input_assignable(xout),
                 xnext = Cvt::make(x + 1, dtype::Int32());
            desc.add_output(xnext * desc.get_counter_var(), OutputMode::SUM);
            desc.set_loop_condition(desc.get_counter_var() < LOOP_TIME);
            desc.assign(x, Cvt::make(xnext, dtype::Float32()));
        };
        auto y = opr::Loop::make(desc_maker)[0];
        bool succ = false;
        auto chk = [&]() {
            ASSERT_EQ(DTypeEnum::Float32, inputs[0].dtype().enumv());
            ASSERT_EQ(DTypeEnum::Int32, y.dtype().enumv());
            succ = true;
        };
        chk();
        mgb_assert(succ);
        return {Cvt::make(y, dtype::Float32())};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        dest[0].resize(inp[0]->shape());
        auto p0 = inp[0]->ptr<float>(), pt = dest[0].ptr<float>();
        for (size_t i = 0, it = dest[0].layout().total_nr_elems();
                i < it; ++ i) {
            float v = p0[i];
            int ret = 0;
            for (int j = 0; j <= LOOP_TIME; ++ j) {
                int vnext = v + 1;
                v = vnext;
                ret += vnext * j;
            }
            pt[i] = ret;
        }
    };

    HostTensorGenerator<> gen;
    auto genx = [&](HostTensorND &dest) {
        dest = *gen(dest.shape());
        auto ptr = dest.ptr<float>();
        for (size_t i = 0, it = dest.layout().total_nr_elems();
                i < it; ++ i) {
            float iv, fv;
            fv = std::modf(ptr[i] * 10, &iv);
            if (fv < 0) {
                fv += 1;
                iv -= 1;
            }
            if (fv <= 0.1f)
                fv += 0.5f;
            else if (fv >= 0.9f)
                fv -= 0.5f;
            ptr[i] = iv + fv;
        }
    };

    Checker{make_graph, fwd}.
        disable_multi_loss_check().
        set_input_generator(0, genx).
        run({TensorShape{2}}).
        run({TensorShape{3}}).
        run({TensorShape{2, 3, 5}});
}

TEST(TestOprLoop, MutableStateSaverOnlyNecessary) {

    using Checker = AutoOprChecker<2, 4>;

    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
        1e-2, 1};
    auto genx = [&](HostTensorND &dest) {
        dest = *gen(dest.shape());
    };

    auto host_loop_time = std::make_shared<HostTensorND>(
            CompNode::load("xpu0"), dtype::Int32());
    int& loop_time = host_loop_time->resize({1}).ptr<int>()[0];
    loop_time = 1;

    std::unordered_map<VarNode*, bool> expected_var_rec_spec;
    cg::OperatorNodeBase *loop_opr = nullptr;
    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
        auto loop_time = opr::Host2DeviceCopy::make(
                *inputs[0].node()->owner_graph(), host_loop_time);

        auto desc_maker = [&expected_var_rec_spec, loop_time,
                xi=inputs[0], yi=inputs[1]](LoopDesc &desc) {
            auto
                // value unused in grad
                x0 = desc.add_input_assignable(xi).rename("x0"),
                // already output all
                x1 = desc.add_input_assignable(xi).rename("x1"),
                // normal
                x2 = desc.add_input_assignable(xi).rename("x2"),
                // grad not taken
                y0 = desc.add_input_assignable(yi).rename("y0");

            auto x = desc.add_input(xi).rename("x"),
                 y = desc.add_input(yi).rename("y"),
                 cnt = desc.get_counter_var();

            desc.assign(x0, x0 + cnt);
            desc.assign(x1, opr::pow(x1, x.make_scalar(.1f)) * x);
            desc.assign(x2, opr::pow(x2, x.make_scalar(.2f)) * x);
            desc.assign(y0, opr::pow(y0, y.make_scalar(.3f)) * y);
            desc.add_output(x0, OutputMode::SUM);
            desc.add_output(x1, OutputMode::ALL);
            desc.add_output(x2, OutputMode::LAST);
            desc.add_output(y0, OutputMode::LAST);
            desc.set_loop_condition(cnt < desc.add_input(loop_time) - 1);

            expected_var_rec_spec = {
                {x0.node(), false},
                {x2.node(), true},
                {y0.node(), false},
            };
        };
        auto y = opr::Loop::make(desc_maker);
        loop_opr = y[0].node()->owner_opr();
        mgb_assert(y.size() == 4);
        return {y[0], y[1], y[2], y[3]};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        dest[0].resize(inp[0]->shape());
        {
            TensorLayout shp1{inp[0]->shape(), dtype::Byte()};
            shp1.add_axis_inplace(0, loop_time, 0);
            dest[1].resize(shp1);
        }
        dest[2].resize(inp[0]->shape());
        dest.back().resize(inp[1]->shape());
        auto px = inp[0]->ptr<float>(),
             o0 = dest[0].ptr<float>(),
             o1 = dest[1].ptr<float>(),
             o2 = dest[2].ptr<float>();
        auto sx = inp[0]->shape().total_nr_elems();
        for (size_t i = 0; i < sx; ++ i) {
            auto x = px[i];
            o0[i] = 0;
            o1[i] = x;
            auto o0cur = x, o2cur = x;
            for (int j = 0, cont = true; cont; ) {
                cont = j < loop_time - 1;
                o0[i] += o0cur;
                o0cur += j;
                if (j)
                    o1[j * sx + i] = std::pow(o1[(j-1) * sx + i], .1f) * x;
                o2[i] = o2cur;
                o2cur = std::pow(o2cur, .2f) * x;

                ++ j;
            }
        }

        auto py = inp[1]->ptr<float>(), o3 = dest[3].ptr<float>();
        auto sy = inp[1]->shape().total_nr_elems();
        for (size_t i = 0; i < sy; ++ i) {
            auto y = py[i], ans = y;
            for (int j = 0; j < loop_time - 1; ++ j)
                ans = std::pow(ans, .3f) * y;
            o3[i] = ans;
        }
    };

    Checker checker{make_graph, fwd};
    checker.disable_multi_loss_check();
    Checker::RunOptions opt;
    opt.numdiff_eps = 1e-3;
    opt.numdiff_max_err = 5e-3;

    bool var_rec_spec_checked = false;
    auto on_grad_computed = [&](cg::ComputingGraph *, cg::AsyncExecutable *) {
        auto var_rec_spec = LoopTest::var_rec_spec(loop_opr);
#define CHK(a, b) \
        do { \
            for (auto &&i: a) { \
                auto iter = b.find(i.first); \
                ASSERT_TRUE(iter != b.end()) << \
                    ssprintf("%s in %s, but not in %s", \
                            i.first->cname(), #a, #b); \
                ASSERT_TRUE(i.second == iter->second) << \
                    ssprintf("var=%s %s=%d %s=%d", \
                            i.first->cname(), #a, i.second, #b, iter->second); \
            }; \
        } while(0)
        CHK(var_rec_spec, expected_var_rec_spec);
        CHK(expected_var_rec_spec, var_rec_spec);
#undef CHK
        var_rec_spec_checked  = true;
    };

    checker.
        on_grad_computed(on_grad_computed).
        set_input_allow_grad(1, false).
        set_input_generator(0, genx).
        set_input_generator(1, genx);

    for (loop_time = 1; loop_time <= 4; ++ loop_time) {
        var_rec_spec_checked = false;
        checker.
            run({TensorShape{2}, {3}}, opt).
            run({TensorShape{3}, {2}}, opt).
            run({TensorShape{2, 3, 2}, {size_t(2 + loop_time)}}, opt).
            run({TensorShape{2}, {3}}, opt);
        ASSERT_TRUE(var_rec_spec_checked);
    }
}

namespace {

void test_null_grad(bool dyn) {
    auto d = [dyn](SymbolVar var) -> SymbolVar {
        if (dyn)
            var = opr::MarkDynamicVar::make(var).node();
        return var;
    };

    auto zg = [](SymbolVar var) {
        return opr::SetGrad::make(var, opr::SetGrad::zero_grad);
    };

    auto spow = [&](SymbolVar a, float p) {
        return opr::pow(d(a), a.make_scalar(p));
    };

    constexpr int LOOP_TIME = 1;
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
        1e-2, 1};
    constexpr size_t SIZE0 = 23, SIZE1 = 32;
    auto host_x0 = gen({SIZE0}),
         host_x1 = gen({SIZE1});

    auto graph = ComputingGraph::make();
    auto outgraph_x0 = opr::Host2DeviceCopy::make(*graph, host_x0),
         outgraph_x1 = opr::Host2DeviceCopy::make(*graph, host_x1);

    auto desc_maker = [&](LoopDesc &desc) {
        auto x0a = desc.add_input_assignable(outgraph_x0).rename("x0a"),
             x0b = desc.add_input_assignable(outgraph_x0).rename("x0b"),

             // one value as multiple assignors
             val0 = (spow(x0a, 0.4) * spow(x0b, 0.5)).rename("val0"),

             x0c = desc.add_input_assignable(outgraph_x0).rename("x0c"),
             x0c1 = desc.add_input(outgraph_x0).rename("x0c1"),
             // assignee null grad
             val1 = (spow(zg(x0c), 0.5) * x0c1).rename("val1"),

             x0d = desc.add_input_assignable(outgraph_x0).rename("x0d"),
             // assignor null grad
             val2 = (spow(x0d, 0.6) * x0c1).rename("val2"),

             // null outgrad in par graph
             x1 = desc.add_input(outgraph_x1).rename("x1"),
             x1a = desc.add_input_assignable(outgraph_x1).rename("x1a"),
             val3 = (spow(x1a, 0.7) * x1).rename("val3");

        desc.assign(x0a, val0);
        desc.assign(x0b, val0);
        desc.assign(x0c, val1);
        desc.assign(x0d, val2);
        desc.assign(x1a, val3);

        desc.add_output(val0, OutputMode::LAST);
        desc.add_output(val1, OutputMode::LAST);
        desc.add_output(zg(val2), OutputMode::LAST);
        desc.add_output(val3, OutputMode::LAST);

        desc.set_loop_condition(desc.get_counter_var() < LOOP_TIME - 1);
    };

    auto y = opr::Loop::make(desc_maker);

    // multi grad
    cg::grad(opr::Dot::make(y[2], y[2]), outgraph_x0);

    auto sum = [](SymbolVar x) {
        return opr::reduce_sum(x, x.make_scalar(1));
    };
    auto loss = sum(y[0]) + sum(y[1]) + sum(y[2]);
    auto gx0 = cg::grad(loss, outgraph_x0),
         gx1 = cg::grad(loss, outgraph_x1, true, false);
    ASSERT_EQ(nullptr, gx1.node());

    std::array<HostTensorND, 4> host_y, expect_y;
    HostTensorND host_gx0;

    auto func = graph->compile({
            make_callback_copy(gx0, host_gx0),
            make_callback_copy(y[0], host_y[0]),
            make_callback_copy(y[1], host_y[1]),
            make_callback_copy(y[2], host_y[2]),
            make_callback_copy(y[3], host_y[3]),
            });
    func->execute();

    for (size_t i = 0; i < 3; ++ i)
        expect_y[i].copy_from(*host_x0);
    expect_y[3].copy_from(*host_x1);

    HostTensorND expect_gx0;
    expect_gx0.copy_from(*host_x0);
    memset(expect_gx0.raw_ptr(), 0, expect_gx0.layout().span().dist_byte());

    auto px0 = host_x0->ptr<float>(), px1 = host_x1->ptr<float>(),
         pgx0 = expect_gx0.ptr<float>();

    {
        // y0
        auto p = expect_y[0].ptr<float>();
        float dpow = std::pow(0.9f, LOOP_TIME);
        for (size_t i = 0; i < SIZE0; ++ i) {
            p[i] = std::pow(px0[i], dpow);
            pgx0[i] += dpow * std::pow(px0[i], dpow - 1.f);
        }
    }
    {
        // y1
        auto p = expect_y[1].ptr<float>();
        for (size_t i = 0; i < SIZE0; ++ i) {
            float x0c = px0[i], grad = 0;
            for (int j = 0; j < LOOP_TIME; ++ j) {
                x0c = std::pow(x0c, .5f);
                grad += x0c;
                x0c *= px0[i];
            }
            pgx0[i] += grad;
            p[i] = x0c;
        }
    }
    {
        // y2
        auto p = expect_y[2].ptr<float>();
        for (size_t i = 0; i < SIZE0; ++ i) {
            float x0d = px0[i];
            for (int j = 0; j < LOOP_TIME; ++ j) {
                x0d = std::pow(x0d, .6f) * px0[i];
            }
            p[i] = x0d;
        }
    }
    {
        // y3
        auto p = expect_y[3].ptr<float>();
        for (size_t i = 0; i < SIZE1; ++ i) {
            float x1 = px1[i];
            for (int j = 0; j < LOOP_TIME; ++ j) {
                x1 = std::pow(x1, .7f) * px1[i];
            }
            p[i] = x1;
        }
    }

    for (size_t i = 0; i < 4; ++ i)
        MGB_ASSERT_TENSOR_EQ(expect_y[i], host_y[i]) << "fail at " << i;

    MGB_ASSERT_TENSOR_EQ(expect_gx0, host_gx0);
}

} // anonymous namespace

TEST(TestOprLoop, NullGrad) {
    test_null_grad(false);
}

TEST(TestOprLoop, NullGradDyn) {
    test_null_grad(true);
}

TEST(TestOprLoop, ImmutableTensorFwd) {
    auto graph = ComputingGraph::make();
    auto x = SymbolVar::make_scalar(132, *graph, CompNode::load("xpu0"));

    auto desc_maker = [&](LoopDesc &desc) {
        auto x0 = desc.add_input(x),
             x1 = desc.add_input_assignable(x);
        ASSERT_TRUE(x0.node()->owner_opr()->same_type<opr::ImmutableTensor>());
        desc.add_output(x0, OutputMode::LAST);
        desc.add_output(x1, OutputMode::LAST);
        desc.assign(x1, x1 + 1);
        desc.set_loop_condition(desc.get_counter_var() < 1);
    };
    auto y = opr::Loop::make(desc_maker);
    HostTensorND host_y[2];
    auto func = graph->compile({
            make_callback_copy(y[0], host_y[0]),
            make_callback_copy(y[1], host_y[1])});
    func->execute();
    ASSERT_EQ(132, host_y[0].ptr<int>()[0]);
    ASSERT_EQ(133, host_y[1].ptr<int>()[0]);
}

TEST(TestOprLoop, InputChangeCompStream) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({23}, cns[0]);
    auto cn1 = cns[1],
         cn1_copy = cn1.change_stream(CompNode::Stream::COPY);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         x1 = opr::Copy::make(x, cn1);
    auto desc_maker = [x1](LoopDesc &desc) {
        auto xsub = desc.add_input(x1);
        desc.add_output(xsub, OutputMode::SUM);
        desc.set_loop_condition(xsub.make_scalar(0));
    };
    auto y = opr::Loop::make(desc_maker)[0];
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    if (cn1.mem_node() != host_x->comp_node().mem_node()) {
        ASSERT_EQ(cn1_copy, x1.node()->comp_node());
    }
    ASSERT_EQ(cn1, y.node()->comp_node());
    MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
}

TEST(TestOprLoop, VisitInpSub) {
    LoopTest::check_output_recorder_sum_optimize_success() = false;
    using Checker = AutoOprChecker<1, 1>;

    // sum(a[i:i+2]**2 for i in range(s.shape[0] - 1))

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        auto desc_maker = [xout=inputs[0]](LoopDesc &desc) {
            auto x = desc.add_input(xout),
                 i = desc.get_counter_var(),
                 xsub = opr::Subtensor::make(
                         x, {opr::Subtensor::AxisIndexer::make_interval(
                                 0, i, i + 2, None)});
            desc.add_output(opr::pow(xsub, x.make_scalar(2)), OutputMode::SUM);
            desc.set_loop_condition(i < opr::GetVarShape::make(x, 0) - 2);
        };
        return {opr::Loop::make(desc_maker)[0]};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto &&x = *inp[0];
        auto dshp = x.shape();
        dshp[0] = 2;
        size_t nr_col = dshp.total_nr_elems() / dshp.shape[0];
        auto px = x.ptr<float>(),
             py0 = dest[0].resize(dshp).ptr<float>(),
             py1 = py0 + nr_col;
        memset(py0, 0, sizeof(float) * nr_col * 2);
        for (size_t i = 0; i < x.shape()[0] - 1; ++ i) {
            auto xrow0 = px + i * nr_col,
                 xrow1 = xrow0 + nr_col;
            for (size_t j = 0; j < nr_col; ++ j) {
                py0[j] += xrow0[j] * xrow0[j];
                py1[j] += xrow1[j] * xrow1[j];
            }
        }
    };

    Checker{make_graph, fwd}.
        disable_multi_loss_check().
        run({TensorShape{3}}).
        run({TensorShape{2}}).
        run({TensorShape{4}}).
        run({TensorShape{5}}).
        run({TensorShape{10, 2, 3}});

    ASSERT_TRUE(LoopTest::check_output_recorder_sum_optimize_success());
}

TEST(TestOprLoop, VisitInpSubMavi) {
    LoopTest::check_output_recorder_sum_optimize_success() = false;
    using Checker = AutoOprChecker<1, 1>;

    // sum(a[[i, i+2]]**2 for i in range(s.shape[0] - 2))

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        auto desc_maker = [xout=inputs[0]](LoopDesc &desc) {
            auto x = desc.add_input(xout),
                 i = desc.get_counter_var(),
                 idx = opr::Concat::make({i, i + 2}, 0),
                 xsub = opr::IndexingMultiAxisVec::make(
                         x, {opr::Subtensor::AxisIndexer::make_index(0, idx)});
            desc.add_output(opr::pow(xsub, x.make_scalar(2)), OutputMode::SUM);
            desc.set_loop_condition(i < opr::GetVarShape::make(x, 0) - 3);
        };
        return {opr::Loop::make(desc_maker)[0]};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto &&x = *inp[0];
        auto dshp = x.shape();
        dshp[0] = 2;
        size_t nr_col = dshp.total_nr_elems() / dshp.shape[0];
        auto px = x.ptr<float>(),
             py0 = dest[0].resize(dshp).ptr<float>(),
             py1 = py0 + nr_col;
        memset(py0, 0, sizeof(float) * nr_col * 2);
        for (size_t i = 0; i < x.shape()[0] - 2; ++ i) {
            auto xrow0 = px + i * nr_col,
                 xrow1 = xrow0 + nr_col * 2;
            for (size_t j = 0; j < nr_col; ++ j) {
                py0[j] += xrow0[j] * xrow0[j];
                py1[j] += xrow1[j] * xrow1[j];
            }
        }
    };

    Checker{make_graph, fwd}.
        disable_multi_loss_check().
        run({TensorShape{3}}).
        run({TensorShape{4}}).
        run({TensorShape{10, 2, 3}});

    ASSERT_TRUE(LoopTest::check_output_recorder_sum_optimize_success());
}

TEST(TestOprLoop, AsyncDispatch) {
    constexpr int LOOP_TIME = 5;
    constexpr double SLEEP_TIME = 0.03;
    HostTensorGenerator<> gen;
    auto host_x = gen({128});
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    RealTimer timer;
    double time_loop_finish = -1;
    auto cb_rec_loop_finish = [&](DeviceTensorND&) {
        time_loop_finish = timer.get_secs();
    };
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto desc_maker = [x0=x](LoopDesc &desc) {
        auto x = desc.add_input_assignable(x0),
             i = desc.get_counter_var();
        i = opr::MarkDynamicVar::make(i);
        set_priority(i, -100);
        desc.add_output(x, OutputMode::SUM);
        desc.assign(x, opr::Sleep::make(x, SLEEP_TIME) + i);
        desc.set_loop_condition(desc.get_counter_var() < LOOP_TIME - 1);
    };
    auto ys = opr::Loop::make(desc_maker);
    auto y = opr::CallbackInjector::make(ys[0], cb_rec_loop_finish);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    timer.reset();
    ASSERT_EQ(time_loop_finish, -1.);
    func->execute();
    EXPECT_GE(time_loop_finish, 0);
    EXPECT_LT(time_loop_finish, SLEEP_TIME);

    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    auto used = timer.get_secs();
    if (used <= LOOP_TIME * SLEEP_TIME) {
        mgb_log_warn("expect time [%f > %f], got %f", used,
                     LOOP_TIME * SLEEP_TIME, used);
    }

    int bias = 0;
    for (int cur = 0, i = 0; i < LOOP_TIME; ++ i) {
        bias += cur;
        cur += i;
    }

    auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
    auto sz = host_x->shape(0);
    for (size_t i = 0; i < sz; ++ i) {
        MGB_ASSERT_FLOAT_EQ(px[i] * LOOP_TIME + bias, py[i]) <<
            ssprintf("failed at idx %zu: x=%g", i, px[i]);
    }
}

TEST(TestOprLoop, UnusedStaticInerCN) {
    REQUIRE_GPU(1);

    auto cn0 = CompNode::load("gpu0"),
         cn1 = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    auto host_x = std::make_shared<HostTensorND>(cn0, dtype::Float32());
    host_x->resize({2}).ptr<float>()[0] = 2.3;
    host_x->ptr<float>()[1] = 4.5;
    auto host_idx = std::make_shared<HostTensorND>(cn1, dtype::Int32());
    host_idx->resize({1}).ptr<int>()[0] = 0;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    // static dep on other comp node should be allowed
    auto desc_maker = [x0=x, &host_idx, cn0](LoopDesc &desc) {
        auto x = desc.add_input(x0);
        auto idx = opr::Host2DeviceCopy::make(
                *x.node()->owner_graph(), host_idx);
        idx = opr::Copy::make(idx, {cn0});
        auto y = opr::Subtensor::make(
                x, {opr::Subtensor::AxisIndexer::make_index(0, idx)});
        desc.add_output(y, OutputMode::LAST);
        desc.set_loop_condition(x.make_scalar(0));
    };
    auto y = opr::Loop::make(desc_maker)[0];
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    ASSERT_EQ(2.3f, host_y.ptr<float>()[0]);

    host_idx->ptr<int>()[0] = 1;
    func->execute();
    ASSERT_EQ(4.5f, host_y.ptr<float>()[0]);
}

TEST(TestOprLoop, ExtraVarDeps) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    int nr_call = 0;
    auto cb = [&nr_call](DeviceTensorND&) {
        ++ nr_call;
    };
    auto desc_maker = [&](LoopDesc &desc) {
        auto xi = desc.add_input(x),
             y = xi * 2 + 1;
        y.node()->owner_graph()->options().extra_vardeps[y.node()].push_back(
                opr::CallbackInjector::make(xi * 3 + 1, cb).node());
        desc.set_loop_condition(y.make_scalar(0));
        desc.add_output(y, OutputMode::LAST);
    };
    auto y = opr::Loop::make(desc_maker)[0];
    HostTensorND host_y, y_expect;
    auto func = graph->compile({make_callback_copy(y, host_y),
            make_callback_copy(x * 2 + 1, y_expect)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
    ASSERT_EQ(1, nr_call);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

