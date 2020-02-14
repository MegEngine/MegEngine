/**
 * \file src/gopt/test/basic_arith.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/numerical_diff.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/tensor_manip.h"

#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/basic_arith.h"

using namespace mgb;

using Elemwise = opr::Elemwise;
using Mode = Elemwise::Mode;

namespace {
    SymbolVar powc(SymbolVar x, float exp) {
        return opr::PowC::make(x, exp);
    }

    /*!
     * get all operands of a chain of element-wise oprs of the same mode
     */
    std::vector<VarNode*> expand_elem_chain(
            VarNode *var, opr::Elemwise::Mode mode) {
        if (var->owner_opr()->same_type<opr::Elemwise>()) {
            auto &&op = var->owner_opr()->cast_final<opr::Elemwise>();
            if (op.param().mode == mode) {
                auto ret = expand_elem_chain(op.input(0), mode);
                for (size_t i = 1; i < op.input().size(); ++ i) {
                    auto cur = expand_elem_chain(op.input()[i], mode);
                    ret.reserve(ret.size() + cur.size());
                    ret.insert(ret.end(), cur.begin(), cur.end());
                }
                return ret;
            }
        }
        return {var};
    }

    SymbolVar fma3(SymbolVar a, SymbolVar b, SymbolVar c) {
        return Elemwise::make({a, b, c}, Mode::FUSE_MUL_ADD3);
    }

    SymbolVar fma4(SymbolVar a, SymbolVar b, SymbolVar c, SymbolVar d) {
        return Elemwise::make({a, b, c, d}, Mode::FUSE_MUL_ADD4);
    }


} // anonymous namespace

TEST(TestGoptBasicArithInplace, EqToUnit) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2, 3}));
    auto a = x - x, b = x / x;
    ASSERT_EQ(a.as_immutable_scalar()->get_cast<float>(), 0.f);
    ASSERT_EQ(b.as_immutable_scalar()->get_cast<float>(), 1.f);
    TensorShape shp{2, 3};
    ASSERT_EQ(a.shape(), shp);
    ASSERT_EQ(b.shape(), shp);
}

TEST(TestGoptBasicArithInplace, ZeroOne) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_zero = gen({1}), host_one = gen({1}),
         host_x = gen({1});
    host_zero->ptr<float>()[0] = 0;
    host_one->ptr<float>()[0] = 1;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         zero = opr::ImmutableTensor::make(
                 *graph, *host_zero).broadcast({2, 3}),
         one = opr::ImmutableTensor::make(
                 *graph, *host_one).broadcast({2, 3});

    auto check_eq_1 = [&](SymbolVar y) {
        ASSERT_EQ(y.shape(), TensorShape({2, 3}));
        ASSERT_EQ(y.as_immutable_scalar()->get_cast<float>(), 1.f);
    };
    auto check_eq_x = [&](SymbolVar y) {
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(host_y.shape(), TensorShape({2, 3}));
        auto val = host_x->ptr<float>()[0];
        auto py = host_y.ptr<float>();
        for (size_t i = 0; i < 6; ++ i) {
            ASSERT_EQ(py[i], val);
        }
    };

    check_eq_x(zero + x);
    check_eq_x(one * x);
    check_eq_1(opr::pow(x, zero));
    check_eq_x(opr::pow(x, one));
    check_eq_1(opr::exp(zero));
}

TEST(TestGoptBasicArithInplace, Absorbing) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({1});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         zero = x.make_scalar(0).broadcast({2, 3});
    auto y = zero * x;
    ASSERT_EQ(y.shape(), TensorShape({2, 3}));
    ASSERT_EQ(y.as_immutable_scalar()->get_cast<float>(), 0.f);
}

TEST(TestGoptBasicArithInplace, LogExpExpand) {
    // test log(exp(a) * (exp(b) / (exp(c) * d**2))) -> a + b - c - log(d**2)

    using Checker = AutoOprChecker<4, 1>;
    using Mode = opr::Elemwise::Mode;
    auto make_graph = [&](const typename Checker::SymInpArray &inp) ->
            Checker::SymOutArray {
        SymbolVar a, b, c, d, x;
        auto chk = [&]() {
            ASSERT_EQ(a + (b - (c + opr::log(opr::powf(d, 2)))), x);
        };
        unpack_vector(SymbolVarArray(inp.begin(), inp.end()),
                a, b, c, d);
        x = opr::log(
                opr::exp(a) * (opr::exp(b) / (opr::exp(c) * opr::powf(d, 2)))
                );
        chk();
        return {x};
    };

    auto fwd = [&](typename Checker::NumOutArray &dest,
            typename Checker::NumInpArray inp) {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        auto i = [&](size_t idx) {
            return opr::Host2DeviceCopy::make(*graph, inp[idx]);
        };
        auto ans = opr::Elemwise::make(
                {opr::exp(i(0)) * opr::exp(i(1)) /
                (opr::exp(i(2)) * i(3) * i(3))},
                Mode::LOG);
        mgb_assert(expand_elem_chain(ans.node(), Mode::LOG).size() == 1);
        graph->compile({make_callback_copy(ans, dest[0])})->execute();
    };

    auto ms = [](const TensorShape &a, const TensorShape &b) ->
        Checker::ShapeInpArray {
        return {a, a, a, b};
    };
    Checker::RunOptions opt;
    opt.numdiff_eps = 1e-2;
    opt.numdiff_eps_single_inp[3] = 1e-3;
    opt.numdiff_max_err_single_inp[3] = 1e-2;
    Checker{make_graph, fwd}.
        run(ms({2, 3}, {2, 3}), opt).
        run(ms({1, 3}, {2, 3}), opt).
        run(ms({3, 2}, {1}), opt);

}

TEST(TestGoptBasicArithInplace, LogSumExp) {
    using Checker = AutoOprChecker<2, 1>;
    using Mode = opr::Elemwise::Mode;

    auto make_graph = [&](const typename Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        SymbolVar
            a = inputs[0], b = inputs[1],
            c = opr::Elemwise::make({opr::exp(a) + opr::exp(b)}, Mode::LOG);
        mgb_assert(Mode::LOG_SUM_EXP == c.node()->owner_opr()->
                cast_final_safe<opr::Elemwise>().param().mode);
        return {c};
    };

    auto fwd = [&](typename Checker::NumOutArray &dest,
            typename Checker::NumInpArray inp) {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = false;
        auto a = opr::Host2DeviceCopy::make(*graph, inp[0]),
             b = opr::Host2DeviceCopy::make(*graph, inp[1]),
             c = opr::Elemwise::make({opr::exp(a) + opr::exp(b)}, Mode::LOG);
        mgb_assert(Mode::LOG_SUM_EXP != c.node()->owner_opr()->
                cast_final_safe<opr::Elemwise>().param().mode);
        graph->compile({make_callback_copy(c, dest[0])})->execute();
    };

    Checker{make_graph, fwd}.
        run({TensorShape{1}, {5, 3}}).
        run({TensorShape{3, 1}, TensorShape{1, 4}}).
        run({TensorShape{5, 4}, TensorShape{5, 4}});
}

TEST(TestGoptBasicArithInplace, Log1pExpm1) {
    using Checker = AutoOprChecker<1, 2>;
    using Mode = opr::Elemwise::Mode;

    auto make_graph = [&](const typename Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        SymbolVar
            x = inputs[0],
            a = opr::Elemwise::make({x + 1}, Mode::LOG),
            b = opr::Elemwise::make({x}, Mode::EXP) - 1;
        mgb_assert(Mode::LOG1P == a.node()->owner_opr()->
                cast_final_safe<opr::Elemwise>().param().mode);
        mgb_assert(Mode::EXPM1 == b.node()->owner_opr()->
                cast_final_safe<opr::Elemwise>().param().mode);
        return {a, b};
    };

    auto fwd = [&](typename Checker::NumOutArray &dest,
            typename Checker::NumInpArray inp) {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = false;
        auto x = opr::Host2DeviceCopy::make(*graph, inp[0]),
             a = opr::Elemwise::make({x + 1}, Mode::LOG),
             b = opr::Elemwise::make({x}, Mode::EXP) - 1;
        mgb_assert(Mode::LOG1P != a.node()->owner_opr()->
                cast_final_safe<opr::Elemwise>().param().mode);
        mgb_assert(Mode::EXPM1 != b.node()->owner_opr()->
                cast_final_safe<opr::Elemwise>().param().mode);
        graph->compile({make_callback_copy(a, dest[0]),
                make_callback_copy(b, dest[1])})->execute();
    };


    auto ensure_noneg = [](Checker::NumInpArray inp) {
        auto sz = inp[0]->layout().total_nr_elems();
        auto ptr = inp[0]->ptr<float>();
        for (size_t i = 0; i < sz; ++ i) {
            ptr[i] = std::fabs(i + 0.5) - 0.5;
        }
    };

    Checker{make_graph, fwd}.
        set_input_coordinator(ensure_noneg).
        run({TensorShape{1}}).
        run({TensorShape{1, 3}}).
        run({TensorShape{5, 1}});
}

TEST(TestGoptBasicArithInplace, FloorDiv) {
    {
        // float: floor_div(x, 1) -> floor(x)
        HostTensorGenerator<> gen;
        auto host_x = gen({2, 1});
        auto graph = ComputingGraph::make();
        using Mode = Elemwise::Mode;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y0 = Elemwise::make({x, x.make_scalar(1)}, Mode::FLOOR_DIV),
             y0_expect = Elemwise::make({x}, Mode::FLOOR),
             y1 = Elemwise::make({x, x.make_scalar(1).broadcast({1, 2})},
                                 Mode::FLOOR_DIV);
        ASSERT_EQ(y0_expect, y0);
        ASSERT_FALSE(y1.node()->owner_opr()->same_type<Elemwise>());
        HostTensorND host_y1;
        auto func = graph->compile({make_callback_copy(y1, host_y1)});
        func->execute();
        ASSERT_EQ(TensorShape({2, 2}), host_y1.shape());

        auto px = host_x->ptr<float>(), py = host_y1.ptr<float>();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                ASSERT_EQ(std::floor(px[i]), py[i * 2 + j]);
            }
        }
    }

    {
        // int: floor_div(x, 1) -> x
        HostTensorGenerator<dtype::Int8> gen;
        auto host_x = gen({2, 1});
        auto graph = ComputingGraph::make();
        using Mode = Elemwise::Mode;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = Elemwise::make({x, x.make_scalar_dt(1).broadcast({1, 2})},
                                 Mode::FLOOR_DIV);
        HostTensorND host_y{dtype::Int8()};
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({2, 2}), host_y.shape());

        auto px = host_x->ptr<int8_t>(), py = host_y.ptr<int8_t>();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                ASSERT_EQ(px[i], py[i * 2 + j]);
            }
        }
    }
}

TEST(TestGoptBasicArith, GradSumMoveBroadcast) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({3, 4, 5}),
         host_l0 = gen({1, 4, 1}),
         host_l1 = gen({4});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         tshp = cg::var_from_tensor_shape(x, {1, 4, 1}).rename("tshp"),
         l0 = opr::Host2DeviceCopy::make(*graph, host_l0).rename("l0"),
         l1 = opr::Host2DeviceCopy::make(*graph, host_l1).rename("l1"),
         loss =
             opr::reduce_sum(
                 opr::MarkNoBroadcastElemwise::make(x) * l0,
                 x.make_scalar(1)) +
             opr::Dot::make(opr::reduce_sum(x, tshp).flatten(), l1),
         gx = cg::grad(loss, x);

    auto gx_opr = gx.node()->owner_opr();
    ASSERT_TRUE(gx_opr->same_type<opr::Broadcast>());

    HostTensorND host_loss, host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->to_json()->writeto_fpath(output_file(
                "TestGoptBasicArith.GradSumMoveBroadcast"));
    func->execute();

    func = graph->compile({make_callback_copy(loss, host_loss)});
    std::vector<HostTensorND*> inp{host_x.get()};
    auto get_loss = [&]() {
        func->execute();
        return host_loss.ptr<float>()[0];
    };
    auto num_gx = numerical_diff_pt2(inp, get_loss, {1e-2f})[0];
    MGB_ASSERT_TENSOR_NEAR(num_gx, host_gx, 1e-3);
}

TEST(TestGoptBasicArith, GradSumMoveIncrSubtensor) {
    constexpr size_t SIZE = 23;
    auto sum_sqr = [](SymbolVar i) {
        return opr::reduce_sum_sqr(i, i.make_scalar(1));
    };

    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE});
    auto mavi_idx0 = std::make_shared<HostTensorND>(host_x->comp_node(),
                TensorShape{2}, dtype::Int32()),
         mavi_idx1 = std::make_shared<HostTensorND>(host_x->comp_node(),
                TensorShape{1}, dtype::Int32());

    mavi_idx0->ptr<int>()[0] = 1;
    mavi_idx0->ptr<int>()[1] = 2;
    mavi_idx1->ptr<int>()[0] = 1;

    auto graph = ComputingGraph::make();
    using AI = opr::indexing::AxisIndexer;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         sub0 = opr::Subtensor::make(x, {AI::make_interval(
                     0, x.make_scalar(2), None, None)}),
         sub1 = opr::Subtensor::make(x, {AI::make_interval(
                     0, None, x.make_scalar(-2), None)}),
         sub2 = opr::IndexingMultiAxisVec::make(x, {AI::make_index(
                     0, opr::Host2DeviceCopy::make(*graph, mavi_idx0))}),
         sub3 = opr::IndexingMultiAxisVec::make(x, {AI::make_index(
                     0, opr::Host2DeviceCopy::make(*graph, mavi_idx1))}),
         loss = sum_sqr(sub0) + sum_sqr(sub1) + sum_sqr(sub2) + sum_sqr(sub3),
         gx = cg::grad(loss, x);

    {
        int nr_incr_sub = 0, nr_incr_mavi = 0;
        auto opr = gx.node()->owner_opr();
        for (; ; ) {
            if (opr->same_type<opr::IncrSubtensor>()) {
                ++ nr_incr_sub;
            } else if (opr->same_type<opr::IndexingIncrMultiAxisVec>()) {
                ++ nr_incr_mavi;
            } else {
                break;
            }
            opr = opr->input(0)->owner_opr();
        }
        ASSERT_EQ(nr_incr_sub, 2);
        ASSERT_EQ(nr_incr_mavi, 2);
    }

    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->execute();

    auto px = host_x->ptr<float>(),
         pgx = host_gx.ptr<float>();

    for (size_t i = 0; i < SIZE; ++ i) {
        float v0 = px[i] * 2, v = v0;
        if (i >= 2 && i < SIZE - 2)
            v += v0;
        if (i == 1)
            v += v0 * 2;
        if (i == 2)
            v += v0;
        MGB_ASSERT_FLOAT_EQ(v, pgx[i]) << "fail at " << i;
    }
}

TEST_PASS(ExpandFusedArithPass, FMA) {
    auto w = mkvar("w"), x = mkvar("x"), y = mkvar("y"), z = mkvar("z"),
         a = fma3(fma3(w, x, y),
                 fma4(w, x + y, y - z, z - w), z),
         b = (w * x + y) * (w * (x + y) + (y - z) * (z - w)) + z;
    check(b, a);
}

TEST_PASS(ExpandFusedArithPass, ADD) {
    using namespace opr;
    using M = opr::Elemwise::Mode;
    auto o = [](SymbolVar a, SymbolVar b, M m) {
        return Elemwise::make({a, b}, m);
    };
    auto w = mkvar("w"), x = mkvar("x"), y = mkvar("y"), z = mkvar("z"),
         a = o(w, x, M::FUSE_ADD_SIGMOID) + o(y, z, M::FUSE_ADD_TANH) +
             o(w, z, M::FUSE_ADD_RELU) + o(z, x, M::FUSE_ADD_H_SWISH),
         b = sigmoid(w + x) + tanh(z + y) + relu(w + z) + hswish(z + x);
    check(b, a);
}

TEST_PASS(NormalizeArithChainPass, 0) {
    // note that groundtruth is given in BFS order
    auto x = mkvar("x"), y = mkvar("y"), z = mkvar("z");
    check(z + y * (-2) + x * 2, x - y + z - (y - x));
    check(opr::Broadcast::make(z, opr::GetVarShape::make({z, y})) *
            opr::powf(x, -1),
            z * y / (x * y));

    // z - x has multiple readers, and thus is replaced as whole
    auto zmx = z + (-x);
    check(opr::powf(zmx, -1) * y * (-zmx + z * 2 + (-x)),
            y * (z - (x - z) - (z - x)) / (z - x));

    // test for leaf nodes with input replaced
    check((x + (-y)) * opr::powf(y + (-z), -1), (x - y) / (y - z));

    // single-inp opr
    check(opr::pow(opr::sin(y + (-x)), x + (-y)),
            opr::pow(opr::sin(y - x), x - y));

    // check x / y in float16 where y would be converted to (y ^ -1), and it
    // should keep the float16 dtype.
    auto x_fp16 = opr::TypeCvt::make(x, dtype::Float16()),
         y_fp16 = opr::TypeCvt::make(y, dtype::Float16());
    check(x_fp16 * opr::powf(y_fp16, -1), x_fp16 / y_fp16);
}

TEST_PASS(NormalizeArithChainPass, EndpointInDep) {
    auto x = mkvar("x"), y = mkvar("y"), z = mkvar("z"),
         a0_ = x - y,
         a1 = x + (-y),
         b0_ = a0_ / z,
         b1 = a1 * opr::powf(z, -1);

    SymbolVar a0, b0;
    unpack_vector(run_opt({a0_, b0_}), a0, b0);
    ASSERT_EQ(a1, a0);
    ASSERT_EQ(b1, b0);
}

TEST_PASS(NormalizeArithChainPass, Collapse) {
    auto a = opr::Host2DeviceCopy::make(*graph, gen({1})),
         b = opr::Host2DeviceCopy::make(*graph, gen({1})),
         m0 = a + a + a,
         m1 = m0 + a + a - a + (-a),
         m2 = m0 - a - a,
         p0 = b * b * b,
         p1 = p0 * b * b / b * opr::powf(b, -1),
         p2 = p0 / b / b;

    SymbolVar n0, n1, n2, q0, q1, q2;
    unpack_vector(run_opt({m0, m1, m2, p0, p1, p2}),
            n0, n1, n2, q0, q1, q2);

    auto check_broadcast = [](SymbolVar src, SymbolVar dst) {
        auto opr = dst.node()->owner_opr();
        ASSERT_TRUE(opr->same_type<opr::Broadcast>());
        ASSERT_EQ(src.node(), opr->input(0));
    };

    ASSERT_EQ(a * 3, n0);
    check_broadcast(n0, n1);
    ASSERT_EQ(n0 + a * (-2), n2);

    ASSERT_EQ(opr::powf(b, 3), q0);
    check_broadcast(q0, q1);
    ASSERT_EQ(q0 * opr::powf(b, -2), q2);
}

TEST_PASS(NormalizeArithChainPass, CoeffMerge) {
    auto a = opr::Host2DeviceCopy::make(*graph, gen({23}));
    SymbolVar b;
    unpack_vector(
            run_opt({1 / a * (1 / opr::powf(opr::powf(a, -33), 0.1))}),
            b);
    ASSERT_EQ(opr::powf(a, 2.3), b);
}

TEST_PASS(NormalizeArithChainPass, MulMerge) {
    auto a = opr::Host2DeviceCopy::make(*graph, gen({23}));
    SymbolVar b;
    unpack_vector(run_opt({1.f + (a * 1.2f) * 0.3f + a}), b);
    ASSERT_EQ(1.f + a * (1.2f * 0.3f + 1.f), b);
}

TEST_PASS(NormalizeArithChainPass, PowMerge) {
    auto a = opr::Host2DeviceCopy::make(*graph, gen({23}));
    SymbolVar b;
    unpack_vector(run_opt({2.3f + opr::powf(a * opr::powf(a, 1.2f), 0.3f)}), b);
    ASSERT_EQ(2.3f + opr::powf(a, (1.f + 1.2f) * 0.3f), b);
}

TEST_PASS(NormalizeArithChainPass, PowCExpand0) {
    auto a = opr::Host2DeviceCopy::make(*graph, gen({23}));
    SymbolVar b;
    unpack_vector(run_opt({powc(a, 2.3f)}), b);
    ASSERT_EQ(opr::powf(a, 2.3f), b);
}

TEST_PASS(NormalizeArithChainPass, PowCExpand1) {
    auto a = opr::Host2DeviceCopy::make(*graph, gen({23}));
    SymbolVar b;
    using opr::powf;
    unpack_vector(run_opt({powc(powf(powc(a, 1.2f) * powf(a, 2.3f), 0.4f) * a,
                                0.8f)}),
                  b);
    ASSERT_EQ(powf(a, ((1.2f + 2.3f) * 0.4f + 1.f) * .8f), b);
}

TEST(TestNormalizeArithChainPass, PowcCExpand2) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto a = opr::Host2DeviceCopy::make(*graph, gen({1}));
    using opr::powf;
    auto loss = a * 2;
    auto grad = powc(opr::VirtualGrad::make(loss, a), 1.6f);
    HostTensorND host_g;
    ASSERT_NO_THROW(
        graph->compile({make_callback_copy(grad, host_g)}));
}

TEST_PASS(ReorderArithChainPass, 0) {
    auto chk = [this](SymbolVar inp, SymbolVar expect) {
        check(expect, inp, gopt::ConstVarType::IMMUTABLE_AND_PARAM);
    };
    auto w1v0 = mkvar("w1v0", {1}), w1v1 = mkvar("w1v1", {1}),
         w5v0 = mkvar("w5v0", {5}), w5v1 = mkvar("w5v1", {5}),
         w5v2 = mkvar("w5v2", {5}),
         w1c0 = mkcvar("w1c0", {1}),
         w5c0 = mkcvar("w5c0", {5}), w5c1 = mkcvar("w5c1", {5}),
         w151v0 = mkvar("w151v0", {1, 5, 1}),
         w155v0 = mkvar("w155v0", {1, 5, 5}),
         w511v0 = mkvar("w511v0", {5, 1, 1});

    // mixed modes with shape match
    chk((w5c0 + (w1v1 + w5v1 + opr::powf(w1v1, 2)) + w5c1) * (w1v0 * w5v0),
         w5v0 * ((w1v1 + opr::powf(w1v1, 2)) + (w5v1 + (w5c0 + w5c1))) * w1v0);

    // const vars
    chk(w5v0 + w5c0 + w5c1,
        w5v0 + (w5c0 + w5c1));


    // const var with compatible shapes
    chk(w5c0 + w1v0 + w1c0,
        w5c0 + w1c0 + w1v0);

    // shape compatibility
    chk(w151v0 + w511v0 + w155v0,
        w151v0 + w155v0 + w511v0);

    // const-nonconst merge
    chk(w1c0 + (w151v0 + w155v0),
        (w1c0 + w151v0) + w155v0);

    {
        using namespace std::placeholders;
        auto run = [&](const SymbolVar& inp) -> SymbolVar {
            return run_opt({inp}, gopt::ConstVarType::IMMUTABLE_AND_PARAM)[0];
        };
        auto x0 = run(w5v0 + w5v1 + w5c0 + w5v2 + w1c0),
             x1 = run(w5c0 + w5v1 + w5v2 + w1c0 + w5v0);
        ASSERT_EQ(x0, x1);

        auto x = w5v1 + w5v2 + w5v0, y0 = run(x), y1 = run(y0);
        ASSERT_EQ(y0, y1);
    }
}

TEST_PASS(ArithFusePass, FMA) {
    auto a = mkvar("a", {1, 1}), b = mkvar("b", {1, 3}),
         c = mkvar("c", {1, 1}), d = mkvar("d", {1, 3}),
         e = mkvar("e", {2, 1}), f = mkvar("f", {2, 3}),
         g = mkvar("g", {1, 3}), h = mkvar("h", {2, 1}),
         i = mkvar("i", {1});
    check(fma4(a, b, c, d) + g + fma3(e, f, h),
            a * b + c * d + e * f + g + h);
    check(fma3(a, b, fma3(a, c, c)), b * a + c * a + c);
    check(opr::pow(opr::sin(fma3(a, b, c)), fma3(a, d, g)),
            opr::pow(opr::sin(a * b + c), a * d + g));
    check(fma3(f, g, fma4(b, i, d, i)),
            f * g + (b * i + d * i));
}

TEST_PASS(ArithFusePass, ADD) {
    auto add_sigmoid = [](SymbolVar a, SymbolVar b) {
        return Elemwise::make({a, b}, opr::Elemwise::Mode::FUSE_ADD_SIGMOID);
    };
    auto a = mkvar("a"), b = mkvar("b"), c = mkvar("c"),

         // fma is preferred
         f0 = opr::sigmoid(opr::relu(a * b + c) + a),
         g0 = add_sigmoid(opr::relu(fma3(a, b, c)), a),

         p0 = c + opr::relu(a + b * c),
         q0 = c + opr::relu(fma3(b, c, a)),

         // uniq reader check
         f1 = opr::sigmoid(opr::sigmoid(p0) + c) + opr::relu(p0),
         g1 = add_sigmoid(opr::sigmoid(q0), c) + opr::relu(q0),

         // triple replace
         f2 = opr::sigmoid(a + opr::relu(b * c + c)) * c + a,
         g2 = fma3(add_sigmoid(opr::relu(fma3(b, c, c)), a), c, a);

    check(g0, f0);
    check(g1, f1);
    check(g2, f2);
}

TEST_PASS(ArithFusePass, ADD_HSWISH) {
    auto add_hswish = [](SymbolVar a, SymbolVar b) {
        return Elemwise::make({b, a}, opr::Elemwise::Mode::FUSE_ADD_H_SWISH);
    };
    auto a = mkvar("a"), b = mkvar("b"), c = mkvar("c"),

         // fma is preferred
         f0 = opr::hswish(opr::relu(a * b + c) + a),
         g0 = add_hswish(opr::relu(fma3(a, b, c)), a),

         p0 = c + opr::relu(a + b * c),
         q0 = c + opr::relu(fma3(b, c, a)),

         // uniq reader check
         f1 = opr::hswish(opr::hswish(p0) + c) + opr::relu(p0),
         g1 = add_hswish(opr::hswish(q0), c) + opr::relu(q0),

         // triple replace
         f2 = opr::hswish(a + opr::relu(b * c + c)) * c + a,
         g2 = fma3(add_hswish(opr::relu(fma3(b, c, c)), a), c, a);

    check(g0, f0);
    check(g1, f1);
    check(g2, f2);
}

TEST_PASS(ArithMulDistributePass, 0) {
    auto a = mkvar("a", {3, 3}), b = mkvar("b", {3, 1}), c = mkvar("c", {3, 3}),
         d = mkvar("d", {3, 1}), e = mkvar("e", {3, 1});
    check(a * (b * e) * c + d * e, (a * b * c + d) * e);

    auto u = (a * b + c * d) * e, v = a * (b * e) + c * (d * e);
    check(v, u);
    check<false>(u + a * b, u + a * b);
}

TEST(TestGoptBasicArithPassFinalArithTransform, MergeNeg) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2, 3})),
         y = opr::Host2DeviceCopy::make(*graph, gen({1}));
    SymbolVar z0, z1;
    unpack_vector(
            gopt::GraphOptimizer{}.
            add_pass<gopt::NormalizeArithChainPass>().
            add_pass<gopt::FinalArithTransformPass>().
            apply({{
                x + (-y),
                x / opr::powf(x.make_scalar(1) / y, -1)
                }}).endpoint_vars(),
            z0, z1);
    ASSERT_EQ(x - y, z0);
    ASSERT_EQ(x / y, z1);
}

TEST(TestGoptBasicArithPassFinalArithTransform, MergeNeg2) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2, 3})),
         y = opr::Host2DeviceCopy::make(*graph, gen({1}));
    SymbolVar z0, z1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::NormalizeArithChainPass>()
                          .add_pass<gopt::FinalArithTransformPass>()
                          .apply({{(-x) + (-y), (1.f / x) * powc(y, -1)}})
                          .endpoint_vars(),
                  z0, z1);
    ASSERT_EQ(-(x + y), z0);
    ASSERT_EQ(powc(x * y, -1), z1);
}

TEST(TestGoptBasicArithPassFinalArithTransform, PowScalarMerge) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2, 3}));
    SymbolVar y0, y1, y2;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass<gopt::NormalizeArithChainPass>()
                    .add_pass<gopt::FinalArithTransformPass>()
                    .apply({{
                            powc(opr::powf(opr::pow(x, x), 1.2f), 0.5f),
                            powc(opr::pow(x.make_scalar(2.3f), x), -1.f),
                            powc(opr::pow(x.make_scalar(2.3f), -opr::sin(x)),
                                 -1.f),
                    }})
                    .endpoint_vars(),
            y0, y1, y2);
    ASSERT_EQ(opr::pow(x, 0.6f * x), y0);
    ASSERT_EQ(opr::pow(x.make_scalar(2.3f), -x), y1);
    ASSERT_EQ(opr::pow(x.make_scalar(2.3f), opr::sin(x)), y2);
}

TEST(TestGoptBasicArithPassFinalArithTransform, SumSqr) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2, 3}));
    SymbolVar y;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::NormalizeArithChainPass>()
                          .add_pass<gopt::FinalArithTransformPass>()
                          .apply({{opr::reduce_sum(x * x, x.make_scalar(1))}})
                          .endpoint_vars(),
                  y);
    auto expect = opr::reduce_sum_sqr(x, x.make_scalar(1));
    ASSERT_EQ(expect, y);
}

TEST(TestGoptBasicArithPassFinalArithTransform, ExpNeg) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2, 3})),
         e = opr::Host2DeviceCopy::make(*graph, gen({1}));
    SymbolVar y;
    unpack_vector(
            gopt::GraphOptimizer{}.
            add_pass<gopt::NormalizeArithChainPass>().
            add_pass<gopt::FinalArithTransformPass>().
            apply({{x.make_scalar(1) / opr::pow(x, e)}}).endpoint_vars(),
            y);
    ASSERT_EQ(opr::pow(x, -e), y);
}

TEST(TestGoptBasicArithPassFinalArithTransform, PowC) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2, 3}));
    SymbolVar y;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::NormalizeArithChainPass>()
                          .add_pass<gopt::FinalArithTransformPass>()
                          .apply({{opr::powf(powc(x, 1.2f) * x, 0.7f) + x * x}})
                          .endpoint_vars(),
                  y);
    ASSERT_EQ(powc(x, 2.f) + powc(x, (1.2f + 1.f) * 0.7f), y);
}

TEST(TestGoptBasicArithPassFinalArithTransform, ConstFoldingDType) {
    auto graph = ComputingGraph::make();
    auto a = SymbolVar::make_scalar(1.f, *graph, CompNode::load("xpu0")),
         b = a * 2,
         x = SymbolVar::make_scalar(1, *graph, CompNode::load("xpu0")),
         y = x * 2;
    ASSERT_EQ(dtype::Float32(), a.dtype());
    ASSERT_EQ(dtype::Float32(), b.dtype());
    ASSERT_EQ(dtype::Int32(), x.dtype());
    ASSERT_EQ(dtype::Int32(), y.dtype());
}

TEST(TestGoptBasicArith, TermCanceling) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto x = opr::Host2DeviceCopy::make_no_value_infer(*graph, host_x),
         y = 1 * x + 1 * (1 - x);
    SymbolVar y_opt;
    unpack_vector(
            gopt::GraphOptimizer{}.
            add_pass<gopt::NormalizeArithChainPass>().
            add_pass<gopt::ArithMulDistributePass>().
            add_pass<gopt::ReorderArithChainPass>(
                gopt::ConstVarType::IMMUTABLE).
            add_pass<gopt::FinalArithTransformPass>().
            apply({{y}}).endpoint_vars(),
            y_opt);
    ASSERT_FALSE(cg::is_static_var_value(y.node()));
    ASSERT_TRUE(cg::is_static_var_value(y_opt.node()));
    ASSERT_EQ(host_x->shape(), y_opt.shape());

    HostTensorND host_y_opt;
    auto func = graph->compile({make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    auto py = host_y_opt.ptr<float>();
    for (int i = 0; i < 6; ++ i) {
        ASSERT_EQ(1.f, py[i]);
    }
}

TEST(TestGoptBasicArith, ElemChainTopologicalOrder) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x0 = opr::Host2DeviceCopy::make(*graph, gen({1})),
         x1 = opr::Host2DeviceCopy::make(*graph, gen({1})),
         x2 = opr::Host2DeviceCopy::make(*graph, gen({1})),
         x3 = opr::Host2DeviceCopy::make(*graph, gen({2, 3})),
         x4 = opr::Host2DeviceCopy::make(*graph, gen({1})),
         tmp = x2 + (x3 + x4),
         y0 = (x0 + x1) + tmp,
         out0 = opr::relu(y0),
         out1 = opr::VirtualDep::make({tmp, out0}),
         out2 = opr::VirtualDep::make({y0, out1});
    auto dest_vars = gopt::GraphOptimizer{}.verbosity(2)
         .add_pass<gopt::ReorderArithChainPass>(gopt::ConstVarType::IMMUTABLE)
         .apply({{out0, out1, out2}})
         .endpoint_vars();
    ASSERT_EQ(dest_vars[0].node()->owner_opr()->input()[0]->id(),
              dest_vars[2].node()->owner_opr()->input()[0]->id());
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
