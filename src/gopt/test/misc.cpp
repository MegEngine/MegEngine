/**
 * \file src/gopt/test/misc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/misc.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/cond.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

using namespace mgb;

TEST_PASS(RemoveNonComputingOprPass, Simple) {
    auto x = mkvar("x");
    check(x, opr::MarkNoBroadcastElemwise::make(x));
}

TEST_PASS(RemoveNonComputingOprPass, Split) {
    auto a = mkvar("a"), b = mkvar("b"),
         loss = opr::reduce_sum(opr::Concat::make({a, b}, 0), a.make_scalar(1)),
         ga = cg::grad(loss, a),
         ga_exp = a.make_scalar(1.f).broadcast(ga.symshape());
    check(ga_exp, ga);
}

TEST_PASS(RemoveNonComputingOprPass, SplitImmOpt) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto cn0 = cns[0], cn1 = cns[1];
    auto host_x0 = gen({2, 3}, cn0),
         host_x1 = gen({2, 3}, cn1);
    auto graph = ComputingGraph::make();
    auto make1 = [&graph](SymbolVar var) {
        auto val = std::make_shared<HostTensorND>(
                var.node()->comp_node(), TensorShape{1}, dtype::Int32());
        val->ptr<int>()[0] = 1;
        return opr::Host2DeviceCopy::make(*graph, val);

    };
    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0),
         x1 = opr::Host2DeviceCopy::make(*graph, host_x1);
    auto splt = opr::Split::make(x0.make_scalar(0.f).broadcast({2}),
            opr::Split::Options::make_partition(0, {
                make1(x0), make1(x1)}),
            OperatorNodeConfig{}.comp_node_arr({cn0, cn1}));
    auto y0 = x0 + splt[0], y1 = x1 + splt[1];
    HostTensorND host_y0, host_y1;
    auto func = graph->compile({make_callback_copy(y0, host_y0),
            make_callback_copy(y1, host_y1)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_x1, host_y1);
}

TEST_PASS(DelayBroadcastPass, Basic) {
    auto x = mkvar("x", {1, 1, 3});
    auto y = mkvar("y", {1, 2, 3});
    auto z = mkvar("z", {2, 2, 3});

    auto relu_maker = [](SymbolVar x) -> SymbolVar {
        using Param = opr::Elemwise::Param;
        Param param;
        param.mode = Param::Mode::RELU;
        return opr::Elemwise::make({x}, param);
    };

    auto typecvt_maker = [](SymbolVar x, bool float16 = true) -> SymbolVar {
        if (float16)
            return opr::TypeCvt::make(x, dtype::Float16());
        else
            return opr::TypeCvt::make(x, dtype::Float32());
    };

    auto broadcast_maker = [](SymbolVar x, SymbolVar from) -> SymbolVar {
        return opr::Broadcast::make(x, opr::GetVarShape::make(from));
    };

    auto get_var_shp_maker = [](SymbolVar x) -> SymbolVar {
        return opr::GetVarShape::make(x);
    };

    // check just two oprs need swapping
    check(broadcast_maker(relu_maker(x), y), relu_maker(broadcast_maker(x, y)));

    // check multiple oprs need shifting
    check(broadcast_maker(typecvt_maker(relu_maker(x)), y),
          typecvt_maker(relu_maker(broadcast_maker(x, y))));

    // check opr::GetVarShape
    check(get_var_shp_maker(broadcast_maker(typecvt_maker(relu_maker(x)), y)),
          get_var_shp_maker(typecvt_maker(relu_maker(broadcast_maker(x, y)))));

    check(get_var_shp_maker(broadcast_maker(typecvt_maker(relu_maker(x)), y)),
          get_var_shp_maker(typecvt_maker(broadcast_maker(relu_maker(x), y))));

    check(typecvt_maker(get_var_shp_maker(broadcast_maker(relu_maker(x), y))),
          typecvt_maker(get_var_shp_maker(relu_maker(broadcast_maker(x, y)))));

    // remains the same after apply the pass.
    check<false>(broadcast_maker(broadcast_maker(x, y), z),
                 broadcast_maker(broadcast_maker(x, y), z));

    // mix.
    check(broadcast_maker(broadcast_maker(relu_maker(typecvt_maker(x)), y), z),
          relu_maker(broadcast_maker(typecvt_maker(broadcast_maker(x, y)), z)));

    // endpoint situation 1. See `DelayBroadcastPass::apply` comments.
    check(y + broadcast_maker(relu_maker(x), z),
          y + relu_maker(broadcast_maker(x, z)));

    // second replaced chain depend on another replaced chain.
    check(broadcast_maker(typecvt_maker(broadcast_maker(typecvt_maker(x), y) +
                                                typecvt_maker(y),
                                        false),
                          z),
          typecvt_maker(broadcast_maker(typecvt_maker(broadcast_maker(x, y)) +
                                                typecvt_maker(y),
                                        z),
                        false));

    // broadcast opr depend on another chain.
    auto shape3 = mkvar("shape3", {2}).symshape() + 1;
    auto shape333 = opr::abs(opr::Broadcast::make(shape3, shape3));
    auto shape333_after = opr::Broadcast::make(opr::abs(shape3), shape3);

    check(broadcast_maker(relu_maker(x), shape333_after),
          relu_maker(broadcast_maker(x, shape333)));
}

TEST_PASS(DelayBroadcastPass, Const) {
    auto x = mkvar("x", {5, 3});
    check(x.make_scalar(-1).broadcast(x.symshape()),
          -x.make_scalar(1).broadcast(x.symshape()));
}

TEST_PASS(DelayBroadcastPass, ScalarInput) {
    auto x = mkvar("x", {1}).reshape({1}), y = mkvar("y", {3, 1});
    check((x - y).broadcast({3, 5}), x - y.broadcast({3, 5}));
}

TEST_PASS(DelayBroadcastPass, LongChain) {
    auto x = mkvar("x", {1, 1, 3});
    auto y = mkvar("y", {1, 2, 3});
    auto z = mkvar("z", {2, 2, 3});

    auto relu = [](SymbolVar x) -> SymbolVar {
        using Param = opr::Elemwise::Param;
        Param param;
        param.mode = Param::Mode::RELU;
        return opr::Elemwise::make({x}, param);
    };

    auto bcast = [](SymbolVar x, SymbolVar from) -> SymbolVar {
        return opr::Broadcast::make(x, opr::GetVarShape::make(from));
    };

    // Do graph optimization first, then construct expected graph.
    // Note: DO NOT call `check` directly here, the \p inp and
    // \p expect of the `check` are in the same graph, some problems
    // would not be exposed due to the cache mechanism
    auto out = bcast(relu(bcast(relu(x), y)), z);
    out = gopt::GraphOptimizer{}.
        add_pass<gopt::DelayBroadcastPass>().
        apply({{out}}).endpoint_vars()[0];
    ASSERT_EQ(bcast(bcast(relu(relu(x)), y), z), out);
}

TEST_PASS(ExpandVirtualGradPass, Simple) {
    auto x = mkvar("x");
    check(x * 2,
          opr::VirtualGrad::make(opr::reduce_sum_sqr(x, x.make_scalar(1)), x));
}

TEST_PASS(ExpandVirtualGradPass, Dyncase) {
    auto x0 = mkvar("x"), x = opr::MarkDynamicVar::make(x0);
    check(opr::MarkDynamicVar::make(x * 2),
            opr::VirtualGrad::make(
                opr::reduce_sum_sqr(x, x.make_scalar(1)),
                x0));
}

TEST_F(TestGoptExpandVirtualGradPass, GradWrt) {
    graph->options().graph_opt_level = 0;
    auto x = mkvar("x", {2, 3});
    SymbolVar wrt;
    auto get_grad = [&wrt](const opr::SetGrad &g) -> SymbolVar {
        auto w = gopt::GraphOptimizer::var_replace_lookup(wrt.node());
        return cg::grad(cg::current_grad_target(*g.owner_graph()), w, false);
    };
    wrt = opr::SetGrad::make(x * 2 + 1, get_grad) * 3 + 1;

    auto gx = opr::VirtualGrad::make(
            opr::reduce_sum(wrt, wrt.make_scalar(1)),
            x);

    SymbolVar gx_opt;
    unpack_vector(
            gopt::GraphOptimizer{}.
            add_pass<gopt::ArithFusePass>().
            add_pass<gopt::ExpandVirtualGradPass>().
            verbosity(2).
            apply({{gx}}).endpoint_vars(),
            gx_opt);

    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx_opt, host_gx)});
    func->execute();
    ASSERT_EQ(x.shape(), host_gx.shape());

    auto pgx = host_gx.ptr<float>();
    for (size_t i = 0, it = host_gx.shape().total_nr_elems();
            i < it; ++ i) {
        ASSERT_EQ(2.f, pgx[i]);
    }
}

TEST_F(TestGoptExpandVirtualGradPass, VarReplaceLookup) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    auto host_x = gen({1});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    SymbolVar y;
    auto grad_getter = [&](const opr::SetGrad &) { return y; };
    auto a = opr::SetGrad::make(x, grad_getter);

    int counter = 0;
    auto callback = [&](DeviceTensorND &) { counter++; };
    y = opr::CallbackInjector::make(a * a, callback);

    auto grad = opr::VirtualGrad::make(y, x);

    HostTensorND host_y, host_grad;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(grad, host_grad)});

    func->execute();
    ASSERT_EQ(counter, 1);
}

TEST_PASS(RecompTypeCvtPass, Basic) {
    auto x = mkvar("x", {2, 3, 3});
    auto x_fp16 = opr::TypeCvt::make(x, dtype::Float16());
    auto sin_x = opr::sin(x_fp16);
    auto x_fp32 = opr::TypeCvt::make(sin_x, dtype::Float32());
    auto f = x_fp32;
    for (size_t i = 0; i < 20; ++i) {
        f = opr::sin(f);
    }
    auto for_pass = f + x_fp32;
    OperatorNodeConfig config = x_fp32.node()->owner_opr()->config();
    config.instance_id(for_pass.node()->owner_opr());
    auto expected = f + opr::TypeCvt::make(sin_x, dtype::Float32(),
            config);

    check(expected, for_pass, 0.1);
}

TEST_PASS(CombineAstypeAndReducePass, Grad) {
        auto data = mkvar("data", {10});
        auto x_fp16 = opr::relu(opr::TypeCvt::make(data, dtype::Float16()));
        auto x = opr::TypeCvt::make(x_fp16, dtype::Float32());
        SymbolVar tshp;
        using namespace opr;
        Reduce::Param param_i16_co32{Reduce::Mode::SUM, 0,
                                     Reduce::Param::DataType::FLOAT_O32xC32};
        Reduce::Param param_default{Reduce::Mode::SUM, 0,
                                    Reduce::Param::DataType::DEFAULT};
        auto y0 = opr::Reduce::make(x_fp16, param_i16_co32, tshp);
        auto y1 = opr::Reduce::make(x, param_default, tshp);
        auto grad0 = cg::grad(y0, data);
        auto grad1 = cg::grad(y1, data);

        HostTensorND host_grad0, host_grad1;
        auto func0 = graph->compile({make_callback_copy(grad0, host_grad0)});
        func0->execute();
        auto func1 = graph->compile({make_callback_copy(grad1, host_grad1)});
        func1->execute();
        MGB_ASSERT_TENSOR_EQ(host_grad0, host_grad1);
}

TEST_PASS(CombineAstypeAndReducePass, Basic) {
    for (auto&& axis : {MEGDNN_MAX_NDIM, 0}) {
        auto x = mkvar("x", {2, 3, 3});
        auto x_fp16 = opr::relu(opr::TypeCvt::make(x, dtype::Float16()));
        x = opr::TypeCvt::make(x_fp16, dtype::Float32());
        SymbolVar tshp;
        if (axis == MEGDNN_MAX_NDIM) {
            tshp = mkvar("tshp", {1, 3, 2}).symshape();
        }
        using namespace opr;
        Reduce::Param param_i16_co32{Reduce::Mode::SUM, axis,
                                     Reduce::Param::DataType::FLOAT_O32xC32};
        Reduce::Param param_default{Reduce::Mode::SUM, axis,
                                    Reduce::Param::DataType::DEFAULT};
        auto expected = opr::Reduce::make(x_fp16, param_i16_co32, tshp);
        auto get = opr::Reduce::make(x, param_default, tshp);
        check(expected, get);
    }
}

#if MGB_ENABLE_COND_EXEC

TEST(TestCondExec, GoptRemoveConstMask) {
    using MergeMode = opr::CondExecMerge::Mode;
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto run = [&](MergeMode merge_mode, int const_mask, int pred_mask,
                   bool expect_change) -> HostTensorND {
        auto host_pred0 = gen({1}), host_pred1 = gen({1});
        host_pred0->ptr<float>()[0] = pred_mask & 1;
        host_pred1->ptr<float>()[0] = pred_mask >> 1;
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        auto make_mark =
                [x, &graph](bool const_pred,
                            const std::shared_ptr<HostTensorND>& host_pred) {
                    SymbolVar pred;
                    if (const_pred) {
                        pred = opr::ImmutableTensor::make(*graph, *host_pred);
                    } else {
                        pred = opr::Host2DeviceCopy::make(*graph, host_pred);
                    }
                    SymbolVar ppv, ret;
                    unpack_vector(opr::CondExecPred::make(
                                          pred, {pred.make_scalar_dt(1)}),
                                  ppv);
                    unpack_vector(opr::CondExecMark::make(ppv, {x}), ret);
                    return ret;
                };
        SymbolVarArray merge_shp;
        if (merge_mode == MergeMode::SUM) {
            merge_shp.push_back(x.symshape());
        }
        auto xmark0 = make_mark(const_mask & 1, host_pred0) + 1.2f,
             xmark1 = make_mark(const_mask >> 1, host_pred1) * 2.3f,
             y = opr::CondExecMerge::make({xmark0, xmark1}, {1, merge_mode},
                                          merge_shp)[0];
        VarNodeArray y_opt_arr{y.node()};
        gopt::GraphOptimizer{}
                .add_pass<gopt::CondExecConstPredicateFolding>()
                .apply_inplace(y_opt_arr);
        SymbolVar y_opt = y_opt_arr[0];
        if (expect_change) {
            EXPECT_NE(y_opt.node(), y.node());
        } else {
            EXPECT_EQ(y_opt, y);
        }
        HostTensorND host_y;
        graph->options().graph_opt_level = 0;
        auto func = graph->compile({make_callback_copy(y_opt, host_y)});
        func->execute();
        return host_y;
    };

    for (size_t mode_num = 0;
         mode_num < opr::CondExecMerge::Param::MODE_NR_MEMBER; ++mode_num) {
        auto mode = static_cast<MergeMode>(mode_num);
        bool exact_one = (mode == MergeMode::EXACT_ONE ||
                          mode == MergeMode::EXACT_ONE_SAME_SHAPE);
        for (int pmask = 0; pmask < 4; ++pmask) {
            if (exact_one && (pmask & 1) + (pmask >> 1) != 1) {
                continue;
            }
            if (mode == MergeMode::SUM_COND_OUT && !pmask) {
                ASSERT_THROW(run(mode, 0b11, 0, false), GraphError);
                continue;
            }
            auto v0 = run(mode, 0b11, pmask, true);
            auto v1 = run(mode, 0b01, pmask, false);
            MGB_ASSERT_TENSOR_EQ(v0, v1);
        }
    }
}

#endif  // MGB_ENABLE_COND_EXEC

TEST_PASS(RemoveRedundantTypeCvtPass, Basic) {
#if !MEGDNN_DISABLE_FLOAT16
    auto x = mkvar("x", {2, 3, 3});
    auto x_fp16 = opr::TypeCvt::make(x, dtype::Float16());
    auto x_fp16_fp32 = opr::TypeCvt::make(x_fp16, dtype::Float32());
    auto x_fp16_fp32_fp16 = opr::TypeCvt::make(x_fp16_fp32, dtype::Float16());
    check(x_fp16, x_fp16_fp32_fp16);
#endif

    auto x_q8 = opr::TypeCvt::make(x, dtype::QuantizedS8(0.1f));
    auto x_q8_fp32 = opr::TypeCvt::make(x_q8, dtype::Float32());
    auto x_q8_fp32_q8 = opr::TypeCvt::make(x_q8_fp32, dtype::QuantizedS8(0.1f));
    auto x_q8_fp32_q8_ = opr::TypeCvt::make(x_q8_fp32, dtype::QuantizedS8(2.f));
    auto x_q8_q8 = opr::TypeCvt::make(x_q8, dtype::QuantizedS8(2.f));
    check(x_q8, x_q8_fp32_q8);
    check(x_q8_q8, x_q8_fp32_q8_);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
