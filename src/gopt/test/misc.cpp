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
#include "megbrain/opr/blas.h"
#include "megbrain/opr/cond.h"
#include "megbrain/opr/io.h"
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
    config.update_instance_id(for_pass.node()->owner_opr());
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

    auto x_i32 = opr::TypeCvt::make(x, dtype::Int32());
    auto x_i32_i16 = opr::TypeCvt::make(x_i32, dtype::Int16());
    auto x_i32_i16_i8 = opr::TypeCvt::make(x_i32_i16, dtype::Int8());
    auto x_i8 = opr::TypeCvt::make(x, dtype::Int8());
    check(x_i8, x_i32_i16_i8);

    auto x_q8 = opr::TypeCvt::make(x, dtype::QuantizedS8(0.1f));
    auto x_q8_fp32 = opr::TypeCvt::make(x_q8, dtype::Float32());
    auto x_q8_fp32_q8 = opr::TypeCvt::make(x_q8_fp32, dtype::QuantizedS8(0.1f));
    auto x_q8_fp32_q8_ = opr::TypeCvt::make(x_q8_fp32, dtype::QuantizedS8(2.f));
    auto x_q8_q8 = opr::TypeCvt::make(x_q8, dtype::QuantizedS8(2.f));
    check(x_q8, x_q8_fp32_q8);
    check(x_q8_q8, x_q8_fp32_q8_);
}

TEST_PASS(RemoveRedundantCopyPass, Basic) {
    auto x = mkvar("x", {2, 3, 3}, CompNode::load("cpu0"));
    {
        auto x_cpu1 = opr::Copy::make(x, CompNode::load("cpu1"));
        auto x_cpu0 = opr::Copy::make(x_cpu1, CompNode::load("cpu0"));
        auto x_cpu2 = opr::Copy::make(x_cpu0, CompNode::load("cpu2"));
        auto x_expected = opr::Copy::make(x, CompNode::load("cpu2"));
        check(x, x_cpu0);
        check(x_expected, x_cpu2);
    }

    {
        auto x_cpu1 = opr::Copy::make(x, CompNode::load("cpu1"));
        auto x_cpu2 = opr::Copy::make(x_cpu1, CompNode::load("cpu2"));
        auto x_cpu3 = opr::Copy::make(x_cpu2, CompNode::load("cpu3"));
        auto x_expected = opr::Copy::make(x, CompNode::load("cpu3"));
        check(x_expected, x_cpu3);
    }

    {
        auto x_cpu1 = opr::Copy::make(x, CompNode::load("cpu0:1"));
        auto x_cpu2 = opr::Copy::make(x_cpu1, CompNode::load("cpu0:2"));
        auto x_cpu3 = opr::Copy::make(x_cpu2, CompNode::load("cpu0:3"));
        auto x_expected = opr::Copy::make(x, CompNode::load("cpu0:3"));
        check(x_expected, x_cpu3);
    }

    {
        auto x_cpu1 = opr::Copy::make(x, CompNode::load("cpu0:1"));
        auto x_mt = opr::Copy::make(x_cpu1, CompNode::load("multithread8:0"));
        auto x_cpu3 = opr::Copy::make(x_mt, CompNode::load("cpu0:3"));
        auto x_expected = opr::Copy::make(x, CompNode::load("cpu0:3"));
        check(x_expected, x_cpu3);
    }

#if MGB_ATLAS
    {
        auto x_atlas0 = opr::Copy::make(x, CompNode::load("atlas0"));
        auto x_cpu2 = opr::Copy::make(x_atlas0, CompNode::load("cpu0:2"));
        auto x_cpu3 = opr::Copy::make(x_cpu2, CompNode::load("cpu0:3"));
        auto x_expected = opr::Copy::make(x, CompNode::load("cpu0:3"));
        check(x_expected, x_cpu3);
    }
#endif

#if MGB_CUDA
    {
        auto x_cuda0 = opr::Copy::make(x, CompNode::load("gpu0"));
        auto x_cpu2 = opr::Copy::make(x_cuda0, CompNode::load("cpu0:2"));
        auto x_cpu3 = opr::Copy::make(x_cpu2, CompNode::load("cpu0:3"));
        auto x_expected = opr::Copy::make(x, CompNode::load("cpu0:3"));
        check(x_expected, x_cpu3);
    }

    {
        auto x_mt = opr::Copy::make(x, CompNode::load("multithread8:0"));
        auto x_cpu2 = opr::Copy::make(x_mt , CompNode::load("gpu0:1"));
        auto x_cpu3 = opr::Copy::make(x_cpu2, CompNode::load("multithread8:0"));
        auto x_expected = opr::Copy::make(x, CompNode::load("multithread8:0"));
        check(x_expected, x_cpu3);
    }

#endif
}

#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/collective_comm.h"
#include "../../opr-mm/test/mock_client.h"

TEST_PASS(PackAllReduceScanPass, Basic) {
    auto graph = ComputingGraph::make();
    graph->options().allreduce_pack_max_size = 5000;

    auto client = std::make_shared<test::MockGroupClient>();
    auto cn = CompNode::load("gpux");

    auto dev_x0 = std::make_shared<DeviceTensorND>(cn, TensorShape{3, 5});
    auto dev_x1 = std::make_shared<DeviceTensorND>(cn, TensorShape{4, 6});
    auto dev_y0 = std::make_shared<DeviceTensorND>(cn, TensorShape{1});
    auto dev_y1 = std::make_shared<DeviceTensorND>(cn, TensorShape{1});

    auto x0 = opr::SharedDeviceTensor::make(*graph, dev_x0);
    auto x1 = opr::VolatileSharedDeviceTensor::make(*graph, dev_x1);
    auto y0 = opr::SharedDeviceTensor::make(*graph, dev_y0);
    auto y1 = opr::VolatileSharedDeviceTensor::make(*graph, dev_y1);

    auto grad0 = opr::VirtualGrad::make(y0, x0);
    auto grad1 = opr::VirtualGrad::make(y0, x1);
    auto grad2 = opr::VirtualGrad::make(y1, x0);
    auto grad3 = opr::VirtualGrad::make(y1, x1);

    auto mode = opr::CollectiveComm::Param::Mode::ALL_REDUCE_SUM;
    auto comm0 = opr::CollectiveComm::make({grad0}, graph.get(), "grad0", 2,
                                           false, 0, false, client, mode)[0];
    auto comm1 = opr::CollectiveComm::make({grad1}, graph.get(), "grad1", 2,
                                           false, 0, false, client, mode)[0];
    auto comm2 = opr::CollectiveComm::make({grad2}, graph.get(), "grad2", 2,
                                           false, 0, false, client, mode)[0];
    auto comm3 = opr::CollectiveComm::make({grad3}, graph.get(), "grad3", 2,
                                           false, 0, false, client, mode)[0];

    gopt::GraphOptimizer()
        .add_pass<gopt::PackAllReduceScanPass>()
        .apply({{comm0, comm1, comm2, comm3}});

    auto get_hash = [] (const SymbolVar& symvar) {
        cg::OperatorNodeBase* opr = symvar.node()->owner_opr();
        return opr->cast_final_safe<opr::CollectiveComm>().pack_hash();
    };
    uint64_t hash0 = get_hash(comm0);
    uint64_t hash1 = get_hash(comm1);
    uint64_t hash2 = get_hash(comm2);
    uint64_t hash3 = get_hash(comm3);

    ASSERT_EQ(hash0, hash1);
    ASSERT_EQ(hash2, hash3);
    ASSERT_NE(hash0, hash2);
}

TEST_PASS(PackAllReduceReplacePass, CollectGroups) {
    REQUIRE_GPU(2);
    auto cns = load_multiple_xpus(2);
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 2;

    auto cli0 = std::make_shared<test::MockGroupClient>("mock_addr0");
    auto cli1 = std::make_shared<test::MockGroupClient>("mock_addr1");

    using GroupInfo = gopt::PackAllReduceReplacePass::GroupInfo;
    ThinHashMap<uint64_t, std::shared_ptr<GroupInfo>> group_info;
    ThinHashMap<uint64_t, cg::OprNodeArray> groups;

    auto add_opr = [&] (const CompNode& cn, TensorShape shape, const DType& dt,
        std::shared_ptr<test::MockGroupClient> client, uint64_t extra_hash) {
        auto dev0 = std::make_shared<DeviceTensorND>(cn, shape, dt);
        auto wrt = opr::SharedDeviceTensor::make(*graph, dev0);

        auto dev1 = std::make_shared<DeviceTensorND>(cn, TensorShape{1}, dt);
        auto target = opr::SharedDeviceTensor::make(*graph, dev1);

        auto grad = opr::VirtualGrad::make(target, wrt);

        auto comm =
                opr::CollectiveComm::make(
                        {grad}, graph.get(), "key", 2, false, 0, false, client,
                        opr::CollectiveComm::Param::Mode::ALL_REDUCE_SUM)[0]
                        .node()
                        ->owner_opr();

        comm->cast_final_safe<opr::CollectiveComm>().set_pack_hash(extra_hash);

        return gopt::PackAllReduceReplacePass::collect_groups(comm, group_info, groups);
    };

    uint64_t hash0 = add_opr(cns[0], TensorShape{1, 3}, dtype::Float32{}, cli0, 1);
    uint64_t hash1 = add_opr(cns[0], TensorShape{2, 4}, dtype::Float32{}, cli0, 1);  // same
    uint64_t hash2 = add_opr(cns[1], TensorShape{3, 5}, dtype::Float32{}, cli0, 1);  // comp_node
    uint64_t hash3 = add_opr(cns[0], TensorShape{4, 6}, dtype::Float16{}, cli0, 1);  // dtype
    uint64_t hash4 = add_opr(cns[0], TensorShape{5, 7}, dtype::Float32{}, cli1, 1);  // client
    uint64_t hash5 = add_opr(cns[0], TensorShape{6, 8}, dtype::Float32{}, cli0, 2);  // extra_hash

    ASSERT_EQ(hash0, hash1);

    std::set<uint64_t> s;
    s.insert(hash0);
    s.insert(hash1);
    s.insert(hash2);
    s.insert(hash3);
    s.insert(hash4);
    s.insert(hash5);
    ASSERT_EQ(5, s.size());

    ASSERT_EQ(1, group_info.count(hash0));
    ASSERT_EQ(1, group_info.count(hash1));
    ASSERT_EQ(1, group_info.count(hash2));
    ASSERT_EQ(1, group_info.count(hash3));
    ASSERT_EQ(1, group_info.count(hash4));
    ASSERT_EQ(1, group_info.count(hash5));

    ASSERT_EQ(2, groups[hash0].size());
    ASSERT_EQ(2, groups[hash1].size());
    ASSERT_EQ(1, groups[hash2].size());
    ASSERT_EQ(1, groups[hash3].size());
    ASSERT_EQ(1, groups[hash4].size());
    ASSERT_EQ(1, groups[hash5].size());
}

TEST_PASS(PackAllReduceReplacePass, DividePacks) {
    auto cn = CompNode::load("gpux");
    auto graph = ComputingGraph::make();
    auto client = std::make_shared<test::MockGroupClient>();
    auto mode = opr::CollectiveComm::Param::Mode::ALL_REDUCE_SUM;

    ThinHashMap<uint64_t, cg::OprNodeArray> groups;
    ThinHashMap<uint64_t, std::vector<cg::OprNodeArray>> packs;

    auto insert_opr = [&] (size_t size) {
        auto dev = std::make_shared<DeviceTensorND>(cn, TensorShape{size / sizeof(float)});
        auto sd = opr::SharedDeviceTensor::make(*graph, dev);
        auto symvar = opr::CollectiveComm::make(
                {sd}, graph.get(), "key", 2, false, 0, false, client, mode)[0];
        auto opr = symvar.node()->owner_opr();
        auto& comm = opr->cast_final_safe<opr::CollectiveComm>();
        comm.set_pack_hash(1);
        return opr;
    };

    auto pack_size = [&] (cg::OprNodeArray& pack) {
        size_t sum = 0;
        for (size_t i = 0; i < pack.size(); i++) {
            auto var = pack[i]->input(0);
            sum += var->dtype().size(var->shape().total_nr_elems());
        }
        return sum;
    };

    groups[0].push_back(insert_opr(100));  // group0, pack0, size=1100
    groups[0].push_back(insert_opr(300));  // group0, pack0, size=1100
    groups[0].push_back(insert_opr(400));  // group0, pack0, size=1100
    groups[0].push_back(insert_opr(300));  // group0, pack0, size=1100
    groups[0].push_back(insert_opr(500));  // group0, pack1, size=800
    groups[0].push_back(insert_opr(200));  // group0, pack1, size=800
    groups[0].push_back(insert_opr(100));  // group0, pack1, size=800

    groups[1].push_back(insert_opr(100));  // group1, pack0, size=900
    groups[1].push_back(insert_opr(400));  // group1, pack0, size=900
    groups[1].push_back(insert_opr(300));  // group1, pack0, size=900
    groups[1].push_back(insert_opr(100));  // group1, pack0, size=900

    gopt::PackAllReduceReplacePass::divide_packs(groups, packs, 1000);

    ASSERT_EQ(2, packs.size());

    ASSERT_EQ(2, packs[0].size());
    ASSERT_EQ(4, packs[0][0].size());
    ASSERT_EQ(1100, pack_size(packs[0][0]));
    ASSERT_EQ(3, packs[0][1].size());
    ASSERT_EQ(800, pack_size(packs[0][1]));

    ASSERT_EQ(1, packs[1].size());
    ASSERT_EQ(4, packs[1][0].size());
    ASSERT_EQ(900, pack_size(packs[1][0]));
}

TEST_PASS(PackAllReduceReplacePass, InsertPackedOprs) {
    auto cn = CompNode::load("gpux");
    auto graph = ComputingGraph::make();
    auto client = std::make_shared<test::MockGroupClient>();
    auto mode = opr::CollectiveComm::Param::Mode::ALL_REDUCE_SUM;

    size_t nr_devices = 2;
    uint32_t rank = 0;

    using GroupInfo = gopt::PackAllReduceReplacePass::GroupInfo;
    ThinHashMap<uint64_t, std::shared_ptr<GroupInfo>> group_info;
    ThinHashMap<uint64_t, cg::OprNodeArray> groups;

    auto insert_opr = [&] (const TensorShape& shape) {
        auto dev = std::make_shared<DeviceTensorND>(cn, shape);
        auto sd = opr::SharedDeviceTensor::make(*graph, dev);
        auto symvar =
                opr::CollectiveComm::make({sd}, graph.get(), "key", nr_devices,
                                          false, rank, false, client, mode)[0];
        auto opr = symvar.node()->owner_opr();
        auto& comm = opr->cast_final_safe<opr::CollectiveComm>();
        comm.set_pack_hash(1);
        gopt::PackAllReduceReplacePass::collect_groups(opr, group_info, groups);
        return symvar;
    };

    auto shape_x = TensorShape{100, 200};
    auto shape_y = TensorShape{200, 400};

    auto x = insert_opr(shape_x);
    auto y = insert_opr(shape_y);

    ASSERT_EQ(1, group_info.size());
    ASSERT_EQ(1, groups.size());
    auto info = group_info.begin()->second;
    auto pack = groups.begin()->second;
    size_t pack_id = 0;
    ThinHashMap<VarNode*, VarNode*> replace_map;
    gopt::PackAllReduceReplacePass::insert_packed_oprs(pack_id, pack, info, replace_map, -1);

    auto grad_x = SymbolVar(x.node()->owner_opr()->input(0));
    auto grad_y = SymbolVar(y.node()->owner_opr()->input(0));

    auto concat = opr::Concat::make({grad_x.flatten(), grad_y.flatten()}, 0);

    std::string key = ssprintf("grad_pack_%zu", pack_id);
    auto allreduce =
            opr::CollectiveComm::make({concat}, graph.get(), key, nr_devices,
                                      false, rank, false, client, mode)[0];

    std::vector<size_t> partition;
    partition.push_back(shape_x.total_nr_elems());
    partition.push_back(shape_y.total_nr_elems());
    auto splits = opr::Split::make(allreduce,
        opr::Split::Options::make_partition(allreduce, 0, partition));

    ASSERT_EQ(2, splits.size());
    auto dest_x = splits[0].reshape(shape_x);
    auto dest_y = splits[1].reshape(shape_y);

    ASSERT_EQ(2, replace_map.size());

    ASSERT_TRUE(replace_map.count(x.node()) > 0);
    ASSERT_EQ(replace_map.at(x.node()), dest_x.node());

    ASSERT_TRUE(replace_map.count(y.node()) > 0);
    ASSERT_EQ(replace_map.at(y.node()), dest_y.node());
}

TEST_PASS(PackAllReduceReplacePass, Equivalence) {
    REQUIRE_GPU(2);
    auto cns = load_multiple_xpus(2);
    auto client = std::make_shared<test::MockGroupClient>();

    auto build_graph = [&] (uint32_t rank, std::shared_ptr<ComputingGraph> graph,
                            SymbolVarArray& array) {
        HostTensorGenerator<> gen;
        auto cn = cns[rank];
        auto host_x = gen({1, 1000});
        auto host_y = gen({1000, 1});

        auto dev_x = std::make_shared<DeviceTensorND>(cn);
        auto dev_y = std::make_shared<DeviceTensorND>(cn);

        dev_x->copy_from(*host_x).sync();
        dev_y->copy_from(*host_y).sync();

        auto x = opr::SharedDeviceTensor::make(*graph, dev_x);
        auto y = opr::VolatileSharedDeviceTensor::make(*graph, dev_y);
        auto loss = opr::MatrixMul::make(x, y).flatten();

        auto grad_x = opr::VirtualGrad::make(loss, x);
        auto grad_y = opr::VirtualGrad::make(loss, y);

        using Mode = opr::CollectiveComm::Param::Mode;
        bool is_root = (rank == 0);
        auto reduced_x = opr::CollectiveComm::make(
                                 {grad_x}, graph.get(), "x", 2, is_root, rank,
                                 false, client, Mode::ALL_REDUCE_SUM)[0] /
                         2;
        auto reduced_y = opr::CollectiveComm::make(
                                 {grad_y}, graph.get(), "y", 2, is_root, rank,
                                 false, client, Mode::ALL_REDUCE_SUM)[0] /
                         2;

        graph->options().allreduce_pack_max_size = 5000;
        graph->options().allreduce_pack_ignore_first = 0;

        auto dest_vars = gopt::GraphOptimizer{}
            .add_pass<gopt::PackAllReduceScanPass>()
            .add_pass<gopt::PackAllReduceReplacePass>()
            .apply({{reduced_x, reduced_y}}).endpoint_vars();

        array.emplace_back(reduced_x);
        array.emplace_back(reduced_y);
        array.emplace_back(dest_vars[0]);
        array.emplace_back(dest_vars[1]);
    };

    auto run = [&] (uint32_t rank) {
        auto graph = ComputingGraph::make();
        SymbolVarArray array;
        build_graph(rank, graph, array);

        HostTensorND host_reduced_x, host_reduced_y, host_dest_0, host_dest_1;

        graph->options().allreduce_pack_max_size = 0;
        auto func = graph->compile({make_callback_copy(array[0], host_reduced_x),
                                    make_callback_copy(array[1], host_reduced_y),
                                    make_callback_copy(array[2], host_dest_0),
                                    make_callback_copy(array[3], host_dest_1)});
        func->execute();

        MGB_ASSERT_TENSOR_EQ(host_reduced_x, host_dest_0);
        MGB_ASSERT_TENSOR_EQ(host_reduced_y, host_dest_1);
    };

    std::thread t0(run, 0);
    std::thread t1(run, 1);

    t0.join();
    t1.join();
}

#endif  // MGB_ENABLE_OPR_MM

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
