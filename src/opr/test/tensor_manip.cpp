/**
 * \file src/opr/test/tensor_manip.cpp
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
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/misc.h"
#include "megbrain/utils/arith_helper.h"

using namespace mgb;
using namespace opr;

TEST(TestTensorManip, GetVarShape) {
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 1}), host_y = gen({1, 2});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z0 = opr::GetVarShape::make({x, y, x.make_scalar(5)}),
         z1 = opr::GetVarShape::make({x, y}, 1);

    // ensure scalar is removed
    ASSERT_EQ(2u, z0.node()->owner_opr()->input().size());

    constexpr auto tdt = cg::OperatorNodeBase::NodeProp::DepType::SHAPE;
    auto &&dt = z0.node()->owner_opr()->node_prop().dep_map();
    ASSERT_EQ(2u, dt.size());
    ASSERT_EQ(tdt, dt.at(x.node()));
    ASSERT_EQ(tdt, dt.at(y.node()));

    auto as_shp = [](const HostTensorND &hv) {
        mgb_assert(hv.dtype() == dtype::Int32());
        mgb_assert(hv.shape().ndim == 1);
        TensorShape ret;
        ret.ndim = hv.shape()[0];
        auto p = hv.ptr<int>();
        for (size_t i = 0; i < ret.ndim; ++ i)
            ret[i] = p[i];
        return ret;
    };
    HostTensorND host_z0, host_z1;
    auto func = graph->compile({
            make_callback_copy(z0, host_z0),
            make_callback_copy(z1, host_z1)});
    func->execute();

    ASSERT_EQ(TensorShape({3, 2}), as_shp(host_z0));
    ASSERT_EQ(TensorShape({2}), as_shp(host_z1));

    *host_x= *gen({5, 1, 6});
    *host_y= *gen({1, 8, 1});
    func->execute();

    ASSERT_EQ(TensorShape({5, 8, 6}), as_shp(host_z0));
    ASSERT_EQ(TensorShape({8}), as_shp(host_z1));
}

TEST(TestTensorManip, GetVarShapeBypass) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, gen({3, 2})),
         t = opr::Host2DeviceCopy::make(*graph, gen({2, 3})),
         tshp = opr::GetVarShape::make(t),
         y = opr::GetVarShape::make(opr::Reshape::make(x, tshp));
    ASSERT_EQ(tshp, y);
}

TEST(TestTensorManip, GetVarShapeNegativeAxis) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 3}), host_y = gen({2, 1});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z0 = opr::GetVarShape::make({x, y}, -1),
         z1 = opr::GetVarShape::make({x, y}, -2);

    // ensure scalar is removed
    ASSERT_EQ(2u, z0.node()->owner_opr()->input().size());

    constexpr auto tdt = cg::OperatorNodeBase::NodeProp::DepType::SHAPE;
    auto&& dt = z0.node()->owner_opr()->node_prop().dep_map();
    ASSERT_EQ(2u, dt.size());
    ASSERT_EQ(tdt, dt.at(x.node()));
    ASSERT_EQ(tdt, dt.at(y.node()));

    auto as_shp = [](const HostTensorND& hv) {
        mgb_assert(hv.dtype() == dtype::Int32());
        mgb_assert(hv.shape().ndim == 1);
        TensorShape ret;
        ret.ndim = hv.shape()[0];
        auto p = hv.ptr<int>();
        for (size_t i = 0; i < ret.ndim; ++i)
            ret[i] = p[i];
        return ret;
    };
    HostTensorND host_z0, host_z1;
    auto func = graph->compile(
            {make_callback_copy(z0, host_z0), make_callback_copy(z1, host_z1)});
    func->execute();

    ASSERT_EQ(TensorShape({3}), as_shp(host_z0));
    ASSERT_EQ(TensorShape({2}), as_shp(host_z1));

    *host_x = *gen({5, 1, 6});
    *host_y = *gen({1, 8, 1});
    func->execute();

    ASSERT_EQ(TensorShape({6}), as_shp(host_z0));
    ASSERT_EQ(TensorShape({8}), as_shp(host_z1));
}

TEST(TestTensorManip, Reshape) {
    constexpr size_t N = 123, C = 456;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({N * C}), host_opr1 = gen({N, C});
    auto graph = ComputingGraph::make();
    SymbolVar opr0 = opr::Host2DeviceCopy::make(*graph, host_opr0, {"opr0"}),
              opr1 = opr::Host2DeviceCopy::make(*graph, host_opr1, {"opr1"}),
              opr0_reshp = opr::Reshape::make(
                      opr0, opr::GetVarShape::make(opr1)),
              sum = opr::add(opr0_reshp, opr1);

    {
        // check dep type
        auto op = opr0_reshp.node()->owner_opr();
        auto &&dep_map = opr0_reshp.node()->owner_opr()->node_prop().dep_map();
        using DT = cg::OperatorNodeBase::NodeProp::DepType;
        ASSERT_EQ(2u, dep_map.size());
        ASSERT_EQ(DT::DEV_VALUE | DT::VALUE_ALLOW_EMPTY, dep_map.at(op->input(0)));
        ASSERT_EQ(DT::HOST_VALUE, dep_map.at(op->input(1)));
    }

    HostTensorND host_sum;
    auto func = graph->compile({make_callback_copy(sum, host_sum)});
    func->execute();
    ASSERT_TRUE(cg::is_static_var_storage(opr0_reshp.node()));
    ASSERT_FALSE(host_sum.layout().eq_layout(host_opr0->layout()));
    ASSERT_TRUE(host_sum.layout().eq_layout(host_opr1->layout()));
    ASSERT_EQ(dev_ptr(opr0), dev_ptr(opr0_reshp));
    auto o0 = host_opr0->ptr<float>(), o1 = host_opr1->ptr<float>(),
         s = host_sum.ptr<float>();
    for (size_t i = 0, it = host_opr0->layout().total_nr_elems();
            i < it; i ++) {
        MGB_ASSERT_FLOAT_EQ(o0[i] + o1[i], s[i]) <<
            ssprintf("failed opr0(%.5f)+opr1(%.5f) at %zd", o0[i], o1[i], i);
    }
}

TEST(TestTensorManip, ReshapeNoncontigValueInfer) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 1});
    auto graph = ComputingGraph::make();
    auto x = opr::ImmutableTensor::make(*graph, *host_x),
         y = x.broadcast({2, 2}),
         z = opr::Reshape::make(y, {1, 0}, 1);
    auto &&mgr = graph->static_infer_manager();
    ASSERT_EQ(cg::static_infer::InferType::CONST,
            mgr.get_infer_type(z.node()).value);
    auto zv = mgr.infer_value(z.node());
    auto xp = host_x->ptr<float>(),
         zp = zv.ptr<float>();
    for (int i = 0; i < 2; ++ i) {
        for (int j = 0; j < 2; ++ j) {
            ASSERT_EQ(xp[i], zp[i * 2 + j]);
        }
    }

    ASSERT_THROW(opr::Reshape::make(y, {3, 0}, 1), TensorReshapeError);
    ASSERT_THROW(opr::Reshape::make(y, {3, 2}), TensorReshapeError);
}

TEST(TestTensorManip, ReshapeSameShapeBypass) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         x1 = x.reshape({6}),
         x2 = x1.reshape({6}),
         x3 = x.reshape(opr::GetVarShape::make(x));
    ASSERT_EQ(x1.node(), x2.node());
    ASSERT_EQ(x.node(), x3.node());
    ASSERT_NE(x.node(), x1.node());
}

TEST(TestTensorManip, ReshapeAndInplace) {
    constexpr size_t C = 456;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({C}), host_opr1 = gen({C / 2, 2});
    auto graph = ComputingGraph::make();
    SymbolVar opr0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_opr0),
              opr1 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_opr1),
              reshape = opr::Reshape::make(opr0, TensorShape{C / 2, 2}),
              sum = reshape + opr1;
    opr1.node()->add_flag(cg::VarNode::Flag::NO_MEM_RECLAIM);
    HostTensorND host_sum(CompNode::load("xpu0"));
    auto func = graph->compile({make_callback_copy(sum, host_sum)});
    func->execute();
    ASSERT_EQ(dev_ptr(reshape), dev_ptr(sum));
    // assert contiguous layout
    ASSERT_EQ(host_opr1->layout(), host_sum.layout());
    auto o0 = host_opr0->ptr<float>(), o1 = host_opr1->ptr<float>(),
         s = host_sum.sync().ptr<float>();
    for (size_t i = 0, it = host_opr0->layout().total_nr_elems();
            i < it; ++ i) {
        MGB_ASSERT_FLOAT_EQ(o0[i] + o1[i], s[i]) <<
            ssprintf("failed opr0(%.5f)+opr1(%.5f) at %zd", o0[i], o1[i], i);
    }
}

TEST(TestTensorManip, DynamicReshape) {
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 4}),
         host_tshp = std::make_shared<HostTensorND>(
                 host_x->comp_node(), dtype::Int32());
    host_tshp->resize({1}).ptr<int>()[0] = 12;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         x_rshp_shp = opr::MarkDynamicVar::make(
                 opr::Host2DeviceCopy::make(*graph, host_tshp).rename(
                     "x_rshp_shp")),
         x_rshp = opr::Reshape::make(x, x_rshp_shp).rename("x_rshp"),
         x_flat = x_rshp.flatten(),
         gx = cg::grad(
                 opr::Dot::make(x_flat, x_flat).rename("loss"), x).rename("gx");
    ASSERT_FALSE(cg::is_static_var_shape(x_rshp.node()));
    ASSERT_TRUE(cg::is_static_var_shape(gx.node()));
    ASSERT_EQ(host_x->shape(), gx.node()->shape());
    HostTensorND host_rshp, host_gx;
    auto func = graph->compile({make_callback_copy(x_rshp, host_rshp),
            make_callback_copy(gx, host_gx)});

    auto check = [&](const TensorShape &ishp, const TensorShape &tshp) {
        host_x->copy_from(*gen(ishp));
        {
            DeviceTensorND tmp;
            cg::copy_shape_to_tensor_value(tmp, tshp);
            host_tshp->copy_from(tmp);
        }
        func->execute();
        ASSERT_EQ(tshp, host_rshp.shape());
        ASSERT_EQ(host_x->shape(), host_gx.shape());
        for (size_t i = 0, it = host_x->shape().total_nr_elems();
                i < it; ++ i)
            MGB_ASSERT_FLOAT_EQ(host_x->ptr<float>()[i] * 2, host_gx.ptr<float>()[i]);
    };

    check({3, 4}, {12});
    check({5, 3}, {15});
    check({3, 4, 35}, {21, 20});
}

TEST(TestTensorManip, ReshapeWithUnspec) {
    HostTensorGenerator<> gen;
    auto host_x = gen({4, 8});
    auto graph = ComputingGraph::make();

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Reshape::make(x, {1, 8}, 0);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    for (size_t ishp: {1, 5, 6}) {
        host_x->copy_from(*gen({ishp * 8}));
        func->execute();
        TensorShape expect_shape({ishp, 8});
        ASSERT_EQ(expect_shape, host_y.shape());
        MGB_ASSERT_TENSOR_EQ(
                host_x->sub(SubTensorSpec::make_from_layout(
                        host_x->layout().reshape(expect_shape))),
                host_y);
    }
}

TEST(TestTensorManip, ReshapeInferShapeForDynamicInput) {
    constexpr size_t N0 = 2, C0 = 3;
    HostTensorGenerator<> gen;
    auto host_x = gen({N0, C0}), host_tshp = gen({1});
    auto graph = ComputingGraph::make();
    host_tshp->ptr<float>()[0] = N0 * C0;
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x),
              xd = opr::MarkDynamicVar::make(x),
              tshp = opr::Host2DeviceCopy::make(*graph, host_tshp),
              y0 = opr::Reshape::make(xd, tshp) + 1,
              y1 = opr::Reshape::make(xd, opr::GetVarShape::make(x)) + 2;

    ASSERT_EQ(y0.shape(), TensorShape({N0 * C0}));
    ASSERT_EQ(y1.shape(), TensorShape({N0, C0}));
    HostTensorND host_y0, host_y1;
    auto func = graph->compile({make_callback_copy(y0, host_y0),
            make_callback_copy(y1, host_y1)});

    auto run = [&](const TensorShape &ishp) {
        auto tot = ishp.total_nr_elems();
        host_x->copy_from(*gen(ishp));
        host_tshp->ptr<float>()[0] = tot;
        func->execute();
        ASSERT_EQ(host_y0.shape(), TensorShape({tot}));
        ASSERT_EQ(host_y1.shape(), ishp);
        for (size_t i = 0; i < tot; ++ i) {
            ASSERT_EQ(host_x->ptr<float>()[i] + 1, host_y0.ptr<float>()[i]);
            ASSERT_EQ(host_x->ptr<float>()[i] + 2, host_y1.ptr<float>()[i]);
        }
    };

    run({3, 2});
    run({23, 12, 5});
}

TEST(TestTensorManip, ReshapeEmptyShape) {
    HostTensorGenerator<> gen;
    constexpr size_t x_length = 233;
    auto host_x = gen({x_length}),
         host_v = gen({2, 3, 3, 3});
    for (size_t i = 0; i < x_length; ++ i) {
        host_x->ptr<float>()[i] = 1.f;
    }
    constexpr auto INVALID_AXIS = opr::Reshape::Param::INVALID_AXIS;
    for (auto unspec_axis: {INVALID_AXIS, 0, 1, 3}) {
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        TensorShape tshape{2, 3, 3, 3};
        auto zero_axis = unspec_axis;
        if (unspec_axis == INVALID_AXIS) {
            tshape[zero_axis = 2] = 0;
        }
        using CondTakeMode = opr::CondTake::Param::Mode;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             x_empty = opr::CondTake::make(x, x, {CondTakeMode::EQ, 0.f})[0],
             v = opr::Host2DeviceCopy::make(*graph, host_v),
             x_reshape = opr::Reshape::make(x_empty, tshape, {unspec_axis}),
             y = opr::Concat::make({x_reshape, v}, zero_axis);
        HostTensorND host_empty, host_y;
        auto func = graph->compile({
            make_callback_copy(x_reshape, host_empty),
            make_callback_copy(y, host_y)});
        func->execute().wait();
        ASSERT_TRUE(host_empty.layout().is_empty());
        MGB_ASSERT_TENSOR_EQ(*host_v, host_y);
    }
}

TEST(TestTensorManip, ReshapeWithNegativeUnspec) {
    HostTensorGenerator<> gen;
    auto host_x = gen({4, 8});
    auto graph = ComputingGraph::make();

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Reshape::make(x, {1, 8}, -2);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    for (size_t ishp : {1, 5, 6}) {
        host_x->copy_from(*gen({ishp * 8}));
        func->execute();
        TensorShape expect_shape({ishp, 8});
        ASSERT_EQ(expect_shape, host_y.shape());
        MGB_ASSERT_TENSOR_EQ(host_x->sub(SubTensorSpec::make_from_layout(
                                     host_x->layout().reshape(expect_shape))),
                             host_y);
    }
}

TEST(TestTensorManip, Broadcast) {
    constexpr size_t N = 20, C = 30;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({1, 1}), host_opr1 = gen({N, C});
    auto graph = ComputingGraph::make();
    SymbolVar opr0 = opr::Host2DeviceCopy::make(*graph, host_opr0, {"opr0"}),
              opr1 = opr::Host2DeviceCopy::make(*graph, host_opr1, {"opr1"}),
              sum = opr::add(
                      opr::Broadcast::make(opr0, host_opr1->shape()), opr1);

    HostTensorND host_sum(CompNode::load("xpu0"));
    auto func = graph->compile({
        {sum, [&](DeviceTensorND &s){
            host_sum.copy_from(s);
        }}});
    func->execute();
    ASSERT_TRUE(host_sum.layout().eq_layout(host_opr1->layout()));
    auto o0 = host_opr0->ptr<float>(), o1 = host_opr1->ptr<float>(),
         s = host_sum.sync().ptr<float>();
    for (size_t i = 0, it = host_opr0->layout().total_nr_elems();
            i < it; i ++) {
        MGB_ASSERT_FLOAT_EQ(o0[0] + o1[i], s[i]) <<
            ssprintf("failed opr0(%.5f)+opr1(%.5f) at %zd", o0[i], o1[i], i);
    }
}

TEST(TestTensorManip, BroadcastEmptyShape) {
    HostTensorGenerator<> gen;
    for (auto&& arg:
        {std::make_pair(TensorShape{1}, TensorShape{0}),
         {{1, 2, 3}, {0, 2, 3}},
         {{2, 3}, {1, 0, 2, 3}},
         {{1, 0, 2, 3}, {4, 0, 2, 3}},
         {{0, 1, 2, 3}, {3, 0, 4, 2, 3}}}) {
        auto host_x = gen(arg.first);
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = opr::Broadcast::make(x, arg.second);
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_TRUE(host_y.shape().eq_shape(arg.second));
    }
}

TEST(TestTensorManip, Dimshuffle) {
    HostTensorGenerator<> gen;
    constexpr size_t S0 = 8, S1 = 3;
    auto host_x = gen({S0, S1}),
         host_prod = gen({S1, 1, S0, 1});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         prod = opr::Host2DeviceCopy::make(*graph, host_prod).rename("prod"),
         x_ds = opr::Dimshuffle::make(x, {1, -1, 0, -1}).rename("x_ds"),
         y = (x_ds * prod).reshape({S0 * S1}).rename("y"),
         loss = opr::Dot::make(y, y).rename("loss"),
         gx = cg::grad(loss, x).rename("gx");

    ASSERT_TRUE(cg::is_static_var_shape(gx.node()));
    ASSERT_EQ(host_x->shape(), gx.node()->shape());
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->execute();

    for (size_t i = 0; i < S0; i ++)
        for (size_t j = 0; j < S1; j ++) {
            float x = host_x->ptr<float>({i, j})[0],
                  prod = host_prod->ptr<float>({j, 0, i, 0})[0],
                  gx = host_gx.ptr<float>({i, j})[0];
            MGB_ASSERT_FLOAT_EQ(2 * prod * prod * x, gx) <<
                ssprintf("failed at (%zd, %zd): x=%g prod=%g gx=%g",
                        i, j, x, prod, gx);
        }
}

TEST(TestTensorManip, DimshuffleEmptyShape) {
    HostTensorGenerator<> gen;
    for (auto&& arg:
        {std::make_pair(
            TensorShape{3, 0},
            std::vector<int>{1, -1, 0, -1}),
         {{3, 1, 0, 4}, {-1, 3, -1, 0, 2}},
         {{2, 0, 3, 0}, {1, 0, 2, 3}}}) {
        auto host_x = gen(arg.first);
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = opr::Dimshuffle::make(x, arg.second);
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        auto&& y_shape = host_y.shape();
        for(size_t idx = 0; idx < arg.second.size(); ++ idx) {
            auto elem = arg.second[idx];
            if (elem == -1) {
                ASSERT_EQ(y_shape[idx], 1u);
            } else {
                ASSERT_EQ(arg.first[elem], y_shape[idx]);
            }
        }
    }
}

TEST(TestTensorManip, DimshuffleCombined) {
    using Checker = AutoOprChecker<1, 1>;
    constexpr int RED0 = 2, RED1 = 3;

    for (bool dyn: {false, true}) {

        auto make_graph = [dyn](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {

                auto x = inputs[0];
                if (dyn)
                    x = opr::MarkDynamicVar::make(x);

                auto cv = [&](int v) {
                    auto rst = x.make_scalar(v);
                    if (dyn)
                        rst = opr::MarkDynamicVar::make(rst);
                    return rst;
                };

                auto xshp = opr::GetVarShape::make(x);
                auto sub = [&](int idx) {
                    return opr::IndexAt::make(xshp, {{0, cv(idx)}});
                };
                auto tshp0 = opr::Concat::make({
                        sub(0), sub(1) / (RED0 * RED1), cv(RED0), cv(RED1),
                        sub(2), sub(3)}, 0),
                     tshp1 = opr::Concat::make({
                             sub(0), sub(1) / (RED0 * RED1),
                             sub(2) * RED0, sub(3) * RED1}, 0);
                auto y0 = opr::Reshape::make(x, tshp0),
                     y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 2, 4, 5}),
                     y2 = opr::Reshape::make(y1, tshp1);
                return {y2.node()};
            };

        auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
            auto &&iv = *inp.at(0);
            auto &&ov = dest.at(0);
            auto ishp = iv.shape();
            auto oshp = ishp;
            oshp.shape[1] /= RED0 * RED1;
            oshp.shape[2] *= RED0;
            oshp.shape[3] *= RED1;
            ov.comp_node(iv.comp_node()).resize(oshp);

            size_t tmpshp[6] = {oshp.shape[0], oshp.shape[1], RED1, RED0,
                ishp.shape[2], ishp.shape[3]},
                   tmpidx[6];
            for (size_t oidx = 0, oidxt = oshp.total_nr_elems();
                    oidx < oidxt; ++ oidx) {
                for (int i = 5, x = oidx; i >= 0; -- i) {
                    tmpidx[i] = x % tmpshp[i];
                    x /= tmpshp[i];
                    mgb_assert(i || !x);
                }
                std::swap(tmpshp[2], tmpshp[3]);
                std::swap(tmpidx[2], tmpidx[3]);
                size_t iidx = 0;
                for (int i = 5, d = 1; i >= 0; -- i) {
                    iidx += d * tmpidx[i];
                    d *= tmpshp[i];
                }
                std::swap(tmpshp[2], tmpshp[3]);
                ov.ptr<float>()[oidx] = iv.ptr<float>()[iidx];
            }
        };

        Checker::RunOptions opt;
        opt.numdiff_eps = 1; // large eps because all linear
        constexpr size_t R = RED0 * RED1;
        Checker(make_graph, fwd).
            run({{{1, R, 1, 1}}}, opt).
            run({{{5, R * 2, 3, 2}}}, opt).
            run({{{2, R * 3, 4, 3}}}, opt);
    }
}

TEST(TestTensorManip, Subtensor) {
    using Checker = AutoOprChecker<1, 1>;

    SymbolVar sub0, sub1, sub2, sub3, sub4;
    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        using AIdx = opr::Subtensor::AxisIndexer;
        auto x = inputs[0];
        x = x.rename("x");
        auto cv = [&](int v, bool dyn = false) {
            auto rst = x.make_scalar(v);
            if (dyn)
                rst = opr::MarkDynamicVar::make(rst);
            return rst;
        };

        // sub0 = (0.9*x)[10:shp0:2]
        sub0 = opr::Subtensor::make(x * 0.9f,
                {AIdx::make_interval(
                        0, cv(10, true), opr::GetVarShape::make(x, 0),
                        cv(2))}).rename("sub0");

        // sub1 = x[:-10:2]
        sub1 = opr::Subtensor::make(opr::MarkDynamicVar::make(x),
                {AIdx::make_interval(
                        0, None, cv(-10), cv(2))}).rename("sub1");

        // sub2_raw = x[5:-5:2, 3]
        auto sub2_raw = opr::Subtensor::make(
                opr::IndexAt::make(x, {{1, cv(3)}}),
                {AIdx::make_interval(0, cv(5), cv(-5), cv(2))});
        {
            auto opr = sub2_raw.node()->owner_opr();
            auto &&inp = opr->input();
            auto &&dmap = opr->node_prop().dep_map();
            for (size_t i = 1; i < inp.size(); ++ i) {
                mgb_assert(dmap.at(inp[i]) &
                        cg::OperatorNodeBase::NodeProp::DepType::HOST_VALUE);
            }
        }
        sub2 = opr::AxisAddRemove::make(sub2_raw,
                {opr::AxisAddRemove::AxisDesc::make_add(1)}).rename("sub2");

        // sub3 = x[4:-6:2, -1:]
        sub3 = opr::Subtensor::make(x, {
                AIdx::make_interval(0, cv(4), cv(-6), cv(2)),
                AIdx::make_interval(1, cv(-1), None, None)});

        // sub4 = (x + 0.1)[-3:7:-2, 1::-3] (negative stride)
        sub4 = opr::Subtensor::make(x + .1f, {
                AIdx::make_interval(0, cv(-3), cv(7), cv(-2)),
                AIdx::make_interval(1, cv(1), None, cv(-3, true))});

        return {(sub0 + sub1 + sub2 + sub3 + sub4).rename("y")};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto iptr = inp[0]->ptr<float>();
        auto ishp = inp[0]->shape();
        auto oshp = ishp;
        auto s0 = ishp.shape[0], s1 = ishp.total_nr_elems() / s0,
             s2 = s1 / ishp.shape[1];
        auto os0 = (s0 - 10 + 1) / 2;
        oshp.shape[0] = os0;
        dest[0].comp_node(inp[0]->comp_node());
        dest[0].resize(oshp);
        auto optr = dest[0].ptr<float>();

        for (size_t i = 0; i < os0; ++ i)
            for (size_t j = 0; j < s1; ++ j) {
                optr[i * s1 + j] =
                    iptr[(i * 2 + 10) * s1 + j] * .9f +
                    iptr[(i * 2) * s1 + j] +
                    iptr[(i * 2 + 5) * s1 + j % s2 + s2 * 3] +
                    iptr[(i * 2 + 4) * s1 + j % s2 + s2 * (ishp.shape[1] - 1)] +
                    iptr[(ishp.shape[0]-3-i*2)*s1 + j % s2 + s2 * 1] + 0.1;
            }
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1; // large eps because all linear
    Checker checker(make_graph, fwd);

    checker.
        run({{{11, 5}}}, opt).
        run({{{20, 6}}}, opt).
        run({{{56, 6, 4}}}, opt);

    ASSERT_FALSE(cg::is_static_var_shape(sub0.node()));
    ASSERT_FALSE(cg::is_static_var_shape(sub1.node()));
    ASSERT_TRUE(cg::is_static_var_storage(sub2.node()));
    ASSERT_TRUE(cg::is_static_var_storage(sub3.node()));
    ASSERT_FALSE(cg::is_static_var_storage(sub4.node()));
}

TEST(TestTensorManip, SubtensorNegativeAxis) {
    using Checker = AutoOprChecker<1, 1>;

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        using AIdx = opr::Subtensor::AxisIndexer;
        auto x = inputs[0];
        return {opr::Subtensor::make(x,
                {AIdx::make_index(-1, x.make_scalar(2))})};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto iptr = inp[0]->ptr<float>();
        auto ishp = inp[0]->shape();
        auto oshp = ishp;
        -- oshp.ndim;
        auto stride = oshp.shape[oshp.ndim];
        if (!oshp.ndim)
            oshp = {1};
        auto optr = dest[0].resize(oshp).ptr<float>();

        for (size_t i = 0, it = oshp.total_nr_elems(); i < it; ++ i) {
            optr[i] = iptr[i * stride + 2];
        }
    };

    Checker checker(make_graph, fwd);
    checker.
        run({TensorShape{5}}).
        run({TensorShape{2, 3}}).
        run({TensorShape{2, 3, 4}}).
        run({TensorShape{2, 3, 4, 5}});
}

TEST(TestTensorManip, SubtensorWithEmptyIndexDesc) {
    using Checker = AutoOprChecker<1, 1>;

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        auto x = inputs[0];
        return {opr::Subtensor::make(x, {})};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto iptr = inp[0]->ptr<float>();
        auto oshp = inp[0]->shape();
        auto optr = dest[0].resize(oshp).ptr<float>();
        for (size_t i = 0, it = oshp.total_nr_elems(); i < it; ++i) {
            optr[i] = iptr[i];
        }
    };

    Checker checker(make_graph, fwd);
    checker.
        run({TensorShape{5}}).
        run({TensorShape{2, 3}}).
        run({TensorShape{2, 3, 4}}).
        run({TensorShape{2, 3, 4, 5}});
}

TEST(TestTensorManip, SubtensorShapeInferForDynAxisIdx) {
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 6, 3});
    auto host_idx = gen({1});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::MarkDynamicVar::make(
                 opr::Host2DeviceCopy::make(*graph, host_idx));
    auto cv = [&](int v) {
        return x.make_scalar(v);
    };
    using Ad = opr::Subtensor::AxisIndexer;
    // y = x[2, 1:-2:2]
    auto y = opr::Subtensor::make(x,
            {Ad::make_interval(1, cv(1), cv(-2), cv(2)),
            Ad::make_index(0, idx)});
    ASSERT_TRUE(cg::is_static_var_shape(y.node()));
    ASSERT_EQ(y.node()->shape(), TensorShape({2, 3}));

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    host_idx->ptr<float>()[0] = 2;

    func->execute();

    HostTensorND expt{host_x->comp_node(), host_x->dtype()};
    expt.resize({2, 3});
    for (size_t i = 0; i < 2; ++ i)
        for (size_t j = 0; j < 3; ++ j) {
            expt.ptr<float>()[i * 3 + j] = host_x->ptr<float>({2, i * 2 + 1, j})[0];
        }
    MGB_ASSERT_TENSOR_EQ(expt, host_y);
}

TEST(TestTensorManip, SubtensorDynCaseMemFwd) {
    auto run = [](int dyn_type) {
        // dyn_type: 0->const idx, 1->static idx, 2->dynamic idx, 3->dynamic inp
        ASSERT_FALSE(HasFailure()) << "already failed before " << dyn_type;
        HostTensorGenerator<> gen;
        auto host_x = gen({2, 3});
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x);
        SymbolVar idx;
        if (dyn_type == 0 || dyn_type == 3) {
            idx = x.make_scalar(1);
            if (dyn_type == 3) {
                // force dynamic storage by reading on another comp node
                auto xrd = opr::Copy::make(
                        x, host_x->comp_node().change_stream(1));
                graph->options().extra_vardeps[x.node()].push_back(xrd.node());
            }
        } else {
            auto host_idx = std::make_shared<HostTensorND>(host_x->comp_node(),
                                                           dtype::Int32{});
            host_idx->resize({1}).ptr<int>()[0] = 1;
            idx = opr::Host2DeviceCopy::make(*graph, host_idx);
            if (dyn_type == 2) {
                idx = opr::MarkDynamicVar::make(idx);
            }
        }
        auto y = opr::Subtensor::make(
                x, {opr::Subtensor::AxisIndexer::make_interval(0, idx, None,
                                                               None)});
        if (dyn_type != 2) {
            ASSERT_EQ(TensorShape({1, 3}), y.shape());
        }
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        auto xsub = host_x->sub(SubTensorSpec::make_from_offset_elem(
                TensorLayout({1, 3}, dtype::Float32{}), 3));
        MGB_ASSERT_TENSOR_EQ(xsub, host_y);
        ASSERT_EQ(dyn_type == 0, cg::is_static_var_storage(y.node()));
        ASSERT_EQ(dyn_type != 2, cg::is_static_var_shape(y.node()));
        ASSERT_EQ(static_cast<const uint8_t*>(prev_dev_ptr(x)) +
                          3 * sizeof(float),
                  prev_dev_ptr(y));
    };
    run(0);
    run(1);
    run(2);
    run(3);
}

TEST(TestTensorManip, SubtensorWithNoValInferInp) {
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 1}), host_idx = gen({1});
    auto graph = ComputingGraph::make();
    using Ad = opr::Subtensor::AxisIndexer;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx = opr::Host2DeviceCopy::make_no_value_infer(*graph, host_idx),
         y = opr::Subtensor::make(x, {Ad::make_index(0, idx)});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    host_idx->ptr<float>()[0] = 2;
    func->execute();

    HostTensorND expt{host_x->comp_node(), host_x->dtype()};
    expt.resize({1}).ptr<float>()[0] = host_x->ptr<float>()[2];
    MGB_ASSERT_TENSOR_EQ(expt, host_y);
}

TEST(TestTensorManip, SubtensorDedup) {
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 5, 5, 5});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto cv = [&](int v) {
        return x.make_scalar(v);
    };

    using S = opr::Subtensor;
    using D = S::AxisIndexer;
    std::unordered_set<VarNode*> nodes;
    for (int i: {0, 1, 1, 0}) {
        nodes.insert(S::make(x, {D::make_index(i, cv(2))}).node());
        nodes.insert(S::make(x, {D::make_interval(
                        i, cv(2), None, None)}).node());
        nodes.insert(S::make(x, {D::make_interval(
                        i, None, cv(2), None)}).node());
        nodes.insert(S::make(x, {D::make_interval(
                        i, None, None, cv(2))}).node());
    }

    ASSERT_EQ(8u, nodes.size());
}

TEST(TestTensorManip, SubtensorIdxChange) {
    auto run = [](bool dyn) {
        HostTensorGenerator<> gen;
        auto host_x = gen({10});
        auto host_idx = std::make_shared<HostTensorND>(host_x->comp_node(),
                                                       dtype::Int32());
        host_idx->resize({1}).ptr<int>()[0] = 1;
        bool idx_exec = false, idx_infered = false;
        auto cb_set_idx_exec = [&](DeviceTensorND& dv) {
            if (dv.comp_node() == CompNode::default_cpu()) {
                idx_infered = true;
            } else {
                idx_exec = true;
            }
        };
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        SymbolVar idx_;
        if (dyn) {
            idx_ = opr::Host2DeviceCopy::make(*graph, host_idx);
        } else {
            idx_ = opr::ImmutableTensor::make(*graph, *host_idx);
        }
        auto idx = opr::CallbackInjector::make(idx_,
                                               {false, true, cb_set_idx_exec}),
             y = opr::Subtensor::make(
                     x, {opr::Subtensor::AxisIndexer::make_interval(
                                0, idx, idx + 1, None)});

        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        ASSERT_TRUE(cg::is_static_var_shape(y.node()));
        ASSERT_TRUE(cg::is_static_var_value(y.node()));
        ASSERT_EQ(!dyn, cg::is_static_var_storage(y.node()));
        ASSERT_EQ(TensorShape({1}), y.node()->shape());

        auto px = host_x->ptr<float>();
        func->execute();
        ASSERT_EQ(px[1], host_y.ptr<float>()[0]);

        host_idx->ptr<int>()[0] = 5;
        func->execute();
        if (dyn) {
            ASSERT_EQ(px[5], host_y.ptr<float>()[0]);
        } else {
            ASSERT_EQ(px[1], host_y.ptr<float>()[0]);
        }
        ASSERT_TRUE(idx_infered);
        ASSERT_FALSE(idx_exec);
    };
    run(true);
    run(false);
}

namespace {

void test_subtensor_fwdonly(bool dyn_inp, bool dyn_idx) {
    constexpr size_t SIZE = 25;
    auto mkhost = [](size_t size, DType dtype) {
        auto rst = std::make_shared<HostTensorND>(
                CompNode::load("xpu0"), dtype);
        rst->resize({size});
        return rst;
    };
    auto host_x = mkhost(SIZE, dtype::Float32()),
         host_idx0 = mkhost(1, dtype::Int32()),
         host_idx1 = mkhost(1, dtype::Int32());
    for (size_t i = 0; i < SIZE; ++ i) {
        host_x->ptr<float>()[i] = i;
    }

    host_idx0->ptr<int>()[0] = 2;
    host_idx1->ptr<int>()[0] = 6;

    auto graph = ComputingGraph::make();
    using AIdx = opr::Subtensor::AxisIndexer;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         idx0 = opr::Host2DeviceCopy::make(*graph, host_idx0),
         idx1 = opr::Host2DeviceCopy::make(*graph, host_idx1);
    float *x_ptr = nullptr, *x_ptr_end = nullptr, *xsub_ptr = nullptr;
    if (dyn_inp)
        x = opr::MarkDynamicVar::make(x);
    x = opr::CallbackInjector::make(x, [&](DeviceTensorND&v){
            x_ptr = v.ptr<float>();
            x_ptr_end = v.ptr<float>() + v.layout().total_nr_elems();
    });
    if (dyn_idx)
        idx0 = opr::MarkDynamicVar::make(idx0);
    auto xsub = opr::Subtensor::make(x, {
            AIdx::make_interval(0, idx0, idx1, None)});
    xsub = opr::CallbackInjector::make(xsub,
            [&](DeviceTensorND&v){xsub_ptr=v.ptr<float>();});

    ASSERT_EQ(!dyn_inp && !dyn_idx, cg::is_static_var_shape(xsub.node()));

    HostTensorND host_sub;
    auto func = graph->compile({make_callback_copy(xsub, host_sub)});

    bool failed = false;
    auto run_and_check = [&](size_t begin, size_t end) {
        ASSERT_FALSE(failed);
        failed = true;
        host_idx0->ptr<int>()[0] = begin;
        host_idx1->ptr<int>()[0] = end;
        func->execute();

        if (!(!dyn_inp && dyn_idx)) {
            ASSERT_GE(xsub_ptr, x_ptr);
            ASSERT_LE(xsub_ptr, x_ptr_end);
        }

        ASSERT_EQ(TensorShape({end - begin}), host_sub.shape());
        for (size_t i = 0; i < end - begin; ++ i)
            ASSERT_EQ(host_x->ptr<float>()[i + begin], host_sub.ptr<float>()[i]) << ssprintf(
                    "failed [%zu, %zu): i=%zu", begin, end, i);
        failed = false;
    };

    run_and_check(0, 1);
    run_and_check(2, 3);
    run_and_check(0, 5);
    run_and_check(1, 6);
    run_and_check(3, 21);
    run_and_check(0, SIZE);
    run_and_check(1, SIZE);
    run_and_check(0, SIZE - 1);
}
} // anonymous namespace

TEST(TestTensorManip, SubtensorFwdOnly00) {
    test_subtensor_fwdonly(false, false);
}

TEST(TestTensorManip, SubtensorFwdOnly01) {
    test_subtensor_fwdonly(false, true);
}

TEST(TestTensorManip, SubtensorFwdOnly10) {
    test_subtensor_fwdonly(true, false);
}

TEST(TestTensorManip, SubtensorFwdOnly11) {
    test_subtensor_fwdonly(true, true);
}

TEST(TestTensorManip, OverlapSetSubtensor) {
    constexpr size_t SIZE = 2048, SIZE_SUB = (SIZE - 4) / 2;
    auto host_x = std::make_shared<HostTensorND>(
            CompNode::load("xpu0"), dtype::Float32());
    host_x->resize({SIZE});
    for (size_t i = 0; i < SIZE; ++ i)
        host_x->ptr<float>()[i] = i;
    auto graph = ComputingGraph::make();
    graph->options().allocate_static_mem_after_graph_compile = true;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x");
    auto cv = [&](int v, bool dyn = false) {
        auto rst = x.make_scalar(v);
        if (dyn)
            rst = opr::MarkDynamicVar::make(rst);
        return rst;
    };
    using AIdx = opr::Subtensor::AxisIndexer;
    auto xsub = opr::Subtensor::make(x, {AIdx::make_interval(0,
                cv(2), cv(-2), cv(2))}).rename("xsub"),
         // y = xsub[:-10] := xsub[10:]
         y = opr::SetSubtensor::make(
                 xsub,
                 opr::Subtensor::make(xsub, {AIdx::make_interval(0,
                         cv(10), None, None)}).rename("xsub[10:]"),
                 {AIdx::make_interval(0, None, cv(-10), None)}).rename("y");

    HostTensorND expected(host_x->comp_node(), dtype::Float32());
    expected.resize({SIZE_SUB});
    for (size_t i = 0; i < SIZE_SUB; ++ i) {
        auto i0 = i;
        if (i0 < SIZE_SUB - 10)
            i0 += 10;
        expected.ptr<float>()[i] = i0 * 2 + 2;
    }

    ASSERT_TRUE(cg::is_static_var_value(y.node()));
    HostTensorND infer_result;
    infer_result.copy_from(graph->static_infer_manager().infer_value(y.node()));
    MGB_ASSERT_TENSOR_EQ(expected, infer_result);

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->to_json()->writeto_fpath(output_file("OverlapSetSubtensor.json"));
    func->execute();
    MGB_ASSERT_TENSOR_EQ(expected, host_y);
}

TEST(TestTensorManip, OverlapSetSubtensor2) {
    constexpr size_t SIZE_X = 20, SIZE_Y = 23;
    auto run = [](bool should_overlap) {
        auto host_x = std::make_shared<HostTensorND>(CompNode::load("xpu0"),
                                                     dtype::Float32());
        host_x->resize({SIZE_X, SIZE_Y});
        for (size_t i = 0; i < SIZE_X * SIZE_Y; ++i)
            host_x->ptr<float>()[i] = i;
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x");
        auto cv = [&](int v) { return x.make_scalar(v); };
        auto make_sub_desc = [&](int begin,
                                 int end) -> opr::Subtensor::IndexDesc {
            using AIdx = opr::Subtensor::AxisIndexer;
            return {AIdx::make_interval(0, cv(begin), cv(end), None)};
        };
        auto slice = [&](SymbolVar inp, int begin, int end) {
            return opr::Subtensor::make(inp, make_sub_desc(begin, end));
        };
        // y = x.copy()
        // y[2:7] = y[4:9].copy()
        // y[1:6] += y[3:8].copy()
        auto xsub = slice(x, 4, 9).rename("xsub"),
             y0 = opr::SetSubtensor::make(x, xsub, make_sub_desc(2, 7))
                          .rename("y0"),
             y0sub = slice(y0, 3, 8).rename("y0sub"),
             ypar = should_overlap ? y0 : y0 + 1,
             y = opr::IncrSubtensor::make(ypar, y0sub, make_sub_desc(1, 6))
                         .rename("y1");

        HostTensorND expect;
        expect.copy_from(*host_x);
        auto ptr = expect.ptr<float>();
        memmove(ptr + 2 * SIZE_Y, ptr + 4 * SIZE_Y, 5 * SIZE_Y * sizeof(float));
        for (size_t i = 1; i < 6; ++i) {
            for (size_t j = 0; j < SIZE_Y; ++j) {
                ptr[i * SIZE_Y + j] += ptr[(i + 2) * SIZE_Y + j];
            }
        }
        if (!should_overlap) {
            for (size_t i = 0; i < SIZE_X * SIZE_Y; ++i) {
                ++ptr[i];
            }
        }

        ASSERT_TRUE(cg::is_static_var_value(y.node()));
        HostTensorND infer_result;
        infer_result.copy_from(
                graph->static_infer_manager().infer_value(y.node()));
        MGB_ASSERT_TENSOR_EQ(expect, infer_result);

        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(expect, host_y);

        if (!should_overlap) {
            ASSERT_EQ(prev_dev_ptr(ypar), prev_dev_ptr(y));
        }
    };
    run(false);
    run(true);
}

TEST(TestTensorManip, SetSubtensor) {
    using Checker = AutoOprChecker<3, 1>;
    auto make_graph = [](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        using AIdx = opr::Subtensor::AxisIndexer;
        auto x = inputs[0], v0 = inputs[1], v1 = inputs[2];
        x = x.rename("x");
        v0 = v0.rename("v0");
        v1 = v1.rename("v1");
        auto cv = [&](int v, bool dyn = false) {
            auto rst = x.make_scalar(v);
            if (dyn)
                rst = opr::MarkDynamicVar::make(rst);
            return rst;
        };
        auto
            // x0 = x[10::2] := v0
            x0 = opr::SetSubtensor::make(x, v0, {AIdx::make_interval(
                        0, cv(10), None, cv(2))}).rename("x0"),
            // x1 = x[:-10:2] := v0[:, 3] := v1
            x1 = opr::SetSubtensor::make(opr::MarkDynamicVar::make(x),
                    opr::SetSubtensor::make(v0, v1,
                        {AIdx::make_index(1, cv(3))}),
                    {AIdx::make_interval(0, None, cv(-10), cv(2))}
                    ).rename("x_sub1"),
            // x2 = (x[:5] := x[4:9])[3:-7:2, -1] := v1
            x2_t = opr::Subtensor::make(x, {
                    AIdx::make_interval(0, cv(4), cv(9), None)}).
                rename("x2_t"),
            x2 = opr::SetSubtensor::make(
                    opr::SetSubtensor::make(x, x2_t,
                        {AIdx::make_interval(0, None, cv(5), None)}),
                    v1, {AIdx::make_interval(0, cv(3), cv(-7), cv(2)),
                    AIdx::make_index(1, cv(-1))}).rename("x2"),
            y = (x0 + x1 + x2).rename("y");
        mgb_assert(cg::is_static_var_storage(x0.node()));
        mgb_assert(!cg::is_static_var_shape(x1.node()));
        mgb_assert(cg::is_static_var_storage(x2.node()));
        return {y};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto px = inp[0]->ptr<float>(), pv0 = inp[1]->ptr<float>(), pv1 = inp[2]->ptr<float>();
        auto ishp = inp[0]->shape();
        dest[0].comp_node(inp[0]->comp_node());
        dest[0].resize(ishp);
        auto optr = dest[0].ptr<float>();
        auto s0 = ishp.shape[0], s1 = ishp.total_nr_elems() / s0,
             s2 = s1 / ishp.shape[1];
        for (size_t i = 0; i < s0; ++ i) {
            for (size_t j = 0; j < s1; ++ j) {
                float x0, x1, x2;
                x0 = x1 = x2 = px[i * s1 + j];
                if (i >= 10 && (i - 10) % 2 == 0)
                    x0 = pv0[((i - 10) / 2)*s1 + j];

                if (i < s0 - 10 && i % 2 == 0) {
                    auto row = i / 2;
                    if (j / s2 == 3)
                        x1 = pv1[row*s2 + j%s2];
                    else
                        x1 = pv0[row*s1 + j];
                }

                if (i >= 3 && i < s0 - 7 && (i - 3) % 2 == 0 &&
                        j / s2 == ishp.shape[1] - 1)
                    x2 = pv1[((i-3)/2)*s2 + j%s2];
                else if (i < 5)
                    x2 = px[(i + 4)*s1 + j];

                optr[i*s1+j] = x0 + x1 + x2;
            }
        }
    };

    auto mkshp = [](const TensorShape &shp0) -> Checker::ShapeInpArray {
        mgb_assert(shp0.shape[0] > 10 && shp0.ndim >= 2 && shp0.shape[1] >= 4);
        auto shp1 = shp0;
        shp1.shape[0] = (shp0.shape[0] - 10) / 2;
        auto shp2 = shp1;
        for (size_t i = 2; i < shp2.ndim; ++ i)
            shp2.shape[i - 1] = shp2.shape[i];
        -- shp2.ndim;
        return {shp0, shp1, shp2};
    };

    Checker::RunOptions opt;
    opt.numdiff_eps = 1;
    Checker(make_graph, fwd).
        run(mkshp({16, 4, 2}), opt).
        run(mkshp({14, 10}), opt).
        run(mkshp({18, 5, 2, 3}), opt);
}

TEST(TestTensorManip, SetSubtensorCheckByShapeInfer) {
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_int;
    auto host_x = gen({12}), host_sub = gen({1}), host_idx = gen_int({1});
    host_idx->ptr<int>()[0] = 13;
    auto graph = ComputingGraph::make();
    using Ad = opr::Subtensor::AxisIndexer;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         sub = opr::Host2DeviceCopy::make(*graph, host_sub);
    auto idx1 = Ad::make_index(0,
                               opr::ImmutableTensor::make(*graph, *host_idx)),
         idx2 = Ad::make_index(0, opr::Host2DeviceCopy::make(*graph, host_idx));

    MGB_MARK_USED_VAR(x);
    MGB_MARK_USED_VAR(sub);
    MGB_MARK_USED_VAR(idx1);
    MGB_MARK_USED_VAR(idx2);
    ASSERT_THROW(opr::SetSubtensor::make(x, sub, {idx1}), MegBrainError);
    ASSERT_THROW(opr::SetSubtensor::make(x, sub, {idx2}), MegBrainError);
}

TEST(TestTensorManip, SetSubtensorShapeInfer) {
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_int;
    auto host_x = gen({12}), host_sub = gen({1}), host_idx = gen_int({1});
    host_idx->ptr<int>()[0] = 13;
    auto graph = ComputingGraph::make();
    auto&& mgr = graph->static_infer_manager();
    using Ad = opr::Subtensor::AxisIndexer;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         sub = opr::Host2DeviceCopy::make(*graph, host_sub),
         index = opr::Host2DeviceCopy::make_no_value_infer(*graph, host_idx);
    auto rt_static_idx = Ad::make_index(0, index * 2);
    auto y = opr::SetSubtensor::make(x, sub, {rt_static_idx});
    ASSERT_TRUE(mgr.infer_shape_fallible(y.node()));
}

TEST(TestTensorManip, SetSubtensorDynIdx) {
    HostTensorGenerator<> gen;
    auto host_x = gen({12}), host_sub = gen({1}),
         host_idx = gen({1});
    host_idx->ptr<float>()[0] = 3;
    auto dev_idx = std::make_shared<DeviceTensorND>();
    dev_idx->copy_from(*host_idx);

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         sub = opr::Host2DeviceCopy::make(*graph, host_sub),
         idx = opr::SharedDeviceTensor::make(*graph, dev_idx),
         y = opr::SetSubtensor::make(x, sub, {
                 opr::SetSubtensor::AxisIndexer::make_index(0, idx)});

    ASSERT_TRUE(cg::is_static_var_storage(y.node()));
    HostTensorND host_y;

    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();

    host_x->ptr<float>()[3] = host_sub->ptr<float>()[0];
    MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
}

TEST(TestTensorManip, SetSubtensorWithEmptyIndexDesc) {
    HostTensorGenerator<> gen;
    auto host_x = gen({12}), host_y = gen({12});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::SetSubtensor::make(x, y, {});

    ASSERT_TRUE(cg::is_static_var_storage(z.node()));
    HostTensorND host_z;

    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(*host_y, host_z);
}

TEST(TestTensorManip, IncrSubtensor) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph = [](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        using AIdx = opr::Subtensor::AxisIndexer;
        auto x = inputs[0];
        return {opr::IncrSubtensor::make(x, inputs[1],
                {AIdx::make_interval(0,
                        x.make_scalar(2), x.make_scalar(-2),
                        x.make_scalar(2))})};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto nr = inp[0]->shape(0);
        auto pv = inp[1]->ptr<float>(),
             pd = dest[0].copy_from(*inp[0]).ptr<float>();
        for (size_t i = 0; i < (nr - 3) / 2; ++ i) {
            pd[i * 2 + 2] += pv[i];
        }
    };

    Checker{make_graph, fwd}.
        run({TensorShape{5}, {1}}).
        run({TensorShape{8}, {2}}).
        run({TensorShape{23}, {10}});
}

TEST(TestTensorManip, Concat) {
    auto cns = load_multiple_xpus(4);

    using Checker = AutoOprChecker<3, 1>;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto sub0 = inputs[0], sub1 = opr::Copy::make(inputs[1], cns[1]),
             sub2 = opr::Copy::make(inputs[2], cns[2]),
             ret = opr::Concat::make({sub0, sub1, sub2}, 1, cns[3]);
        return {opr::Copy::make(ret, cns[0])};
    };

    auto fwd = [](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        size_t n = inp[0]->shape(0), c0 = inp[0]->shape(1),
               c1 = inp[1]->shape(1), c2 = inp[2]->shape(1), c = c0 + c1 + c2;
        auto i0 = inp[0]->ptr<float>(), i1 = inp[1]->ptr<float>(),
             i2 = inp[2]->ptr<float>(), o = dest[0].resize({n, c}).ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < c; ++j) {
                float cur;
                if (j < c0) {
                    cur = i0[i * c0 + j];
                } else if (j < c0 + c1) {
                    cur = i1[i * c1 + j - c0];
                } else {
                    cur = i2[i * c2 + j - c0 - c1];
                }
                o[i * c + j] = cur;
            }
        }
    };
    Checker checker{make_graph, fwd, cns[0]};
    checker.run({TensorShape{2, 3}, {2, 4}, {2, 5}})
            .run({TensorShape{2, 8}, {2, 3}, {2, 9}})
            .run({TensorShape{5, 10}, {5, 3}, {5, 4}});
}

TEST(TestTensorManip, ConcatWithNegativeAxis) {
    auto cns = load_multiple_xpus(4);

    using Checker = AutoOprChecker<3, 1>;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto sub0 = inputs[0], sub1 = opr::Copy::make(inputs[1], cns[1]),
             sub2 = opr::Copy::make(inputs[2], cns[2]),
             ret = opr::Concat::make({sub0, sub1, sub2}, -1, cns[3]);
        return {opr::Copy::make(ret, cns[0])};
    };

    auto fwd = [](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        size_t n = inp[0]->shape(0), c0 = inp[0]->shape(1),
               c1 = inp[1]->shape(1), c2 = inp[2]->shape(1), c = c0 + c1 + c2;
        auto i0 = inp[0]->ptr<float>(), i1 = inp[1]->ptr<float>(),
             i2 = inp[2]->ptr<float>(), o = dest[0].resize({n, c}).ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < c; ++j) {
                float cur;
                if (j < c0) {
                    cur = i0[i * c0 + j];
                } else if (j < c0 + c1) {
                    cur = i1[i * c1 + j - c0];
                } else {
                    cur = i2[i * c2 + j - c0 - c1];
                }
                o[i * c + j] = cur;
            }
        }
    };
    Checker checker{make_graph, fwd, cns[0]};
    checker.run({TensorShape{2, 3}, {2, 4}, {2, 5}})
            .run({TensorShape{2, 8}, {2, 3}, {2, 9}})
            .run({TensorShape{5, 10}, {5, 3}, {5, 4}});
}

TEST(TestTensorManip, ConcatEmpty) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3, 5}),
         host_y = gen({2, 0, 5});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::Concat::make({x, y}, 1);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(*host_x, host_z);
    host_x->resize({2, 0, 5});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(*host_y, host_z);
}

TEST(TestTensorManip, ConcatEmpty2) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 0, 5}),
         host_y = gen({2, 0, 6});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::Concat::make({x, y}, 2);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    ASSERT_EQ(TensorShape({2, 0, 11}), host_z.shape());
}

TEST(TestTensorManip, AxisAddRemove) {
    HostTensorGenerator<> gen;
    for (bool dyn_shape : {false, true}) {
        auto host_x = gen({2, 1, 5});
        using AD = opr::AxisAddRemove::AxisDesc;
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        if (dyn_shape) {
            x = opr::MarkDynamicVar::make(x);
        }
        auto y = opr::AxisAddRemove::make(x, {AD::make_add(0)}),
             z = opr::AxisAddRemove::make(x, {AD::make_remove(1)});
        HostTensorND host_y, host_z;
        auto func = graph->compile(
                {make_callback_copy(y, host_y), make_callback_copy(z, host_z)});
        func->execute();
        ASSERT_EQ(TensorShape({1, 2, 1, 5}), host_y.shape());
        ASSERT_EQ(TensorShape({2, 5}), host_z.shape());
        MGB_ASSERT_TENSOR_EQ(*host_x, host_y.resize(host_x->shape()));
        MGB_ASSERT_TENSOR_EQ(*host_x, host_z.resize(host_x->shape()));

        // test empty tensor
        host_x->resize({2, 1, 0});
        func->execute();
        ASSERT_EQ(TensorShape({1, 2, 1, 0}), host_y.shape());
        ASSERT_EQ(TensorShape({2, 0}), host_z.shape());
    }
}

TEST(TestTensorManip, Split) {
    auto cns = load_multiple_xpus(3);
    constexpr size_t C1 = 20, C2 = 30;
    constexpr size_t N = 2, C = C1 + C2;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({N, C}, cns[0]);
    auto graph = ComputingGraph::make();
    SymbolVar opr0 = opr::Host2DeviceCopy::make(*graph, host_opr0, {"opr0"});

    auto spl = opr::Split::make(
            opr0, Split::Options::make_partition(opr0, 1, {C1, C2}),
            OperatorNodeConfig("split").comp_node_arr({cns[1], cns[2]}));

    auto cost0 = opr::Dot::make(spl[0].flatten(), spl[0].flatten()),
         cost1_ = opr::Dot::make(spl[1].flatten(), spl[1].flatten()),
         cost1 = opr::Copy::make(cost1_,
                 OperatorNodeConfig().follow_comp_node(cost0)),
         cost = opr::Copy::make(cost0 + cost1,
                 OperatorNodeConfig().follow_comp_node(opr0)),
         grad = cg::grad(cost, opr0);

    HostTensorND host_spl0, host_spl1, host_grad;
    auto func = graph->compile({
        {spl[0], [&](DeviceTensorND &s) {
            host_spl0.copy_from(s);
        }},
        {spl[1], [&](DeviceTensorND &s) {
            host_spl1.copy_from(s);
        }},
        {grad, [&](DeviceTensorND &s) {
            host_grad.copy_from(s);
        }}
    });
    func->execute();

    auto o0 = host_spl0.sync().ptr<float>(), o1 = host_spl1.sync().ptr<float>(),
         c = host_opr0->ptr<float>(), g = host_grad.sync().ptr<float>();
    for (size_t i = 0, it = host_opr0->layout().total_nr_elems();
            i < it; i ++) {
        auto ch = i % C;
        auto n = i / C;
        if (ch < C1) {
            MGB_ASSERT_FLOAT_EQ(o0[n * C1 + ch], c[i]) <<
                ssprintf("failed at %zd", i);
        } else {
            MGB_ASSERT_FLOAT_EQ(o1[n * C2 + ch - C1], c[i]) <<
                ssprintf("failed at %zd", i);
        }
        MGB_ASSERT_FLOAT_EQ(c[i] * 2, g[i]) <<
            ssprintf("grad failed at %zd", i);
    }
}

TEST(TestTensorManip, SplitWithNegativeAxis) {
    auto cns = load_multiple_xpus(3);
    constexpr size_t C1 = 20, C2 = 30;
    constexpr size_t N = 2, C = C1 + C2;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({N, C}, cns[0]);
    auto graph = ComputingGraph::make();
    SymbolVar opr0 = opr::Host2DeviceCopy::make(*graph, host_opr0, {"opr0"});

    auto spl = opr::Split::make(
            opr0, Split::Options::make_partition(opr0, -1, {C1, C2}),
            OperatorNodeConfig("split").comp_node_arr({cns[1], cns[2]}));

    auto cost0 = opr::Dot::make(spl[0].flatten(), spl[0].flatten()),
         cost1_ = opr::Dot::make(spl[1].flatten(), spl[1].flatten()),
         cost1 = opr::Copy::make(cost1_,
                 OperatorNodeConfig().follow_comp_node(cost0)),
         cost = opr::Copy::make(cost0 + cost1,
                 OperatorNodeConfig().follow_comp_node(opr0)),
         grad = cg::grad(cost, opr0);

    HostTensorND host_spl0, host_spl1, host_grad;
    auto func = graph->compile({
        {spl[0], [&](DeviceTensorND &s) {
            host_spl0.copy_from(s);
        }},
        {spl[1], [&](DeviceTensorND &s) {
            host_spl1.copy_from(s);
        }},
        {grad, [&](DeviceTensorND &s) {
            host_grad.copy_from(s);
        }}
    });
    func->execute();

    auto o0 = host_spl0.sync().ptr<float>(), o1 = host_spl1.sync().ptr<float>(),
         c = host_opr0->ptr<float>(), g = host_grad.sync().ptr<float>();
    for (size_t i = 0, it = host_opr0->layout().total_nr_elems();
            i < it; i ++) {
        auto ch = i % C;
        auto n = i / C;
        if (ch < C1) {
            MGB_ASSERT_FLOAT_EQ(o0[n * C1 + ch], c[i]) <<
                ssprintf("failed at %zd", i);
        } else {
            MGB_ASSERT_FLOAT_EQ(o1[n * C2 + ch - C1], c[i]) <<
                ssprintf("failed at %zd", i);
        }
        MGB_ASSERT_FLOAT_EQ(c[i] * 2, g[i]) <<
            ssprintf("grad failed at %zd", i);
    }
}

TEST(TestTensorManip, SplitToDynOutShape) {
    using Checker = AutoOprChecker<1, 2>;
    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        auto x = inputs[0];
        auto y = opr::Split::make(x,
                opr::Split::Options::make_partition(0, {
                    x.make_scalar(3),
                    opr::MarkDynamicVar::make(
                            opr::GetVarShape::make(x, 0) - x.make_scalar(3))}));
        return {y[0], y[1]};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto sub = [&](size_t begin, Maybe<ptrdiff_t> end) {
            auto &&iv = inp[0];
            return iv->sub(Slice(begin, end, None).apply(iv->layout(), 0));
        };
        dest[0].copy_from(sub(0, 3));
        dest[1].copy_from(sub(3, None));
    };

    Checker{make_graph, fwd}.
        run({TensorShape{5}}).
        run({TensorShape{8}}).
        run({TensorShape{9, 3}});
}

TEST(TestTensorManip, SplitToDynOutStorage) {
    using Checker = AutoOprChecker<1, 2>;
    auto make_graph = [&](
            const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
        auto x = inputs[0];
        auto y = opr::Split::make(x,
                opr::Split::Options::make_partition(0, {
                    x.make_scalar(3),
                    opr::GetVarShape::make(x, 0) - x.make_scalar(3)}));
        auto y0 = opr::Copy::make(y[0], x.node()->comp_node().change_stream(1));
        y0 = opr::Copy::make(y0, x.node()->comp_node());
        return {y0, y[1]};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto sub = [&](size_t begin, Maybe<ptrdiff_t> end) {
            auto &&iv = inp[0];
            return iv->sub(Slice(begin, end, None).apply(iv->layout(), 0));
        };
        dest[0].copy_from(sub(0, 3));
        dest[1].copy_from(sub(3, None));
    };

    Checker{make_graph, fwd}.
        run({TensorShape{5}}).
        run({TensorShape{8}}).
        run({TensorShape{9, 3}});
}

namespace {

void do_test_dynamic_split(bool multiple_cn, bool force_dynamic) {
    auto cns = load_multiple_xpus(3);
    constexpr size_t N = 2, C = 51;
    HostTensorGenerator<> gen;
    auto host_x = gen({N, C}, cns[0]),
         host_sub_begin = gen({1}, cns[0]),
         host_sub_end = gen({1}, cns[0]);
    host_sub_begin->ptr<float>()[0] = 0;
    host_sub_end->ptr<float>()[0] = 2;
    auto graph = ComputingGraph::make();

    SymbolVar x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x, {"x"}),
              sub_begin = opr::Host2DeviceCopy::make_no_fwd(
                      *graph, host_sub_begin, {"sub_begin"}),
              sub_end = opr::Host2DeviceCopy::make_no_fwd(
                      *graph, host_sub_end, {"sub_end"}),
              xsub = opr::Subtensor::make(x, {
                      opr::Subtensor::AxisIndexer::make_interval(
                              1, sub_begin, sub_end, None)}).rename("xsub");

    OperatorNodeConfig split_config("split");
    if (multiple_cn) {
        split_config.comp_node_arr({cns[1], cns[2]});
    }

    if (force_dynamic)
        xsub = opr::MarkDynamicVar::make(xsub);

    auto spl = opr::Split::make(
            xsub, Split::Options::make_callback(1, 2,
                [](size_t s) {return std::vector<size_t>{s / 2, s - s / 2};}),
            split_config);

    if (multiple_cn) {
        spl[0] = opr::Sleep::make(spl[0], 0.1);
        spl[1] = opr::Sleep::make(spl[1], 0.2);
    }
    auto cost0 = opr::Dot::make(spl[0].flatten(), spl[0].flatten()),
         cost1_ = opr::Dot::make(spl[1].flatten(), spl[1].flatten()),
         cost1 = opr::Copy::make(cost1_,
                 OperatorNodeConfig().follow_comp_node(cost0)),
         cost = opr::Copy::make(cost0 + cost1,
                 OperatorNodeConfig().follow_comp_node(x)) * 0.5f,
         grad = cg::grad(cost, x);

    HostTensorND host_spl0, host_spl1, host_grad;
    auto func = graph->compile({
            make_callback_copy(spl[0], host_spl0),
            make_callback_copy(spl[1], host_spl1),
            make_callback_copy(grad, host_grad)
    });

    if (force_dynamic)
        ASSERT_TRUE(!cg::is_static_var_shape(spl[0].node()));
    else {
        auto cb = [](cg::OperatorNodeBase *op) {
            for (auto i: op->output()) {
                mgb_assert(cg::is_static_var_shape(i),
                        "dynamic var: %s", cg::dump_var_info({i}).c_str());
            }
            return true;
        };
        func->iter_opr_seq(cb);
    }

    bool failed = false, fwd_checked = false;
    auto run_and_check = [&](size_t begin, size_t end) {
        ASSERT_FALSE(failed);
        failed = true;

        host_sub_begin->ptr<float>()[0] = begin;
        host_sub_end->ptr<float>()[0] = end;
        func->execute();

        auto mid = begin + (end - begin) / 2;

        auto inp = host_x->ptr<float>(), grad = host_grad.ptr<float>();
        ASSERT_EQ(host_spl0.shape(), TensorShape({N, mid - begin}));
        ASSERT_EQ(host_spl1.shape(), TensorShape({N, end - mid}));
        if (!force_dynamic && !multiple_cn && !begin && mid - begin == 1) {
            // check mem fwd for spl[0]
            // do not check for spl[1] since flatten() causes copy
            ASSERT_EQ(prev_dev_ptr(spl[0]), static_cast<const
                    dt_float32*>(prev_dev_ptr(x)));
            fwd_checked = true;
        }
        for (size_t i = 0, it = host_x->layout().total_nr_elems();
                i < it; ++ i) {
            auto ch = i % C;
            auto n = i / C;
            float expect_grad;
            if (ch >= begin && ch < mid) {
                MGB_ASSERT_FLOAT_EQ(inp[i],
                        *host_spl0.ptr<float>({n, ch - begin})
                        ) << ssprintf("failed at (%zu, %zu),sub=[: ,%zu:%zu]",
                            i, ch, begin, end);
                expect_grad = inp[i];
            } else if (ch >= mid && ch < end) {
                MGB_ASSERT_FLOAT_EQ(inp[i],
                        *host_spl1.ptr<float>({n, ch - mid})
                        ) << ssprintf("failed at (%zu, %zu),sub=[: ,%zu:%zu]",
                            i, ch, begin, end);
                expect_grad = inp[i];
            } else {
                expect_grad = 0;
            }
            MGB_ASSERT_FLOAT_EQ(expect_grad, grad[i]) <<
                ssprintf("grad failed at (%zu, %zu), sub=x[:, %zu:%zu]",
                        n, ch, begin, end);
        }

        failed = false;
    };

    run_and_check(0, 3);
    run_and_check(2, 8);
    run_and_check(5, 12);
    run_and_check(1, C - 1);
    run_and_check(0, C);
    run_and_check(C - 2, C);
    run_and_check(0, 2);

    if (!multiple_cn && !force_dynamic) {
        ASSERT_TRUE(fwd_checked);
    }
}

}

TEST(TestTensorManip, DynamicSplit00) {
    do_test_dynamic_split(false, false);
}

TEST(TestTensorManip, DynamicSplit01) {
    do_test_dynamic_split(false, true);
}

TEST(TestTensorManip, DynamicSplit10) {
    do_test_dynamic_split(true, false);
}

TEST(TestTensorManip, DynamicSplit11) {
    do_test_dynamic_split(true, true);
}

TEST(TestTensorManip, SplitFromDynStorage) {
    HostTensorGenerator<> gen;
    auto host_x = gen({4});
    auto graph = cg::ComputingGraph::make();
    auto x = opr::MarkDynamicVar::make(
            opr::Host2DeviceCopy::make(*graph, host_x)).reshape({4});
    ASSERT_TRUE(cg::is_static_var_shape(x.node()));
    auto y = opr::Split::make(x, opr::Split::Options::make_partition(
                x, 0, {1, 3}));
    HostTensorND y0, y1;
    auto func = graph->compile({
            make_callback_copy(y[0], y0), make_callback_copy(y[1], y1)});

    func->execute();
    ASSERT_FALSE(cg::is_static_var_storage(x.node()));
    HostTensorND expt{host_x->comp_node(), host_x->dtype()};
    expt.resize({1}).ptr<float>()[0] = host_x->ptr<float>()[0];
    MGB_ASSERT_TENSOR_EQ(expt, y0);
    expt.resize({3});
    for (int i = 0; i < 3; ++ i)
        expt.ptr<float>()[i] = host_x->ptr<float>()[i + 1];
    MGB_ASSERT_TENSOR_EQ(expt, y1);
}

TEST(TestTensorManip, SplitPreAllocatedMultiCN) {
    auto cns = load_multiple_xpus(3);
    HostTensorGenerator<> gen;
    auto host_x = gen({3}, cns[0]);
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x).sync();
    auto graph = cg::ComputingGraph::make();
    auto x = opr::SharedDeviceTensor::make(*graph, dev_x);
    auto ys = opr::Split::make(x, opr::Split::Options::make_average(0, 3),
            OperatorNodeConfig{}.comp_node_arr({cns.begin(), cns.end()}));
    ASSERT_EQ(3u, ys.size());
    HostTensorND y0, y1, y2;
    auto func = graph->compile({
            make_callback_copy(ys[0], y0),
            make_callback_copy(opr::Copy::make(ys[1], {cns[0]}), y1),
            make_callback_copy(ys[2], y2)});
    func->execute();
    ASSERT_TRUE(cg::is_static_var_storage(ys[0].node()));
    ASSERT_FALSE(cg::is_static_var_storage(ys[1].node()));
    ASSERT_EQ(x.node()->prev_dev_ptr(), ys[0].node()->prev_dev_ptr());
    ASSERT_EQ(host_x->ptr<float>()[0], y0.ptr<float>()[0]);
    ASSERT_EQ(host_x->ptr<float>()[1], y1.ptr<float>()[0]);
    ASSERT_EQ(host_x->ptr<float>()[2], y2.ptr<float>()[0]);
}

TEST(TestTensorManip, SplitMemfwdMultipleTimesWithOffset) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({4}, cns[0]);
    auto graph = cg::ComputingGraph::make();
    auto x0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         x = opr::Subtensor::make(x0,
                 {opr::Subtensor::AxisIndexer::make_interval(
                         0, x0.make_scalar(1), None, None)});
    auto ys = opr::Split::make(x, opr::Split::Options::make_average(0, 3));
    ASSERT_EQ(3u, ys.size());
    HostTensorND y0, y1, y2;
    auto func = graph->compile({
            make_callback_copy(ys[0], y0),
            make_callback_copy(opr::Copy::make(ys[1], {cns[1]}), y1),
            make_callback_copy(ys[2], y2)});
    func->execute();
    ASSERT_FALSE(cg::is_static_var_storage(ys[0].node()));
    ASSERT_TRUE(cg::is_static_var_shape(ys[0].node()));
    ASSERT_FALSE(cg::is_static_var_storage(ys[1].node()));
    ASSERT_EQ(host_x->ptr<float>()[1], y0.ptr<float>()[0]);
    ASSERT_EQ(host_x->ptr<float>()[2], y1.ptr<float>()[0]);
    ASSERT_EQ(host_x->ptr<float>()[3], y2.ptr<float>()[0]);
    ASSERT_EQ(static_cast<const float*>(prev_dev_ptr(x0)) + 3,
            prev_dev_ptr(ys[2]));
}

TEST(TestTensorManip, SplitValueInfer) {
    auto cns = load_multiple_xpus(3);
    HostTensorGenerator<> gen;
    auto host_x = gen({3});
    auto graph = cg::ComputingGraph::make();
    auto x = opr::ImmutableTensor::make(*graph, *host_x);

    auto ys = opr::Split::make(x, opr::Split::Options::make_average(0, 3),
            OperatorNodeConfig{}.comp_node_arr({cns.begin(), cns.end()}));
    for (size_t i = 0; i < 3; ++ i) {
        // split itself does not replace imm vars; use +0 to trigger optimizer
        auto var = (ys[i] + 0).node();
        ASSERT_TRUE(var->owner_opr()->same_type<opr::ImmutableTensor>());
        ASSERT_EQ(cns[i], var->comp_node());
        HostTensorND hv;
        hv.copy_from(var->owner_graph()->static_infer_manager().infer_value(
                    var));
        ASSERT_EQ(TensorShape{1}, hv.shape());
        ASSERT_EQ(host_x->ptr<float>()[i], hv.ptr<float>()[0]);
    }
}

TEST(TestTensorManip, SplitZeroGrad) {
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 2});
    auto graph = cg::ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto ys = opr::Split::make(x, opr::Split::Options::make_average(0, 3));
    auto loss = opr::reduce_sum(ys[2] * ys[2], x.make_scalar(1)),
         gx = cg::grad(loss, x);
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->execute();
    auto px = host_x->ptr<float>(), pgx = host_gx.ptr<float>();
    for (int i = 0; i < 2; ++ i) {
        MGB_ASSERT_FLOAT_EQ(0.f, pgx[i]);
        MGB_ASSERT_FLOAT_EQ(0.f, pgx[2 + i]);
        MGB_ASSERT_FLOAT_EQ(px[4 + i] * 2, pgx[4 + i]);
    }
}

TEST(TestTensorManip, DynamicFill) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    auto graph = cg::ComputingGraph::make();
    auto x = opr::MarkDynamicVar::make(
            opr::Host2DeviceCopy::make(*graph, host_x)),
         y = x.fill_retain_dtype(23);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    bool failed = false;
    auto check = [&](const TensorShape &ishp) {
        ASSERT_FALSE(failed);
        failed = true;
        host_x->resize(ishp);
        func->execute();
        ASSERT_EQ(ishp, host_y.shape());
        auto ptr = host_y.ptr<float>();
        for (size_t i = 0, it = host_y.shape().total_nr_elems();
                i < it; ++ i)
            ASSERT_EQ(23, ptr[i]);
        failed = false;
    };
    check({4, 2});
    check({2, 4});
    check({23});
}

TEST(TestTensorManip, Pooling2DBySetSub) {
    constexpr int PH = 4, PW = 3;

    using Checker = AutoOprChecker<1, 1>;

    bool run_dyn = false;

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {

        auto x = inputs.at(0);

        if (run_dyn)
            x = opr::MarkDynamicVar::make(x);

        x.rename("x");
        auto cv = [&](int v, bool dyn = false) {
            auto rst = x.make_scalar(v);
            if (dyn)
                rst = opr::MarkDynamicVar::make(rst);
            return rst;
        };

        auto oh = (opr::GetVarShape::make(x, 0) / PH).rename("oh"),
             ow = (opr::GetVarShape::make(x, 1) / PW).rename("ow"),
             y_tmp_shape = opr::Concat::make({cv(PH * PW), oh, ow}, 0),
             y_tmp = opr::Alloc::make(y_tmp_shape, dtype::Float32());

        if (!run_dyn)
            mgb_assert(cg::is_static_var_storage(y_tmp.node()));

        using Ad = opr::Subtensor::AxisIndexer;
        for (size_t i = 0, num = 0; i < (size_t)PH; ++ i) {
            for (size_t j = 0; j < (size_t)PW; ++ j) {
                bool dyn = run_dyn && num % 2;
                auto xsub = opr::Subtensor::make(x,
                        {Ad::make_interval(0, cv(i, dyn), None, cv(PH)),
                        Ad::make_interval(1, cv(j), None, cv(PW))}).rename(
                            ssprintf("sub(%zu, %zu)", i, j));
                y_tmp = opr::SetSubtensor::make(y_tmp, xsub,
                        {Ad::make_index(0, cv(num, dyn))}).rename(
                            ssprintf("y(%zu, %zu)", i, j));
                if (!run_dyn) {
                    mgb_assert(cg::is_static_var_storage(xsub.node()));
                    mgb_assert(cg::is_static_var_storage(y_tmp.node()));
                } else if (dyn)
                    y_tmp = opr::MarkDynamicVar::make(y_tmp);
                ++ num;
            }
        }
        auto y = opr::Reduce::make(y_tmp, {opr::Reduce::Mode::SUM, 0});
        y = opr::AxisAddRemove::make(y,
                {opr::AxisAddRemove::AxisDesc::make_remove(0)});
        if (!run_dyn)
            mgb_assert(cg::is_static_var_storage(y.node()));
        return {y};
    };

    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto &&ishp = inp.at(0)->shape();
        auto oshp = ishp;
        mgb_assert(oshp.shape[0] % PH == 0);
        mgb_assert(oshp.shape[1] % PW == 0);
        oshp.shape[0] /= PH;
        oshp.shape[1] /= PW;

        auto optr = dest.at(0).comp_node(inp[0]->comp_node()).
            resize(oshp).ptr<float>();

        auto &&iv = *inp.at(0);
        for (size_t i = 0; i < oshp.shape[0]; ++ i)
            for (size_t j = 0; j < oshp.shape[1]; ++ j) {
                auto ii = i * PH, ij = j * PW;
                float sum = 0;
                for (size_t di = 0; di < PH; ++ di)
                    for (size_t dj = 0; dj < PW; ++ dj) {
                        sum += *iv.ptr<float>({ii + di, ij + dj});
                    }
                *(optr ++) = sum;
            }
    };

    auto run = [&](bool dyn) {
        run_dyn = dyn;
        Checker(make_graph, fwd).
            run({TensorShape{PH * 1, PW * 2}}).
            run({TensorShape{PH * 4, PW * 3}}).
            run({TensorShape{PH * 2, PW * 2}});
    };

    run(false);
    run(true);
}

TEST(TestTensorManip, Flatten) {
    HostTensorGenerator<> gen;
    auto host_x = gen({20});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         y = x.flatten();
    y = y + x.reshape(y.symshape());
    ASSERT_EQ(TensorShape{20}, y.node()->shape());
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    for (auto &&ishp: {
            TensorShape{2, 5}, TensorShape{6, 8, 1}, TensorShape{3}}) {
        *host_x = *gen(ishp);
        func->execute();
        auto expected = host_x->sub(SubTensorSpec::make_from_layout({{
                    ishp.total_nr_elems()}, host_x->dtype()}));
        auto ptr = expected.ptr<float>();
        for (size_t i = 0; i < expected.shape()[0]; ++ i)
            ptr[i] *= 2;
        MGB_ASSERT_TENSOR_EQ(expected, host_y);
    }
}

TEST(TestTensorManip, FillWithDtypeDedup) {
    HostTensorGenerator<> gen;
    auto host_x = gen({20});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    SymbolVar vals[] = {
        x.fill_retain_dtype(0), x.fill_retain_dtype(1),
        x.fill_retain_dtype(0), x.fill_retain_dtype(1),
        x.fill_retain_dtype(0.f), x.fill_retain_dtype(1.f),
        x.fill_retain_dtype(0.f), x.fill_retain_dtype(1.f),
    };
    for (int i: {0, 1})
        for (int j = 2; j < 8; j += 2)
            ASSERT_EQ(vals[i].node(), vals[i + j].node()) << i << ' ' << i + j;
    ASSERT_NE(vals[0].node(), vals[1].node());
}

TEST(TestTensorManip, StrongContig) {
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 1});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Dimshuffle::make(x, {1, 0});
    auto cb = [](DeviceTensorND &dv) {
        TensorLayout expect{{1, 5}, dv.dtype()};
        ASSERT_EQ(expect, dv.layout());
    };
    auto func = graph->compile({{y, cb}});
    func->execute();
}

namespace{
void test_param_pack_concat(const TensorShapeArray &shapes, DType type){
    auto cn = CompNode::load("xpu0");
    auto graph = ComputingGraph::make();
    auto align = cn.get_mem_addr_alignment() / type.size();

    size_t size = 0;
    std::vector<size_t> begins;
    for (auto &&shape : shapes){
        size = get_aligned_power2(size, align);
        begins.push_back(size);
        size += shape.total_nr_elems();
    }

    SmallVector<SymbolVar> srcs;
    for(size_t i = 0; i < shapes.size(); i++){
        auto data = std::make_shared<HostTensorND>();
        data->comp_node(cn).dtype(dtype::Int32()).resize(shapes[i]);
        auto ptr = data->ptr<dt_int32>();
        for(size_t j = 0; j < shapes[i].total_nr_elems(); j++){
            ptr[j] = j;
        }
        auto nd = opr::Host2DeviceCopy::make(*graph, data);
        srcs.push_back(nd);
    }

    auto host_offsets_gen = megdnn::ParamPackConcat::gen_offsets(shapes,
            cn.get_mem_addr_alignment(), 4);
    ASSERT_EQ(host_offsets_gen.back(), size);
    auto host_offsets = std::make_shared<HostTensorND>();
    host_offsets->comp_node(cn).dtype(dtype::Int32{}).resize({srcs.size() * 2});
    memcpy(host_offsets->raw_ptr(), host_offsets_gen.data(), srcs.size() * 8);
    auto offsets = opr::Host2DeviceCopy::make(*graph, host_offsets);

    auto z = opr::ParamPackConcat::make(srcs, offsets, host_offsets_gen);
    HostTensorND host_z;

    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();

    HostTensorND expected;
    expected.comp_node(cn).dtype(dtype::Int32()).resize({size});
    {
        auto ptr = expected.ptr<dt_int32>();

        memset(ptr, 0, sizeof(int32_t)*size);
        for(size_t i = 0; i < begins.size(); i++){
            auto begin = begins[i];
            auto shape = shapes[i];
            for(size_t j = 0; j < shape.total_nr_elems(); j++){
                ptr[begin + j] = j;
            }
        }
    }
    MGB_ASSERT_TENSOR_EQ(expected, host_z);
}

template <size_t nr_out>
void test_param_pack_split(const TensorShapeArray& shapes) {
    auto cn = CompNode::load("xpu0");
    auto align = std::max<size_t>(cn.get_mem_addr_alignment() / 4, 1);
    size_t concat_size = 0;
    mgb_assert(shapes.size() == nr_out);
    for (auto&& i : shapes) {
        concat_size =
                get_aligned_power2(concat_size, align) + i.total_nr_elems();
    }

    using Checker = AutoOprChecker<1, nr_out>;

    auto make_graph = [&](const typename Checker::SymInpArray& inputs) ->
            typename Checker::SymOutArray {
        auto offsets_val = megdnn::ParamPackConcat::gen_offsets(
                shapes, cn.get_mem_addr_alignment(), 4);
        HostTensorND offsets;
        std::copy_n(offsets_val.data(), offsets_val.size(),
                    offsets.dtype(dtype::Int32{})
                            .comp_node(cn)
                            .resize({offsets_val.size()})
                            .ptr<dt_int32>());
        auto out = opr::ParamPackSplit::make(inputs[0], offsets_val,
                                             shapes);
        mgb_assert(out.size() == nr_out);
        typename Checker::SymOutArray ret;
        for (size_t i = 0; i < nr_out; ++i) {
            ret[i] = out[i];
        }
        return ret;
    };

    auto fwd = [&](typename Checker::NumOutArray& dest,
                   typename Checker::NumInpArray inp) {
        size_t offset = 0;
        auto ptr = inp[0]->template ptr<float>();
        for (size_t i = 0; i < nr_out; ++i) {
            dest[i].resize(shapes[i]);
            offset = get_aligned_power2(offset, align);
            auto nr_elem = shapes[i].total_nr_elems();
            memcpy(dest[i].template ptr<float>(), ptr + offset, nr_elem * 4);
            offset += nr_elem;
        }
    };

    Checker{make_graph, fwd}
            .run({TensorShape{concat_size}})
            .run({TensorShape{concat_size}})
            .run({TensorShape{concat_size}});
}

} // anonymous namespace

TEST(TestParamPack, Concat){
    TensorShapeArray array = {{129}, {21}};
    test_param_pack_concat(array, dtype::Int32());

    array = {{23}, {32}, {75}, {45}};
    test_param_pack_concat(array, dtype::Int32());

    array = {{129}, {512}, {513}, {27}};
    test_param_pack_concat(array, dtype::Int32());
}

TEST(TestParamPack, Split) {
    test_param_pack_split<2>({{2, 3}, {4, 5, 6}});
    test_param_pack_split<3>({{2, 9}, {123}, {5, 3}});
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
