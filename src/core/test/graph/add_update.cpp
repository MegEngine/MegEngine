/**
 * \file src/core/test/graph/add_update.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/tensor_manip.h"

#include "megbrain/test/helper.h"

using namespace mgb;

#if MGB_ENABLE_EXCEPTION
TEST(TestGraph, ForceUpdate0) {
    // opr taking both versions of var
    auto t0 = std::make_shared<DeviceTensorND>(
            CompNode::load("xpu0"), TensorShape{2, 2});
    auto t1 = std::make_shared<HostTensorND>(
            CompNode::load("xpu0"), TensorShape{2, 2});
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::SharedDeviceTensor::make(*graph, t0),
        y = opr::Host2DeviceCopy::make(*graph, t1),
        z = opr::AddUpdate::make(x, y),
        zz = opr::add(x, z);
    EXPECT_THROW(graph->compile({{zz, [](DeviceTensorND&){}}}),
            GraphError);
}
#endif

TEST(TestGraph, ForceUpdate1) {
    auto t0 = std::make_shared<DeviceTensorND>(
            CompNode::load("xpu0"), TensorShape{2, 2});
    auto t1 = std::make_shared<DeviceTensorND>(
            CompNode::load("xpu0"), TensorShape{2, 2});
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::SharedDeviceTensor::make(*graph, t0),
        y = opr::SharedDeviceTensor::make(*graph, t1),
        // x' = x + y
        z = opr::AddUpdate::make(x, y),
        // y' = y + x'
        zz = opr::AddUpdate::make(y, z);

    EXPECT_NO_THROW(graph->compile({{zz, [](DeviceTensorND&){}}}));
}

TEST(TestGraph, ForceUpdate2) {
    // check for mem fwd with force update
    HostTensorGenerator<> gen;
    auto host_x0 = gen({4}), host_y = gen({4});
    auto t0 = std::make_shared<DeviceTensorND>();
    t0->copy_from(*host_x0);
    auto graph = ComputingGraph::make();
    auto pri = [&](SymbolVar var, int pri) {
        set_priority(var, pri);
        return var;
    };
    SymbolVar
        x = pri(opr::SharedDeviceTensor::make(*graph, t0).rename("x"), -100),
        y = pri(opr::Host2DeviceCopy::make(*graph, host_y).rename("y"), -100),
        z = opr::AddUpdate::make(x, y).rename("z"),
        x1 = pri(x.reshape({1, 4}).rename("x1"), -50),
        x2 = pri(x1.reshape({4, 1}).rename("x2"), -50),
        x3 = pri(x2.reshape({2, 2}).rename("x3"), -50),
        x4 = pri(x3.reshape({4}).rename("x4"), 50),
        zz = (x4 + z).rename("zz");

    HostTensorND host_zz;
    auto func = graph->compile({make_callback_copy(zz, host_zz)});
    func->execute();

    EXPECT_EQ(dev_ptr(x), dev_ptr(x1));
    EXPECT_EQ(dev_ptr(x1), dev_ptr(x2));
    EXPECT_NE(dev_ptr(x1), dev_ptr(x3));
    EXPECT_EQ(dev_ptr(x3), dev_ptr(x4));

    HostTensorND t0_updated;
    t0_updated.copy_from(*t0).sync();

    auto px0 = host_x0->ptr<float>(), py = host_y->ptr<float>(),
         pzz = host_zz.ptr<float>(), pt0 = t0_updated.ptr<float>();
    for (int i = 0; i < 4; i ++) {
        MGB_ASSERT_FLOAT_EQ(px0[i] * 2 + py[i], pzz[i]);
        MGB_ASSERT_FLOAT_EQ(px0[i] + py[i], pt0[i]);
    }
}

TEST(TestGraph, ForceUpdate3) {
    // opr relies on both versions, solve by copy
    constexpr size_t SIZE = 123;
    HostTensorGenerator<> gen;
    auto host_t0 = gen({SIZE});
    auto t0 = std::make_shared<DeviceTensorND>();
    auto t1 = gen({SIZE});
    t0->copy_from(*host_t0);
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::SharedDeviceTensor::make(*graph, t0),
        y = opr::Host2DeviceCopy::make(*graph, t1),
        z = opr::AddUpdate::make(x, y),
        zz = opr::add(opr::Copy::make(x), z);
    HostTensorND host_zz, host_t0u;
    auto func = graph->compile({make_callback_copy(zz, host_zz)});
    func->execute();
    host_t0u.copy_from(*t0).sync();
    ASSERT_EQ(host_t0->shape(), host_t0u.shape());
    ASSERT_EQ(host_t0->shape(), host_zz.shape());
    auto pt0 = host_t0->ptr<float>(), pt0u = host_t0u.ptr<float>(),
         pt1 = t1->ptr<float>(), pz = host_zz.ptr<float>();
    for (size_t i = 0; i < SIZE; ++ i) {
        MGB_ASSERT_FLOAT_EQ(pt0[i] + pt1[i], pt0u[i]);
        MGB_ASSERT_FLOAT_EQ(pt0[i] + pt0u[i], pz[i]);
    }
}

TEST(TestGraph, ForceUpdate4) {
    // waiting for multiple comp nodes
    auto cns = load_multiple_xpus(2);
    constexpr size_t SIZE = 432;
    HostTensorGenerator<> gen;
    auto host_t0 = gen({SIZE}, cns[0]);
    auto t0 = std::make_shared<DeviceTensorND>();
    auto t1 = gen({SIZE}, cns[0]);
    t0->copy_from(*host_t0);
    OperatorNodeConfig conf1(cns[1]);
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::SharedDeviceTensor::make(*graph, t0),
        y = opr::Host2DeviceCopy::make(*graph, t1),
        z = opr::AddUpdate::make(x, y),
        x1 = opr::Copy::make(x, conf1),
        x2 = opr::Copy::make(x1),
        x3 = opr::Copy::make(
                opr::Sleep::make(x.reshape({SIZE / 2, 2}), 0.1).reshape({SIZE}),
                conf1),
        z1 = opr::Copy::make(z, conf1),
        s = x1 + z1 + x2 + x3;
    HostTensorND host_s, host_t0u;
    auto func = graph->compile({make_callback_copy(s, host_s)});

    // check that z waits on x1
    bool found = false;
    for (auto &&spec: z.node()->owner_opr()->input_waiting_spec()) {
        if (spec.comp_node == cns[0]) {
            found = true;
            auto &&v = spec.dev_ready;
            ASSERT_EQ(1u, v.size());
            ASSERT_EQ(x1.node(), v[0]);
        }
    }
    ASSERT_TRUE(found);

    func->execute();
    host_t0u.copy_from(*t0).sync();

    ASSERT_EQ(host_t0->shape(), host_t0u.shape());
    ASSERT_EQ(host_t0->shape(), host_s.shape());
    auto px = host_t0->ptr<float>(), py = t1->ptr<float>(),
         pz = host_t0u.ptr<float>(), ps = host_s.ptr<float>();
    for (size_t i = 0; i < SIZE; ++ i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + py[i], pz[i]);
        MGB_ASSERT_FLOAT_EQ(px[i] * 3 + pz[i], ps[i]);
    }
}

TEST(TestGraph, ForceUpdate5) {
    // unused reader for force_update src
    constexpr size_t SIZE = 5;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE});
    auto dev_x = std::make_shared<DeviceTensorND>();
    auto host_dx = gen({SIZE});
    bool called = false;
    dev_x->copy_from(*host_x);
    auto graph = ComputingGraph::make();
    SymbolVar x = opr::SharedDeviceTensor::make(*graph, dev_x),
              dx = opr::Host2DeviceCopy::make(*graph, host_dx),
              xu = opr::AddUpdate::make(x, dx);
    opr::CallbackInjector::make(x, [&](DeviceTensorND&){called = true;});
    HostTensorND host_xu;
    auto func = graph->compile({make_callback_copy(xu, host_xu)});
    func->execute();
    ASSERT_FALSE(called);
}

TEST(TestGraph, ForceUpdateMultiple) {
    constexpr size_t SIZE = 5;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE});
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x);
    auto graph = ComputingGraph::make();
    SymbolVar x = opr::SharedDeviceTensor::make(*graph, dev_x).rename("x"),
              xu1 = opr::AddUpdate::make(x, x.make_scalar(1.f)).rename("xu1"),
              xu2 = opr::AddUpdate::make(x, x.make_scalar(2.f)).rename("xu2"),
              xrd = opr::SetGrad::make(x, opr::SetGrad::zero_grad),
              y = xrd * 2;
    set_priority(xu1, -100);
    set_priority(xu2, -100);

    ASSERT_THROW(graph->compile({{xu1, {}}, {xu2, {}}}), GraphError);

    auto check = [&](SymbolVar dest, float delta) {
        HostTensorND host_y, expect, y_expect;
        auto func = graph->compile({
                {dest, {}},
                make_callback_copy(y, host_y)});
        func->execute();

        ASSERT_NE(xrd.node()->prev_dev_ptr(), x.node()->prev_dev_ptr());
        ASSERT_EQ(dest.node()->prev_dev_ptr(), x.node()->prev_dev_ptr());

        expect.copy_from(*host_x);
        y_expect.copy_from(*host_x);
        auto ptr0 = expect.ptr<float>(), ptr1 = y_expect.ptr<float>();
        for (size_t i = 0, it = expect.shape().total_nr_elems(); i < it; ++ i) {
            ptr0[i] += delta;
            ptr1[i] = (ptr0[i] - (1 + (dest.node() == xu2.node()))) * 2;
        }
        HostTensorND get;
        get.copy_from(*dev_x).sync();
        MGB_ASSERT_TENSOR_EQ(expect, get);
        MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
    };
    check(xu1, 1);
    check(xu2, 3);
    ASSERT_THROW(graph->compile({{xu1, {}}, {xu2, {}}}), GraphError);
    check(xu2, 5);
    check(xu1, 6);
}

TEST(TestGraph, ForceUpdateOtherCn) {
    auto cn1 = CompNode::load("xpu0:1");
    CompNodeSyncManager sync;
    sync.comp_node(cn1).add_waiter_record(true);

    auto set_finish = [&](DeviceTensorND&) {
        sync.set_ready();
    };
    auto wait_finish = [&](DeviceTensorND &dv) {
        dv.comp_node().device_wait_event(
                sync.busy_wait_set_ready_and_get_event());
    };

    HostTensorGenerator<> gen;
    auto host_x = gen({16}, "xpu0");
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x);
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    graph->options().async_exec_level |= 0b10;
    SymbolVar x = opr::SharedDeviceTensor::make(*graph, dev_x).rename("x"),
              x0 = x.reshape({4, 4}).rename("x0"),
              x1 = (opr::CallbackInjector::make(x0, wait_finish).rename("xslp").
                      reshape({16}) + 1).rename("x1"),
              xud0 = opr::AddUpdate::make(x, x.make_scalar(3), {}, cn1),
              xud = opr::CallbackInjector::make(xud0, set_finish);
    set_priority(xud0, 100);
    HostTensorND host_x1;
    auto func = graph->compile({make_callback_copy(x1, host_x1), {xud, {}}});
    func->execute();

    HostTensorND host_xnow;
    host_xnow.copy_from(*dev_x).sync();
    auto px = host_x->ptr<float>(), px1 = host_x1.ptr<float>(),
         pxnow = host_xnow.ptr<float>();
    for (int i = 0; i < 16; ++ i) {
        ASSERT_FLOAT_EQ(px[i] + 1, px1[i]);
        ASSERT_FLOAT_EQ(px[i] + 3, pxnow[i]);
    }

    ASSERT_NE(dev_ptr(x0), dev_ptr(x));
}

TEST(TestGraph, ForceUpdateExtendGraph) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto func = graph->compile({{x + 2, {}}});
    func->execute();

    auto dv = std::make_shared<DeviceTensorND>();
    dv->copy_from(*host_x);
    auto sv = opr::SharedDeviceTensor::make(*graph, dv);
    func = graph->compile({{opr::AddUpdate::make(sv, x), {}}});
    func->execute();
    ASSERT_EQ(host_x->ptr<float>()[0] * 2,
            HostTensorND{}.copy_from(*dv).sync().ptr<float>()[0]);
}

TEST(TestGraph, ForceUpdateWithMemFwd) {
    HostTensorGenerator<> gen;
    auto init_val = gen({2, 3});
    auto check = [&](bool should_fwd) {
        auto graph = ComputingGraph::make();
        auto xv = std::make_shared<DeviceTensorND>();
        xv->copy_from(*init_val);
        auto dev_x = opr::SharedDeviceTensor::make(*graph, xv),
             xrshp = dev_x.reshape({2, 3, 1}),
             tmp = xrshp.reshape({2, 3}),
             dev_y = xrshp.reshape({3, 2}),
             dev_xu = opr::AddUpdate::make(dev_x, dev_x.make_scalar_dt(1));
        set_priority(xrshp, -100);
        set_priority(dev_y, -100);
        set_priority(tmp, should_fwd ? -10 : 10);
        HostTensorND host_y;
        auto on_dev_y = [&](DeviceTensorND &val) {
            if (should_fwd) {
                EXPECT_EQ(xv->raw_ptr(), val.raw_ptr());
            } else {
                EXPECT_NE(xv->raw_ptr(), val.raw_ptr());
            }
            host_y.copy_from(val).sync();
        };
        auto func = graph->compile({{dev_xu, {}}, {tmp, {}},
                {dev_y, on_dev_y}});
        func->execute();

        auto get = host_y.sub(SubTensorSpec::make_from_layout(
                    init_val->layout()));
        MGB_ASSERT_TENSOR_EQ(get, *init_val);
        auto px = init_val->ptr<float>();
        auto py = get.copy_from(*xv).sync().ptr<float>();
        for (size_t i = 0; i < 6; ++ i) {
            MGB_ASSERT_FLOAT_EQ(px[i] + 1, py[i]);
        }
    };

    check(false);
    check(true);
}

TEST(TestGraph, ForceUpdateWithMemFwdArg) {
    HostTensorGenerator<> gen;
    constexpr size_t SIZE = 12345;
    auto init_val = gen({SIZE});
    auto graph = ComputingGraph::make();
    auto xv = std::make_shared<DeviceTensorND>();
    xv->copy_from(*init_val);
    auto dev_x = opr::SharedDeviceTensor::make(*graph, xv),
         delta = opr::Subtensor::make(dev_x,
                 {opr::Subtensor::AxisIndexer::make_interval(
                         0, None, None, dev_x.make_scalar(-1))}),
         dev_xu = opr::AddUpdate::make(dev_x, delta);
    auto func = graph->compile({{dev_xu, {}}});
    func->execute();
    ASSERT_NE(reinterpret_cast<const float*>(dev_ptr(delta)),
            xv->ptr<float>() + SIZE - 1);

    auto px = init_val->ptr<float>();
    HostTensorND yval;
    auto py = yval.copy_from(*xv).sync().ptr<float>();
    for (size_t i = 0; i < SIZE; ++ i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + px[SIZE - 1 - i], py[i]);
    }
}

TEST(TestGraph, ForceUpdateWithMemFwdOtherCN) {
    HostTensorGenerator<> gen;
    auto init_val = gen({2, 3});
    auto cn0 = init_val->comp_node(),
         cn1 = cn0.change_stream(1);
    auto check = [&](bool should_fwd) {
        auto graph = ComputingGraph::make();
        auto xv = std::make_shared<DeviceTensorND>();
        xv->copy_from(*init_val);
        auto dev_x = opr::SharedDeviceTensor::make(*graph, xv).rename("x"),
             dev_y = dev_x.reshape({2, 3, 1}).rename("y");
        SymbolVar xsum;
        if (should_fwd) {
            // dev_xu waits on dev_y.sum(), so it is safe to forward dev_y
            xsum = opr::reduce_sum(
                    // + 0 to avoid y dynamic alloc
                    dev_y + opr::MarkDynamicVar::make(dev_y.make_scalar(0)),
                    dev_y.make_scalar(1)).rename("xsum");
        } else {
            auto h2d = opr::Host2DeviceCopy::make(*graph, init_val);
            xsum = opr::reduce_sum(h2d, dev_y.make_scalar(1));
            set_priority(h2d, -100);
            set_priority(xsum, -100);
        }
        auto dev_xu = opr::AddUpdate::make(dev_x, xsum, {}, cn1);
        HostTensorND host_y;
        auto on_dev_y = [&](DeviceTensorND &val) {
            if (should_fwd) {
                EXPECT_EQ(xv->raw_ptr(), val.raw_ptr());
            } else {
                EXPECT_NE(xv->raw_ptr(), val.raw_ptr());
            }
            host_y.copy_from(val).sync();
        };
        auto func = graph->compile({{dev_y, on_dev_y}, {dev_xu, {}}});
        func->execute();
        auto get = host_y.sub(SubTensorSpec::make_from_layout(
                    init_val->layout()));
        MGB_ASSERT_TENSOR_EQ(get, *init_val);
        auto px = init_val->ptr<float>();
        // need to wait because AddUpdate is performed on another comp node
        func->wait();
        auto py = get.copy_from(*xv).sync().ptr<float>();
        float xsumv = 0;
        for (size_t i = 0; i < 6; ++ i) {
            xsumv += px[i];
        }
        for (size_t i = 0; i < 6; ++ i) {
            MGB_ASSERT_FLOAT_EQ(px[i] + xsumv, py[i]) <<
                "should_fwd=" << should_fwd << "\n" <<
                "i=" << i << "\n" <<
                "px[i]=" << px[i] << "\n" <<
                "xsumv=" << xsumv;
        }
    };
    check(false);
    check(true);
}

TEST(TestGraph, ForceUpdateWithMemFwdOtherCNOrderAnalyze) {
    HostTensorGenerator<> gen;
    constexpr size_t SIZE = 12345;
    auto init_val = gen({SIZE});
    auto cn1 = init_val->comp_node().change_stream(1);
    auto check = [&](bool should_fwd) {
        auto graph = ComputingGraph::make();
        auto xv = std::make_shared<DeviceTensorND>();
        xv->copy_from(*init_val);
        auto x = opr::SharedDeviceTensor::make(*graph, xv),
             xu = opr::AddUpdate::make(x, x.make_scalar(.5f), {}, cn1),
             xrshp = x.reshape({SIZE, 1}),
             y = xrshp + 1.4f,
             z = x + 2.3f; // xu waits for xrshp and z
        set_priority(z, should_fwd ? 10 : -10);
        HostTensorND host_y, host_z;
        auto func = graph->compile({make_callback_copy(y, host_y),
                make_callback_copy(z, host_z), {xu, {}}});
        func->execute().wait();

        cg::OperatorNodeBase::InputWaitingSpecElem ws;
        unpack_vector(xu.node()->owner_opr()->input_waiting_spec(), ws);
        VarNode* waited;
        unpack_vector(ws.dev_ready, waited);
        if (should_fwd) {
            EXPECT_EQ(dev_ptr(x), dev_ptr(xrshp));
            EXPECT_EQ(waited, z.node());
        } else {
            EXPECT_NE(dev_ptr(x), dev_ptr(xrshp));
            EXPECT_EQ(waited, xrshp.node());
        }

        HostTensorND cur_x;
        auto px = init_val->ptr<float>(),
             py = host_y.ptr<float>(), pz = host_z.ptr<float>(),
             pcx = cur_x.copy_from(*xv).sync().ptr<float>();
        for (size_t i = 0; i < SIZE; ++ i) {
            MGB_ASSERT_FLOAT_EQ(px[i] + 1.4f, py[i]);
            MGB_ASSERT_FLOAT_EQ(px[i] + 2.3f, pz[i]);
            MGB_ASSERT_FLOAT_EQ(px[i] + 0.5f, pcx[i]);
        }
    };
    check(false);
    check(true);
}

TEST(TestGraph, UnusedAsyncAddUpdateReader) {
    auto cn1 = CompNode::load("xpu0:1");
    HostTensorGenerator<> gen;
    auto host_dev_v = gen({3, 2});
    auto dev_v = std::make_shared<DeviceTensorND>();
    dev_v->copy_from(*host_dev_v);
    auto graph = ComputingGraph::make();
    auto tgt = opr::SharedDeviceTensor::make(*graph, dev_v),
         x_shp = opr::GetVarShape::make(tgt, {}, {cn1}),
         delta = opr::Host2DeviceCopy::make(*graph, host_dev_v),
         y = opr::AddUpdate::make(tgt, delta),
         z = opr::Reshape::make(y, x_shp, -1, {cn1});
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
