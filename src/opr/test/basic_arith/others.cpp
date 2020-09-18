/**
 * \file src/opr/test/basic_arith/others.cpp
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

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/utils/timer.h"

#include "megdnn/tensor_iter.h"

#include <cmath>

using namespace mgb;

TEST(TestOprBasicArith, AddUpdate) {
    constexpr size_t SIZE = 123456;
    opr::AddUpdate::Param param{2, -1, 0.5f};
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({SIZE});
    auto dev_x = std::make_shared<DeviceTensorND>(CompNode::load("xpu0"));
    dev_x->copy_from(*host_x);

    auto graph = ComputingGraph::make();
    SymbolVar dev_x_shared = opr::SharedDeviceTensor::make(
                    *graph, dev_x, {"x"}),
              dev_y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"}),
              dev_x_updated = opr::AddUpdate::make(dev_x_shared, dev_y, param);
    auto func = graph->compile({{
            dev_x_updated, [&](DeviceTensorND &){}}});
    func->execute();
    ASSERT_EQ(dev_x->raw_ptr(), dev_x_updated.node()->prev_dev_ptr());

    func->to_json()->writeto_fpath(output_file("add_update_graph.json"));

    HostTensorND get{CompNode::load("xpu0")};
    get.copy_from(*dev_x).sync();
    ASSERT_TRUE(get.layout().eq_layout(host_x->layout()));

    auto x = host_x->ptr<float>(), y = host_y->ptr<float>(),
         z = get.ptr<float>();
    for (size_t i = 0; i < SIZE; i ++) {
        auto expect = x[i] * param.alpha->get_cast<float>() +
            y[i] * param.beta->get_cast<float>() +
            param.bias->get_cast<float>();
        MGB_ASSERT_FLOAT_EQ(expect, z[i]);
    }
}

TEST(TestOprBasicArith, AddUpdateInt) {
    constexpr size_t SIZE = 123;
    opr::AddUpdate::Param param{2, -1, 3};
    HostTensorGenerator<dtype::Int32> gen;
    auto host_x = gen({SIZE}), host_y = gen({SIZE});
    auto dev_x = std::make_shared<DeviceTensorND>(CompNode::load("xpu0"));
    dev_x->copy_from(*host_x);

    auto graph = ComputingGraph::make();
    SymbolVar dev_x_shared = opr::SharedDeviceTensor::make(
                    *graph, dev_x, {"x"}),
              dev_y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"}),
              dev_x_updated = opr::AddUpdate::make(dev_x_shared, dev_y, param);
    auto func = graph->compile({{
            dev_x_updated, [&](DeviceTensorND &){}}});
    func->execute();
    ASSERT_EQ(dev_x->raw_ptr(), dev_x_updated.node()->prev_dev_ptr());

    HostTensorND get{CompNode::load("xpu0")};
    get.copy_from(*dev_x).sync();
    ASSERT_TRUE(get.layout().eq_layout(host_x->layout()));

    auto x = host_x->ptr<int>(), y = host_y->ptr<int>(),
         z = get.ptr<int>();
    for (size_t i = 0; i < SIZE; i ++) {
        auto expect = x[i] * param.alpha->get_cast<int>() +
            y[i] * param.beta->get_cast<int>() +
            param.bias->get_cast<int>();
        ASSERT_EQ(expect, z[i]) << ssprintf("i=%zu x=%d y=%d", i, x[i], y[i]);
    }

    ASSERT_NO_THROW(func->execute());
    param.bias->set(2.3f);
    ASSERT_THROW(func->execute(), MegDNNError);
}

TEST(TestOprBasicArith, DynAddUpdate) {
    constexpr size_t SIZE = 10;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({SIZE});
    auto dev_x = std::make_shared<DeviceTensorND>(CompNode::load("xpu0"));
    dev_x->copy_from(*host_x);

    auto graph = ComputingGraph::make();
    auto x = opr::SharedDeviceTensor::make(*graph, dev_x, {"x"}),
         y = opr::MarkDynamicVar::make(opr::Host2DeviceCopy::make(*graph,
                     host_y, {"y"})),
         x_updated = opr::AddUpdate::make(x, y, {});
    ASSERT_FALSE(cg::is_static_var_shape(y.node()));
    ASSERT_TRUE(cg::is_static_var_shape(x_updated.node()));
    auto func = graph->compile({{x_updated, [&](DeviceTensorND &){}}});
    func->execute();

    HostTensorND host_xu;
    host_xu.copy_from(*dev_x).sync();
    ASSERT_TRUE(host_xu.layout().eq_layout(host_x->layout()));

    {
        auto x = host_x->ptr<float>(), y = host_y->ptr<float>(),
             z = host_xu.ptr<float>();
        for (size_t i = 0; i < SIZE; i ++) {
            MGB_ASSERT_FLOAT_EQ(x[i] + y[i], z[i]);
        }
    }
}

TEST(TestOprBasicArith, AddUpdateBroadcast) {
    constexpr size_t SIZE = 123456;
    opr::AddUpdate::Param param{-1.2f, 2.1f, -4};
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE});
    auto dev_x = std::make_shared<DeviceTensorND>(CompNode::load("xpu0"));
    dev_x->copy_from(*host_x);

    auto graph = ComputingGraph::make();
    SymbolVar x = opr::SharedDeviceTensor::make(*graph, dev_x, {"x"}),
              delta = opr::Subtensor::make(x,
                      {opr::Subtensor::AxisIndexer::make_index(0,
                              x.make_scalar(3))}),
              x_updated = opr::AddUpdate::make(x, delta, param);
    auto func = graph->compile({{x_updated, {}}});
    func->execute();

    HostTensorND get{CompNode::load("xpu0")};
    get.copy_from(*dev_x).sync();
    ASSERT_TRUE(get.layout().eq_layout(host_x->layout()));

    auto xp = host_x->ptr<float>(), z = get.ptr<float>();
    for (size_t i = 0; i < SIZE; ++ i) {
        auto expect = xp[i] * param.alpha->get_cast<float>() +
            xp[3] * param.beta->get_cast<float>() +
            param.bias->get_cast<float>();
        MGB_ASSERT_FLOAT_EQ(expect, z[i]);
    }
}

TEST(TestOprBasicArith, AddUpdateNan) {
    constexpr size_t SIZE = 23;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}),
         host_src = gen({1});

    host_x->ptr<float>()[0] = NAN;
    auto dev_x = std::make_shared<DeviceTensorND>(CompNode::load("xpu0"));
    dev_x->copy_from(*host_x);

    auto graph = ComputingGraph::make();
    SymbolVar x = opr::SharedDeviceTensor::make(*graph, dev_x, {"x"}),
              dest = opr::Host2DeviceCopy::make(*graph, host_src),
              xu = opr::AddUpdate::make(x, dest, {0.f, 1});
    auto func = graph->compile({{xu, {}}});
    func->execute();

    HostTensorND host_y;
    host_y.copy_from(*dev_x).sync();
    for (size_t i = 0; i < SIZE; ++ i)
        MGB_ASSERT_FLOAT_EQ(host_src->ptr<float>()[0], host_y.ptr<float>()[i]);
}

TEST(TestOprBasicArith, AddInplace) {
    constexpr size_t SIZE = 102400;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({SIZE}), host_opr1 = gen({SIZE}),
         host_opr2 = gen({SIZE});

    // for operations with commutable input, must check both input order:
    // opr1 + opr0, opr1 + opr2
    auto graph = ComputingGraph::make();
    auto opr0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_opr0, {"opr0"}),
         opr1 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_opr1, {"opr1"}),
         opr2 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_opr2, {"opr2"}),
         sum0 = opr::add(opr1, opr0).rename("sum0"),
         sum1 = opr::add(opr1, opr2).rename("sum1"),
         sum2 = opr::add(opr2, opr0).rename("sum2");

    // check dedup
    ASSERT_EQ(sum0.node(), (opr0 + opr1).node());

    HostTensorND host_sum0, host_sum1;
    auto func = graph->compile({make_callback_copy(sum0, host_sum0),
            make_callback_copy(sum1, host_sum1)});

    func->execute();

    EXPECT_TRUE(dev_ptr(sum0) == dev_ptr(opr1) ||
            dev_ptr(sum0) == dev_ptr(opr0));
    EXPECT_TRUE(dev_ptr(sum1) == dev_ptr(opr1) ||
            dev_ptr(sum1) == dev_ptr(opr2));
    func->to_json()->writeto_fpath(output_file("TestAddInplaceFunc0.json"));

    ASSERT_TRUE(host_sum0.layout().eq_layout(host_opr0->layout()));
    ASSERT_TRUE(host_sum1.layout().eq_layout(host_opr0->layout()));

    auto o0 = host_opr0->ptr<float>(), o1 = host_opr1->ptr<float>(),
         o2 = host_opr2->ptr<float>(),
         s0 = host_sum0.sync().ptr<float>(), s1 = host_sum1.sync().ptr<float>();
    for (size_t i = 0; i < SIZE; i ++) {
        MGB_ASSERT_FLOAT_EQ(o1[i] + o0[i], s0[i]) <<
            ssprintf("failed opr1(%.5f)+opr0(%.5f) at %zd", o1[i], o0[i], i);
        MGB_ASSERT_FLOAT_EQ(o1[i] + o2[i], s1[i]) <<
            ssprintf("failed opr1(%.5f)+opr2(%.5f) at %zd", o1[i], o2[i], i);
    }

    *host_opr0 = *gen({SIZE});
    *host_opr1 = *gen({SIZE});
    *host_opr2 = *gen({SIZE});
    HostTensorND host_sum2;
    func = graph->compile({make_callback_copy(sum0, host_sum0),
            make_callback_copy(sum1, host_sum1),
            make_callback_copy(sum2, host_sum2)});
    func->execute();
    func->to_json()->writeto_fpath(output_file("TestAddInplaceFunc1.json"));
    ASSERT_TRUE(host_sum0.layout().eq_layout(host_opr0->layout()));
    ASSERT_TRUE(host_sum1.layout().eq_layout(host_opr0->layout()));
    ASSERT_TRUE(host_sum2.layout().eq_layout(host_opr0->layout()));

    o0 = host_opr0->ptr<float>(); o1 = host_opr1->ptr<float>();
    o2 = host_opr2->ptr<float>();
    s0 = host_sum0.ptr<float>(); s1 = host_sum1.ptr<float>();
    auto s2 = host_sum2.sync().ptr<float>();
    for (size_t i = 0; i < SIZE; i ++) {
        MGB_ASSERT_FLOAT_EQ(o1[i] + o0[i], s0[i]) <<
            ssprintf("failed opr1(%.5f)+opr0(%.5f) at %zd", o1[i], o0[i], i);
        MGB_ASSERT_FLOAT_EQ(o1[i] + o2[i], s1[i]) <<
            ssprintf("failed opr1(%.5f)+opr2(%.5f) at %zd", o1[i], o2[i], i);
        MGB_ASSERT_FLOAT_EQ(o2[i] + o0[i], s2[i]) <<
            ssprintf("failed opr2(%.5f)+opr0(%.5f) at %zd", o2[i], o0[i], i);
    }
}

TEST(TestOprBasicArith, AddUpdateOtherStream) {
    REQUIRE_GPU(1);
    constexpr size_t SIZE = 60;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    std::atomic_bool flag{false};
    auto set_flag = [&flag](DeviceTensorND&) {
        flag = true;
    };

    auto wait_flag = [&flag](DeviceTensorND&) {
        while (!flag) {
            using namespace std::literals;
            std::this_thread::sleep_for(0.2s);
        }
    };

    std::shared_ptr<HostTensorND> host_val = gen({SIZE});
    auto cn1 = CompNode::load("gpu0:0").change_stream(1);
    auto param = opr::SharedDeviceTensor::make(*graph, *host_val);
    param.node()->owner_opr()->node_prop().attribute().priority =
            std::numeric_limits<int>::max();
    auto copy = opr::Copy::make(param, cn1);
    auto add = (copy + 3) * 5;
    auto add_update = opr::AddUpdate::make(param, add, {}, {cn1});

    auto callback = opr::CallbackInjector::make(add_update, set_flag);

    auto waiter = opr::CallbackInjector::make(
            opr::SharedDeviceTensor::make(*graph, *host_val),
            wait_flag);

    HostTensorND host_out0;
    HostTensorND host_out1;
    auto func = graph->compile({make_callback_copy(callback, host_out0),
            make_callback_copy(waiter, host_out1)});
    func->execute();
}

TEST(TestOprBasicArith, DisableAddUpdate) {
    constexpr size_t SIZE = 10;
    opr::AddUpdate::Param param{2, -1, 0.5f, 1};
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({SIZE});
    auto dev_x = std::make_shared<DeviceTensorND>(CompNode::load("xpu0"));
    dev_x->copy_from(*host_x);

    auto graph = ComputingGraph::make();
    SymbolVar dev_x_shared = opr::SharedDeviceTensor::make(
                    *graph, dev_x, {"x"}),
              dev_y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"}),
              dev_x_updated = opr::AddUpdate::make(dev_x_shared, dev_y, param);
    auto func = graph->compile({{
            dev_x_updated, [&](DeviceTensorND &){}}});
    func->execute();
    ASSERT_EQ(dev_x->raw_ptr(), dev_x_updated.node()->prev_dev_ptr());

    func->to_json()->writeto_fpath(output_file("add_update_graph.json"));

    HostTensorND get{CompNode::load("xpu0")};
    get.copy_from(*dev_x).sync();
    ASSERT_TRUE(get.layout().eq_layout(host_x->layout()));

    auto x = host_x->ptr<float>(), y = get.ptr<float>();
    for (size_t i = 0; i < SIZE; i ++) {
        MGB_ASSERT_FLOAT_EQ(x[i], y[i]);
    }
}

TEST(TestOprBasicArith, AddUpdateVolatile) {
    constexpr int SIZE = 12222;
    opr::AddUpdate::Param param{2, -1, 0.5f};
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("xpu0");

    for (auto dynamic_alloc : {false, true}) {
        // test on both static and dynamic allocation
        auto host_x = gen({SIZE << 1}), host_y = gen({SIZE << 1});
        auto dev_x = std::make_shared<DeviceTensorND>(cn);
        DeviceTensorND dev_x0, dev_x1;
        HostTensorND host_sub;
        dev_x0.copy_from(*host_x).sync();
        dev_x1.copy_from(*host_x).sync();
        *dev_x = dev_x0;
        auto graph = ComputingGraph::make();
        graph->options().force_dynamic_alloc = dynamic_alloc;
        SymbolVar dev_x_shared = opr::VolatileSharedDeviceTensor::make(
                        *graph, dev_x, {"x"}),
                dev_y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"}),
                dev_x_updated = opr::AddUpdate::make(dev_x_shared, dev_y, param),
                // check read-only forward on force updated var
                dev_x_updated_sub = opr::Subtensor::make(dev_x_updated, {
                    opr::Subtensor::AxisIndexer::make_interval(-1, None, None,
                       dev_x_shared.make_scalar(SIZE >> 1))});
        auto func = graph->compile({
                {dev_x_updated, [&](DeviceTensorND &){}},
                {make_callback_copy(dev_x_updated_sub, host_sub)}});
        auto run = [&] {
            HostTensorND origin_x{cn}, get{cn};
            origin_x.copy_from(*dev_x).sync();
            func->execute().wait();
            ASSERT_EQ(dev_x->raw_ptr(), dev_x_updated.node()->prev_dev_ptr());
            ASSERT_EQ(dev_x->raw_ptr(), dev_x_updated_sub.node()->prev_dev_ptr());
            get.copy_from(*dev_x).sync();
            ASSERT_TRUE(get.layout().eq_layout(origin_x.layout()));

            mgb_assert(origin_x.layout().is_contiguous() &&
                        get.layout().is_contiguous() &&
                        host_y->layout().is_contiguous());
            auto x = origin_x.ptr<float>(), y = host_y->ptr<float>(),
                z = get.ptr<float>();
            bool bcast = dev_x->shape().ndim > 1;
            auto expect = [&](size_t i) {
                return x[i] * param.alpha->get_cast<float>() +
                    (bcast ? y[i / SIZE] : y[i]) *
                    param.beta->get_cast<float>() +
                    param.bias->get_cast<float>();
            };
            for (size_t i = 0; i < SIZE * 2; i ++) {
                MGB_ASSERT_FLOAT_EQ(expect(i), z[i]);
            }
            mgb_assert(host_sub.shape().total_nr_elems() == 4 &&
                host_sub.layout().is_contiguous());
            for (size_t i = 0; i < 4; ++ i) {
                size_t idx = i * (SIZE >> 1);
                MGB_ASSERT_FLOAT_EQ(expect(idx), host_sub.ptr<float>()[i]);
            }
        };
        run();
        run();
        *dev_x = dev_x1; // ptr change
        run();
        host_x = gen({2, SIZE});
        host_y->copy_from(*gen({2, 1})).sync();
        dev_x->copy_from(*host_x).sync(); // shape change
        run();
    }
}

// AddUpdate in gradient path but no gradient flows through it
TEST(TestOprBasicArith, AddUpdateInGradPath) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto dest = opr::SharedDeviceTensor::make(*graph, *gen({42}));
    auto host_x = gen({42});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    // delta depends on x, but not differentiable wrt x
    // a invalid grad is registered for AddUpdate to fix this case
    auto delta = opr::VirtualDep::make({opr::SetGrad::make(x, nullptr), x});
    auto updated = opr::AddUpdate::make(dest, delta);
    auto y = opr::reduce_ax_sum(updated + x, 0);
    auto dx = cg::grad(y, x);
    HostTensorND host_dx;
    auto func = graph->compile({make_callback_copy(dx, host_dx)});
    func->execute();
    for (size_t i = 0; i < host_dx.shape(0); ++i) {
        MGB_ASSERT_FLOAT_EQ(host_dx.ptr<float>()[i], 1.f);
    }
}

TEST(TestOprBasicArith, MemFwd) {
    constexpr size_t SIZE = 12321;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x).rename("x"),
         y = opr::sin(x),
         z = y + 1;
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();

    ASSERT_EQ(dev_ptr(x), dev_ptr(y));
    ASSERT_EQ(dev_ptr(x), dev_ptr(z));
    for (size_t i = 0; i < SIZE; ++ i) {
        MGB_ASSERT_FLOAT_EQ(host_z.ptr<float>()[i],
                std::sin(host_x->ptr<float>()[i]) + 1.f);
    };
}

TEST(TestOprBasicArith, BinaryGradWithBroadcast) {
    using Checker = AutoOprChecker<3, 1>;
    auto make_graph = [](const Checker::SymInpArray &inputs) ->
            Checker::SymOutArray {
        return {inputs[0] + (opr::MarkDynamicVar::make(inputs[1]) + inputs[2])};
    };
    auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        host_add(dest[0], *inp[0], *inp[1]);
        host_add(dest[0], dest[0], *inp[2]);
    };
    Checker(make_graph, fwd).
        run({TensorShape{2, 3}, TensorShape{2, 3}, TensorShape{1}}).
        run({TensorShape{1, 5}, TensorShape{1, 1}, TensorShape{5, 1}}).
        run({TensorShape{2, 1, 1}, TensorShape{1, 3, 1}, TensorShape{1, 1, 4}}).
        run({TensorShape{1, 1, 1}, TensorShape{1, 3, 1}, TensorShape{2, 3, 4}});
}

TEST(TestOprBasicArith, BinaryBroadcastCorrectness) {
    using Checker = AutoOprChecker<2, 1>;

    auto run = [&](bool dyn_inp) {
        auto make_graph = [&](const Checker::SymInpArray &inputs) ->
                Checker::SymOutArray {
            auto x = inputs[0], y = inputs[1];
            if (dyn_inp) {
                x = opr::MarkDynamicVar::make(x);
                y = opr::MarkDynamicVar::make(y);
            }
            x.rename("x");
            y.rename("y");

            return {x * y};
        };

        auto fwd = [](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
            TensorShape oshp;
            megdnn::Elemwise::deduce_shape({inp[0]->shape(), inp[1]->shape()},
                    oshp);
            auto &&dv = dest[0].comp_node(inp[0]->comp_node()).resize(oshp);
            auto &&iv0 = inp[0]->sub(SubTensorSpec::make_from_layout(
                        inp[0]->layout().broadcast(oshp))),
                 &&iv1 = inp[1]->sub(SubTensorSpec::make_from_layout(
                             inp[1]->layout().broadcast(oshp)));

            auto it0 = megdnn::tensor_iter_valonly<float>(
                    iv0.as_megdnn()).begin(),
                 it1 = megdnn::tensor_iter_valonly<float>(
                    iv1.as_megdnn()).begin();
            for (size_t i = 0, it = oshp.total_nr_elems(); i < it; ++ i) {
                dv.ptr<float>()[i] = *it0 * *it1;
                ++ it0;
                ++ it1;
            }
        };

        Checker::RunOptions opt;
        opt.numdiff_eps = 1;
        Checker(make_graph, fwd).
            run({TensorShape{5, 3}, {5, 3}}, opt).
            run({TensorShape{2, 2, 1, 1}, {1, 2, 1, 1}}, opt).
            run({TensorShape{1, 2}, {2, 1}}, opt).
            run({TensorShape{3, 2, 5}, {1}}, opt).
            run({TensorShape{4, 5, 1, 1}, {4, 5, 6, 7}}, opt).
            run({TensorShape{8, 4, 1, 1}, {1, 4, 5, 1}}, opt);
    };

    run(false);
    run(true);
}

TEST(TestOprBasicArith, Optimize) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({23});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         x_sum2 = opr::reduce_sum(
                 opr::pow(x, x.make_scalar(2)), x.make_scalar(1));

    ASSERT_EQ(opr::Reduce::Mode::SUM_SQR,
            x_sum2.node()->owner_opr()->cast_final_safe<opr::Reduce>().
            param().mode);

    float sum2 = 0;
    auto xptr = host_x->ptr<float>();
    for (size_t i = 0, it = host_x->shape().total_nr_elems(); i < it;  ++ i) {
        sum2 += xptr[i] * xptr[i];
    }
    HostTensorND host_x_sum2;
    auto func = graph->compile({make_callback_copy(x_sum2, host_x_sum2)});
    func->execute();
    ASSERT_EQ(TensorShape{1},  host_x_sum2.shape());
    MGB_ASSERT_FLOAT_EQ(sum2, host_x_sum2.ptr<float>()[0]);
}

TEST(TestOprBasicArith, TypeCvt) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen{0, 1000};
    auto host_x = gen({23});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::TypeCvt::make(x, dtype::Int32{});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();

    auto px = host_x->ptr<float>();
    auto py = host_y.ptr<int>();
    for (size_t i = 0; i < 23; ++i) {
        ASSERT_EQ(static_cast<int>(px[i]), py[i]);
    }

    host_x->resize({3, 0});
    func->execute();
    ASSERT_EQ(TensorShape({3, 0}), host_y.shape());
}

TEST(TestOprBasicArith, TypeCvtBool) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Int32> gen;
    auto host_x = gen({3});
    auto px = host_x->ptr<int>();
    px[0] = -1;
    px[1] = 0;
    px[2] = 1;

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::TypeCvt::make(x, dtype::Bool{});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();

    auto py = host_y.ptr<bool>();
    for (size_t i = 0;i < 3;i ++) {
        ASSERT_EQ(static_cast<bool>(px[i]), py[i]);
    }
    ASSERT_EQ(TensorShape({3}), host_y.shape());
}

TEST(TestOprBasicArith, TypeCvtFromBool) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Bool> gen;
    auto host_x = gen({2});
    auto px = host_x->ptr<bool>();
    px[0] = true;
    px[1] = false;

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::TypeCvt::make(x, dtype::Int32{});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();

    auto py = host_y.ptr<int>();
    for (size_t i = 0;i < 2;i ++) {
        ASSERT_EQ(static_cast<int>(px[i]), py[i]);
    }
    ASSERT_EQ(TensorShape({2}), host_y.shape());
}

TEST(TestOprBasicArith, ElemwiseMemFwd) {
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 3}),
         host_y = gen({3, 3});

    // x[:, ::-1]
    auto rev = [](SymbolVar x) {
        return opr::Subtensor::make(x,
                                    {opr::Subtensor::AxisIndexer::make_interval(
                                            1, None, None, x.make_scalar(-1))});
    };
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         y = opr::Host2DeviceCopy::make_no_fwd(*graph, host_y),
         y0 = rev(y),
         y1 = rev(x),
         z0 = x + y0,
         z1 = x + y1,
         z2 = x + x;

    auto check = [&graph, &host_x, x](SymbolVar y, SymbolVar z, float* py,
                                      bool rev_y, bool should_fwd) {
        HostTensorND host_z;
        auto func = graph->compile({make_callback_copy(z, host_z)});
        func->execute();
        HostTensorND expect;
        expect.copy_from(*host_x);
        auto pe = expect.ptr<float>();
        for (size_t i = 0; i < 3; ++i) {
            auto cur_py = py + i * 3 + static_cast<int>(rev_y) * 2;
            for (size_t j = 0; j < 3; ++j) {
                pe[i * 3 + j] += *cur_py;
                cur_py += rev_y ? -1 : 1;
            }
        }
        MGB_ASSERT_TENSOR_EQ(expect, host_z);

        auto xptr = dev_ptr(x), yptr = dev_ptr(y), zptr = dev_ptr(z);
        if (should_fwd) {
            ASSERT_EQ(zptr, xptr);
        } else {
            ASSERT_NE(zptr, xptr);
            ASSERT_NE(zptr, yptr);
        }
    };

    check(y0, z0, host_y->ptr<float>(), true, true);
    ASSERT_EQ(dev_ptr(y) + 2 * sizeof(float), dev_ptr(y0));

    check(y1, z1, host_x->ptr<float>(), true, false);
    ASSERT_EQ(dev_ptr(x) + 2 * sizeof(float), dev_ptr(y1));

    check(x, z2, host_x->ptr<float>(), false, true);
}

TEST(TestOprBasicArith, ElemwiseRequireContig) {
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    HostTensorGenerator<> gen;
    auto host_x = gen({3, 3}), host_y = gen({1, 3});

    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         y = opr::Host2DeviceCopy::make_no_fwd(*graph, host_y),
         xt = opr::Dimshuffle::make(x, {1, 0}),
         yb = y.broadcast({3, 3}),
         z = xt + yb;

    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    HostTensorND expect{host_x->comp_node(), host_x->dtype()};
    expect.resize({3, 3});
    auto px = host_x->ptr<float>(), py = host_y->ptr<float>(),
         pe = expect.ptr<float>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            pe[i * 3 + j] = px[j * 3 + i] + py[j];
        }
    }
    MGB_ASSERT_TENSOR_EQ(expect, host_z);

    ASSERT_NE(dev_ptr(x), dev_ptr(xt));
    ASSERT_EQ(dev_ptr(y), dev_ptr(yb));
    ASSERT_EQ(dev_ptr(xt), dev_ptr(z));
}

TEST(TestOprBasicArith, TypeCvtDedup) {
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 5, 5, 5});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    dtype::Quantized8Asymm dtype1(0.01f, (uint8_t) 123);
    dtype::Quantized8Asymm dtype2(0.02f, (uint8_t) 234);
    auto cvt1 = opr::TypeCvt::make(x, dtype1);
    auto cvt2 = opr::TypeCvt::make(x, dtype2);
    ASSERT_NE(cvt1.node(), cvt2.node());

    dtype::Quantized8Asymm dtype3(0.01f, (uint8_t) 123);
    auto cvt3 = opr::TypeCvt::make(x, dtype3);
    ASSERT_EQ(cvt1.node(), cvt3.node());
}

TEST(TestOprBasicArith, PowC) {
    using Checker = AutoOprChecker<1, 1>;
    SymbolVar inp, sub;
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        // test non-contig
        inp = inputs[0];
        sub = opr::Subtensor::make(
                inp, {opr::Subtensor::AxisIndexer::make_interval(
                             1, None, inputs[0].make_scalar(-2), None)});
        return {opr::PowC::make(sub, 2.f)};
    };
    auto fwd = [](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        TensorShape oshp = inp[0]->shape();
        oshp[1] -= 2;
        size_t size_x = oshp[0],
               strd_x = inp[0]->shape().total_nr_elems() / size_x,
               size_y = oshp.total_nr_elems() / size_x;

        auto px = inp[0]->ptr<float>(), py = dest[0].resize(oshp).ptr<float>();
        for (size_t i = 0; i < size_x; ++i) {
            for (size_t j = 0; j < size_y; ++j) {
                float xv = px[i * strd_x + j], yv = xv * xv;
                py[i * size_y + j] = yv;
            }
        }
    };
    Checker checker{make_graph, fwd};
    checker.run({TensorShape{2, 3}})
            .run({TensorShape{12, 33}})
            .run({TensorShape{5, 33, 7}});

    ASSERT_EQ(prev_dev_ptr(inp), prev_dev_ptr(sub));
}

TEST(TestOprBasicArith, PowCInfer) {
    HostTensorGenerator<> gen;
    auto run = [&](bool contig) {
        auto host_x = gen({3, contig ? 4u : 5u});
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
             xsub = opr::Subtensor::make(
                     x, {opr::Subtensor::AxisIndexer::make_interval(
                                1, None, x.make_scalar(4), None)}),
             y = opr::PowC::make(xsub, 4.f);
        auto y_infer = graph->static_infer_manager().infer_value(y.node());
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_y, HostTensorND::make_proxy(y_infer));

        ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(xsub));
        if (contig) {
            // inplace computing
            ASSERT_EQ(prev_dev_ptr(xsub), prev_dev_ptr(y));
        } else {
            ASSERT_NE(prev_dev_ptr(xsub), prev_dev_ptr(y));
        }
    };
    run(false);
    run(true);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
