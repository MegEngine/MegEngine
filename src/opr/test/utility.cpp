/**
 * \file src/opr/test/utility.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/utility.h"
#include "megbrain/gopt/framework.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/test/helper.h"

using namespace mgb;

TEST(TestOprUtility, AssertEqual) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 2, 3}),
         host_y = gen({1, 2, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::AssertEqual::make(x, y);
    auto func = graph->compile({{z, {}}});

    ASSERT_THROW(func->execute().wait(), MegBrainError);
    host_y->copy_from(*host_x);
    ASSERT_NO_THROW(func->execute().wait());
}

TEST(TestOprUtility, VirtualGradCompNode) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({1}, cns[0]);
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         loss = opr::Copy::make(x, cns[1]),
         gx = opr::VirtualGrad::make(loss, x);

    HostTensorND gx_host;
    auto func = graph->compile({make_callback_copy(gx, gx_host)});
    func->execute();
    ASSERT_EQ(1.f, gx_host.ptr<float>()[0]);
    ASSERT_EQ(host_x->comp_node(), gx_host.comp_node());
    ASSERT_EQ(host_x->comp_node(), gx.node()->comp_node());
}

TEST(TestOprUtility, VirtualLoss) {
    auto cns = load_multiple_xpus(2);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2}), cns[0]),
         x1 = x + x * 2 - x,  // test shallow copy
            x2 = (opr::Copy::make(x, cns[1]) + 1) * 3,
         x1g = opr::Host2DeviceCopy::make(*graph, gen({2}), cns[0]),
         x2g = opr::Host2DeviceCopy::make(*graph, gen({2}), cns[1]),
         loss = opr::VirtualLoss::make({x1, x2}, {x1g, x2g}),
         gx = opr::VirtualGrad::make(loss, x),
         expect = 2 * x1g + 3 * opr::Copy::make(x2g, cns[0]);
    HostTensorND host_gx, host_expect;
    auto func = graph->compile({make_callback_copy(gx, host_gx),
                                make_callback_copy(expect, host_expect)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_expect, host_gx);
}

TEST(TestOprUtility, Timestamp) {
    auto cns = load_multiple_xpus(2);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto times = gen({5}, cns[0]);
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2}), cns[0]),
         x0 = opr::Timestamp::make(x, times, 0),
         y0 = opr::Timestamp::make(opr::Sleep::make(x0, 0.1), times, 1),
         z0 = opr::Timestamp::make(opr::Sleep::make(y0, 0.1), times, 2),

         x_cn1 = opr::Copy::make(x, cns[1]),
         x1 = opr::Timestamp::make(x_cn1, times, 3),
         y1 = opr::Timestamp::make(opr::Sleep::make(x1, 0.2), times, 4);

    graph->options().var_sanity_check_first_run = false;

    // check normal case with multiple comp nodes
    auto func = graph->compile({{z0, {}}, {y1, {}}});
    auto p = times->ptr<float>();
    memset(p, -1, sizeof(float) * 5);
    func->execute().wait();
    auto check_near = [&](float a, float b) {
        // sleep kernel in cuda is easily affected by the frequency change of
        // GPU, so we just print warn log instead assert. more refer to
        // XPU-226
        if ((a - b) >= 0.05) {
            mgb_log_warn("expect time [a - b < 0.05], got %f", a - b);
        }
    };

    double factor = 0.1 / (p[1] - p[0]);
    check_near(0.2, factor * (p[2] - p[0]));
    check_near(0.2, factor * (p[4] - p[3]));

    // one node is not used in compile
    p[2] = -1;
    func = graph->compile({{y0, {}}, {y1, {}}});
    func->execute().wait();
    check_near(0.2, factor * (p[4] - p[3]));
    ASSERT_EQ(-1.f, p[2]);
}

TEST(TestOprUtility, VirtualDep) {
    auto cns = load_multiple_xpus(2);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;

    auto host_x0 = gen({32, 3, 56, 56}, cns[0]);
    auto host_x1 = gen({32, 3, 56, 56}, cns[1]);

    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0);
    auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1);
    auto y0 = x0 + 3;
    auto y1 = x0 * 5;
    auto y2 = x1 * x1;

    bool called = false;
    auto cb = [&called](DeviceTensorND&) {
        called = true;
    };

    auto virtual_dep = opr::VirtualDep::make({y0, y1,
            opr::CallbackInjector::make(y2, cb)});


    HostTensorND host_y0, host_virtual_dep;

    auto func = graph->compile(
            {make_callback_copy(y0, host_y0),
            make_callback_copy(virtual_dep, host_virtual_dep)});

    func->execute();

    ASSERT_TRUE(called);

    MGB_ASSERT_TENSOR_EQ(host_y0, host_virtual_dep);
}

TEST(TestOprUtility, VirtualDepSideEffect) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    for (auto cn : load_multiple_xpus(2)) {
        bool called = false;
        auto cb = [&called](DeviceTensorND&) { called = true; };

        auto x = opr::ImmutableTensor::make(*graph, *gen({1}, cn));
        auto y = opr::VirtualDep::make({x, opr::CallbackInjector::make(x, cb)});
        ASSERT_TRUE(cg::is_static_var_value(y.node()));

        auto func = graph->compile({{y, {}}});
        func->execute();
        ASSERT_TRUE(called);
    }
}

TEST(TestOprUtility, CallbackInjector) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    int nr_cb_shp = 0, nr_cb_val = 0;
    auto host_x = gen({1, 2, 3});

    auto cb_shp = [&nr_cb_shp, &host_x](const DeviceTensorND& dv) {
        ASSERT_EQ(1u, dv.shape().ndim);
        ASSERT_EQ(CompNode::default_cpu(), dv.comp_node());
        TensorShape got;
        got.ndim = dv.shape()[0];
        std::copy(dv.ptr<int>(), dv.ptr<int>() + got.ndim, got.shape);
        ASSERT_EQ(host_x->shape(), got);
        ++nr_cb_shp;
    };

    auto cb_val = [&nr_cb_val, &host_x](const DeviceTensorND& dv) {
        ASSERT_EQ(host_x->comp_node(), dv.comp_node());
        HostTensorND hv;
        hv.copy_from(dv).sync();
        MGB_ASSERT_TENSOR_EQ(*host_x, hv);
        ++nr_cb_val;
    };

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         yshp = x.reshape(opr::CallbackInjector::make(x.symshape(), cb_shp)),
         yval = opr::CallbackInjector::make(x, cb_val) + 1;
    auto func = graph->compile({{yshp, {}}, {yval, {}}});

    func->execute();
    *host_x = *gen({4, 2});
    func->execute();
    host_x->copy_from(*gen({2, 3}));
    func->execute();
    func->execute();

    ASSERT_EQ(3, nr_cb_shp);
    ASSERT_EQ(4, nr_cb_val);
}


TEST(TestOprUtility, MultiInputCallbackInjector) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    int nr_cb_val = 0;
    auto host_x = gen({1, 2, 3});
    auto cb_val = [&nr_cb_val, &host_x](const SmallVector<DeviceTensorND>& dv) {
        ASSERT_EQ(size_t(3), dv.size());
        ASSERT_EQ(host_x->comp_node(), dv[0].comp_node());
        HostTensorND hv;
        hv.copy_from(dv[0]).sync();
        MGB_ASSERT_TENSOR_EQ(*host_x, hv);
        ++nr_cb_val;
    };

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         yval = opr::CallbackInjector::make({x, x, x}, cb_val) + 1;
    auto func = graph->compile({{yval, {}}});

    func->execute();
    *host_x = *gen({4, 2});
    func->execute();
    host_x->copy_from(*gen({2, 3}));
    func->execute();
    func->execute();

    ASSERT_EQ(4, nr_cb_val);

    auto a = gen({1, 2});
    auto b = gen({2, 3});
    auto c = gen({1, 3});
    auto cb_mul = [&a, &b, &c](const SmallVector<DeviceTensorND>& dv) {
        ASSERT_EQ(a->comp_node(), dv[0].comp_node());
        ASSERT_EQ(b->comp_node(), dv[1].comp_node());
        ASSERT_EQ(c->comp_node(), dv[2].comp_node());
    };
    auto test_v = opr::CallbackInjector::make({
        opr::Host2DeviceCopy::make(*graph, a),
        opr::Host2DeviceCopy::make(*graph, b),
        opr::Host2DeviceCopy::make(*graph, c),
        }, cb_mul);
    auto test_func = graph->compile({{test_v, {}}, {test_v, {}}});
    test_func->execute();

    for (auto ignore : {false, true}) {
        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen;
        int nr_cb = 0;
        auto host_x = gen({1, 2, 3});
        auto host_z = gen({2, 1, 3});

        auto cb_val = [&nr_cb](const DeviceTensorND&) { ++nr_cb; };

        auto y = opr::CallbackInjector::make({
            opr::ImmutableTensor::make(*graph, *host_x),
            opr::ImmutableTensor::make(*graph, *host_z)
            }, {true, ignore, cb_val}) + 1;
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});

        auto chk = [ignore, &nr_cb](int nu) {
            if (ignore) {
                ASSERT_EQ(1, nr_cb);
            } else {
                ASSERT_EQ(nu, nr_cb);
            }
        };

        chk(0);
        func->execute();
        chk(1);
        func->execute();
        chk(2);

        HostTensorND y_expect;
        y_expect.copy_from(*host_x);
        for (size_t i = 0; i < 6; ++i) {
            ++y_expect.ptr<float>()[i];
        }
        MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
    }
}

TEST(TestOprUtility, CallbackInjectorSideEffect) {
    for (auto ignore : {false, true}) {
        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen;
        int nr_cb = 0;
        auto host_x = gen({1, 2, 3});

        auto cb_val = [&nr_cb](const DeviceTensorND&) { ++nr_cb; };

        auto x = opr::ImmutableTensor::make(*graph, *host_x),
             y = opr::CallbackInjector::make(x, {true, ignore, cb_val}) + 1;
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});

        auto chk = [ignore, &nr_cb](int nu) {
            if (ignore) {
                ASSERT_EQ(1, nr_cb);
            } else {
                ASSERT_EQ(nu, nr_cb);
            }
        };

        chk(0);
        func->execute();
        chk(1);
        func->execute();
        chk(2);
        func->execute();
        chk(3);

        HostTensorND y_expect;
        y_expect.copy_from(*host_x);
        for (size_t i = 0; i < 6; ++i) {
            ++y_expect.ptr<float>()[i];
        }
        MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
    }
}

TEST(TestOprUtility, PersistentOutputStorageShapes) {
    HostTensorGenerator<> gen;
    auto host_x = gen({123});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         xp = opr::PersistentOutputStorage::make(x), y0 = xp + 1, y1 = x + 1;
    HostTensorND host_y0, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y0, host_y1);
    auto ptr0 = prev_dev_ptr(xp), ptr1 = prev_dev_ptr(y1);

    // allocate some storage to avoid y1 reuses previous ptr
    func->clear_device_memory();
    DeviceTensorStorage mem_occupy{host_x->comp_node()};
    mem_occupy.ensure_size(123 * 4).ptr();

    // shape change in one func
    *host_x = *gen({23});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y0, host_y1);
    ASSERT_EQ(ptr0, prev_dev_ptr(xp));

    // use another func to check if ptr is persistent
    *host_x = *gen({45});
    auto func1 = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func1->execute();
    MGB_ASSERT_TENSOR_EQ(host_y0, host_y1);
    ASSERT_EQ(ptr0, prev_dev_ptr(xp));
    ASSERT_NE(ptr1, prev_dev_ptr(y1));

    // realloc if shape grows
    *host_x = *gen({450});
    func1->execute();
    MGB_ASSERT_TENSOR_EQ(host_y0, host_y1);
    ASSERT_NE(ptr0, prev_dev_ptr(xp));

    // fail on dynamic shape
    auto y2 = opr::PersistentOutputStorage::make(opr::MarkDynamicVar::make(x));
    auto func2 = graph->compile({{y2, {}}});
    ASSERT_THROW(func2->execute(), GraphError);
}

TEST(TestOprUtility, PersistentOutputStorageMultiCn) {
    HostTensorGenerator<> gen;
    auto cns = load_multiple_xpus(2);
    auto host_x = gen({23}, cns[0]);
    auto graph = ComputingGraph::make();
    // disable copy stream
    graph->options().seq_opt.enable_seq_comp_node_opt = false;

    SymbolVarArray dptr_vars;
    auto copy_add = [&dptr_vars](SymbolVar x, CompNode cn, int share) {
        auto var = opr::PersistentOutputStorage::make(x, share);
        dptr_vars.push_back(var);
        return opr::Copy::make(var + 1, cn);
    };
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y0 = copy_add(opr::Sleep::make(x, 0.01), cns[1], 0),
         y1 = copy_add(y0, cns[0], 0), y2 = copy_add(y1, cns[1], 1),
         y3 = copy_add(y2, cns[0], -1), y4 = copy_add(y3, cns[1], 0),
         y5 = copy_add(y4, cns[0], -1), z = x + 6;
    HostTensorND host_y5, host_z;
    auto func = graph->compile(
            {make_callback_copy(y5, host_y5), make_callback_copy(z, host_z)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_z, host_y5);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < i; ++j) {
            auto a = prev_dev_ptr(dptr_vars[i]), b = prev_dev_ptr(dptr_vars[j]);
            if (i == 4 && j == 0) {
                ASSERT_EQ(a, b);
            } else {
                ASSERT_NE(a, b);
            }
        }
    }
}

TEST(TestOprUtility, InvliadGradCopy) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({2, 3});
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x),
              y = opr::InvalidGrad::make(*((x * 2 + 3) * 4).node()->owner_opr(),
                                         0),
              y1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_preset_passes()
                          .apply({{y}})
                          .endpoint_vars(),
                  y1);
    ASSERT_NE(y.node(), y1.node());
}

TEST(TestOprUtility, RequireInputDynamicStorage) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({2, 3});

    auto x = opr::Host2DeviceCopy::make(*graph, host_x), x2 = x * 2, y = x2 + 3;

    bool called = false;
    auto cb = [&called](DeviceTensorND&) { called = true; };

    auto ycb = opr::CallbackInjector::make(y, cb),
         require_input_dynamic_storage =
                 opr::RequireInputDynamicStorage::make(ycb);

    HostTensorND host_y, host_require_input_dynamic_storage;
    graph->options().graph_opt_level = 0;

    ComputingGraph::OutputSpec out_spec{
            make_callback_copy(y, host_y),
            make_callback_copy(require_input_dynamic_storage,
                               host_require_input_dynamic_storage)};
    auto func = graph->compile(out_spec);
    func->execute();
    auto nr_opr = [](const std::unique_ptr<cg::AsyncExecutable>& func) {
        size_t ret = 0;
        func->iter_opr_seq([&](cg::OperatorNodeBase*) {
            ++ret;
            return true;
        });
        return ret;
    };
    size_t nr0 = nr_opr(func);

    ASSERT_TRUE(called);
    ASSERT_TRUE(cg::is_static_var_storage(x2.node()));
    ASSERT_FALSE(cg::is_static_var_storage(y.node()));
    ASSERT_EQ(prev_dev_ptr(y), prev_dev_ptr(require_input_dynamic_storage));
    ASSERT_EQ(prev_dev_ptr(ycb), prev_dev_ptr(require_input_dynamic_storage));
    MGB_ASSERT_TENSOR_EQ(host_y, host_require_input_dynamic_storage);

    graph->options().graph_opt_level = 2;
    // shallow copy in graph opt
    func = graph->compile(out_spec);
    ASSERT_LT(nr_opr(func), nr0);
}

TEST(TestOprUtility, ShapeHint) {
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_int;
    constexpr size_t length = 233;
    { // basic
        for (bool dynamic : {false, true}) {
            auto host_x = gen_int({length});
            auto graph = ComputingGraph::make();
            SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x), x_shape_hint, y;
            if (dynamic) {
                x_shape_hint = opr::ShapeHint::make(opr::MarkDynamicVar::make(x), TensorShape{length * 2});
            } else {
                x_shape_hint = opr::ShapeHint::make(x, TensorShape{length * 2});
            }
            y = x_shape_hint * 2 + 1;
            if (dynamic) {
                ASSERT_TRUE(y.shape().eq_shape({length * 2}));
            } else {
                ASSERT_TRUE(y.shape().eq_shape({length}));
            }
            HostTensorND host_y;
            auto func = graph->compile({make_callback_copy(y, host_y)});
            func->execute();
            ASSERT_TRUE(host_y.shape().eq_shape({length}));
            for (size_t i = 0; i < length; ++ i) {
                ASSERT_EQ((*host_x->ptr<int32_t>()) * 2 + 1, *host_y.ptr<int32_t>());
            }
        }
    }
    { // shallow copy
        auto graph = ComputingGraph::make();
        auto host_x = gen({length});
        SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x),
                  y = opr::ShapeHint::make(x, TensorShape{length * 2}),
                  x_unknown = opr::MarkDynamicVar::make(x),
                  y_copy = serialization::copy_opr_shallow(
                        *y.node()->owner_opr(), {x_unknown.node()})->output(0);
        ASSERT_TRUE(y.shape().eq_shape({length}));
        ASSERT_TRUE(y_copy.shape().eq_shape({length * 2}));
    }
    { // grad
        auto host_x = gen({1}), host_y = gen({1});
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = opr::Host2DeviceCopy::make(*graph, host_y),
             x_shape_hint = opr::ShapeHint::make(opr::MarkDynamicVar::make(x), TensorShape{1}),
             y_shape_hint = opr::ShapeHint::make(y, TensorShape{1}),
             t = x_shape_hint * y_shape_hint;
        HostTensorND host_gx, host_gy;
        auto func = graph->compile({
            make_callback_copy(cg::grad(t, x), host_gx),
            make_callback_copy(cg::grad(t, y), host_gy)
        });
        func->execute();
        ASSERT_TRUE(host_gx.shape().is_scalar());
        ASSERT_TRUE(host_gy.shape().is_scalar());
        ASSERT_FLOAT_EQ(*host_x->ptr<float>(), *host_gy.ptr<float>());
        ASSERT_FLOAT_EQ(*host_y->ptr<float>(), *host_gx.ptr<float>());
    }
}
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
