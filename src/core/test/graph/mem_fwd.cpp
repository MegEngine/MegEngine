/**
 * \file src/core/test/graph/mem_fwd.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/event.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/helper.h"

using namespace mgb;

namespace {
class TrackableDynamicMemAlloc final : public cg::DeviceMemoryAllocator {
    ThinHashSet<VarNode*> m_alive_vars;
    std::mutex m_mtx;

public:
    void alloc_dynamic(VarNode* var, DeviceTensorStorage& dest,
                       size_t size) override {
        ASSERT_LT(dest.size(), size);
        MGB_LOCK_GUARD(m_mtx);
        auto ptr = dest.comp_node().alloc_device(size);
        auto ins = m_alive_vars.insert(var);
        ASSERT_TRUE(ins.second);
        auto del = [ this, var, size, cn = dest.comp_node() ](void* ptr) {
            // modify the data to detect access after free
            DeviceTensorND tensor;
            DeviceTensorStorage storage;
            storage.reset(cn, size,
                          {DeviceTensorStorage::RawStorage{},
                           static_cast<dt_byte*>(ptr)});
            tensor.reset(storage, {TensorShape{size}, dtype::Byte{}});
            dev_tensor_memset(tensor, -1);

            storage.comp_node().free_device(ptr);
            MGB_LOCK_GUARD(m_mtx);
            auto nr = m_alive_vars.erase(var);
            ASSERT_EQ(1u, nr);
        };
        dest.reset(dest.comp_node(), size, {static_cast<dt_byte*>(ptr), del});
    }

    const ThinHashSet<VarNode*>& alive_vars() const { return m_alive_vars; }

    ~TrackableDynamicMemAlloc() { EXPECT_TRUE(m_alive_vars.empty()); }
};

MGB_DEFINE_OPR_CLASS(DynFwdInpToOutOpr, cg::SingleCNOperatorNodeBase) // {
    TrackableDynamicMemAlloc* const m_alloc;
    void init_output_static_infer_desc() override {}

    void scn_do_execute() override {
        auto succ = output(0)->reset_dev_tensor_from_other_var(input(0));
        size_t base_size = 1;
        if (input(0)->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
            base_size = 0;
        }

        auto&& vars = m_alloc->alive_vars();
        if (succ) {
            ASSERT_EQ(base_size, vars.size());
        } else {
            ASSERT_EQ(base_size + 1, vars.size());
            ASSERT_EQ(1u, vars.count(output(0)));
        }
        if (base_size) {
            auto ivar = input(0);
            if (ivar->owner_opr()->same_type<opr::Subtensor>()) {
                ivar = ivar->owner_opr()->input(0);
            }
            ASSERT_EQ(1u, vars.count(ivar));
        }
    }

public:
    DynFwdInpToOutOpr(VarNode* inp, TrackableDynamicMemAlloc* alloc,
                      const OperatorNodeConfig& config)
            : Super(inp->owner_graph(), config, "dyn_fwd", {inp}),
              m_alloc{alloc} {
        add_input({inp});
        add_output(None)
                ->dtype(inp->dtype())
                .add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
    }

    static SymbolVar make(SymbolVar inp, TrackableDynamicMemAlloc* alloc,
                          const OperatorNodeConfig& config = {}) {
        return inp.insert_single_output_opr<DynFwdInpToOutOpr>(inp.node(),
                                                               alloc, config);
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(DynFwdInpToOutOpr);
}  // anonymous namespace

TEST(TestGraph, ShareDevMem) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1234});

    auto make_graph = [&]() {
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = x + 1;
        return std::make_pair(graph, y);
    };

    auto run = [&](bool share) {
        HostTensorND host_y0, host_y1;
        auto g0 = make_graph(), g1 = make_graph();
        if (share)
            g0.first->share_device_memory_with(*g1.first);
        auto f0 = g0.first->compile({make_callback_copy(g0.second, host_y0)});
        auto f1 = g1.first->compile({make_callback_copy(g1.second, host_y1)});
        f0->execute();
        f1->execute();
        f0->wait();
        f1->wait();
        if (share) {
            ASSERT_EQ(dev_ptr(g0.second), dev_ptr(g1.second));
        } else {
            ASSERT_NE(dev_ptr(g0.second), dev_ptr(g1.second));
        }
        MGB_ASSERT_TENSOR_EQ(host_y0, host_y1);
    };

    run(false);
    run(true);
}

TEST(TestGraph, MemFwd0) {
    HostTensorGenerator<> gen;
    auto host_x = gen({3000, 300});
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x).rename("x"),
        x1 = x.reshape({900000, 1, 1, 1}).rename("x1"),
        y = opr::relu(x1).reshape({3000, 300}).rename("y"),
        y1 = y.reshape({900000}).reshape({3000, 300}).rename("y1"),
        z = (y + y1).rename("z");

    HostTensorND host_z;
    auto func = graph->compile({{
        z, [&](DeviceTensorND &s){
            host_z.copy_from(s);
    }}});

    func->execute();

    EXPECT_EQ(dev_ptr(x), dev_ptr(z));

    ASSERT_TRUE(host_x->layout().eq_layout(host_z.layout()));
    ASSERT_TRUE(host_x->layout().is_contiguous());
    auto px = host_x->ptr<float>(),
         pz = host_z.sync().ptr<float>();
    for (size_t i = 0, it = host_z.layout().total_nr_elems(); i < it; ++ i) {
        ASSERT_FLOAT_EQ(std::max(px[i] * 2.f, 0.f), pz[i]);
    }
}

TEST(TestGraph, MemFwd1) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 1});
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
        x1 = x.broadcast({5, 5}).rename("x1"),
        y = x1 * 2;

    HostTensorND host_y;
    graph->options().graph_opt_level = 0;
    auto func = graph->compile({{
        y, [&](DeviceTensorND &s){
            host_y.copy_from(s);
    }}});

    func->execute();

    EXPECT_NE(dev_ptr(x), dev_ptr(y));
    ASSERT_TRUE(host_y.layout().is_contiguous());
    auto val = host_x->ptr<float>()[0] * 2;
    auto ptr = host_y.sync().ptr<float>();
    for (size_t i = 0, it = host_y.layout().total_nr_elems(); i < it; ++ i) {
        ASSERT_FLOAT_EQ(val, ptr[i]);
    }
}

TEST(TestGraph, MemFwd2) {
    HostTensorGenerator<> gen;
    auto host_x1 = gen({1, 1}), host_x2 = gen({1});
    host_x1->ptr<float>()[0] = 1;
    host_x2->ptr<float>()[0] = 2;
    auto graph = ComputingGraph::make();
    using MMul = opr::MatrixMul;
    SymbolVar
        x1 = opr::Host2DeviceCopy::make(*graph, host_x1).rename("x1"),
        x2 = opr::Host2DeviceCopy::make(*graph, host_x2).rename("x2"),
        x2_ = opr::mul(x2, x1.reshape({1})).reshape({1, 1}).rename("x2_"),
        y = MMul::make(x1, MMul::make(x2.reshape({1, 1}), x2_)).rename("y");

    HostTensorND host_y;
    auto func = graph->compile({{
        y, [&](DeviceTensorND &s){
            host_y.copy_from(s);
    }}});

    func->execute();
    host_y.sync();
    ASSERT_EQ(1u, host_y.layout().total_nr_elems());
    ASSERT_EQ(4.f, host_y.ptr<float>()[0]);
}

TEST(TestGraph, MemFwd3) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    host_x->ptr<float>()[0] = 2;
    auto graph = ComputingGraph::make();
    SymbolVar
        x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
        xcpy = opr::Sleep::make(x, 0.001).rename("xcpy"),
        y = (x + xcpy).rename("y");

    HostTensorND host_y;
    auto func = graph->compile({{
        y, [&](DeviceTensorND &s){
            host_y.copy_from(s);
    }}});

    func->execute();
    func->to_json()->writeto_fpath(output_file("TestMemFwd3.json"));
    host_y.sync();
    ASSERT_TRUE(host_y.layout().is_scalar());
    ASSERT_EQ(4.f, host_y.ptr<float>()[0]);

    if (dev_ptr(y) != dev_ptr(x))
        ASSERT_EQ(dev_ptr(y), dev_ptr(xcpy));
    else
        ASSERT_NE(dev_ptr(y), dev_ptr(xcpy));
}

TEST(TestGraph, InplaceWithDynStorage) {
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> geni(0, 456);

    auto run_test = [&](bool dyn) {
        auto host_x = gen({123, 456}),
             host_val = gen({123, 1}),
             host_idx = geni({123});

        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        auto cn1 = host_x->comp_node();
        if (dyn)
            cn1 = cn1.change_stream(1);
        SymbolVar
            x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
            val = opr::Host2DeviceCopy::make(*graph, host_val),
            idx = opr::Host2DeviceCopy::make(*graph, host_idx),
            out = opr::IndexingSetOneHot::make(x, idx, val, {1}),
            delta = opr::MarkDynamicVar::make(out.make_scalar(2.3f), cn1),
            // out is dyn alloc because delta is on another cn
            y = opr::add(out, delta, delta.node()->comp_node());

        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});

        func->execute();

        if (dyn) {
            ASSERT_TRUE(out.node()->contain_flag(
                        VarNode::Flag::RT_FORCE_DYNAMIC_MEM_ALLOC));
            ASSERT_NE(prev_dev_ptr(x), prev_dev_ptr(out));
        } else {
            ASSERT_FALSE(out.node()->contain_flag(
                        VarNode::Flag::RT_FORCE_DYNAMIC_MEM_ALLOC));
            ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(out));
        }

        auto px = host_x->ptr<float>(), pval = host_val->ptr<float>(),
             py = host_y.ptr<float>();
        auto pidx = host_idx->ptr<int>();

        for (int i = 0; i < 123; ++ i) {
            auto idx = pidx[i];
            for (int j = 0; j < 456; ++ j) {
                auto val = px[i * 456 + j];
                if (j == idx)
                    val = pval[i];
                val += 2.3;
                MGB_ASSERT_FLOAT_EQ(val, py[i * 456 + j]) <<
                    "failed at " << i << "," << j;
            }
        }
    };
    run_test(false);
    run_test(true);
}

TEST(TestGraph, ImpureMemPlanFwd) {
    auto dv = std::make_shared<DeviceTensorND>();
    DeviceTensorND dv0, dv1;
    {
        HostTensorGenerator<> gen;
        auto hv = gen({46});    // use a temp storage to avoid mem deallocation
        dv0.copy_from(*hv).sync();
        hv = gen({46});
        dv1.copy_from(*hv).sync();
    }

    auto graph = ComputingGraph::make();
    *dv = dv0;
    auto x = opr::VolatileSharedDeviceTensor::make(*graph, dv),
         xrshp = opr::Reshape::make(x, TensorShape{2, 0}, 1),
         xsub = opr::Subtensor::make(
                 xrshp, {opr::Subtensor::AxisIndexer::make_interval(
                                0, x.make_scalar(1), None, None)}),
         y1 = xsub + 1, y2 = x + 1, y3 = opr::MarkDynamicVar::make(x) + 1;
    HostTensorND host_y1, host_y2, host_y3;
    auto func = graph->compile({make_callback_copy(y1, host_y1),
                                make_callback_copy(y2, host_y2),
                                make_callback_copy(y3, host_y3)});
    bool mem_alloc_called = false;

    graph->event().register_receiver_permanent<cg::event::StaticMemAlloc>(
            [&](const cg::event::StaticMemAlloc&) { mem_alloc_called = true; });

    auto _check = [&]() {
        HostTensorND hv;
        hv.copy_from(*dv).sync();
        func->execute();
        auto px = hv.ptr<float>(), py1 = host_y1.ptr<float>(),
             py2 = host_y2.ptr<float>(), py3 = host_y3.ptr<float>();
        auto elems = hv.layout().total_nr_elems();
        for (size_t i = 0; i < elems; ++i) {
            if (i >= elems / 2) {
                MGB_ASSERT_FLOAT_EQ(px[i] + 1, py1[i - elems / 2]);
            }
            MGB_ASSERT_FLOAT_EQ(px[i] + 1, py2[i]);
            MGB_ASSERT_FLOAT_EQ(px[i] + 1, py3[i]);
        }
        ASSERT_EQ(dv->raw_ptr(), prev_dev_ptr(x));
        ASSERT_EQ(dv->raw_ptr(), prev_dev_ptr(xrshp));
        ASSERT_EQ(dv->raw_ptr() +
                          elems / 2 * sizeof(float) * dv->layout().stride[0],
                  prev_dev_ptr(xsub));
    };
#define check(expect_alloc)                           \
    do {                                              \
        _check();                                     \
        bool expect_alloc_bv = expect_alloc;          \
        ASSERT_EQ(expect_alloc_bv, mem_alloc_called); \
        mem_alloc_called = false;                     \
    } while (0)

    check(true);
    check(false);
    *dv = dv1;  // change ptr
    check(false);
    check(false);

    TensorLayout ly_new{TensorShape{20}, dtype::Float32{}};
    *dv = dv1.sub(SubTensorSpec::make_from_layout(ly_new));
    ASSERT_EQ(dv1.raw_ptr(), dv->raw_ptr());
    // change shape
    check(true);
    check(false);

    ly_new.stride[0] = 2;
    *dv = dv1.sub(SubTensorSpec::make_from_layout(ly_new));
    ASSERT_EQ(dv1.raw_ptr(), dv->raw_ptr());

    // change only stride
    check(true);
    check(false);

#undef check
}

TEST(TestGraph, CrossCNMemFwd) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x).sync();
    for (int casenum : {0, 1, 2}) {
        // case0: h2d
        // case1: persist
        // case2: persist, with add update
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        auto cn0 = host_x->comp_node(), cn1 = cn0.change_stream(1);
        auto x = casenum ? opr::SharedDeviceTensor::make(*graph, dev_x)
                         : opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
             y1 = x + 1, y2 = opr::Copy::make(x, cn1), y2p = y2 + .5f,
             y2pcn0 = opr::Copy::make(y2p, cn0), z = y1 * y2pcn0;
        HostTensorND expect;
        graph->compile({make_callback_copy(((x + 1) * (x + .5f)), expect)})
                ->execute();
        HostTensorND host_z;
        ComputingGraph::OutputSpec out_spec{make_callback_copy(z, host_z)};
        if (casenum == 2) {
            auto xud = opr::AddUpdate::make(x, z);
            out_spec.push_back({xud, {}});
        }
        auto func = graph->compile(out_spec);
        func->execute();
        MGB_ASSERT_TENSOR_EQ(expect, host_z);
        ASSERT_EQ(prev_dev_ptr(y2p), prev_dev_ptr(y2pcn0));

        if (casenum < 2) {
            ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(y2));
        } else {
            ASSERT_NE(prev_dev_ptr(x), prev_dev_ptr(y2));
            HostTensorND xget, expect;
            xget.copy_from(*dev_x).sync();
            expect.copy_from(host_z);
            for (int i = 0; i < 6; ++i) {
                expect.ptr<float>()[i] += host_x->ptr<float>()[i];
            }
            MGB_ASSERT_TENSOR_EQ(expect, xget);
        }
    }
}

TEST(TestGraph, MemResetFwdAsync) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x).sync();
    auto cn0 = host_x->comp_node(), cn1 = cn0.change_stream(1);
    for (int casenum : {0, 1, 2, 3}) {
        // case0: h2d, sys dynamic alloc
        // case1: h2d, no sys alloc
        // case2: persist, no sys alloc
        // case3: persist, no sys alloc, add update
        auto graph = ComputingGraph::make();
        graph->options().var_sanity_check_first_run = false;
        graph->options().graph_opt_level = 0;
        graph->options().seq_opt.enable_seq_comp_node_opt = false;
        auto tracker = std::make_shared<TrackableDynamicMemAlloc>();
        graph->set_device_memory_allocator(tracker);

        auto x = casenum < 2 ? opr::Host2DeviceCopy::make_no_fwd(*graph, host_x)
                             : opr::SharedDeviceTensor::make(*graph, dev_x),
             x_fwd = casenum ? DynFwdInpToOutOpr::make(x, tracker.get(), cn1)
                             : opr::RequireInputDynamicStorage::make(x, cn1),
             y = opr::Sleep::make(x_fwd, 0.02);
        ASSERT_EQ(cn1, x_fwd.node()->comp_node());
        HostTensorND host_y;
        ComputingGraph::OutputSpec out_spec{make_callback_copy(y, host_y)};
        if (casenum == 3) {
            auto xud = opr::AddUpdate::make(
                    x, x.make_scalar(2.3f).broadcast(x.symshape()));
            out_spec.push_back({xud, {}});
        }
        auto func = graph->compile(out_spec);
        if (casenum < 2) {
            ASSERT_FALSE(cg::is_static_var_storage(x.node()));
        }
        for (size_t i = 0; i < 3; ++i) {
            if (casenum < 2) {
                *host_x = *gen({2 + i, 3});
            } else {
                host_x->copy_from(*gen(host_x->shape()));
            }
            dev_x->copy_from(*host_x).sync();
            func->execute().wait();
            ASSERT_TRUE(tracker->alive_vars().empty());
            if (casenum <= 2) {
                ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(x_fwd));
            } else {
                // case3: fail due to add update
                ASSERT_NE(prev_dev_ptr(x), prev_dev_ptr(x_fwd));
            }
            MGB_ASSERT_TENSOR_EQ(*host_x, host_y) << "casenum=" << casenum;
            if (casenum >= 2) {
                HostTensorND xv;
                xv.copy_from(*dev_x).sync();
                HostTensorND expect;
                expect.copy_from(*host_x);
                if (casenum == 3) {
                    auto ptr = expect.ptr<float>();
                    for (size_t i = 0, it = host_x->shape().total_nr_elems();
                         i < it; ++i) {
                        ptr[i] += 2.3f;
                    }
                }
                MGB_ASSERT_TENSOR_EQ(expect, xv);
            }
        }
    }
}

TEST(TestGraph, MemResetFwdNonContig) {
    HostTensorGenerator<> gen;
    auto host_x = gen({23});
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    graph->options().graph_opt_level = 0;
    auto tracker = std::make_shared<TrackableDynamicMemAlloc>();
    graph->set_device_memory_allocator(tracker);
    auto host_step = std::make_shared<HostTensorND>(
            host_x->comp_node(), TensorShape{1}, dtype::Int32{});
    host_step->ptr<int>()[0] = 1;

    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         step = opr::Host2DeviceCopy::make(*graph, host_step),
         xsub = opr::Subtensor::make(
                 x, {opr::Subtensor::AxisIndexer::make_interval(0, None, None,
                                                                step)}),
         y = DynFwdInpToOutOpr::make(xsub, tracker.get());
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();
    MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
    ASSERT_EQ(prev_dev_ptr(xsub), prev_dev_ptr(y));

    HostTensorND expect;
    {
        auto p = expect.copy_from(*host_x).ptr<float>();
        for (size_t i = 0; i < 11; ++i) {
            std::swap(p[i], p[22 - i]);
        }
    }

    host_step->ptr<int>()[0] = -1;
    func->execute();
    MGB_ASSERT_TENSOR_EQ(expect, host_y);
    ASSERT_NE(prev_dev_ptr(xsub), prev_dev_ptr(y));
}

TEST(TestGraph, MemFwdPersistToDynamic) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto dev_x = std::make_shared<DeviceTensorND>();

    for (bool should_fwd : {false, true}) {
        dev_x->copy_from(*host_x);

        auto allocator = std::make_shared<TrackableDynamicMemAlloc>();

        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        graph->set_device_memory_allocator(allocator);

        auto x = opr::SharedDeviceTensor::make(*graph, dev_x),
             xdelta = x * 1.3f, xud = opr::AddUpdate::make(x, xdelta),
             xrshp = x.reshape({6});

        auto xrshp_cb = [&](DeviceTensorND&) {
            ASSERT_EQ(should_fwd ? 0u : 1u,
                      allocator->alive_vars().count(xrshp.node()));
        };
        auto y = opr::CallbackInjector::make(xrshp, xrshp_cb) + 2.3f;
        HostTensorND host_y;

        if (should_fwd) {
            set_priority(xdelta, 100);
            set_priority(xud, 100);
        } else {
            set_priority(xdelta, -100);
            set_priority(xud, -100);
        }

        xrshp.node()->add_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC);

        auto func = graph->compile({make_callback_copy(y, host_y), {xud, {}}});
        func->execute();

        ASSERT_TRUE(allocator->alive_vars().empty());

        HostTensorND expect_x, expect_y;

        auto g1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*g1, host_x);
        g1->compile({make_callback_copy(x1 * 2.3f, expect_x),
                     make_callback_copy(x1.flatten() + 2.3f, expect_y)})
                ->execute();
        HostTensorND xgot;
        xgot.copy_from(*dev_x).sync();
        MGB_ASSERT_TENSOR_EQ(expect_x, xgot);
        MGB_ASSERT_TENSOR_EQ(expect_y, host_y);

        if (should_fwd) {
            ASSERT_EQ(dev_x->raw_ptr(), prev_dev_ptr(xrshp));
        } else {
            ASSERT_NE(dev_x->raw_ptr(), prev_dev_ptr(xrshp));
        }
    }
}

TEST(TestGraph, MemFwdPersistSysAlloc) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x).sync();
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto cn1 = host_x->comp_node().change_stream(1);

    auto x = opr::SharedDeviceTensor::make(*graph, dev_x);
    auto cb = [&](const DeviceTensorND&) {
        ASSERT_TRUE(x.node()->dev_tensor_valid());
    };

    auto xcn1 = opr::Copy::make(x, cn1),
         xcn1_dyn = opr::RequireInputDynamicStorage::make(xcn1),
         xdyn = opr::RequireInputDynamicStorage::make(x),
         ycb = opr::CallbackInjector::make(x.make_scalar(0), cb);
    set_priority(x, 100);
    auto func = graph->compile({{ycb, {}}, {xcn1_dyn, {}}, {xdyn, {}}});

    int cur_step = 0, cb_step = -1, dv_step = -1;
    auto on_opr = [&](cg::OperatorNodeBase* opr) {
        if (opr->same_type<opr::CallbackInjector>()) {
            cb_step = cur_step;
        }
        if (opr->same_type<opr::SharedDeviceTensor>()) {
            dv_step = cur_step;
        }
        ++cur_step;
        return true;
    };
    func->iter_opr_seq(on_opr);

    ASSERT_LT(cb_step, dv_step);
    ASSERT_GT(cb_step, 0);

    func->execute();

    ASSERT_EQ(dev_x->raw_ptr(), prev_dev_ptr(xcn1));
    ASSERT_TRUE(cg::is_static_var_storage(x.node()));
    ASSERT_FALSE(cg::is_static_var_storage(xcn1.node()));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

