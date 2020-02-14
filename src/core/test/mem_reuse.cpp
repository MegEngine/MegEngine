/**
 * \file src/core/test/mem_reuse.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/event.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/blas.h"

#include "megbrain/test/helper.h"

using namespace mgb;

namespace {

SymbolVar make_conv(SymbolVar inp, SymbolVar kern) {
    using Conv = opr::Convolution;
    Conv::ExecutionPolicy poly;
    poly.workspace_limit = 0;
    return Conv::make(inp, kern, {}, poly);
}

// used for test NO_SYS_MEM_ALLOC
MGB_DEFINE_OPR_CLASS(SharedDeviceTensorDirect, cg::SingleCNOperatorNodeBase)// {
    DeviceTensorND m_dv;

    void init_output_comp_node() override {
        output(0)->comp_node(m_dv.comp_node());
        comp_node(m_dv.comp_node());
    }

    void scn_do_execute() override {
        output(0)->reset_dev_tensor_from_tensor(m_dv);
    }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto &&mgr = owner_graph()->static_infer_manager();
        mgr.register_shape_infer(output(0),
                ShapeInferDesc::make_const(m_dv.shape()));
    }

    public:
        SharedDeviceTensorDirect(ComputingGraph &graph,
                const DeviceTensorND &dv, const OperatorNodeConfig &config):
            Super(&graph, config, "shared_nsm", {}),
            m_dv{dv}
        {
            add_output(None)
                ->add_flag(cg::VarNode::Flag::NO_SYS_MEM_ALLOC)
                .dtype(dv.dtype());
        }

        static SymbolVar make(ComputingGraph &graph, const DeviceTensorND &dv,
                const OperatorNodeConfig &config = {}) {
            return graph.insert_opr(std::make_unique<SharedDeviceTensorDirect>(
                        graph, dv, config))->output(0);
        }
};

}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SharedDeviceTensorDirect);

TEST(TestMemReuse, PureMLP0) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_inp = gen({256, 1, 64, 64}),
         host_kern0 = gen({32, 1, 1, 1}),
         host_kern1 = gen({32, 32, 1, 1});
    auto inp = opr::SharedDeviceTensor::make(*graph, *host_inp, {"inp"}),
         kern0 = opr::SharedDeviceTensor::make(*graph, *host_kern0, {"kern0"}),
         kern1 = opr::SharedDeviceTensor::make(*graph, *host_kern1, {"kern1"});
    constexpr size_t NR_LAYER = 7;
    SymbolVar layers[NR_LAYER];
    layers[0] = make_conv(inp, kern0).rename("l0");
    for (size_t i = 1; i < NR_LAYER; i ++)
        layers[i] = make_conv(layers[i - 1], kern1).rename(ssprintf("l%zu", i));
    size_t alloc_size = 0;
    auto hdl = graph->event().register_receiver<cg::event::StaticMemAlloc>(
            [&](const cg::event::StaticMemAlloc &s) {
                if (s.comp_node.valid()) {
                    alloc_size = s.alloc_size;
                }
            });

    graph->options().allocate_static_mem_after_graph_compile = true;
    graph->compile({{layers[NR_LAYER - 1], [](DeviceTensorND&){}}});

    EXPECT_EQ(host_inp->layout().span().dist_byte() * 32 * 2, alloc_size);
}

TEST(TestMemReuse, PureMLP1) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_inp = gen({256, 1, 64, 64}),
         host_kern0 = gen({32, 1, 1, 1}),
         host_kern1 = gen({32, 32, 1, 1});
    auto inp = opr::Host2DeviceCopy::make(*graph, host_inp, {"inp"}),
         kern0 = opr::SharedDeviceTensor::make(*graph, *host_kern0, {"kern0"}),
         kern1 = opr::SharedDeviceTensor::make(*graph, *host_kern1, {"kern1"}),
         layer0 = make_conv(inp, kern0).rename("l0"),
         layer1 = make_conv(layer0, kern1).rename("l1"),
         layer2 = make_conv(layer1, kern1).rename("l2");
    size_t alloc_size = 0;
    auto hdl = graph->event().register_receiver<cg::event::StaticMemAlloc>(
            [&](const cg::event::StaticMemAlloc &s) {
                if (s.comp_node.valid()) {
                    alloc_size = s.alloc_size;
                }
            });

    graph->options().allocate_static_mem_after_graph_compile = true;
    graph->compile({{layer2, [](DeviceTensorND&){}}});

    EXPECT_EQ(host_inp->layout().span().dist_byte() * 32 * 2, alloc_size);
}

TEST(TestMemReuse, MultiCardSafety) {
    auto cns = load_multiple_xpus(3);
    static constexpr size_t N = 4;
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;

    auto host_x0 = gen({N}, cns[0]), host_x1 = gen({N}, cns[1]);
    SymbolVar
        dev_x0_orig = opr::SharedDeviceTensor::make(*graph, *host_x0),
        dev_x0 = opr::Sleep::make(dev_x0_orig, 0.1).rename("x0"),
        dev_x1 = opr::Host2DeviceCopy::make(*graph, host_x1).rename("x1"),
        dev_x1_ = opr::SharedDeviceTensor::make(*graph, *host_x1).rename("x1_"),
        dev_cat = opr::Concat::make({dev_x0, dev_x1}, 0, {cns[2]}),
        ds0 = dev_x1 + dev_x1_,
        ds1 = ds0 + dev_x1_,
        ds2 = ds1 + dev_x1_,
        dev_x1_use = opr::Copy::make(ds2, {cns[2]}),
        dev_dest = opr::Dot::make(dev_cat, dev_cat) +
                opr::Dot::make(dev_x1_use, dev_x1_use);

    {
        auto &&opt = graph->options().seq_opt;
        opt.enable_mem_plan_opt = false;
    }
    HostTensorND host_dest;
    auto func = graph->compile({{dev_dest,
            [&](DeviceTensorND&s){host_dest.copy_from(s);}}});

    func->execute();

    float expected = 0;
    for (size_t i = 0; i < N; i ++) {
        auto v = host_x0->ptr<float>()[i];
        expected += v * v;
    }
    for (size_t i = 0; i < N; i ++) {
        auto v = host_x1->ptr<float>()[i];
        expected += v * v + (4 * v * 4 * v);
    }

    float got = host_dest.sync().ptr<float>()[0];
    MGB_ASSERT_FLOAT_EQ(expected, got);
}

TEST(TestMemReuse, DeviceHolderReuse) {
    HostTensorGenerator<> gen;
    auto host = gen({1});
    host->ptr<float>()[0] = 0;
    auto dev = std::make_shared<DeviceTensorND>();
    dev->copy_from(*host);

    auto host_one = gen({1});
    host_one->ptr<float>()[0] = 1;
    auto dev_one = std::make_shared<DeviceTensorND>();
    dev_one->copy_from(*host_one);

    auto check = [&](thin_function<SymbolVar(ComputingGraph&)> maker,
            bool expect_reuse) {
        auto graph = ComputingGraph::make();
        auto g_x = maker(*graph),
             one = opr::SharedDeviceTensor::make(*graph, dev_one),
             g_y = g_x + one;
        HostTensorND rst;
        auto func = graph->compile({make_callback_copy(g_y, rst)});
        func->execute();
        ASSERT_EQ(1.f, rst.ptr<float>()[0]);
        ASSERT_NE(dev_ptr(one), dev_ptr(g_y));
        if (expect_reuse) {
            ASSERT_EQ(dev_ptr(g_x), dev_ptr(g_y)) <<
                "mem not reused";
        } else {
            ASSERT_NE(dev_ptr(g_x), dev_ptr(g_y));
        }
        HostTensorND orig;
        ASSERT_EQ(orig.copy_from(*dev).sync().ptr<float>()[0], 0);
    };

    check([&](ComputingGraph &g){return
            opr::Host2DeviceCopy::make_no_fwd(g, host);},
            true);

    check([&](ComputingGraph &g){return
            opr::SharedDeviceTensor::make(g, dev);},
            false);
}

TEST(TestMemReuse, SubOverwrite) {
    HostTensorGenerator<> gen;

    auto host_one = gen({1});
    host_one->ptr<float>()[0] = 1;
    auto dev_one = std::make_shared<DeviceTensorND>();
    dev_one->copy_from(*host_one);

    auto host_x = gen({4, 5, 6});
    auto graph = ComputingGraph::make();
    auto sub = [](SymbolVar x, int idx) {
        using O = opr::Subtensor;
        return O::make(x, {O::AxisIndexer::make_index(
                    0, x.make_scalar(idx))});
    };
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         y0 = sub(x, 2),
         y1 = sub(y0, 3),
         y2 = sub(y1, 4),
         z = y2 + opr::SharedDeviceTensor::make(*graph, dev_one);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    ASSERT_TRUE(host_z.layout().is_scalar());
    auto zoffset = host_x->ptr<float>({2, 3, 4}) - host_x->ptr<float>();
    ASSERT_EQ(host_x->ptr<float>()[zoffset] + 1, host_z.ptr<float>()[0]);
    ASSERT_EQ(dev_ptr(z), dev_ptr(x) + zoffset * sizeof(float));
}

TEST(TestMemReuse, WritableFwd) {
    HostTensorGenerator<> gen;
    auto host_x0 = gen({200}), host_x1 = gen({100});

    auto make_y = [&](ComputingGraph &graph) {
        using S = opr::SetSubtensor;
        auto x0 = opr::Host2DeviceCopy::make_no_fwd(graph, host_x0),
             x1 = opr::Host2DeviceCopy::make_no_fwd(graph, host_x1),
             a = x0 * 2,
             b = S::make(a, x1, {S::AxisIndexer::make_interval(
                         0, a.make_scalar(50), a.make_scalar(150), None)});
        auto chk_overwrite = [x0, a, b]() {
            auto p = b.node()->prev_dev_ptr();
            return p == x0.node()->prev_dev_ptr() &&
                p == a.node()->prev_dev_ptr();
        };
        return std::make_pair(b, chk_overwrite);
    };
    auto g0 = ComputingGraph::make(), g1 = ComputingGraph::make();
    g1->options().seq_opt.enable_mem_plan_opt = false;
    auto y0 = make_y(*g0), y1 = make_y(*g1);
    HostTensorND host_y0, host_y1;
    auto f0 = g0->compile({make_callback_copy(y0.first, host_y0)}),
         f1 = g1->compile({make_callback_copy(y1.first, host_y1)});


    f0->execute();
    f1->execute();
    ASSERT_EQ(host_y1.shape(), TensorShape{200});
    MGB_ASSERT_TENSOR_EQ(host_y1, host_y0);
    ASSERT_TRUE(y0.second());
    ASSERT_FALSE(y1.second());
}

TEST(TestMemReuse, RtDynamicMemFwdSubgraph) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({8, 4}, cns[0]);
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         x0 = x.reshape({2, 16}),
         x1 = x0.reshape({4, 8}),
         x2 = x1.reshape({32}),
         y0 = x2 + 1,
         y1 = opr::Copy::make(x1, cns[1]) + 2;
    ASSERT_TRUE(cg::is_static_var_storage(x.node()));
    ASSERT_TRUE(cg::is_static_var_storage(x0.node()));
    ASSERT_TRUE(cg::is_static_var_storage(x1.node()));
    ASSERT_TRUE(cg::is_static_var_storage(x2.node()));
    HostTensorND host_y0, host_y1;
    auto func = graph->compile({
            make_callback_copy(y0, host_y0),
            make_callback_copy(y1, host_y1)});
    func->execute();
    ASSERT_FALSE(cg::is_static_var_storage(x.node()));
    ASSERT_FALSE(cg::is_static_var_storage(x0.node()));
    ASSERT_FALSE(cg::is_static_var_storage(x1.node()));
    ASSERT_FALSE(cg::is_static_var_storage(x2.node()));
    ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(x0));
    ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(x1));
    ASSERT_EQ(prev_dev_ptr(x), prev_dev_ptr(x2));

    auto px = host_x->ptr<float>(),
         py0 = host_y0.ptr<float>(), py1 = host_y1.ptr<float>();
    for (int i = 0; i < 32; ++ i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + 1, py0[i]);
        MGB_ASSERT_FLOAT_EQ(px[i] + 2, py1[i]);
    }
}

TEST(TestMemReuse, FwdNoSysMemAlloc) {
    HostTensorGenerator<> gen;
    auto host_x = gen({8, 4});
    DeviceTensorND dev_x;
    dev_x.copy_from(*host_x);
    auto graph = ComputingGraph::make();
    auto x = SharedDeviceTensorDirect::make(*graph, dev_x),
         y = x.reshape({4, 8}),
         z = y + 1;
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    ASSERT_EQ(prev_dev_ptr(y), prev_dev_ptr(x));
    ASSERT_NE(prev_dev_ptr(z), prev_dev_ptr(x));
    ASSERT_EQ(dev_x.raw_ptr(), prev_dev_ptr(x));

    HostTensorND cur_host_x;
    cur_host_x.copy_from(dev_x).sync();

    auto px0 = host_x->ptr<float>(), px1 = cur_host_x.ptr<float>(),
         pz = host_z.ptr<float>();
    for (size_t i = 0; i < 32; ++ i) {
        MGB_ASSERT_FLOAT_EQ(px0[i], px1[i]);
        MGB_ASSERT_FLOAT_EQ(px0[i] + 1.0f, pz[i]);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

