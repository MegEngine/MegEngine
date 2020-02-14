/**
 * \file src/core/test/static_infer.cpp
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
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/helper.h"

using namespace mgb;

namespace {

//! set source value for testing static infer
MGB_DEFINE_OPR_CLASS(StaticInferSrcValueInjector,
        cg::SingleCNOperatorNodeBase) // {

    bool m_infer_called = false;
    HostTensorND &m_val;

    void scn_do_execute() override {
        mgb_assert(0);
    }

    void init_output_comp_node() override {
    }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto infer_shp = [this](TensorShape &dest, const InpVal &) {
            dest = m_val.shape();
            return true;
        };
        auto infer_val = [this](DeviceTensorND &dest, const InpVal &) {
            m_infer_called = true;
            dest = DeviceTensorND::make_proxy(m_val);
            return true;
        };
        auto &&mgr = owner_graph()->static_infer_manager();
        mgr.register_shape_infer(
                output(0), {SourceType::MUTABLE, {}, infer_shp});
        mgr.register_value_infer(
                output(0), {SourceType::MUTABLE, {}, infer_val});
    }

    public:
        StaticInferSrcValueInjector(
                ComputingGraph *owner, HostTensorND &val, CompNode cn):
            Super{owner, OperatorNodeConfig{}, "src_value_inj", {}},
            m_val{val}
        {
            add_equivalence_component<ScalarHash<void*>>(this);
            add_output(None)->dtype(val.dtype());
            comp_node(cn);
        }

        static StaticInferSrcValueInjector& make(
                ComputingGraph *owner, HostTensorND &val, CompNode cn) {
            return
                owner->insert_opr(
                        std::make_unique<StaticInferSrcValueInjector>(
                            owner, val, cn))
                ->cast_final_safe<StaticInferSrcValueInjector>();
        }

        //! set m_infer_called to false and return current value
        bool reset_infer_called() {
            auto ret = m_infer_called;
            m_infer_called = false;
            return ret;
        }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(StaticInferSrcValueInjector);

//! forward unchanged value and set m_infer_called flag
MGB_DEFINE_OPR_CLASS(StaticInferMidValueInjector,
        cg::SingleCNOperatorNodeBase) // {

    const DeviceTensorND *m_prev_value = nullptr;

    void scn_do_execute() override {
        mgb_assert(0);
    }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto infer_val = [this](DeviceTensorND &dest, const InpVal &inp) {
            m_prev_value = &inp.val.at(0).value();
            dest = *m_prev_value;
            return true;
        };
        auto &&mgr = owner_graph()->static_infer_manager();
        auto ivar = input(0), ovar = output(0);
        mgr.register_shape_infer(ovar, ShapeInferDesc::make_identity(ivar));
        mgr.register_value_infer(
                ovar,
                {SourceType::DEP, {{ivar, DepType::VALUE}}, infer_val});
    }

    public:
        StaticInferMidValueInjector(ComputingGraph *owner, VarNode *inp):
            Super{owner, OperatorNodeConfig{}, "mid_value_inj", {inp}}
        {
            add_input({inp});
            add_output(None);
        }

        static StaticInferMidValueInjector& make(SymbolVar inp) {
            auto owner = inp.node()->owner_graph();
            return
                owner->insert_opr(
                        std::make_unique<StaticInferMidValueInjector>(
                            owner, inp.node()))
                ->cast_final_safe<StaticInferMidValueInjector>();
        }

        //! reset m_prev_value and return current
        const DeviceTensorND* reset_prev_val() {
            auto ret = m_prev_value;
            m_prev_value = nullptr;
            return ret;
        }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(StaticInferMidValueInjector);

class TrackableStaticMemAlloc final : public cg::DeviceMemoryAllocator {
    size_t m_nr_call = 0;

public:
    void alloc_static(ComputingGraph*, DeviceTensorStorage& dest,
                      size_t size) override {
        dest.ensure_size(size);
        ++m_nr_call;
    }

    size_t nr_call() const { return m_nr_call; }
};

} // anonymous namespace

TEST(TestStaticInfer, ValueInfer) {
    using namespace cg::static_infer;
    HostTensorGenerator<> gen;
    constexpr size_t SIZE = 3;
    auto host_x0 = gen({SIZE}), host_x1 = gen({SIZE});

    auto graph = ComputingGraph::make();
    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0),
         x1 = opr::Host2DeviceCopy::make(*graph, host_x1),
         x2 = x0 + x1,
         y0 = x0.make_scalar(2.f),
         y1 = x0.make_scalar(3.f),
         y2 = opr::pow(y0, y1);
    auto &&mgr = x0.node()->owner_graph()->static_infer_manager();
    ASSERT_EQ(InferType::RT_STATIC, mgr.get_infer_type(x2.node()).value);
    ASSERT_EQ(InferType::CONST, mgr.get_infer_type(y0.node()).value);
    ASSERT_EQ(InferType::CONST, mgr.get_infer_type(y1.node()).value);
    ASSERT_EQ(InferType::CONST, mgr.get_infer_type(y2.node()).value);
    auto x2v = mgr.infer_value(x2.node());
    ASSERT_EQ(host_x0->shape(), x2v.shape());
    for (size_t i = 0; i < SIZE; i ++)
        MGB_ASSERT_FLOAT_EQ(host_x0->ptr<float>()[i] + host_x1->ptr<float>()[i],
                x2v.ptr<float>()[i]);

    auto y2v = mgr.infer_value(y2.node());
    ASSERT_TRUE(y2v.shape().is_scalar());
    MGB_ASSERT_FLOAT_EQ(8.f, y2v.ptr<float>()[0]);
}

TEST(TestStaticInfer, ValueNonContig) {
    using namespace cg::static_infer;
    HostTensorGenerator<> gen;
    auto host_x0 = gen({1}), host_x1 = gen({5, 5});

    auto graph = ComputingGraph::make();
    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0),
         x1 = opr::Host2DeviceCopy::make(*graph, host_x1),
         y0 = x0.broadcast({10}),
         y1 = opr::Subtensor::make(x1,
                 {opr::Subtensor::AxisIndexer::make_interval(1,
                         x0.make_scalar(1), x0.make_scalar(4),
                         x0.make_scalar(1))}),
         y2 = y0 + 1;
    auto &&mgr = x0.node()->owner_graph()->static_infer_manager();
    ASSERT_EQ(InferType::RT_STATIC, mgr.get_infer_type(y0.node()).value);
    ASSERT_EQ(InferType::RT_STATIC, mgr.get_infer_type(y1.node()).value);
    ASSERT_EQ(InferType::RT_STATIC, mgr.get_infer_type(y2.node()).value);

    auto &&y0v = mgr.infer_value(y0.node()),
         &&y1v = mgr.infer_value(y1.node()),
         &&y2v = mgr.infer_value(y2.node());
    auto x0v = host_x0->ptr<float>()[0];
    ASSERT_EQ(y0v.layout().stride[0], 0);
    ASSERT_EQ(y0v.ptr<float>()[0], x0v);
    ASSERT_FALSE(y1v.layout().is_contiguous());
    auto y1v_expect = (*host_x1)[{{}, {1, 4}}];
    MGB_ASSERT_TENSOR_EQ(y1v_expect, HostTensorND::make_proxy(y1v));

    ASSERT_TRUE(y2v.layout().is_contiguous());
    auto py2 = y2v.ptr<float>();
    for (size_t i = 0; i < 10; ++ i) {
        ASSERT_EQ(x0v + 1.f, py2[i]);
    }
}

TEST(TestStaticInfer, SrcChangeDetection) {
    using namespace cg::static_infer;
    HostTensorGenerator<> gen;

    HostTensorND host_tshp(CompNode::default_cpu());
    host_tshp.dtype(dtype::Int32()).resize({1});
    host_tshp.ptr<int>()[0] = 2;

    auto graph = ComputingGraph::make();
    auto x0 = opr::Host2DeviceCopy::make(*graph, gen({1}));
    auto &&tshp_src = StaticInferSrcValueInjector::make(graph.get(), host_tshp,
            x0.node()->comp_node());
    auto &&tshp_mid = StaticInferMidValueInjector::make(tshp_src.output(0));


    auto y = x0.broadcast(tshp_mid.output(0));
    ASSERT_TRUE(tshp_src.reset_infer_called());
    ASSERT_TRUE(tshp_mid.reset_prev_val());
    ASSERT_EQ(TensorShape{2}, y.node()->shape());
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();
    ASSERT_TRUE(tshp_src.reset_infer_called());
    ASSERT_EQ(nullptr, tshp_mid.reset_prev_val());

    host_tshp.resize({4});
    {
        auto ptr = host_tshp.ptr<int>();
        ptr[0] = 2; ptr[1] = 23; ptr[2] = 3; ptr[3] = 23;
    }
    host_tshp = host_tshp[{{None, None, 2}}];
    func->execute();
    ASSERT_EQ(TensorShape({2, 3}), host_y.shape());
    ASSERT_TRUE(tshp_src.reset_infer_called());
    ASSERT_TRUE(tshp_mid.reset_prev_val()->layout().is_contiguous());

    host_tshp.ptr<int>()[1] = 32;
    func->execute();
    ASSERT_EQ(TensorShape({2, 3}), host_y.shape());
    ASSERT_TRUE(tshp_src.reset_infer_called());
    ASSERT_EQ(nullptr, tshp_mid.reset_prev_val());

    host_tshp.resize({2});
    {
        auto ptr = host_tshp.ptr<int>();
        ptr[0] = 3; ptr[1] = 2;
    }
    host_tshp = host_tshp[{{None, None, -1}}];
    func->execute();
    ASSERT_EQ(TensorShape({2, 3}), host_y.shape());
    ASSERT_TRUE(tshp_src.reset_infer_called());
    ASSERT_EQ(nullptr, tshp_mid.reset_prev_val());

    host_tshp.ptr<int>()[-1] = 4;
    func->execute();
    ASSERT_EQ(TensorShape({2, 4}), host_y.shape());
    ASSERT_TRUE(tshp_src.reset_infer_called());
    ASSERT_TRUE(tshp_mid.reset_prev_val()->layout().is_contiguous());

    host_tshp.reset(host_tshp.storage(),
            TensorLayout({1}, dtype::Int32()).broadcast({2}));
    host_tshp.ptr<int>()[0] = 2;
    func->execute();
    ASSERT_EQ(TensorShape({2, 2}), host_y.shape());
    ASSERT_TRUE(tshp_src.reset_infer_called());
    ASSERT_EQ(0, tshp_mid.reset_prev_val()->layout().stride[0]);
}

TEST(TestStaticInfer, AsImmutableScalar) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Int32> gen;
    auto host_one = gen({1});
    host_one->ptr<int>()[0] = 1;
    auto one = opr::ImmutableTensor::make(*graph, *host_one),
         x = one + 1,
         y = opr::Subtensor::make((one * 3).broadcast({2, 3}),
                 {opr::Subtensor::AxisIndexer::make_index(
                         1, x.make_scalar(1))}),
         z = opr::Concat::make({one, one}, 0).reshape({2, 1}).broadcast({2, 3});
    auto xv = x.as_immutable_scalar(),
         yv = y.as_immutable_scalar(),
         zv = z.as_immutable_scalar();
    ASSERT_EQ(2, xv->get<int>());
    auto &&mgr = graph->static_infer_manager();
    auto &&yv_infer = mgr.infer_value(y.node());
    ASSERT_EQ(TensorShape{2}, yv_infer.shape());
    ASSERT_EQ(0, yv_infer.layout().stride[0]);
    ASSERT_EQ(3, yv->get<int>());
    ASSERT_FALSE(zv.valid());
    ASSERT_FALSE(y.as_immutable_scalar_require_shape().valid());
}

TEST(TestStaticInfer, EagerConstShape) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}),
         host_y = gen({1, 3});
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::ImmutableTensor::make(*graph, *host_y),
         y1 = y + 2.3f,
         z = x * y1;

    ASSERT_EQ(TensorShape({1, 3}), y1.shape());
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    for (size_t i: {2, 5}) {
        *host_x = *gen({i, 3});
        func->execute();
        ASSERT_EQ(TensorShape({i, 3}), host_z.shape());
        auto px = host_x->ptr<float>(), py = host_y->ptr<float>(),
             pz = host_z.ptr<float>();
        for (size_t x = 0; x < i; ++ x) {
            for (size_t y = 0; y < 3; ++ y) {
                MGB_ASSERT_FLOAT_EQ(px[x * 3 + y] * (py[y] + 2.3f),
                        pz[x * 3 + y]);
            }
        }
    }
}

TEST(TestStaticInfer, Updater) {
    using namespace cg::static_infer;
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;

    HostTensorND host_tshp(CompNode::default_cpu());
    host_tshp.dtype(dtype::Int32()).resize({1});
    host_tshp.ptr<int>()[0] = 1;

    auto host_x = gen({1, 2});
    auto&& tshp = StaticInferSrcValueInjector::make(graph.get(), host_tshp,
                                                    host_x->comp_node());
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = x.reshape(SymbolVar{tshp.output(0)} + 1) + 2.3f;

    HostTensorND host_y;
    auto check = [&]() {
        auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
        size_t num = host_x->shape().total_nr_elems();
        ASSERT_EQ(TensorShape{num}, host_y.shape());
        for (size_t i = 0; i < num; ++i) {
            ASSERT_EQ(px[i] + 2.3f, py[i]);
        }
    };

    auto allocator = std::make_shared<TrackableStaticMemAlloc>();
    graph->set_device_memory_allocator(allocator);
    auto func = graph->compile({make_callback_copy(y, host_y)});
    auto updater = StaticInferUpdater::make();
    updater->add_dest({y.node(), DepType::SHAPE});

    auto run = [&](size_t nr_alloc) {
        func->execute();
        ASSERT_EQ(nr_alloc, allocator->nr_call());
        check();
    };

    run(1);
    ASSERT_TRUE(tshp.reset_infer_called());
    updater->update();
    run(1);
    ASSERT_TRUE(tshp.reset_infer_called());

    *host_x = *gen({4, 256});
    host_tshp.ptr<int>()[0] = 1023;
    ASSERT_FALSE(tshp.reset_infer_called());
    updater->update();
    ASSERT_TRUE(tshp.reset_infer_called());
    ASSERT_EQ(TensorShape{2}, y.shape());
    ASSERT_EQ(TensorShape{1024},
              graph->static_infer_manager().infer_shape(y.node()));

    run(2);
    ASSERT_EQ(TensorShape{1024}, y.shape());

    auto src = graph->static_infer_manager().get_rt_static_source_deps(
            {y.node(), DepType::SHAPE});
    ASSERT_EQ(1u, src.size());
    ASSERT_EQ(tshp.output(0), src[0].dest);
    ASSERT_EQ(DepType::VALUE, src[0].type);
}

TEST(TestStaticInfer, NeedSharedDeviceTensorHostValueCrossCN) {
    constexpr size_t SIZE = 42;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().seq_opt.enable_seq_comp_node_opt=0;
    // force performing async dispatch on CPU
    graph->options().async_exec_level = 0b10;

    auto cb_sleep = [](DeviceTensorND&) {
        using namespace std::literals;
        std::this_thread::sleep_for(0.2s);
    };

    std::shared_ptr<HostTensorND> host_val = gen({SIZE});
    for (size_t i = 0; i < SIZE; ++ i)
        host_val->ptr<float>()[i] = i ? 0.0 : 1.0;

    auto cn0 = CompNode::load("xpu0"), cn1 = CompNode::load("xpu1");
    auto param0 = opr::SharedDeviceTensor::make(*graph,
            *host_val, {"param0", cn0});
    param0.node()->owner_opr()->node_prop().attribute().priority =
            std::numeric_limits<int>::max();
    auto idx0 = opr::TypeCvt::make(
        opr::Reduce::make(param0, {}, param0.make_scalar(1), {cn0}),
        dtype::Int32());
    auto idx1 = opr::Copy::make(idx0, cn1);
    auto param1 = opr::SharedDeviceTensor::make(*graph,
            *host_val, {"param1", cn1});
    auto sub = opr::Subtensor::make(param1,
        {opr::Subtensor::AxisIndexer::make_interval(
            0, idx1, idx1 + 1, None)});

    auto sleeper = opr::CallbackInjector::make(
            opr::SharedDeviceTensor::make(*graph, *host_val,
                {"sleeper", cn0}),
            cb_sleep);

    HostTensorND host_out;
    auto func = graph->compile({
        make_callback_copy(sub, host_out),
        {sleeper, [](DeviceTensorND&){}}
    });
    func->execute().wait();
    ASSERT_EQ(1u, host_out.shape().ndim);
    MGB_ASSERT_FLOAT_EQ(0.0f, host_out.ptr<float>()[0]);
}
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

