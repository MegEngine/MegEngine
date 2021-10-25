/**
 * \file src/core/test/graph/misc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/misc.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/execution_mask.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/graph/helper.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/utils/timer.h"

#include "megbrain/test/helper.h"
#include "megdnn/heuristic_cache.h"
#include "megdnn/oprs/base.h"

#include <array>
#include <atomic>
#include <chrono>
#include <memory>

using namespace mgb;

namespace mgb {
namespace cg {
// declaration of impl class to access its methods
class ComputingGraphImpl : public ComputingGraph {
public:
    GraphExecutable::ExecEnv* current_exec_env();
};
class SeqCompNodeOptimizerImpl : public SeqCompNodeOptimizer {
    ~SeqCompNodeOptimizerImpl() = default;

public:
    void optimize_comp_nodes(const VarNodeArray& endpoints);
};
}  // namespace cg
}  // namespace mgb

namespace {

MGB_DEFINE_OPR_CLASS(PODDedupTestOpr, cg::SingleCNOperatorNodeBase) // {
public:
    struct Param {
        int v0;
        char v1;
    } MGB_PACKED;

    PODDedupTestOpr(ComputingGraph* owner, const Param& param)
            : Super{owner, OperatorNodeConfig{}, "node", {}}, m_param(param) {
        add_equivalence_component<PODHash<Param>>(&m_param);
        add_output(None)->dtype(dtype::Byte());
    }

    static SymbolVar make(ComputingGraph& owner, const Param& param) {
        return owner.insert_opr(std::make_unique<PODDedupTestOpr>(&owner, param))
                ->output(0);
    }

private:
    Param m_param;

    void scn_do_execute() override {}

    void init_output_comp_node() override {
        output(0)->comp_node(CompNode::load("xpu0"));
    }

    void init_output_static_infer_desc() override {
        using namespace mgb::cg::static_infer;
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0),
                {SourceType::CONSTANT, {}, [](TensorShape& dest, const InpVal&) {
                     dest = {1};
                     return true;
                 }});
    }
};

MGB_DEFINE_OPR_CLASS(
        WorkspaceAllocTestOpr, cg::SingleCNOutshapePureByInshapeOprBase) // {
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override {
        MGB_MARK_USED_VAR(inp_shape);
        out_shape.at(0) = {2};
        out_shape.at(1) = {3};
    }

    void scn_do_execute() override {
        ASSERT_EQ(TensorShape{2}, output(0)->dev_tensor().shape());
        ASSERT_EQ(TensorShape{3}, output(1)->dev_tensor().shape());
        executed = true;
    }

public:
    bool executed = false;

    WorkspaceAllocTestOpr(VarNode* inp) : Super(inp->owner_graph(), {}, "test", {inp}) {
        add_input({inp});
        add_output("out")->dtype(dtype::Float32());
        cg::add_workspace_output(this);
    }
};

MGB_DEFINE_OPR_CLASS(AllInputGradOpr, cg::SingleCNOutshapePureByInshapeOprBase) // {
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override {
        out_shape.at(0) = {2};
    }

    void scn_do_execute() override {}

public:
    size_t nr_grad_call = 0;
    VarNode* prev_out_grad = nullptr;

    AllInputGradOpr(VarNode* a, VarNode* b)
            : Super(a->owner_graph(), {}, "all_inp_grad", {a, b}) {
        add_input({a, b});
        add_output(None);
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(AllInputGradOpr);
MGB_IMPL_OPR_GRAD(AllInputGradOpr) {
    auto&& ncopr = const_cast<AllInputGradOpr&>(opr);
    ncopr.prev_out_grad = out_grad[0];
    ++ncopr.nr_grad_call;
    SymbolVar x = opr.input(0), y = opr.input(1);
    if (ncopr.nr_grad_call & 1) {
        return VarNodeArray{(x + y).node(), nullptr};
    } else {
        return VarNodeArray{nullptr, (x * y).node()};
    }
}

template <bool dynamic, typename dtype>
void test_aplusb() {
    using Gen = HostTensorGenerator<dtype>;
    using ctype = typename Gen::ctype;
    Gen gen;
    constexpr size_t SIZE = 1234;
    auto host_x = gen({SIZE}), host_y = gen({SIZE});
    auto graph = ComputingGraph::make();
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
              y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y");
    if (dynamic) {
        x = opr::MarkDynamicVar::make(x).rename("xd");
        y = opr::MarkDynamicVar::make(y).rename("yd");
    }
    auto z = opr::add(x, y).rename("z");
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});

    for (ctype delta = 0; delta < 2; ++delta) {
        auto px = host_x->template ptr<ctype>();
        px[0] += delta;  // test change input data
        func->execute();
        auto py = host_y->template ptr<ctype>(), pz = host_z.template ptr<ctype>();
        ASSERT_EQ(host_x->shape(), host_z.shape());
        for (size_t i = 0; i < SIZE; ++i) {
            MGB_ASSERT_FLOAT_EQ(px[i] + py[i], pz[i])
                    << ssprintf("failed at %zu: %g+%g", i, float(px[i]), float(py[i]));
        }
    }
}

class TrackableStaticMemAlloc final : public cg::DeviceMemoryAllocator {
    SmallVector<DeviceTensorStorage> m_refhold;

public:
    size_t version_num = 0, size_expect = 0;

    void alloc_static(
            ComputingGraph*, DeviceTensorStorage& dest, size_t size) override {
        dest.ensure_size(size);
        m_refhold.emplace_back(dest);
        if (size_expect) {
            ASSERT_EQ(size_expect, size);
        }
    }

    size_t nr_call() const { return m_refhold.size(); }

    size_t static_alloc_version(ComputingGraph*) const override { return version_num; }
};

class TrackableDynamicMemAlloc final : public cg::DeviceMemoryAllocator {
    ThinHashSet<VarNode*> m_alive_vars;
    std::mutex m_mtx;

public:
    void alloc_dynamic(VarNode* var, DeviceTensorStorage& dest, size_t size) override {
        ASSERT_LT(dest.size(), size);
        MGB_LOCK_GUARD(m_mtx);
        auto ptr = dest.comp_node().alloc_device(size);
        auto ins = m_alive_vars.insert(var);
        ASSERT_TRUE(ins.second);
        auto del = [this, var, cn = dest.comp_node()](void* ptr) {
            cn.free_device(ptr);
            MGB_LOCK_GUARD(m_mtx);
            auto nr = m_alive_vars.erase(var);
            ASSERT_EQ(1u, nr);
        };
        dest.reset(dest.comp_node(), size, {static_cast<dt_byte*>(ptr), del});
    }

    const ThinHashSet<VarNode*>& alive_vars() const { return m_alive_vars; }

    ~TrackableDynamicMemAlloc() { EXPECT_TRUE(m_alive_vars.empty()); }
};

}  // anonymous namespace

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PODDedupTestOpr);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(WorkspaceAllocTestOpr);

TEST(TestGraphBasic, APlusBF32) {
    test_aplusb<false, dtype::Float32>();
}

TEST(TestGraphBasic, APlusBI32) {
    test_aplusb<false, dtype::Int32>();
}

TEST(TestGraphBasic, DynAPlusBF32) {
    test_aplusb<true, dtype::Float32>();
}

TEST(TestGraphBasic, DynAPlusBI32) {
    test_aplusb<true, dtype::Int32>();
}

TEST(TestGraph, APlusBOnCPU) {
    HostTensorGenerator<> gen;
    constexpr size_t SIZE = 1234;
    auto host_x = gen({SIZE}, "cpu0"), host_y = gen({SIZE}, "cpu0");
    auto graph = ComputingGraph::make();
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
              y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y");
    auto z = (x + y).rename("z");
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();

    ASSERT_EQ(host_x->shape(), host_z.shape());
    auto px = host_x->ptr<float>(), py = host_y->ptr<float>(), pz = host_z.ptr<float>();
    for (size_t i = 0; i < SIZE; ++i)
        MGB_ASSERT_FLOAT_EQ(px[i] + py[i], pz[i]);
}

TEST(TestGraph, DeDup) {
    auto t0 = std::make_shared<DeviceTensorND>(
                 CompNode::load("xpu0"), TensorShape{2, 2}),
         t1 = std::make_shared<DeviceTensorND>(
                 CompNode::load("xpu0"), TensorShape{2, 2}),
         t2 = std::make_shared<DeviceTensorND>(
                 CompNode::load("xpu0"), TensorShape{2, 2});
    auto graph = ComputingGraph::make();
    auto st0 = opr::SharedDeviceTensor::make(*graph, t0),
         st1 = opr::SharedDeviceTensor::make(*graph, t1);
    SymbolVar x = opr::add(st0, st1),
              y = opr::add(
                      opr::SharedDeviceTensor::make(*graph, t1),
                      opr::SharedDeviceTensor::make(*graph, t0)),
              z = opr::add(
                      opr::SharedDeviceTensor::make(*graph, t0),
                      opr::SharedDeviceTensor::make(*graph, t2));
    EXPECT_EQ(x.node(), y.node());
    EXPECT_NE(x.node(), z.node());
}

TEST(TestGraph, PODDeDup) {
    auto graph = ComputingGraph::make();
    PODDedupTestOpr::Param param{42, 'x'};
    auto var0 = PODDedupTestOpr::make(*graph, param),
         var1 = PODDedupTestOpr::make(*graph, param);
    param.v1 = 'y';
    auto var2 = PODDedupTestOpr::make(*graph, param);
    EXPECT_NE(var0.node(), var2.node());
    EXPECT_NE(var1.node(), var2.node());
    EXPECT_EQ(var0.node(), var1.node());
}

TEST(TestGraph, MultiCard) {
    auto cns = load_multiple_xpus(2);
    constexpr size_t SIZE = 123456;
    constexpr double SLEEP_TIME = 0.8, MAX_EXE_TIME = 0.5;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({SIZE}, cns[0]), host_opr1 = gen({SIZE}, cns[1]);
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    SymbolVar opr0 = opr::Host2DeviceCopy::make(*graph, host_opr0, {"opr0"}),
              opr1 = opr::Host2DeviceCopy::make(*graph, host_opr1, {"opr1"}),
              opr0_delay = opr::Sleep::make(opr0, SLEEP_TIME),
              opr1_delay = opr::Sleep::make(opr1, SLEEP_TIME),
              opr1_card0 = opr::Copy::make(
                      opr1_delay, OperatorNodeConfig{"opr1_card0"}.comp_node(
                                          cns[0].change_stream(1))),
              opr0_double = opr::add(opr0_delay, opr0_delay, {"opr0_double"}),
              sum = opr::add(
                      opr0_double, opr1_card0,
                      OperatorNodeConfig{"sum"}.comp_node(cns[0].change_stream(2))),
              sum_delay = opr::Sleep::make(sum, SLEEP_TIME);
    HostTensorND host_sum;
    auto func = graph->compile(
            {{sum_delay, [&](DeviceTensorND& s) { host_sum.copy_from(s); }}});

    RealTimer timer;
    func->execute();
    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    auto use_time = timer.get_secs();
    if (use_time >= MAX_EXE_TIME) {
        mgb_log_warn("expect time [%f < %f], got %f", use_time, MAX_EXE_TIME, use_time);
    }

    ASSERT_EQ(host_sum.layout(), host_opr0->layout());

    auto p0 = host_opr0->ptr<float>(), p1 = host_opr1->ptr<float>(),
         ps = host_sum.sync().ptr<float>();
    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    use_time = timer.get_secs();
    if (use_time <= SLEEP_TIME * 2) {
        mgb_log_warn(
                "expect time [%f > %f], got %f", use_time, SLEEP_TIME * 2, use_time);
    }
    use_time = timer.get_secs();
    if (use_time >= SLEEP_TIME * 3) {
        mgb_log_warn(
                "expect time [%f < %f], got %f", use_time, SLEEP_TIME * 3, use_time);
    }
    for (size_t i = 0; i < SIZE; i++)
        ASSERT_FLOAT_EQ(p0[i] * 2 + p1[i], ps[i]);
}

TEST(TestGraph, AsyncExec) {
    static constexpr double SLEEP_TIME = 0.1;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    auto host_x = gen({1});
    SymbolVar x0 = opr::Host2DeviceCopy::make(*graph, host_x),
              xs = opr::Sleep::make(x0, SLEEP_TIME);
    auto func = graph->compile({{xs, [](DeviceTensorND&) {}}});

    RealTimer timer;
    double t0, t1, t2, t3, t4, t5;
    t0 = timer.get_secs();
    func->execute();
    t1 = timer.get_secs();
    func->wait();
    t2 = timer.get_secs();
    func->execute();
    t3 = timer.get_secs();
    func->execute();
    t4 = timer.get_secs();
    func->wait();
    t5 = timer.get_secs();

    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    if ((t1 - t0) >= SLEEP_TIME / 2 || (t2 - t0) <= SLEEP_TIME ||
        (t3 - t2) >= SLEEP_TIME / 2 || (t4 - t2) <= SLEEP_TIME ||
        (t5 - t4) <= SLEEP_TIME / 2 || func->get_prev_exec_time() <= SLEEP_TIME ||
        func->get_prev_exec_time() >= SLEEP_TIME * 1.5) {
        mgb_log_warn(
                "time issue, pls check detail: [t0: %f, t1:%f, t2:%f, t3: %f, "
                "t4: %f, t5: %f]",
                t0, t1, t2, t3, t4, t5);
    }
}

TEST(TestGraph, VSizeTensor) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1}), host_y = gen({1});

    auto graph = ComputingGraph::make();
    auto dev_x = opr::Host2DeviceCopy::make(*graph, host_x),
         dev_y = opr::Host2DeviceCopy::make(*graph, host_y), dev_z = dev_x + dev_y;

    HostTensorND host_z;
    auto func = graph->compile(
            {{dev_z, [&](DeviceTensorND& z) { host_z.copy_from(z).sync(); }}});

    auto check = [&](size_t inp_sz) {
        *host_x = *gen({inp_sz});
        *host_y = *gen({inp_sz});
        func->execute();
        ASSERT_EQ(host_z.shape(), TensorShape({inp_sz}));
        auto px = host_x->ptr<float>(), py = host_y->ptr<float>(),
             pz = host_z.ptr<float>();
        for (size_t i = 0; i < inp_sz; i++)
            ASSERT_EQ(px[i] + py[i], pz[i]);
    };

    check(100);
    check(456);
    check(456);
    check(10);
}

TEST(TestGraph, CompileTwice) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x), y1 = x + 1, y2 = x + 2,
         z1 = opr::Copy::make(y1), z2 = opr::Copy::make(y2);
    EXPECT_TRUE(graph->var_receiver_in_current_comp_seq(y1.node()).empty());
    EXPECT_TRUE(graph->var_receiver_in_current_comp_seq(y2.node()).empty());

    HostTensorND host_z1, host_z2;
    auto func = graph->compile({make_callback_copy(z1, host_z1)});
    EXPECT_FALSE(graph->var_receiver_in_current_comp_seq(y1.node()).empty());
    EXPECT_TRUE(graph->var_receiver_in_current_comp_seq(y2.node()).empty());
    func->execute();
    EXPECT_EQ(host_x->ptr<float>()[0] + 1, host_z1.ptr<float>()[0]);
    EXPECT_FALSE(host_z2.storage().comp_node_valid());
    host_z1.ptr<float>()[0]++;

    func = graph->compile({make_callback_copy(z2, host_z2)});
    EXPECT_TRUE(graph->var_receiver_in_current_comp_seq(y1.node()).empty());
    EXPECT_FALSE(graph->var_receiver_in_current_comp_seq(y2.node()).empty());
    func->execute();
    EXPECT_NE(host_x->ptr<float>()[0] + 1, host_z1.ptr<float>()[0]);
    EXPECT_EQ(host_x->ptr<float>()[0] + 2, host_z2.ptr<float>()[0]);
}

TEST(TestGraph, MultiCNDynamicInputs) {
    auto cns = load_multiple_xpus(3);
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 2}, cns[0]), host_y = gen({5, 3}, cns[1]);
    auto graph = ComputingGraph::make();

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         xd = opr::Sleep::make(opr::MarkDynamicVar::make(x), 0.1),
         yd = opr::Sleep::make(opr::MarkDynamicVar::make(y), 0.2),
         z = opr::Concat::make({xd, yd}, 1, OperatorNodeConfig().comp_node(cns[2]));

    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    ASSERT_EQ(host_z.shape(), TensorShape({5, 5}));
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j) {
            float expect;
            if (j < 2)
                expect = *host_x->ptr<float>({i, j});
            else
                expect = *host_y->ptr<float>({i, j - 2});
            ASSERT_FLOAT_EQ(expect, *host_z.ptr<float>({i, j}));
        }
}

TEST(TestGraph, DepMapSameNode) {
    auto run = [](bool dyn) {
        auto graph = ComputingGraph::make();
        auto cn = CompNode::load("xpu0");
        auto x = SymbolVar::make_scalar(1, *graph, cn);
        if (dyn)
            x = opr::MarkDynamicVar::make(x);
        auto y = opr::Reshape::make(x, x);
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape{1}, host_y.shape());
        ASSERT_EQ(1, host_y.ptr<dt_int32>()[0]);
    };
    run(false);
    run(true);
}

TEST(TestGraph, DoubleThrowOnInit) {
    HostTensorGenerator<> gen;
    auto host_x = gen({23});
    auto graph = ComputingGraph::make();

    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    using Ad = opr::SetSubtensor::AxisIndexer;
    std::vector<Ad> axis_desc{Ad::make_index(0, x.make_scalar(0.f))};

    ASSERT_THROW(opr::SetSubtensor::make(x, x, axis_desc), MegBrainError);
    ASSERT_THROW(opr::SetSubtensor::make(x, x, axis_desc), MegBrainError);
}

TEST(TestGraph, ShapeOnlyDep) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto graph = ComputingGraph::make();

    using Ad = opr::AxisAddRemove::AxisDesc;
    bool shp_dep_exec = false;
    auto cb_set_shp_dep_exec = [&](DeviceTensorND&) { shp_dep_exec = true; };
    auto add_chk = [&](SymbolVar var) {
        return opr::CallbackInjector::make(var, cb_set_shp_dep_exec);
    };
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         xd_ = opr::MarkDynamicVar::make(x),
         xd = add_chk(x.make_scalar(0)).broadcast(opr::GetVarShape::make(xd_)),
         axadd = add_chk(opr::AxisAddRemove::make(xd, {Ad::make_add(0)})),
         y = opr::GetVarShape::make(axadd);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    TensorShape y_as_shp;
    DeviceTensorND yv{CompNode::default_cpu()};
    yv.copy_from(host_y);
    cg::copy_tensor_value_to_shape(y_as_shp, yv);
    ASSERT_EQ(TensorShape({1, 2, 3}), y_as_shp);
    ASSERT_FALSE(shp_dep_exec);
}

TEST(TestGraph, MemAllocForAsyncRead) {
    auto cns = load_multiple_xpus(2);
    auto cn1 = cns[1];
    HostTensorGenerator<> gen;
    auto host_x = gen({4, 3}, cns[0]);

    std::atomic_bool copy_issued = ATOMIC_VAR_INIT(false);

    RealTimer timer;
    auto cb_wait_copy_issue = [&](DeviceTensorND&) {
        while (!copy_issued.load())
            ;
        auto t = timer.get_secs();
        mgb_assert(t <= 0.1, "copy issue time too long: %.2f", t);
    };

    auto cb_set_copy_issue = [&](DeviceTensorND&) { copy_issued.store(true); };

    auto make_cb_async = [](SymbolVar dev, HostTensorND& host) {
        return std::make_pair(dev, [&](DeviceTensorND& d) { host.copy_from(d); });
    };

    auto graph = ComputingGraph::make();
    // disable var check to avoid stram sync
    graph->options().var_sanity_check_first_run = false;
    graph->options().seq_opt.enable_seq_comp_node_opt = false;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x), xrshp = x.reshape({6, 2}),
         xv0_static = xrshp + 3 - 2,
         xv0_after_copy = opr::CallbackInjector::make(xv0_static, cb_wait_copy_issue),
         xdyn = opr::MarkDynamicVar::make(xv0_after_copy), y0 = xdyn + 1,
         xcp_cn1 = opr::CallbackInjector::make(
                 opr::Copy::make(x, {cn1}), cb_set_copy_issue),
         y1 = xcp_cn1 + 3;

    HostTensorND host_y0, host_y1;
    auto func =
            graph->compile({make_cb_async(y0, host_y0), make_cb_async(y1, host_y1)});

    timer.reset();
    opr::Sleep::sleep(cn1, 0.2);
    func->execute().wait();
    ASSERT_EQ(x.node()->prev_dev_ptr(), xrshp.node()->prev_dev_ptr());
    ASSERT_NE(x.node()->prev_dev_ptr(), xdyn.node()->prev_dev_ptr());
    ASSERT_EQ(TensorShape({6, 2}), host_y0.shape());
    ASSERT_EQ(TensorShape({4, 3}), host_y1.shape());
    for (size_t i = 0; i < 12; ++i) {
        auto xv = host_x->ptr<float>()[i];
        MGB_ASSERT_FLOAT_EQ(xv + 2, host_y0.ptr<float>()[i]);
        MGB_ASSERT_FLOAT_EQ(xv + 3, host_y1.ptr<float>()[i]);
    }
}

TEST(TestGraph, EmptyStaticAlloc) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    for (int i = 0; i < 2; ++i) {
        auto host_x = gen({2, 3});
        auto dev_x = std::make_shared<DeviceTensorND>();
        dev_x->copy_from(*host_x);
        auto x = opr::SharedDeviceTensor::make(*graph, dev_x), y = x.reshape({6});
        auto func = graph->compile({{y, {}}});
        func->execute();
        ASSERT_EQ(dev_x->raw_ptr(), prev_dev_ptr(y));
    }
}

TEST(TestGraph, MultiOutRelease) {
    // output(0) released before output(1) started execution, while output(2) is
    // forwarded but not used
    auto cns = load_multiple_xpus(4);

    auto cn0 = cns[1], cn1 = cns[2], cn2 = cns[3];
    HostTensorGenerator<> gen;
    auto host_x = gen({6, 3}, cns[0]), host_one = gen({1}, cns[0]);
    host_one->ptr<float>()[0] = 1;
    auto graph = ComputingGraph::make();

    // disable var check to avoid stram sync
    graph->options().var_sanity_check_first_run = false;
    graph->options().async_exec_level = 0b10;

    std::atomic_bool cn0_finished{false};

    float* splt2_dev_ptr_produced = nullptr;
    DeviceTensorStorage splt2_alloc;
    splt2_alloc.comp_node(cn2.change_stream(CompNode::Stream::COPY)).ensure_size(6);

    VarNode* split_out0 = nullptr;

    auto cb_set_cn0_finish = [&](DeviceTensorND&) {
        mgb_assert(split_out0->contain_flag(VarNode::Flag::RT_FORCE_DYNAMIC_MEM_ALLOC));
        // wait for async releaser
        while (split_out0->mem_plan().valid()) {
            asm volatile("" : : : "memory");
        }
        mgb_assert(!split_out0->dev_tensor_valid());

        splt2_alloc = {};
        cn0_finished.store(true);
    };

    auto cb_wait_cn0_finish = [&](DeviceTensorND&) {
        while (!cn0_finished.load())
            ;
    };

    auto cb_record_ptr = [&](DeviceTensorND& dv) {
        splt2_dev_ptr_produced = dv.ptr<float>();
    };

    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         tmp = opr::CallbackInjector::make(
                 SymbolVar::make_scalar(
                         2.3f, *graph, cn1.change_stream(CompNode::Stream::COPY)),
                 cb_wait_cn0_finish),
         one0 = opr::Host2DeviceCopy::make(*graph, host_one, {cn0}),
         one1 = opr::Host2DeviceCopy::make(*graph, host_one, {cn1}),
         one2 = opr::Host2DeviceCopy::make(*graph, host_one, {cn2});
    set_priority(tmp, -100);
    // use Host2DeviceCopy to make constant values for multistream add
    auto splt = opr::Split::make(
            x, opr::Split::Options::make_average(0, 3),
            OperatorNodeConfig{}.comp_node_arr(
                    {cn0.change_stream(23), cn1.change_stream(23),
                     cn2.change_stream(23)}));
    HostTensorND host_y1;
    split_out0 = splt[0].node();

    auto func = graph->compile({
            {opr::add(splt[0], one0, cn0), cb_set_cn0_finish},
            {tmp, {}},
            make_callback_copy(opr::add(splt[1], one1, cn1), host_y1),
            {opr::add(splt[2], one2, cn2), {}},  // mark dynamic
            {splt[2], cb_record_ptr},
    });

    func->execute();
    func->to_json()->writeto_fpath(output_file("TestGraph.MultiOutRelease.json"));
    ASSERT_EQ(TensorShape({2, 3}), host_y1.shape());
    auto py1 = host_y1.ptr<float>(), px = host_x->ptr<float>({2});
    for (size_t i = 0; i < 6; ++i)
        MGB_ASSERT_FLOAT_EQ(px[i] + 1, py1[i]);

    ASSERT_EQ(splt2_dev_ptr_produced, splt[2].node()->prev_dev_ptr());
}

TEST(TestGraph, MemAllocForRemoteReadVars) {
    auto cn1 = CompNode::load("xpu0:1");
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 6}), host_y = gen({5, 6});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x).rename("x"),
         y = opr::Host2DeviceCopy::make_no_fwd(*graph, host_y).rename("y"),
         sum0 = (opr::Sleep::make(x, 0.2) * x + opr::Sleep::make(y, 0.2) * y)
                        .rename("sum0"),
         sum1 = opr::add(x, y, {cn1}).rename("sum1");
    HostTensorND host_sum0, host_sum1;
    auto func = graph->compile(
            {make_callback_copy(sum0, host_sum0), make_callback_copy(sum1, host_sum1)});
    func->execute();
    func->wait();
    for (bool sleep_cn1 : {false, true}) {
        host_sum0 = {};
        host_sum1 = {};
        if (sleep_cn1)
            opr::Sleep::sleep(cn1, 0.5);

        func->execute();

        auto px = host_x->ptr<float>(), py = host_y->ptr<float>(),
             ps0 = host_sum0.ptr<float>(), ps1 = host_sum1.ptr<float>();
        for (int i = 0; i < 30; ++i) {
            auto x = px[i], y = py[i];
            ASSERT_FLOAT_EQ(x * x + y * y, ps0[i]);
            ASSERT_FLOAT_EQ(x + y, ps1[i]);
        }
    }

    ASSERT_FALSE(cg::is_static_var_storage(x.node()));
    ASSERT_FALSE(cg::is_static_var_storage(y.node()));
}

TEST(TestGraph, ShapeOnlyInput) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x), y = opr::GetVarShape::make(x);
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    for (size_t sz : {1, 3, 5}) {
        *host_x = *gen({sz});
        func->execute();
        ASSERT_EQ(sz, size_t(host_y.ptr<dt_int32>()[0]));
    }
}

TEST(TestGraph, HostAndDevValueDep) {
    HostTensorGenerator<dtype::Int32> gen;
    auto host_idx = gen({1}), host_x = gen({3});
    host_idx->ptr<dt_int32>()[0] = 0;
    for (int i = 0; i < 3; ++i)
        host_x->ptr<dt_int32>()[i] = i + 1;

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         idx = opr::Host2DeviceCopy::make(*graph, host_idx).rename("idx"),
         xsub = opr::IndexAt::make(x, {{0, idx}}).rename("xsub"),
         idx2 = (idx * idx).rename("idx2"), y = (xsub + idx2).rename("y");

    set_priority(xsub, -10);
    set_priority(idx2, 10);

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    bool found = false;
    for (auto&& i : func->get_rt_static_source_deps()) {
        constexpr auto V = cg::static_infer::DepType::VALUE;
        if (i.dest == idx.node() && i.type == V) {
            found = true;
            break;
        }
    }
    ASSERT_TRUE(found);

    for (int i = 0; i < 3; ++i) {
        host_idx->ptr<dt_int32>()[0] = i;
        func->execute();
        ASSERT_EQ(i + 1 + i * i, host_y.ptr<dt_int32>()[0]) << "fail at " << i;
    }
}

TEST(TestGraph, ExtraVarDeps) {
    HostTensorND hv{dtype::Float32()};
    hv.comp_node(CompNode::load("xpu0")).resize({1}).ptr<float>()[0] = 0;
    auto dv = std::make_shared<DeviceTensorND>();
    dv->copy_from(hv);

    float cbv0 = -1, cbv1 = -1;

    auto cb0 = [&](DeviceTensorND& v) {
        cbv0 = HostTensorND().copy_from(v).sync().ptr<float>()[0];
    };
    auto cb1 = [&](DeviceTensorND& v) {
        cbv1 = HostTensorND().copy_from(v).sync().ptr<float>()[0];
    };

    auto graph = ComputingGraph::make();
    auto x = opr::SharedDeviceTensor::make(*graph, dv),
         xu = opr::AddUpdate::make(x, x.make_scalar(1.f)),
         y0 = opr::CallbackInjector::make(x, cb0),
         y1 = opr::CallbackInjector::make(xu, cb1);
    graph->options().extra_vardeps[xu.node()].push_back(y0.node());
    graph->options().extra_vardeps[xu.node()].push_back(y1.node());
    auto func = graph->compile({{xu, {}}});
    for (int i = 0; i < 3; ++i) {
        func->execute();
        MGB_ASSERT_FLOAT_EQ(i, cbv0);
        MGB_ASSERT_FLOAT_EQ(i + 1, cbv1);
    }
}

TEST(TestGraph, WorkspaceAlloc) {
    auto graph = ComputingGraph::make();
    auto x = SymbolVar::make_scalar(0, *graph, CompNode::load("xpu0"));
    auto opr = graph->insert_opr(std::make_unique<WorkspaceAllocTestOpr>(x.node()));
    ASSERT_EQ(2u, opr->output().size());
    ASSERT_EQ(TensorShape{2}, opr->output(0)->shape());
    ASSERT_EQ(TensorShape{}, opr->output(1)->shape());
    auto func = graph->compile({{opr->output(0), {}}});
    func->execute();
    ASSERT_TRUE(opr->cast_final_safe<WorkspaceAllocTestOpr>().executed);
}

TEST(TestGraph, ConstFolding) {
    auto graph = ComputingGraph::make();
    auto a = SymbolVar::make_scalar(3, *graph, CompNode::load("xpu0")),
         b = SymbolVar::make_scalar(3, *graph, CompNode::load("xpu0")), c = a + b,
         d = a + b;
    ASSERT_EQ(a.node(), b.node());
    ASSERT_EQ(c.node(), d.node());
    ASSERT_NE(a.node(), c.node());
    ASSERT_EQ(d.node()->owner_opr()->dyn_typeinfo(), opr::ImmutableTensor::typeinfo());
}

TEST(TestGraph, MergeBroadcast) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    auto graph = ComputingGraph::make();
    auto a = opr::Host2DeviceCopy::make(*graph, host_x), b = a.broadcast({1, 2}),
         c = b.broadcast({3, 4});
    ASSERT_EQ(b.node(), b.node());
    ASSERT_EQ(c.node()->shape(), TensorShape({3, 4}));
}

TEST(TestGraph, SwapTypeCvtAndBcast) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    auto graph = ComputingGraph::make();
    auto a = opr::Host2DeviceCopy::make(*graph, host_x), b = a.broadcast({1, 2}),
         c = opr::TypeCvt::make(b, dtype::Int32());
    ASSERT_EQ(b.node()->owner_opr()->dyn_typeinfo(), opr::Broadcast::typeinfo());
    ASSERT_EQ(c.node()->dtype(), dtype::Int32());
}

TEST(TestGraph, SingleGraphMultipleCompile) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::MarkDynamicVar::make(opr::Host2DeviceCopy::make(*graph, host_x)),
         y = x + 1;
    HostTensorND host_y0, host_y1, host_y_expect;
    host_y_expect.copy_from(*host_x);
    for (size_t i = 0, it = host_x->shape().total_nr_elems(); i < it; ++i)
        host_y_expect.ptr<float>()[i]++;

    auto func0 = graph->compile({make_callback_copy(y, host_y0)});
    func0->execute();
    auto func1 = graph->compile({make_callback_copy(y, host_y1)});
    func1->execute();

    ASSERT_THROW(func0->execute(), MegBrainError);

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
}

TEST(TestGraph, VarVirtualReceiverGrad) {
    HostTensorGenerator<> gen;
    constexpr size_t SIZE = 23;

    // add a virtual opr that takes (i0, i1, i2) and outputs
    // (i0^2, (i1+1)^3, (i2+2)^4)
    // in this test, i0 = i2 = x, i1 = x * .9f
    //
    // test for var multiple receivers and same input var of virtual opr

    auto graph = ComputingGraph::make();
    auto bind_vo = [&](const std::array<SymbolVar, 3>& inp,
                       const std::array<SymbolVar, 3>& out) {
        HostTensorND iv[3];
        ComputingGraph::OutputSpec outspec;
        for (int i = 0; i < 3; ++i) {
            outspec.push_back(make_callback_copy(inp[i], iv[i]));
            inp[i].rename(ssprintf("vinp%d", i));
            out[i].rename(ssprintf("vout%d", i));
        }
        graph->compile(outspec)->execute();

        auto grad = [](const VarNodeArray& inp, const VarNodeArray&, size_t idx,
                       const VarNodeArray& out_grad) {
            SymbolVar x = inp[idx], exp = x.make_scalar(float(idx + 2)),
                      gx = exp * opr::pow(x + float(idx), exp - 1.f) * out_grad[idx];
            return gx.node();
        };

        VarNodeArray vinp(3), vout(3);
        for (int i = 0; i < 3; ++i) {
            vinp[i] = inp[i].node();
            vout[i] = out[i].node();
        }
        cg::add_var_virtual_receiver(vinp, vout, grad);

        float *iptr[3], *optr[3];
        for (int i = 0; i < 3; ++i) {
            iptr[i] = iv[i].ptr<float>();
            optr[i] = out[i].node()
                              ->owner_opr()
                              ->cast_final_safe<opr::Host2DeviceCopy>()
                              .host_data()
                              ->ptr<float>();
        }
        for (size_t i = 0; i < SIZE; ++i) {
            for (int j = 0; j < 3; ++j)
                optr[j][i] = std::pow(iptr[j][i] + j, 2.0 + j);
        }
    };
    std::shared_ptr<HostTensorND> host_x = gen({SIZE}), host_vo[3], host_loss_p[5];
    for (int i = 0; i < 5; ++i) {
        if (i < 3)
            host_vo[i] = gen({SIZE});
        host_loss_p[i] = gen({SIZE});
    }

    auto mkl = [&](SymbolVar x, size_t idx) {
        return opr::Dot::make(x, opr::Host2DeviceCopy::make(*graph, host_loss_p[idx]));
    };

    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         loss = mkl(x, 3) + mkl(opr::pow(x, x.make_scalar(-1.f)), 4);
    std::array<SymbolVar, 3> vout;
    for (int i = 0; i < 3; ++i) {
        vout[i] = opr::Host2DeviceCopy::make(*graph, host_vo[i]);
        loss = loss + mkl(vout[i], i);
    }
    bind_vo({x, x * .9f, x}, vout);

    HostTensorND gx, host_loss;
    auto func = graph->compile(
            {make_callback_copy(cg::grad(loss, x), gx),
             make_callback_copy(loss, host_loss)});
    func->execute();

    auto px = host_x->ptr<float>(), pgx = gx.ptr<float>();
    float *plp[5], *pvo[3], scale[5], bias[5], exp[5];
    for (int i = 0; i < 5; ++i) {
        plp[i] = host_loss_p[i]->ptr<float>();
        scale[i] = 1;
        bias[i] = 0;
        exp[i] = 1;
        if (i < 3)
            pvo[i] = host_vo[i]->ptr<float>();
    }
    exp[0] = 2;
    scale[1] = 0.9;
    bias[1] = 1;
    exp[1] = 3;
    bias[2] = 2;
    exp[2] = 4;
    exp[4] = -1;
    float loss_expect = 0;
    for (size_t i = 0; i < SIZE; ++i) {
        float gx = 0, x = px[i];
        for (int j = 0; j < 5; ++j) {
            auto a = scale[j], b = bias[j], c = exp[j];
            // (ax + b)**c
            auto base = a * x + b;
            gx += plp[j][i] * c * a * std::pow(base, c - 1.f);
            loss_expect += plp[j][i] * std::pow(base, c);

            if (j < 3) {
                MGB_ASSERT_FLOAT_EQ(std::pow(base, c), pvo[j][i]);
            }
        }
        MGB_ASSERT_FLOAT_EQ(gx, pgx[i]);
    }
    MGB_ASSERT_FLOAT_EQ(loss_expect, host_loss.ptr<float>()[0]);
}

TEST(TestGraph, ClearDeviceMemory) {
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x), y = x + 1;
    auto func = graph->compile({{y, {}}});
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(0u, graph->clear_device_memory());
        func->execute();
        ASSERT_EQ(1u, graph->clear_device_memory());
        ASSERT_EQ(0u, graph->clear_device_memory());
    }
}

TEST(TestGraph, CopyStream) {
    REQUIRE_GPU(2);

    HostTensorGenerator<> gen;
    auto cn0 = CompNode::load("gpu0"), cn1 = CompNode::load("gpu1");
    auto host_x = gen({23}, cn0);
    auto sum_sqr = [](SymbolVar x) { return opr::reduce_sum_sqr(x, x.make_scalar(1)); };
    auto graph = ComputingGraph::make();
    graph->options().log_level = 3;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         loss0 = opr::Copy::make(
                 sum_sqr(x) + opr::reduce_sum(x, x.make_scalar(1)), cn1),
         loss1 = sum_sqr(opr::Copy::make(x, cn1)),
         gx = opr::VirtualGrad::make(loss0 + loss1, x);
    HostTensorND host_gx;
    auto func = graph->compile({make_callback_copy(gx, host_gx)});
    func->execute();
    ASSERT_EQ(host_gx.shape(), host_x->shape());
    auto px = host_x->ptr<float>(), pgx = host_gx.ptr<float>();
    for (size_t i = 0; i < 23; ++i) {
        MGB_ASSERT_FLOAT_EQ(px[i] * 4 + 1, pgx[i]);
    }

    ASSERT_EQ(int(CompNode::Stream::COPY), host_gx.comp_node().locator().stream);
}

TEST(TestGraph, DynShapeDepCrossCN) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({23}, cns[0]);
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Sleep::make(x, 0.1, {true, true}), a = opr::MarkDynamicVar::make(y),
         ao = opr::Copy::make(a, cns[1].change_stream(1)),
         b = opr::GetVarShape::make(ao, {}, cns[1]) + 1;
    graph->options().var_sanity_check_first_run = false;
    graph->options().async_exec_level |= 0b10;
    set_priority(b, -100);
    HostTensorND host_a, host_b;
    auto func = graph->compile(
            {make_callback_copy(a, host_a, false),
             make_callback_copy(b, host_b, false)});
    func->execute().wait();
    MGB_ASSERT_TENSOR_EQ(*host_x, host_a);
    ASSERT_EQ(TensorShape{1}, host_b.shape());
    ASSERT_EQ(24.f, host_b.ptr<int>()[0]);
}

namespace {
MGB_DEFINE_OPR_CLASS(CustomCopy, cg::SingleCNOperatorNodeBase) // {
    std::shared_ptr<DeviceTensorND> m_data;

    void scn_do_execute() override {
        using namespace std::literals;
        std::this_thread::sleep_for(100ms);
        m_data->copy_from(input(0)->dev_tensor());
    }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), ShapeInferDesc::make_const({}));
    }

public:
    CustomCopy(VarNode* x, std::shared_ptr<DeviceTensorND> dv)
            : Super{x->owner_graph(), {dv->comp_node()}, "d2h", {x}}, m_data(dv) {
        add_input({x});
        using F = VarNode::Flag;
        add_output(None)->add_flag(F::ALLOW_EMPTY_SHAPE).add_flag(F::VOLATILE_CONTENT);
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CustomCopy);
}  // namespace

TEST(TestGraph, DependentOnVolatileContent) {
    HostTensorGenerator<> gen;
    auto cn0 = CompNode::load("xpu0"), cn1 = cn0.change_stream(1);
    auto host_x = gen({233}, cn0);
    auto dev_y = std::make_shared<DeviceTensorND>(cn1);

    auto graph = ComputingGraph::make();
    auto x = opr::SharedDeviceTensor::make(*graph, *host_x),
         y = x.insert_single_output_opr<CustomCopy>(x.node(), dev_y),
         x_new = opr::AddUpdate::make(x, x.make_scalar(1));

    auto func = graph->compile({{y, {}}, {x_new, {}}});
    func->execute().wait();
    HostTensorND host_y;
    host_y.copy_from(*dev_y).sync();
    MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
}

namespace {
void check_wait(SymbolVar dest, SymbolVar dep) {
    if (!dep.node()) {
        ASSERT_EQ(0u, dest.node()->owner_opr()->input_waiting_spec().size());
        return;
    }
    cg::OperatorNodeBase::InputWaitingSpecElem ws;
    unpack_vector(dest.node()->owner_opr()->input_waiting_spec(), ws);
    ASSERT_EQ(ws.comp_node, dest.node()->comp_node());
    VarNode* get;
    unpack_vector(ws.dev_ready, get);
    ASSERT_EQ(dep, get);
};
}  // namespace

TEST(TestGraph, InputWaitingSpec) {
    auto cns = load_multiple_xpus(2);
    constexpr size_t SIZE = 12345;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}, cns[0]);
    auto graph = ComputingGraph::make();
    graph->options().seq_opt.enable_seq_comp_node_opt = false;  // no copy stream
    auto cn0 = cns[0], cn1 = cns[1];
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         y0 = opr::Copy::make(x, cn1), y1 = opr::Copy::make(x + 1, cn1),
         z1 = opr::Copy::make(y1 + 1, cn0), z0 = opr::Copy::make(y0 + 1, cn0);
    set_priority(y0, 5);
    set_priority(y1, 10);
    set_priority(z1, 15);
    set_priority(z0, 20);

    HostTensorND host_z0, host_z1;
    auto func = graph->compile(
            {make_callback_copy(z0, host_z0), make_callback_copy(z1, host_z1)});
    func->execute();

    auto px = host_x->ptr<float>(), pz0 = host_z0.ptr<float>(),
         pz1 = host_z1.ptr<float>();
    for (size_t i = 0; i < SIZE; ++i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + 1, pz0[i]);
        MGB_ASSERT_FLOAT_EQ(px[i] + 2, pz1[i]);
    }
    check_wait(y0, x);
    check_wait(y1, x + 1);
    check_wait(z1, y1 + 1);
    check_wait(z0, {});
}

TEST(TestGraph, InputWaitingSpecMultiOut) {
    auto cn0 = CompNode::load("xpu0:0"), cn1 = CompNode::load("xpu0:1");
    HostTensorGenerator<> gen;
    auto graph = cg::ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    graph->options().var_sanity_check_first_run = 0;
    graph->options().async_exec_level = 0b100;
    graph->options().seq_opt.enable_seq_comp_node_opt = false;
    size_t nr_out = 1024, length = 32;
    auto hv = gen({nr_out * length}, cn0);
    auto x = opr::Host2DeviceCopy::make(*graph, hv);
    auto outs = opr::Split::make(x, opr::Split::Options::make_average(0, nr_out));
    cg::ComputingGraph::OutputSpec output_spec;
    for (size_t i = 0; i < nr_out; ++i) {
        auto y = opr::Copy::make(outs[i], cn1);
        y.node()->owner_opr()->node_prop().attribute().priority = i ? nr_out - i : 0;
        output_spec.push_back({y, {}});
    }
    auto func = graph->compile(output_spec);
    func->execute().wait();

    check_wait(output_spec[0].first, outs[0]);
    check_wait(output_spec[nr_out - 1].first, outs[nr_out - 1]);
    for (size_t i = 1; i < nr_out - 1; ++i) {
        check_wait(output_spec[i].first, {});
    }
}

TEST(TestGraph, GradStaticShape) {
    for (bool enable : {false, true}) {
        auto graph = ComputingGraph::make();
        graph->options().enable_grad_var_static_reshape = enable;
        HostTensorGenerator<> gen;
        auto host_x = gen({234});
        auto x = opr::Host2DeviceCopy::make(*graph, host_x), x1 = x + 1.f,
             y = opr::MarkDynamicVar::make(x1) * x1,
             gx = cg::grad(opr::reduce_sum(y, y.make_scalar(1)), x);
        ASSERT_FALSE(cg::is_static_var_shape(y.node()));
        ASSERT_EQ(enable, cg::is_static_var_shape(gx.node()));

        HostTensorND host_gx;
        auto func = graph->compile({make_callback_copy(gx, host_gx)});
        func->execute();
        auto px = host_x->ptr<float>(), pgx = host_gx.ptr<float>();
        for (size_t i = 0; i < 234; ++i) {
            MGB_ASSERT_FLOAT_EQ(2 * (px[i] + 1), pgx[i]);
        }
    }
}

TEST(TestGraph, AllInputGrad) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2}), host_y = gen({2});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"}),
         y = opr::Host2DeviceCopy::make(*graph, host_y, {"y"}),
         z = x.insert_single_output_opr<AllInputGradOpr>(x.node(), y.node()),
         loss0 = opr::reduce_sum_sqr(z, z.make_scalar(1)),
         loss1 = opr::reduce_sum_sqr(z * 2, z.make_scalar(1));

    auto&& op = z.node()->owner_opr()->cast_final_safe<AllInputGradOpr>();
    auto grad = [](SymbolVar x, SymbolVar y) { return cg::grad(x, y, true, false); };
    auto gx0 = grad(loss0, x), gy0 = grad(loss0, y);
    ASSERT_EQ(1u, op.nr_grad_call);
    ASSERT_EQ(x + y, gx0);
    ASSERT_EQ(nullptr, gy0.node());

    auto gx1 = grad(loss1, x), gy1 = grad(loss1, y);
    ASSERT_EQ(2u, op.nr_grad_call);
    ASSERT_EQ(nullptr, gx1.node());
    ASSERT_EQ(x * y, gy1);
}

TEST(TestGraph, CPPMemLeak) {
    auto run = []() {
        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen;
        auto host_x = gen({23}, "cpux");
        auto x = opr::Host2DeviceCopy::make(*graph, host_x), y0 = x + 1.f, y1 = x + 1.f;
        ASSERT_EQ(y0, y1);  // opr dedup calls clear() in static inference
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y0, host_y)});
        func->execute();
        auto px = host_x->ptr<float>(), py = host_y.ptr<float>();
        for (size_t i = 0; i < 23; ++i) {
            MGB_ASSERT_FLOAT_EQ(px[i] + 1, py[i]);
        }
    };
    // initialize global objects
    CompNode::finalize();
    run();
    run();  // memleak should be caught by asan, if there is any
}

TEST(TestGraph, ReplaceVarHelper) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto x = opr::Host2DeviceCopy::make(*graph, gen({2})), y = x + 1.f, z = y + 1.f;
    auto x1 = opr::Host2DeviceCopy::make(*graph, gen({3}));
    SymbolVar y1, z1;
    unpack_vector(cg::replace_vars({y, z}, {{x, x1}}), y1, z1);
    ASSERT_EQ(x1 + 1.f, y1);
    ASSERT_EQ(y1 + 1.f, z1);
}

TEST(TestGraph, ReplaceVarWithDeps) {
    auto cn = CompNode::load("xpu0");
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;

    using Iter = std::pair<std::shared_ptr<DeviceTensorND>, SymbolVar>;

    auto make_iter = [&]() -> Iter {
        HostTensorND host(cn, {1});
        host.ptr<float>()[0] = 0.0;
        auto dev = opr::SharedDeviceTensor::make(*graph, host);
        auto iter = opr::AddUpdate::make(dev, dev.make_scalar(1));
        return {dev.node()
                        ->owner_opr()
                        ->cast_final_safe<opr::SharedDeviceTensor>()
                        .dev_data(),
                iter};
    };
    auto check_iter = [&](float val, const Iter& iter) {
        HostTensorND host(cn, {1});
        host.copy_from_fixlayout(*iter.first);
        host.sync();
        MGB_ASSERT_FLOAT_EQ(val, host.ptr<float>()[0]);
    };

    auto iter0 = make_iter();
    auto iter1 = make_iter();
    auto iter2 = make_iter();
    auto iter3 = make_iter();

    auto a = iter0.second + 1;
    auto b = iter1.second + 2;
    auto c = b * 5;

    graph->options().extra_vardeps[b.node()].push_back(a.node());

    auto y = cg::replace_vars(
            {c}, {{iter0.second.node(), iter2.second.node()},
                  {iter1.second.node(), iter3.second.node()}});

    ASSERT_EQ(y.size(), 1u);

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y[0], host_y)});

    func->execute();

    check_iter(0, iter0);
    check_iter(0, iter1);
    check_iter(1, iter2);
    check_iter(1, iter3);
}

TEST(TestGraph, EmptyShapeCheck) {
    auto cn = CompNode::load("xpux");
    auto graph = ComputingGraph::make();
    auto host_x = std::make_shared<HostTensorND>(cn, TensorShape{1});
    host_x->ptr<float>()[0] = 2;
    using Param = opr::CondTake::Param;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::CondTake::make(x, x, {Param::Mode::GT})[0],
         z = opr::reduce_max(y, y.make_scalar(1));
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();
    MGB_ASSERT_FLOAT_EQ(2.f, host_z.ptr<float>()[0]);

    host_x->ptr<float>()[0] = -2;
    ASSERT_THROW(
            {
                try {
                    func->execute();
                } catch (const MegBrainError& exc) {
                    std::string msg{exc.what()};
                    ASSERT_TRUE(
                            msg.find("empty input is not allowed") != std::string::npos)
                            << "bad message " << msg;
                    throw;
                }
            },
            MegBrainError);
}

TEST(TestGraph, RefCntManage) {
    HostTensorGenerator<> gen;
    auto cns = load_multiple_xpus(2);
    auto graph = ComputingGraph::make();
    auto host_x = gen({2, 3}, cns[0]), host_y = gen({1, 3}, cns[1]);
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x).rename("x"),
         y = opr::Host2DeviceCopy::make_no_fwd(*graph, host_y).rename("y"),
         x_cn1 = opr::Copy::make(x, {cns[1]}).rename("x_cn1"),
         z = (x_cn1 + y).rename("z");
    HostTensorND host_z;
    // disable comp node opt to avoid copy stream
    graph->options().seq_opt.enable_seq_comp_node_opt = false;
    graph->compile({make_callback_copy(z, host_z)})->execute();
    auto chk_dyn = [](SymbolVar var) {
        auto v = var.node();
        ASSERT_FALSE(cg::is_static_var_storage(v)) << v->name();
        ASSERT_FALSE(v->dev_tensor_valid()) << v->name();
        ASSERT_EQ(0u, v->refcnt()) << v->name();
    };

    bool cross_cn_mem_share = cns[0].mem_node() == cns[1].mem_node();

    for (auto i : {x, y, x_cn1, z}) {
        ASSERT_EQ(0u, i.node()->refcnt()) << i.node()->name();
        if (i.node() == x.node() || (cross_cn_mem_share && i.node() == x_cn1.node())) {
            chk_dyn(i);
        } else {
            ASSERT_TRUE(cg::is_static_var_storage(i.node())) << i.node()->name();
            ASSERT_TRUE(i.node()->dev_tensor_valid()) << i.node()->name();
        }
    }

    graph->options().force_dynamic_alloc = true;
    HostTensorND host_z1;
    graph->compile({make_callback_copy(z, host_z1)})->execute();
    MGB_ASSERT_TENSOR_EQ(host_z, host_z1);
    for (auto i : {x, y, x_cn1, z}) {
        chk_dyn(i);
    }

    // var with refcnt and without reader
    graph->compile({{z, {}}})->execute().wait();
    chk_dyn(z);
}

TEST(TestGraph, CompNodeFinalize) {
    for (int rec = 0; rec < 3; ++rec) {
        auto cn = CompNode::load(rec ? "cpu0" : "xpux");
        HostTensorGenerator<> gen;
        auto graph = ComputingGraph::make();
        auto host_x = gen({1}, cn), host_y = gen({1}, cn);
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = opr::Host2DeviceCopy::make(*graph, host_y), z = x + y;
        HostTensorND host_z;
        if (rec) {
            graph->options().var_sanity_check_first_run = false;
            graph->options().comp_node_seq_record_level = rec;
        }
        auto sync = (rec != 1);
        auto func = graph->compile({make_callback_copy(z, host_z, sync)});
        if (rec == 2) {
            ComputingGraph::assert_destroy(graph);
        }
        for (int i = 0; i < 5; ++i) {
            host_x->copy_from(*gen({1}, cn));
            func->execute();
            if (!sync) {
                func->wait();
            }
            MGB_ASSERT_FLOAT_EQ(
                    host_x->ptr<float>()[0] + host_y->ptr<float>()[0],
                    host_z.ptr<float>()[0]);
        }
        CompNode::finalize();
        ASSERT_THROW(func->execute(), InternalError);
    }
}

namespace {
class GraphHolder final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    std::shared_ptr<ComputingGraph> m_graph;
    int* m_del_chk;

public:
    GraphHolder(std::shared_ptr<ComputingGraph> graph, int* del_chk)
            : m_graph{std::move(graph)}, m_del_chk{del_chk} {}
    ~GraphHolder() { ++*m_del_chk; }
};
MGB_TYPEINFO_OBJ_IMPL(GraphHolder);
}  // anonymous namespace

TEST(TestGraph, CompNodeFinalizeRecursive) {
    // recursive case may occur in python
    int del_chk = 0;
    auto graph = ComputingGraph::make();
    graph->options().user_data.get_user_data_or_create<GraphHolder>([&]() {
        return std::make_shared<GraphHolder>(std::move(graph), &del_chk);
    });
    graph.reset();
    ASSERT_EQ(0, del_chk);
    CompNode::finalize();
    ASSERT_EQ(1, del_chk);
}

#if MGB_NEED_MEGDNN_ASYNC_ERROR
TEST(TestGraph, SignalCompSeqExecFinishedAsyncError) {
    REQUIRE_GPU(1);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Int32> gen;
    auto host_x = gen({10});
    auto host_y = gen({1});
    host_y->ptr<int>()[0] = 20;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto y = opr::Host2DeviceCopy::make(*graph, host_y);
    y = opr::MarkDynamicVar::make(y);
    using AIdx = opr::indexing::AxisIndexer;
    auto out1 = opr::IndexingMultiAxisVec::make({x}, {AIdx::make_index(0, y)});
    size_t exec_cnt = 0;
    auto cb = [&exec_cnt](const cg::event::CompSeqExecFinished& ev) {
        MGB_MARK_USED_VAR(ev);
        exec_cnt++;
    };
    auto handle = graph->event().register_receiver<cg::event::CompSeqExecFinished>(cb);
    auto func = graph->compile({{out1, {}}});
    for (size_t i = 0; i < 3; ++i) {
        ASSERT_THROW(
                {
                    try {
                        func->execute().wait();
                    } catch (const MegBrainError&) {
                        ASSERT_EQ(exec_cnt, i + 1);
                        throw;
                    }
                },
                MegBrainError);
    }
}

TEST(TestGraph, RecoverFromAsyncError) {
    REQUIRE_GPU(1);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Int32> gen;
    auto host_x = gen({10});
    auto host_y = gen({1});
    host_y->ptr<int>()[0] = 5;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto y = opr::Host2DeviceCopy::make(*graph, host_y);
    y = opr::MarkDynamicVar::make(y);
    using AIdx = opr::indexing::AxisIndexer;
    auto out1 = opr::IndexingMultiAxisVec::make({x}, {AIdx::make_index(0, y)});

    auto func = graph->compile({{out1, {}}});

    func->execute().wait();

    ASSERT_THROW(
            {
                try {
                    host_y->ptr<int>()[0] = 20;
                    func->execute().wait();
                } catch (const MegBrainError&) {
                    host_y->ptr<int>()[0] = 5;
                    throw;
                }
            },
            MegBrainError);

    func->execute().wait();
}

TEST(TestGraph, AsyncErrorMultiCompGraph) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Int32> gen;
    auto host_x = gen({10});
    auto host_y0 = gen({1}), host_y1 = gen({1});

    auto gen_func = [&](decltype(host_y0) host_y) {
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        auto y = opr::Host2DeviceCopy::make(*graph, host_y);
        y = opr::MarkDynamicVar::make(y);
        using AIdx = opr::indexing::AxisIndexer;
        auto out1 = opr::IndexingMultiAxisVec::make({x}, {AIdx::make_index(0, y)});
        return graph->compile({{out1, {}}});
    };

    auto func0 = gen_func(host_y0);
    auto func1 = gen_func(host_y1);

    ASSERT_THROW(
            {
                host_y0->ptr<int>()[0] = 20;
                host_y1->ptr<int>()[0] = 5;
                ASSERT_NO_THROW({
                    func0->execute();
                    func1->execute().wait();
                });
                func0->wait();
            },
            MegBrainError);

    ASSERT_NO_THROW({
        host_y0->ptr<int>()[0] = 5;
        host_y1->ptr<int>()[0] = 5;
        func0->execute().wait();
        func1->execute().wait();
    });
}
#endif

TEST(TestGraph, WaitAfterException) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Int32> gen;
    auto host_x = gen({10});
    auto host_y = gen({10});
    size_t flag;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto y = opr::Host2DeviceCopy::make(*graph, host_y);
    auto z = opr::CallbackInjector::make(x + y, [&](DeviceTensorND&) {
        mgb_throw_if(flag, MegBrainError, "throw exception after a + b.");
    });
    auto cb = [&](const cg::event::CompSeqExecFinished& ev) {
        MGB_MARK_USED_VAR(ev);
        mgb_throw_if(
                flag, MegBrainError,
                "It should not signal CompSeqExecFinished "
                "if any exception is thrown during execution.");
    };
    auto handle = graph->event().register_receiver<cg::event::CompSeqExecFinished>(cb);
    auto func = graph->compile({{z, {}}});

    flag = 1;
    ASSERT_THROW(func->execute(), MegBrainError);
    ASSERT_NO_THROW(func->wait());
    flag = 0;
    ASSERT_NO_THROW(func->execute().wait());
}

TEST(TestGraph, PauseExecEnv) {
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    graph->options().async_exec_level = 0b100;
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}, CompNode::default_cpu());
    std::atomic_bool flag0{false}, flag1{false};
    auto cb0 = [&flag0](DeviceTensorND&) {
        flag0 = true;
        while (flag0.load()) {
            std::this_thread::yield();
        }
    };
    auto cb1 = [&flag1](DeviceTensorND&) { flag1 = true; };
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::CallbackInjector::make(x, cb0),
         z = opr::CallbackInjector::make(y, cb1);
    auto func = graph->compile({{z, {}}});

    auto exec_env =
            static_cast<cg::ComputingGraphImpl*>(graph.get())->current_exec_env();
    auto worker = [&flag0, &flag1, exec_env]() {
        while (!flag0.load()) {
            std::this_thread::yield();
        }
        exec_env->pause_exec();
        flag0 = false;

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
        ASSERT_FALSE(flag1.load());
        exec_env->resume_exec();
        std::this_thread::sleep_for(100ms);
        ASSERT_TRUE(flag1.load());
    };
    std::thread worker_th{worker};

    func->execute();
    func->wait();
    worker_th.join();
}

TEST(TestGraph, CustomStaticDeviceMemoryAllocator) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = x + opr::ImmutableTensor::make(*graph, *gen({2, 1}));
    auto func = graph->compile({{y, {}}});
    auto allocator = std::make_shared<TrackableStaticMemAlloc>();
    graph->set_device_memory_allocator(allocator);

    ASSERT_EQ(0u, allocator->nr_call());
    ThinHashSet<const void*> y_addrs;
    size_t expected_nr_call = 1;
    auto check = [&]() {
        func->execute();
        y_addrs.insert(prev_dev_ptr(y));
        ASSERT_EQ(expected_nr_call, allocator->nr_call());
        ASSERT_EQ(expected_nr_call, y_addrs.size());
    };

    for (int i = 1; i < 12; ++i) {
        if (i % 3 == 0) {
            ++expected_nr_call;
            ++allocator->version_num;
        }
        check();
    }

    *host_x = *gen({1, 1023});
    ++expected_nr_call;
    check();

    *host_x = *gen({1, 2047});
    allocator->size_expect =
            func->update_static_alloc_plan_and_get_size().at(host_x->comp_node());
    ASSERT_EQ(expected_nr_call, allocator->nr_call());
    ++expected_nr_call;
    check();

    allocator->version_num = TrackableStaticMemAlloc::VERSION_INVALID;
    ASSERT_THROW(func->execute(), MegBrainError);
}

TEST(TestGraph, CustomDynamicDeviceMemoryAllocator) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    auto allocator = std::make_shared<TrackableDynamicMemAlloc>();
    SymbolVar x, xp1, y, z;
    auto cb = [&](DeviceTensorND& dv) {
        HostTensorND hv;
        hv.copy_from(dv).sync();
        ASSERT_EQ(host_x->ptr<float>()[0] + 1.f, hv.ptr<float>()[0]);
        // CallbackInjector output should reuse its input, so only one var here
        EXPECT_EQ(1u, allocator->alive_vars().count(xp1.node()));
        EXPECT_EQ(1u, allocator->alive_vars().size());
    };
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    graph->options().force_dynamic_alloc = true;

    x = opr::Host2DeviceCopy::make(*graph, host_x);
    xp1 = x + 1;
    y = opr::CallbackInjector::make(xp1, cb);
    z = y * 2;
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    graph->set_device_memory_allocator(allocator);

    ASSERT_TRUE(allocator->alive_vars().empty());
    func->execute();
    ASSERT_EQ(2.f * (host_x->ptr<float>()[0] + 1.f), host_z.ptr<float>()[0]);
    ASSERT_TRUE(allocator->alive_vars().empty());

    *host_x = *gen({1});
    func->execute();
    ASSERT_EQ(2.f * (host_x->ptr<float>()[0] + 1.f), host_z.ptr<float>()[0]);
    ASSERT_TRUE(allocator->alive_vars().empty());
}

TEST(TestGraph, ExecutionMask) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    int called = 0;
    auto cb = [&](DeviceTensorND&) { ++called; };
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::CallbackInjector::make(x, cb);
    auto exec_mask = std::make_shared<cg::ExecutionMask>(nullptr);
    exec_mask->register_to_opr(y.node()->owner_opr());
    auto func = graph->compile({{y, {}}});
    func->execute();
    ASSERT_EQ(0, called);
    exec_mask->enable(true);
    func->execute();
    ASSERT_EQ(1, called);
    func->execute();
    ASSERT_EQ(2, called);
    exec_mask->enable(false);
    func->execute();
    ASSERT_EQ(2, called);
}

TEST(TestGraph, AsyncRelease) {
    // check that async release happens before reset var mem plan (when mem plan
    // is reset, var
    HostTensorGenerator<> gen;
    auto host_x = gen({1024});
    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(*host_x).sync();
    auto cn1 = host_x->comp_node().change_stream(1);

    auto host_tshp = std::make_shared<HostTensorND>(
            host_x->comp_node(), TensorShape{2}, dtype::Int32{});
    auto set_shape = [p = host_tshp->ptr<int>()](int x) {
        p[0] = 1 << x;
        p[1] = 1 << (10 - x);
    };
    set_shape(0);

    auto graph = ComputingGraph::make();
    auto x = opr::SharedDeviceTensor::make(*graph, dev_x),
         tshp = opr::Host2DeviceCopy::make(*graph, host_tshp), x_fwd = x.reshape(tshp),
         y = opr::Sleep::make(x_fwd, 0.05, {}, cn1);
    auto func = graph->compile({{y, {}}});

    ASSERT_TRUE(cg::is_static_var_storage(x.node()));
    ASSERT_FALSE(cg::is_static_var_storage(x_fwd.node()));

    for (int i = 0; i < 3; ++i) {
        set_shape(i + 1);
        func->execute();
        ASSERT_EQ(prev_dev_ptr(x_fwd), dev_x->raw_ptr());
        ASSERT_EQ(TensorShape({2u << i, 1u << (9 - i)}), y.shape());
    }
}

TEST(TestGraph, UpdateStaticAllocPlan) {
    HostTensorGenerator<> gen;
    auto host_x = gen({3});
    auto graph = ComputingGraph::make();
    auto x = opr::Sleep::make(opr::Host2DeviceCopy::make(*graph, host_x), 0.5),
         y = x + opr::ImmutableTensor::make(*graph, *gen({1}));
    auto func = graph->compile({{y, {}}});
    func->update_static_alloc_plan_and_get_size();
    func->execute();

    *host_x = *gen({1023});
    func->execute();

    *host_x = *gen({2047});
    func->update_static_alloc_plan_and_get_size();
    func->execute();
}

TEST(TestGraph, CPUGPUHybrid) {
    REQUIRE_GPU(1);
    auto cn_gpu = CompNode::load("gpu0");
    for (auto&& cn_cpu : {CompNode::load("cpu0"), CompNode::default_cpu()}) {
        auto graph = ComputingGraph::make();
        HostTensorGenerator<> gen;
        constexpr size_t length = 23333;
        auto host_x = gen({length});
        graph->options().var_sanity_check_first_run = false;
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {cn_cpu}),
             y = opr::Sleep::make(x, 0.5) * 2, z_gpu = opr::Copy::make(y, cn_gpu) + 1,
             z = opr::Copy::make(z_gpu, cn_cpu) * 2;
        HostTensorND host_z;
        auto func = graph->compile({make_callback_copy(z, host_z)});
        func->execute();
        for (size_t i = 0; i < length; ++i) {
            MGB_ASSERT_FLOAT_EQ(
                    (host_x->ptr<float>()[i] * 2 + 1) * 2, host_z.ptr<float>()[i]);
        }
    }
}

TEST(TestGraph, In2OutOpStreamPropagate) {
    REQUIRE_GPU(1);  // seq_comp_node_opt works on comp_node with HAS_COPY_STREAM
    HostTensorGenerator<> gen;
    SmallVector<std::shared_ptr<HostTensorND>> host_v = {gen({233}), gen({23})};
    using PropType = cg::SeqCompNodeOptimizer::StreamPropType;
    for (auto type : {PropType::STRONG, PropType::WEAK})
        for (size_t idx : {0, 1}) {
            auto graph = ComputingGraph::make();
            SymbolVarArray inp(2);
            for (size_t i = 0; i < 2; ++i) {
                inp[i] = opr::Host2DeviceCopy::make(*graph, host_v[i]);
            }
            auto out = opr::VirtualDep::make(inp);
            auto&& mgr = static_cast<cg::SeqCompNodeOptimizerImpl&>(
                    graph->seq_comp_node_optimizer());
            mgr.register_stream_var(
                    inp[idx].node(), PropType{CompNode::Stream::COPY, type});
            mgr.optimize_comp_nodes({out.node()});
            ASSERT_EQ(inp[0].node()->comp_node(), out.node()->comp_node());
            auto o_stream = out.node()->comp_node().locator().stream;
            int expect = idx ? 0 : int(CompNode::Stream::COPY);
            ASSERT_EQ(o_stream, expect);
        }
}

TEST(TestGraph, OperatorNodeConfigInstanceID) {
    OperatorNodeConfig config0, config1;
    void *p0 = &config0, *p1 = &config1;
    {  // set and reset
        ASSERT_EQ(config0.instance_id(), config1.instance_id());
        config0.update_instance_id(p0);
        ASSERT_NE(config0.instance_id(), config1.instance_id());
        config0.reset_instance_id();
        ASSERT_EQ(config0.instance_id(), config1.instance_id());
    }
    {  // set to the same pointer
        config0.reset_instance_id();
        config0.update_instance_id(p1);
        config1.reset_instance_id();
        config1.update_instance_id(p1);
        ASSERT_EQ(config0.instance_id(), config1.instance_id());
    }
    {  // check update semantics
        config0.reset_instance_id();
        config0.update_instance_id(p0);
        config1.reset_instance_id();
        config1.update_instance_id(p1);
        ASSERT_NE(config0.instance_id(), config1.instance_id());
        config0.update_instance_id(p1);
        ASSERT_NE(config0.instance_id(), config1.instance_id());
    }
    {  // set in different order
        config0.reset_instance_id();
        config0.update_instance_id(p1);
        config0.update_instance_id(p0);
        config1.reset_instance_id();
        config1.update_instance_id(p0);
        config1.update_instance_id(p1);
        ASSERT_NE(config0.instance_id(), config1.instance_id());
    }
}

TEST(TestGraph, NaiveRecord2NCHW44) {
    auto cn = CompNode::load("cpu0");
    using ConvParam = megdnn::ConvBias::Param;
    ConvParam param;
    param.sparse = ConvParam::Sparse::DENSE;
    param.format = ConvParam::Format::NCHW44;
    HostTensorGenerator<> gen;
    auto host_x = gen({1, 2, 12, 12, 4}, cn), host_w = gen({2, 2, 3, 3, 4, 4}, cn),
         host_b = gen({1, 2, 1, 1, 4}, cn);

    HostTensorND host_z;
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         w = opr::Host2DeviceCopy::make(*graph, host_w),
         b = opr::Host2DeviceCopy::make(*graph, host_b),
         z = opr::ConvBiasForward::make(x, w, b, param, {});
    graph->options().comp_node_seq_record_level = 2;
    graph->options().var_sanity_check_first_run = false;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    ComputingGraph::assert_destroy(graph);
    host_x->copy_from_fixlayout(*gen(host_x->shape(), cn));
    func->execute().wait();
}

namespace {
template <typename DnnOp, typename... Args>
typename megdnn::ExecutionPolicy try_find_any_weight_preprocess_algo(
        DnnOp* dnn_op, const char* mgb_info, Maybe<bool>& found, Args&&... args) {
    if (found.valid()) {
        if (found.val()) {
            return dnn_op->execution_policy();
        } else {
            return {};
        }
    }
    for (auto&& algo :
         dnn_op->get_all_algorithms_info_safe(std::forward<Args>(args)...)) {
        dnn_op->execution_policy().algo = algo.desc;
        auto layouts =
                dnn_op->deduce_preprocessed_filter_layout(std::forward<Args>(args)...);
        if (layouts.empty())
            continue;
        bool valid = false;
        for (auto&& l : layouts) {
            if (!l.is_empty()) {
                valid = true;
                break;
            }
        }
        if (valid) {
            found.emplace(true);
            return {algo.desc, {}};
        }
    }
    found.emplace(false);
    mgb_log_warn("Can't find weight preprocess algo for op %s", mgb_info);
    return {};
}

template <typename DnnOp, typename... Args>
typename megdnn::ExecutionPolicy try_find_any_bias_preprocess_algo(
        DnnOp* dnn_op, const char* mgb_info, Maybe<bool>& found, Args&&... args) {
    if (found.valid()) {
        if (found.val()) {
            return dnn_op->execution_policy();
        } else {
            return {};
        }
    }
    for (auto&& algo :
         dnn_op->get_all_algorithms_info_safe(std::forward<Args>(args)...)) {
        dnn_op->execution_policy().algo = algo.desc;
        auto layouts =
                dnn_op->deduce_preprocessed_filter_layout(std::forward<Args>(args)...);
        if (layouts.size() <= 1)
            continue;
        bool valid = false;
        if (!layouts[1].is_empty()) {
            valid = true;
        }
        if (valid) {
            found.emplace(true);
            return {algo.desc, {}};
        }
    }
    found.emplace(false);
    mgb_log_warn("Can't find bias preprocess algo for op %s", mgb_info);
    return {};
}

void test_free_memory_in_weight_preprocess(int record_level, CompNode cn) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
#if MGB_ENABLE_JSON
    std::unique_ptr<GraphProfiler> profiler;
    if (!record_level) {
        profiler = std::make_unique<GraphProfiler>(graph.get());
    }
#endif
    graph->options().graph_opt.weight_preprocess = true;
    graph->options().comp_node_seq_record_level = record_level;
    auto sync = (record_level != 1);
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make_const(*graph, *gen(shp, cn)).rename(name);
    };
    auto x = mkvar("x", {1, 32, 16, 16});
    // ConvBias test dense
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 0;
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w1 = mkcvar("w1", {32, 32, 1, 1}), b1 = mkcvar("b1", {1, 32, 1, 1});
    auto conv1 = opr::ConvBias::make(x, w1, b1, param_conv_bias);
    Maybe<bool> wp1, wp2;
    conv1.node()->owner_opr()->cast_final_safe<opr::ConvBias>().setup_algo_chooser(
            [&](const cg::OperatorNodeBase* opr) {
                return try_find_any_weight_preprocess_algo(
                        opr->cast_final_safe<opr::ConvBias>().megdnn_opr(),
                        opr->cname(), wp1, opr->input(0)->layout(),
                        opr->input(1)->layout(), opr->input(2)->layout(),
                        TensorLayout{}, opr->output(0)->layout());
            });
    // Convolution
    opr::Convolution::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 0;
    param_conv.sparse = opr::Convolution::Param::Sparse::DENSE;
    auto w2 = mkcvar("w2", {32, 32, 1, 1});
    auto y = opr::Convolution::make(conv1, w2, param_conv);
    y.node()->owner_opr()->cast_final_safe<opr::Convolution>().setup_algo_chooser(
            [&](const cg::OperatorNodeBase* opr) {
                return try_find_any_weight_preprocess_algo(
                        opr->cast_final_safe<opr::Convolution>().megdnn_opr(),
                        opr->cname(), wp2, opr->input(0)->layout(),
                        opr->input(1)->layout(), opr->output(0)->layout());
            });

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y, sync)});
    //! flag the no need memory of var
    func->execute();
    if (!sync) {
        func->wait();
    }
    //! free the no need memory of var
    func->execute();
    if (!sync) {
        func->wait();
    }
    auto check = [&](SymbolVar v) {
        ASSERT_TRUE(v.node()->contain_flag(VarNode::Flag::MEMORY_NO_NEED));
        ASSERT_TRUE(v.node()->dev_tensor().empty());
        ASSERT_TRUE(v.node()->owner_opr()
                            ->cast_final_safe<opr::SharedDeviceTensor>()
                            .get_dev_tensor()
                            .empty());
    };
    ASSERT_TRUE(wp1.valid() && wp2.valid());
    if (wp1.val()) {
        check(w1);
    }
    if (wp2.val()) {
        check(w2);
    }
#if MGB_ENABLE_JSON
    if (profiler) {
        func->wait();
        profiler->to_json_full(func.get())
                ->writeto_fpath(output_file("weight_preprocess.json"));
    }
#endif
}
}  // anonymous namespace

TEST(TestGraph, FreeMemoryInWeightPreprocess) {
    test_free_memory_in_weight_preprocess(0, CompNode::load("xpu0"));
    megdnn::HeuristicCache::instance().clear();
}

TEST(TestGraph, RecordFreeMemoryInWeightPreprocess) {
    test_free_memory_in_weight_preprocess(1, CompNode::load("cpu0"));
    megdnn::HeuristicCache::instance().clear();
}

namespace {
MGB_DEFINE_OPR_CLASS(HostValueReader, cg::SingleCNOutshapePureByInshapeOprBase) // {
    void scn_do_execute() override {
        auto&& hv = owner_graph()->static_infer_manager().infer_value(input(0));
        MGB_MARK_USED_VAR(hv);
    }

    NodeProp* do_make_node_prop() const override {
        auto ret = Super::do_make_node_prop();
        ret->dep_map()[input(0)] = NodeProp::DepType::HOST_VALUE;
        return ret;
    }

    void get_output_var_shape(
            const TensorShapeArray&, TensorShapeArray& out_shape) const override {
        out_shape.at(0) = {};
    }

public:
    HostValueReader(VarNode* inp)
            : Super{inp->owner_graph(), {}, "host_value_reader", {inp}} {
        add_input({inp});
        using F = VarNode::Flag;
        add_output(None)->add_flag(F::ALLOW_EMPTY_SHAPE).add_flag(F::VOLATILE_CONTENT);
    }

    static SymbolVar make(SymbolVar inp) {
        return inp.node()
                ->owner_graph()
                ->insert_opr(std::make_unique<HostValueReader>(inp.node()))
                ->output(0);
    }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(HostValueReader);
}  // namespace

TEST(TestGraph, FreeMemoryInWeightPreprocessWithValueInfer) {
    HostTensorGenerator<> gen;
    CompNode cn = CompNode::load("xpux");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt.weight_preprocess = true;
    graph->options().var_sanity_check_first_run = false;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make_const(*graph, *gen(shp, cn)).rename(name);
    };
    auto x = mkvar("x", {1, 32, 16, 16});
    auto w = mkcvar("w", {32, 32, 1, 1});
    auto y = opr::Convolution::make(x, w);
    Maybe<bool> found;
    y.node()->owner_opr()->cast_final_safe<opr::Convolution>().setup_algo_chooser(
            [&](const cg::OperatorNodeBase* opr) {
                return try_find_any_weight_preprocess_algo(
                        opr->cast_final_safe<opr::Convolution>().megdnn_opr(),
                        opr->cname(), found, opr->input(0)->layout(),
                        opr->input(1)->layout(), opr->output(0)->layout());
            });
    auto reader = HostValueReader::make(w);

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y), {reader, {}}});
    func->execute();
    // FIXME: failed on second execution due to requiring host value of the empty
    // tensor which was freed in weight preprocess
    func->execute();
    ASSERT_FALSE(w.node()->contain_flag(VarNode::Flag::MEMORY_NO_NEED));
    ASSERT_FALSE(w.node()->dev_tensor().empty());
    ASSERT_FALSE(w.node()->owner_opr()
                         ->cast_final_safe<opr::SharedDeviceTensor>()
                         .get_dev_tensor()
                         .empty());
    megdnn::HeuristicCache::instance().clear();
}

TEST(TestGraph, FreeMemoryInWeightPreprocessWithMultiReader) {
    HostTensorGenerator<> gen;
    CompNode cn = CompNode::load("xpux");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt.weight_preprocess = true;
    graph->options().var_sanity_check_first_run = false;
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make_const(*graph, *gen(shp, cn)).rename(name);
    };
    auto x = mkvar("x", {1, 32, 16, 16});
    auto w = mkcvar("w", {32, 32, 1, 1});
    auto y = opr::Convolution::make(x, w);
    Maybe<bool> found;
    y.node()->owner_opr()->cast_final_safe<opr::Convolution>().setup_algo_chooser(
            [&](const cg::OperatorNodeBase* opr) {
                return try_find_any_weight_preprocess_algo(
                        opr->cast_final_safe<opr::Convolution>().megdnn_opr(),
                        opr->cname(), found, opr->input(0)->layout(),
                        opr->input(1)->layout(), opr->output(0)->layout());
            });
    auto y1 = w * 2 + 1;

    HostTensorND host_y, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y1, host_y1)});
    func->execute();
    // FIXME: failed on second execution due to calculate expression
    // (w * 2 + 1) with empty tensor
    func->execute();
    ASSERT_FALSE(w.node()->contain_flag(VarNode::Flag::MEMORY_NO_NEED));
    ASSERT_FALSE(w.node()->dev_tensor().empty());
    ASSERT_FALSE(w.node()->owner_opr()
                         ->cast_final_safe<opr::SharedDeviceTensor>()
                         .get_dev_tensor()
                         .empty());
    megdnn::HeuristicCache::instance().clear();
}

TEST(TestGraph, FreeBias) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto cn = CompNode::load("xpu0");
    graph->options().graph_opt.weight_preprocess = true;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make_const(*graph, *gen(shp, cn)).rename(name);
    };
    auto x = mkvar("x", {1, 32, 16, 16});
    // ConvBias test dense
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 0;
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w1 = mkcvar("w1", {32, 32, 1, 1}), b1 = mkcvar("b1", {1, 32, 1, 1});
    auto conv1 = opr::ConvBias::make(x, w1, b1, param_conv_bias);
    auto w2 = mkcvar("w2", {32, 32, 1, 1});
    auto conv2 = opr::ConvBias::make(conv1, w2, param_conv_bias);
    Maybe<bool> wp1;
    conv1.node()->owner_opr()->cast_final_safe<opr::ConvBias>().setup_algo_chooser(
            [&](const cg::OperatorNodeBase* opr) {
                return try_find_any_bias_preprocess_algo(
                        opr->cast_final_safe<opr::ConvBias>().megdnn_opr(),
                        opr->cname(), wp1, opr->input(0)->layout(),
                        opr->input(1)->layout(), opr->input(2)->layout(),
                        TensorLayout{}, opr->output(0)->layout());
            });

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(conv2, host_y)});
    //! flag the no need memory of var
    func->execute();
    //! free the no need memory of var
    func->execute();
    auto check = [&](SymbolVar v) {
        ASSERT_TRUE(v.node()->contain_flag(VarNode::Flag::MEMORY_NO_NEED));
        ASSERT_TRUE(v.node()->dev_tensor().empty());
        ASSERT_TRUE(v.node()->owner_opr()
                            ->cast_final_safe<opr::SharedDeviceTensor>()
                            .get_dev_tensor()
                            .empty());
    };
    ASSERT_TRUE(wp1.valid());
    if (wp1.val()) {
        check(b1);
    }
}

TEST(TestGraph, CallbackCaller) {
    using namespace opr;
    auto cns = load_multiple_xpus(3);
    constexpr size_t C1 = 20, C2 = 30, C3 = 10, C4 = 40;
    constexpr size_t N = 2, C = C1 + C2;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({N, C}, cns[0]);
    auto graph = ComputingGraph::make();
    SymbolVar opr0 = opr::Host2DeviceCopy::make(*graph, host_opr0, {"opr0"});

    auto spl0 = opr::Split::make(
            opr0, Split::Options::make_partition(opr0, 1, {C1, C2}),
            OperatorNodeConfig("split0").comp_node_arr({cns[1], cns[2]}));

    auto spl1 = opr::Split::make(
            opr0, Split::Options::make_partition(opr0, 1, {C3, C4}),
            OperatorNodeConfig("split1"));

    HostTensorND host_spl00, host_spl01, host_spl10, host_spl11;
    auto func = graph->compile(
            {make_callback_copy(spl0[0], host_spl00),
             make_callback_copy(spl0[1], host_spl01),
             make_callback_copy(spl1[0], host_spl10),
             make_callback_copy(spl1[1], host_spl11)});
    func->execute();
    auto o00 = host_spl00.ptr<float>(), o01 = host_spl01.ptr<float>(),
         o10 = host_spl10.ptr<float>(), o11 = host_spl11.ptr<float>(),
         c = host_opr0->ptr<float>();
    for (size_t i = 0, it = host_opr0->layout().total_nr_elems(); i < it; i++) {
        auto ch = i % C;
        auto n = i / C;
        if (ch < C1) {
            MGB_ASSERT_FLOAT_EQ(o00[n * C1 + ch], c[i]) << ssprintf("failed at %zd", i);
        } else {
            MGB_ASSERT_FLOAT_EQ(o01[n * C2 + ch - C1], c[i])
                    << ssprintf("failed at %zd", i);
        }
        if (ch < C3) {
            MGB_ASSERT_FLOAT_EQ(o10[n * C3 + ch], c[i]) << ssprintf("failed at %zd", i);
        } else {
            MGB_ASSERT_FLOAT_EQ(o11[n * C4 + ch - C3], c[i])
                    << ssprintf("failed at %zd", i);
        }
    }
}

TEST(TestGraph, DynamicOutput) {
    using namespace opr;
    REQUIRE_GPU(1);
    auto cn0 = CompNode::load("gpu0");
    constexpr size_t C1 = 20, C2 = 20;
    constexpr size_t C = C1 + C2;
    HostTensorGenerator<> gen;
    auto host_opr0 = gen({C}, cn0);
    auto graph = ComputingGraph::make();
    graph->options().force_output_dynamic_alloc = true;
    SymbolVar opr0 = opr::Host2DeviceCopy::make(*graph, host_opr0);

    auto spl_0 =
            opr::Split::make(opr0, Split::Options::make_partition(opr0, 0, {C1, C2}));

    auto sum = opr::add(spl_0[1], spl_0[1]);

    HostTensorND expect_sum, expect_spl_0_0, result_sum, result_spl_0_0;

    auto func1 = graph->compile(
            {make_callback_copy(sum, expect_sum),
             make_callback_copy(spl_0[0], expect_spl_0_0)});

    func1->execute().wait();

    auto func2 = graph->compile({{sum, nullptr}, {spl_0[0], nullptr}});
    auto&& dest_vars = func2->get_output_vars();

    func2->execute().wait();

    result_sum.copy_from(dest_vars[0]->dev_tensor()).sync();
    MGB_ASSERT_TENSOR_NEAR(expect_sum, result_sum, 1e-4);
    result_spl_0_0.copy_from(dest_vars[1]->dev_tensor()).sync();
    MGB_ASSERT_TENSOR_NEAR(expect_spl_0_0, result_spl_0_0, 1e-4);
}

namespace {
// used for test reset_dev_tensor_from_tensor
MGB_DEFINE_OPR_CLASS(MaybeEmptyTensorOpr, cg::SingleCNOperatorNodeBase) // {
    DeviceTensorND m_dv;

    void init_output_comp_node() override {
        output(0)->comp_node(m_dv.comp_node());
        comp_node(m_dv.comp_node());
    }

    void scn_do_execute() override { output(0)->reset_dev_tensor_from_tensor(m_dv); }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto&& mgr = owner_graph()->static_infer_manager();
        mgr.register_shape_infer(output(0), ShapeInferDesc::make_const(m_dv.shape()));
    }

public:
    MaybeEmptyTensorOpr(
            ComputingGraph& graph, const DeviceTensorND& dv,
            const OperatorNodeConfig& config)
            : Super(&graph, config, "", {}), m_dv{dv} {
        add_output(None)
                ->add_flag(cg::VarNode::Flag::NO_SYS_MEM_ALLOC)
                .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                .dtype(dv.dtype());
    }

    static SymbolVar make(
            ComputingGraph& graph, const DeviceTensorND& dv,
            const OperatorNodeConfig& config = {}) {
        return graph
                .insert_opr(std::make_unique<MaybeEmptyTensorOpr>(graph, dv, config))
                ->output(0);
    }
};

}  // anonymous namespace

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MaybeEmptyTensorOpr);

TEST(TestMemReuse, ResetEmptyDevTensor) {
    // reciver opr allow empty tensor as input
    auto allow_empty = [](const TensorShape& inp_shp) {
        HostTensorGenerator<> gen;
        auto g = ComputingGraph::make();
        auto host_x1 = gen(inp_shp), host_x2 = gen(inp_shp);
        DeviceTensorND dev_x1, dev_x2;
        dev_x1.copy_from(*host_x1), dev_x2.copy_from(*host_x2);
        auto x1 = MaybeEmptyTensorOpr::make(*g, dev_x1, {"x1"}),
             x2 = MaybeEmptyTensorOpr::make(*g, dev_x2, {"x2"}), y = x1 + x2;
        HostTensorND host_y;
        auto func = g->compile({make_callback_copy(y, host_y)});
        auto&& recv =
                x1.node()->owner_graph()->var_receiver_in_current_comp_seq(x1.node());
        ASSERT_TRUE(recv.is_empty_allowed());
        ASSERT_NO_THROW(func->execute().wait());
        if (inp_shp.is_empty()) {
            ASSERT_TRUE(host_y.empty());
            ASSERT_TRUE(host_y.shape().is_empty());
        }
    };

    // reciver opr do not allow empty tensor as input
    auto forbid_empty = [](const TensorShape& inp_shp) {
        HostTensorGenerator<> gen;
        auto g = ComputingGraph::make();
        auto host_x = gen(inp_shp);
        DeviceTensorND dev_x;
        dev_x.copy_from(*host_x);
        auto x = MaybeEmptyTensorOpr::make(*g, dev_x, {"x"}),
             y = opr::Reduce::make(x, {opr::Reduce::Mode::MAX, 0});
        HostTensorND host_y;
        auto func = g->compile({make_callback_copy(y, host_y)});
        if (inp_shp.is_empty()) {
            ASSERT_ANY_THROW(func->execute().wait());
        } else {
            ASSERT_NO_THROW(func->execute().wait());
        }
    };

    allow_empty({2, 3, 4, 5});
    allow_empty({2, 0, 3, 4});
    forbid_empty({4, 5, 6, 7});
    forbid_empty({8, 0, 0, 9});
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
