/**
 * \file src/core/test/graph/partial_execution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph.h"

#if MGB_ENABLE_PARTIAL_EXECUTION

#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/timer.h"

using namespace mgb;

namespace mgb {
namespace cg {
class ComputingGraphImpl {
public:
    class MultiPartCompiler {
    public:
        static SmallVector<Typeinfo*> test_get_internal_opr_types();
    };
};
}  // namespace cg
}  // namespace mgb

// declare some opr types so ASSERT_OPR could work
namespace mgb {
namespace opr {
namespace {
static const SmallVector<Typeinfo*>& internal_opr_types() {
    static SmallVector<Typeinfo*> ret = cg::ComputingGraphImpl::
            MultiPartCompiler::test_get_internal_opr_types();
    return ret;
}
#define DEF(name, idx)                                                       \
    struct name {                                                            \
        static Typeinfo* typeinfo() { return internal_opr_types().at(idx); } \
    }
DEF(ShapeProvider, 0);
DEF(DeviceDataProvider, 1);
DEF(EmptyExecuteOpr, 2);
DEF(VarSinkOpr, 3);
#undef DEF
}  // anonymous namespace
}  // namespace opr
}  // namespace mgb

namespace {
ThinHashMap<Typeinfo*, size_t> get_opr_types(
        const std::unique_ptr<cg::AsyncExecutable>& func) {
    ThinHashMap<Typeinfo*, size_t> ret;
    cg::DepOprIter opr_iter{
            [&ret](cg::OperatorNodeBase* opr) { ++ret[opr->dyn_typeinfo()]; }};

    auto on_opr = [&opr_iter](cg::OperatorNodeBase* opr) {
        opr_iter.add(opr);
        return true;
    };
    func->iter_opr_seq(on_opr);
    return ret;
}
#define ASSERT_OPR(_set, _type, _num) \
    ASSERT_EQ(_num##u, _set.at(opr::_type::typeinfo()))
#define ASSERT_NO_OPR(_set, _type) \
    ASSERT_EQ(0u, _set.count(opr::_type::typeinfo()))

class TrackableDynamicMemAlloc final : public cg::DeviceMemoryAllocator {
    std::atomic_size_t m_nr_alive{0};

public:
    void alloc_dynamic(VarNode*, DeviceTensorStorage& dest,
                       size_t size) override {
        auto ptr = dest.comp_node().alloc_device(size);
        ++m_nr_alive;
        auto del = [ this, cn = dest.comp_node() ](void* ptr) {
            cn.free_device(ptr);
            --m_nr_alive;
        };
        dest.reset(dest.comp_node(), size, {static_cast<dt_byte*>(ptr), del});
    }

    size_t nr_alive() const { return m_nr_alive; }

    ~TrackableDynamicMemAlloc() { EXPECT_EQ(0u, nr_alive()); }
};

}  // anonymous namespace

TEST(TestPartialExecution, Simple) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_delta = gen({1});
    int call0 = 0, call1 = 0;
    auto make_expect = [&host_x](float delta) {
        HostTensorND hv;
        auto ptr = hv.copy_from(*host_x).ptr<float>();
        for (int i = 0; i < 6; ++i)
            ptr[i] += delta;
        return hv;
    };
    auto cb0 = [&call0, &make_expect](DeviceTensorND& dv) {
        HostTensorND hv;
        hv.copy_from(dv).sync();
        MGB_ASSERT_TENSOR_EQ(make_expect(0), hv);
        ++call0;
    };
    auto cb1 = [&call1, &make_expect](DeviceTensorND& dv) {
        HostTensorND hv;
        hv.copy_from(dv).sync();
        MGB_ASSERT_TENSOR_EQ(make_expect(1), hv);
        ++call1;
    };
    host_delta->ptr<float>()[0] = -1;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         delta = opr::Host2DeviceCopy::make(*graph, host_delta),
         y0 = opr::CallbackInjector::make(x, cb0),
         y1 = opr::CallbackInjector::make(x + delta, cb1) + delta;

    // it should execute in part2 albeit with high priority
    set_priority(delta, -100);
    HostTensorND host_y1;
    auto funcs = graph->compile_multi_part(
            {{{y0, {}}}, {make_callback_copy(y1, host_y1)}});
    ASSERT_EQ(2u, funcs.size());

    for (int i = 0; i < 4; ++i) {
        *host_x = *gen({2, 3});
        ASSERT_EQ(0, call0);
        funcs[0]->execute();
        ASSERT_TRUE(host_y1.empty());
        ASSERT_EQ(1, call0);
        ASSERT_EQ(0, call1);

        host_delta->ptr<float>()[0] = 1;
        funcs[1]->execute();
        ASSERT_EQ(1, call0);
        ASSERT_EQ(1, call1);
        MGB_ASSERT_TENSOR_EQ(make_expect(2), host_y1);

        call0 = call1 = 0;
        host_y1.resize({});
    }
}

TEST(TestPartialExecution, AddUpdate) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto dv = std::make_shared<DeviceTensorND>();
    auto hv = gen({2, 3});
    dv->copy_from(*hv);
    auto make_expect = [&hv](float delta) {
        HostTensorND ret;
        auto ptr = ret.copy_from(*hv).ptr<float>();
        for (int i = 0; i < 6; ++i)
            ptr[i] += delta;
        return ret;
    };
    auto cur_dv = [&dv]() { return HostTensorND{}.copy_from(*dv).sync(); };
    auto x = opr::SharedDeviceTensor::make(*graph, dv), y0 = x + 2.3f,
         y1 = opr::AddUpdate::make(x, x.make_scalar(-1.2f)) + 0.3f;

    HostTensorND host_y0, host_y1;
    auto funcs = graph->compile_multi_part({{make_callback_copy(y0, host_y0)},
                                            {make_callback_copy(y1, host_y1)}});

    funcs[0]->execute();
    MGB_ASSERT_TENSOR_EQ(make_expect(2.3), host_y0);
    MGB_ASSERT_TENSOR_EQ(*hv, cur_dv());

    funcs[1]->execute();
    MGB_ASSERT_TENSOR_EQ(make_expect(-1.2f), cur_dv());
    MGB_ASSERT_TENSOR_EQ(make_expect(-0.9f), host_y1);
}

TEST(TestPartialExecution, CompOrderDep) {
    constexpr float SLEEP_TIME = 0.3;
    auto graph = ComputingGraph::make();
    graph->options().var_sanity_check_first_run = false;
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto dv = std::make_shared<DeviceTensorND>();
    auto hv = gen({2, 3}, cns[0]), host_bias = gen({1}, cns[1]);
    dv->copy_from(*hv).sync();
    auto make_expect = [&hv](float delta) {
        HostTensorND ret;
        auto ptr = ret.copy_from(*hv).ptr<float>();
        for (int i = 0; i < 6; ++i)
            ptr[i] += delta;
        return ret;
    };
    auto cur_dv = [&dv]() { return HostTensorND{}.copy_from(*dv).sync(); };
    auto x = opr::SharedDeviceTensor::make(*graph, dv),
         bias = opr::Host2DeviceCopy::make(*graph, host_bias),
         y0 = opr::Copy::make(x, cns[1]) + opr::Sleep::make(bias, SLEEP_TIME),
         y1 = opr::AddUpdate::make(x, x.make_scalar(-1.2f)) + 0.3f;

    HostTensorND host_y0, host_y1;
    auto funcs =
            graph->compile_multi_part({{make_callback_copy(y0, host_y0, false)},
                                       {make_callback_copy(y1, host_y1)}});

    RealTimer timer;
    funcs[0]->execute();
    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    auto use_time = timer.get_secs();
    if (use_time >= SLEEP_TIME / 2) {
        mgb_log_warn("expect time [%f < %f], got %f", use_time, SLEEP_TIME / 2,
                     use_time);
    }
    MGB_ASSERT_TENSOR_EQ(*hv, cur_dv());
    ASSERT_EQ(hv->shape(), host_y0.shape());

    funcs[1]->execute();
    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    use_time = timer.get_secs();
    if (use_time <= SLEEP_TIME) {
        mgb_log_warn("expect time [%f > %f], got %f", use_time, SLEEP_TIME,
                     use_time);
    }
    MGB_ASSERT_TENSOR_EQ(make_expect(-1.2f), cur_dv());
    MGB_ASSERT_TENSOR_EQ(make_expect(-0.9f), host_y1);
    host_y0.sync();
    MGB_ASSERT_TENSOR_EQ(make_expect(host_bias->ptr<float>()[0]), host_y0);
}

TEST(TestPartialExecution, MultiDepType) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_y = gen({6});
    auto p0_x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         p0_y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y"),
         p0_y_shp = p0_y.symshape(), p0_z = p0_x.reshape(p0_y_shp) + p0_y,

         // host value dep
            p1_z = opr::MarkDynamicVar::make(p0_x).reshape(p0_y_shp) + p0_y,

         // shape dep
            p2_z = p0_x.reshape(p0_z.symshape()) + p0_y;

    HostTensorND host_z0, host_z1, host_z2;
    auto funcs =
            graph->compile_multi_part({{make_callback_copy(p0_z, host_z0)},
                                       {make_callback_copy(p1_z, host_z1)},
                                       {make_callback_copy(p2_z, host_z2)}});

    auto oprs_1 = get_opr_types(funcs[1]), oprs_2 = get_opr_types(funcs[2]);

    ASSERT_OPR(oprs_1, Host2DeviceCopy, 1);
    ASSERT_OPR(oprs_1, MarkDynamicVar, 1);
    ASSERT_OPR(oprs_1, DeviceDataProvider, 2);
    ASSERT_NO_OPR(oprs_1, ShapeProvider);
    ASSERT_NO_OPR(oprs_1, GetVarShape);

    ASSERT_NO_OPR(oprs_2, Host2DeviceCopy);
    ASSERT_OPR(oprs_2, GetVarShape, 1);
    ASSERT_OPR(oprs_2, DeviceDataProvider, 2);
    ASSERT_OPR(oprs_2, Reshape, 1);
    ASSERT_OPR(oprs_2, ShapeProvider, 1);

    for (size_t i = 0; i < 3; ++i) {
        funcs[0]->execute();
        auto host_z0_cp = host_z0;
        host_z0.resize({});
        ASSERT_TRUE(host_z1.empty());
        funcs[1]->execute();
        ASSERT_TRUE(host_z2.empty());
        funcs[2]->execute();
        ASSERT_TRUE(host_z0.empty());

        MGB_ASSERT_TENSOR_EQ(host_z0_cp, host_z1);
        MGB_ASSERT_TENSOR_EQ(host_z0_cp, host_z2);

        host_z1.resize({});
        host_z2.resize({});

        *host_x = *gen({i + 5, 3});
        *host_y = *gen({(i + 5) * 3});
    }
}

TEST(TestPartialExecution, InternalValue) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x), y = x + 1, z = x * 2;
    HostTensorND host_y0, host_y1, host_z;
    auto funcs = graph->compile_multi_part(
            {{make_callback_copy(y, host_y0)},
             {make_callback_copy(y, host_y1), make_callback_copy(z, host_z)}});
    funcs[0]->execute();
    ASSERT_FALSE(host_y0.empty());
    ASSERT_TRUE(host_y1.empty());
    funcs[1]->execute();
    ASSERT_FALSE(host_y1.empty());

    auto oprs_0 = get_opr_types(funcs[0]), oprs_1 = get_opr_types(funcs[1]);
    ASSERT_OPR(oprs_0, Elemwise, 1);
    ASSERT_OPR(oprs_1, Elemwise, 1);
    ASSERT_OPR(oprs_1, DeviceDataProvider, 2);

    auto px = host_x->ptr<float>(), py0 = host_y0.ptr<float>(),
         py1 = host_y1.ptr<float>(), pz = host_z.ptr<float>();
    for (size_t i = 0; i < 6; ++i) {
        auto xv = px[i];
        ASSERT_EQ(xv + 1.f, py0[i]);
        ASSERT_EQ(xv + 1.f, py1[i]);
        ASSERT_EQ(xv * 2, pz[i]);
    }
}

TEST(TestPartialExecution, ValueReuse) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_y = gen({2, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);
    HostTensorND out0, out1, out2;
    auto funcs =
            graph->compile_multi_part({{make_callback_copy(x, out0)},
                                       {make_callback_copy(x * y + 2, out1)},
                                       {make_callback_copy(y, out2)}});

    funcs[0]->execute();
    MGB_ASSERT_TENSOR_EQ(*host_x, out0);

    funcs[1]->execute();
    HostTensorND out1_expect;
    graph->compile({make_callback_copy(x * y + 2, out1_expect)})->execute();
    MGB_ASSERT_TENSOR_EQ(out1_expect, out1);
    ASSERT_TRUE(out2.empty());

    funcs[2]->execute();
    MGB_ASSERT_TENSOR_EQ(*host_y, out2);
}

TEST(TestPartialExecution, MemoryManagement) {
    auto graph = ComputingGraph::make();
    auto allocator = std::make_shared<TrackableDynamicMemAlloc>();
    graph->set_device_memory_allocator(allocator);
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto cb0 = [&](DeviceTensorND&) { ASSERT_EQ(1u, allocator->nr_alive()); };
    auto cb1 = [&](DeviceTensorND&) { ASSERT_EQ(0u, allocator->nr_alive()); };
    auto x = opr::Host2DeviceCopy::make(*graph, host_x), y = x + 1,
         z = opr::CallbackInjector::make(
                 opr::CallbackInjector::make(y, cb0) * 2, cb1);
    HostTensorND host_y, host_z;
    auto funcs = graph->compile_multi_part(
            {{make_callback_copy(y, host_y)}, {make_callback_copy(z, host_z)}});

    for (size_t i = 0; i < 3; ++i) {
        funcs[0]->execute();
        ASSERT_EQ(1u, allocator->nr_alive());
        funcs[1]->execute();
        ASSERT_EQ(0u, allocator->nr_alive());

        auto px = host_x->ptr<float>(), py = host_y.ptr<float>(),
             pz = host_z.ptr<float>();
        for (size_t i = 0, it = host_x->layout().total_nr_elems(); i < it;
             ++i) {
            ASSERT_EQ(px[i] + 1.f, py[i]);
            ASSERT_EQ((px[i] + 1.f) * 2.f, pz[i]);
        }

        *host_x = *gen({i / 2 + 4, 5});
    }
}

TEST(TestPartialExecution, MemoryManagementAbort) {
    auto graph = ComputingGraph::make();
    auto allocator = std::make_shared<TrackableDynamicMemAlloc>();
    graph->set_device_memory_allocator(allocator);
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3});
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x), y = x + 1;
    graph->options().graph_opt_level = 0;
    HostTensorND out0, out1, out2;
    auto funcs = graph->compile_multi_part({{make_callback_copy(x, out0)},
                                            {make_callback_copy(y, out1)},
                                            {make_callback_copy(y * 2, out2)}});

    funcs[0]->execute();
    ASSERT_EQ(1u, allocator->nr_alive());
    funcs[1]->execute();
    ASSERT_EQ(1u, allocator->nr_alive());

    // memory should be reclaimed when execution aborts

    *host_x = *gen({4, 5});
    funcs[0]->execute();
    ASSERT_EQ(1u, allocator->nr_alive());
    ASSERT_TRUE(out2.empty());
    funcs[1]->execute();
    ASSERT_EQ(1u, allocator->nr_alive());
    funcs[2]->execute();
    ASSERT_EQ(0u, allocator->nr_alive());

    HostTensorND out1_expect, out2_expect;
    graph->compile({make_callback_copy(y, out1_expect),
                    make_callback_copy(y * 2, out2_expect)})
            ->execute();
    MGB_ASSERT_TENSOR_EQ(*host_x, out0);
    MGB_ASSERT_TENSOR_EQ(out1_expect, out1);
    MGB_ASSERT_TENSOR_EQ(out2_expect, out2);
}

TEST(TestPartialExecution, Priority) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_y = gen({2, 3});
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         y = opr::Host2DeviceCopy::make_no_fwd(*graph, host_y), z = x + y;
    set_priority(x, 3);
    set_priority(y, -5);
    set_priority(z, -100);
    auto funcs = graph->compile_multi_part({{{x, {}}, {y, {}}}, {{z, {}}}});
    SmallVector<opr::Host2DeviceCopy*> oprs_f0;
    funcs[0]->iter_opr_seq([&](cg::OperatorNodeBase* opr) {
        if (opr->same_type<opr::VarSinkOpr>()) {
            return true;
        }
        oprs_f0.emplace_back(&opr->cast_final_safe<opr::Host2DeviceCopy>());
        return true;
    });

    int nr_dev_data = 0;
    opr::Elemwise* opr_f1 = nullptr;
    funcs[1]->iter_opr_seq([&](cg::OperatorNodeBase* opr) {
        if (opr->same_type<opr::DeviceDataProvider>()) {
            ++nr_dev_data;
            return true;
        }
        EXPECT_EQ(nullptr, opr_f1);
        opr_f1 = &opr->cast_final_safe<opr::Elemwise>();
        return true;
    });
    ASSERT_EQ(2, nr_dev_data);

    ASSERT_EQ(2u, oprs_f0.size());
    ASSERT_EQ(host_y.get(), oprs_f0[0]->host_data().get());
    ASSERT_EQ(host_x.get(), oprs_f0[1]->host_data().get());
    ASSERT_NE(nullptr, opr_f1);

    // priorities are remapped to consecutive integers
    ASSERT_EQ(-3, oprs_f0[0]->node_prop().attribute().priority);
    ASSERT_EQ(-2, oprs_f0[1]->node_prop().attribute().priority);
    ASSERT_EQ(-1, opr_f1->node_prop().attribute().priority);
}

TEST(TestPartialExecution, OrderCheck) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_y = gen({2, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);
    auto funcs =
            graph->compile_multi_part({{{x, {}}}, {{y, {}}}, {{x + y, {}}}});

    funcs[0]->execute();
    funcs[1]->execute();
    funcs[2]->execute();

    funcs[0]->execute();
    funcs[1]->execute();

    // cancel previous execution
    funcs[0]->execute();
    funcs[1]->execute();
    funcs[2]->execute();

    // order violation
    ASSERT_THROW(funcs[1]->execute(), GraphError);

    funcs[0]->execute();
    funcs[1]->execute();
    // duplicated
    ASSERT_THROW(funcs[1]->execute(), GraphError);
}

#if MGB_ENABLE_EXCEPTION
TEST(TestPartialExecution, AsyncError) {
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}), host_y = gen({2, 3});
    host_y->ptr<float>()[0] = host_x->ptr<float>()[0] + 1;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y);

    for (int i = 0; i < 2; ++i) {
        auto funcs = graph->compile_multi_part(
                {{{x, {}}}, {{opr::AssertEqual::make(x, y), {}}}, {{y, {}}}});

        funcs[0]->execute();
        funcs[1]->execute();
        funcs[2]->execute();

        if (i == 0) {
            funcs[0]->wait();
            funcs[2]->wait();
            ASSERT_THROW(funcs[1]->wait(), MegBrainError);
        } else {
            // implicit wait
            ASSERT_THROW(funcs[0]->execute(), MegBrainError);
        }
    }
}
#endif  // MGB_ENABLE_EXCEPTION

#endif  // MGB_ENABLE_PARTIAL_EXECUTION

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
