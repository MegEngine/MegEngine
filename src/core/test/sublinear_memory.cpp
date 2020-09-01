/**
 * \file src/core/test/sublinear_memory.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/sereg.h"
#include "megbrain/test/helper.h"

using namespace mgb;

#if MGB_ENABLE_SUBLINEAR

namespace mgb {
namespace cg {

class SeqModifierForSublinearMemory {
public:
    const CompNode::UnorderedMap<size_t>& prev_min_bottleneck();
};

class ComputingGraphImpl : public ComputingGraph {
public:
    SeqModifierForSublinearMemory& seq_modifier_for_sublinear_memory();
};

}; // namespace cg
}; // namespace mgb

namespace {

MGB_DEFINE_OPR_CLASS(SublinearBadOpr, cg::SingleCNOperatorNodeBase) // {

    bool m_flag;
    size_t m_scale;

    void scn_do_execute() override {
        mgb_assert(0);
    }

    NodeProp* do_make_node_prop() const override {
        auto prop = Super::do_make_node_prop();
        if (m_flag) {
            prop->add_flag(NodeProp::Flag::NO_AUTOMATIC_DUP);
        }
        return prop;
    }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto &&mgr = owner_graph()->static_infer_manager();
        auto infer_shape = [this](TensorShape& dst, const InpVal &inp) {
            size_t n = inp.val.at(0).shape().total_nr_elems();
            dst = TensorShape{n * m_scale};
            return true;
        };
        mgr.register_shape_infer(output(0),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
    }

    public:
        SublinearBadOpr(VarNode* inp, bool bad, size_t scale,
                OperatorNodeConfig config = {}):
            Super{inp->owner_graph(), config, "subliner_bad_op", {inp}},
            m_flag{bad}, m_scale{scale}
        {
            add_input({inp});
            add_output(None);
        }

        static SymbolVar make(SymbolVar inp, bool bad, size_t scale,
                OperatorNodeConfig config = {}) {
            return inp.node()->owner_graph()->insert_opr(
                std::make_unique<SublinearBadOpr>(inp.node(), bad, scale, config))
                ->output(0);
        }

        bool flag() const { return m_flag; }
        size_t scale() const { return m_scale; }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(SublinearBadOpr);

cg::OperatorNodeBase* bad_opr_shallow_copy(
        const serialization::OprShallowCopyContext &ctx,
        const cg::OperatorNodeBase &opr_,
        const VarNodeArray &inputs,
        const OperatorNodeConfig& config) {
    mgb_assert(inputs.size() == 1);
    auto &&opr = opr_.cast_final_safe<SublinearBadOpr>();
    return SublinearBadOpr::make(
            inputs[0], opr.flag(), opr.scale(), config).node()->owner_opr();
}

MGB_REG_OPR_SHALLOW_COPY(SublinearBadOpr, bad_opr_shallow_copy);

}; // anonymous namespace

#if MGB_CUDA
#define CHECK_REQ                                                   \
    do {                                                            \
        /* force use gpu because on CPU it is too slow */           \
        REQUIRE_GPU(1);                                             \
        if (CompNode::load("gpu0").get_mem_status_bytes().second <= \
            5ull * 1024 * 1024 * 1024) {                            \
            mgb_log_warn(                                           \
                    "test skipped due to "                          \
                    "insufficient available gpu memory");           \
            return;                                                 \
        }                                                           \
    } while (0)

TEST(TestSublinearMemory, FullConv) {
    CHECK_REQ;

    HostTensorGenerator<> gen_;
    auto gen = [&](const TensorShape& shp) { return gen_(shp, "gpu0"); };
    constexpr size_t N = 128, H = 256, W = 256;
    auto host_data = gen({N, 1, H, W});

    auto graph = ComputingGraph::make();
    SymbolVarArray params;

    auto data = opr::Host2DeviceCopy::make(*graph, host_data).rename("data"),
         out = data;
    size_t out_chl = host_data->shape(1), layer_count = 0;
    auto add_layer = [&](size_t oc, size_t h, size_t w) {
        gen_.std(sqrt(2.0 / (out_chl * h * w)));
        auto host_kern = gen({oc, out_chl, h, w});
        auto dev_kern = std::make_shared<DeviceTensorND>();
        dev_kern->copy_from(*host_kern);
        params.emplace_back(opr::SharedDeviceTensor::make(*graph, dev_kern));
        out = opr::relu(opr::Convolution::make(
                out, params.back().rename(ssprintf("param%zu", layer_count)),
                {}));
        out.rename(ssprintf("out%zu", layer_count));
        ++layer_count;
        out_chl = oc;
    };

    for (int i = 0; i < 10; ++i)
        add_layer(5, 3, 3);

    auto loss = opr::Dot::make(out.flatten(), out.flatten());
    std::vector<HostTensorND> grad_params_get(params.size());
    ComputingGraph::OutputSpec out_spec;
    for (size_t i = 0; i < params.size(); ++i) {
        out_spec.emplace_back(make_callback_copy(cg::grad(loss, params[i]),
                                                 grad_params_get[i]));
    }

    std::vector<HostTensorND> grad_params_expect(grad_params_get.size());
    for (bool sublinear : {false, true}) {
        graph->options().enable_sublinear_memory_opt = sublinear;
        auto func = graph->compile(out_spec);
        func->execute();
        if (!sublinear) {
            for (size_t i = 0; i < grad_params_get.size(); ++i)
                grad_params_expect[i].copy_from(grad_params_get[i]);
        }
    }

    for (size_t i = 0; i < grad_params_get.size(); ++i)
        MGB_ASSERT_TENSOR_NEAR(grad_params_get[i], grad_params_expect[i], 1e-3);
}

TEST(TestSublinearMemory, ConcatSplit) {
    CHECK_REQ;

    HostTensorGenerator<> gen_;
    auto gen = [&](const TensorShape& shp) { return gen_(shp, "gpu0"); };
    constexpr size_t N = 128, H = 256, W = 256;
    auto host_data = gen({N, 2, H, W});

    auto graph = ComputingGraph::make();
    SymbolVarArray params;

    auto data = opr::Host2DeviceCopy::make(*graph, host_data).rename("data"),
         out = data;
    size_t out_chl = host_data->shape(1), layer_count = 0;
    auto add_layer = [&](size_t oc, size_t h, size_t w) {
        auto prev =
                opr::Split::make(out, opr::Split::Options::make_average(1, 2));
        SymbolVarArray cur_out(2);
        size_t cur_in_chl[] = {out_chl / 2, out_chl - out_chl / 2};
        size_t cur_out_chl[] = {oc / 2, oc - oc / 2};
        for (int i = 0; i < 2; ++i) {
            gen_.std(sqrt(2.0 / (cur_in_chl[i] * h * w)));
            auto host_kern = gen({cur_out_chl[i], cur_in_chl[i], h, w});
            auto dev_kern = std::make_shared<DeviceTensorND>();
            dev_kern->copy_from(*host_kern);
            params.emplace_back(
                    opr::SharedDeviceTensor::make(*graph, dev_kern));
            cur_out[i] =
                    opr::relu(opr::Convolution::make(
                                      prev[i],
                                      params.back().rename(ssprintf(
                                              "param%zu:%d", layer_count, i)),
                                      {}))
                            .rename(ssprintf("out%zu:%d", layer_count, i));
        }
        ++layer_count;
        out_chl = oc;
        out = opr::Concat::make(cur_out, 1);
    };

    for (int i = 0; i < 10; ++i)
        add_layer(6, 3, 3);

    auto loss = opr::Dot::make(out.flatten(), out.flatten());
    std::vector<HostTensorND> grad_params_get(params.size());
    ComputingGraph::OutputSpec out_spec;
    for (size_t i = 0; i < params.size(); ++i) {
        out_spec.emplace_back(make_callback_copy(cg::grad(loss, params[i]),
                                                 grad_params_get[i]));
    }

    std::vector<HostTensorND> grad_params_expect(grad_params_get.size());
    for (bool sublinear : {false, true}) {
        graph->options().enable_sublinear_memory_opt = sublinear;
        auto func = graph->compile(out_spec);
        func->execute();
        if (!sublinear) {
            for (size_t i = 0; i < grad_params_get.size(); ++i)
                grad_params_expect[i].copy_from(grad_params_get[i]);
        }
    }

    for (size_t i = 0; i < grad_params_get.size(); ++i)
        MGB_ASSERT_TENSOR_NEAR(grad_params_get[i], grad_params_expect[i], 1e-3);
}

TEST(TestSublinearMemory, MultiOutputOpr) {
    CHECK_REQ;

    HostTensorGenerator<> gen_;
    auto gen = [&](const TensorShape& shp) { return gen_(shp, "gpu0"); };
    constexpr size_t N = 128, H = 256, W = 256;
    auto host_data = gen({N, 3, H, W});

    auto graph = ComputingGraph::make();
    SymbolVarArray params;

    auto data = opr::Host2DeviceCopy::make(*graph, host_data).rename("data"),
         out = data;
    size_t out_chl = host_data->shape(1), layer_count = 0;
    auto add_layer = [&](size_t oc, size_t h, size_t w) {
        auto prev =
                opr::Split::make(out, opr::Split::Options::make_average(1, 3));
        SymbolVarArray cur_out(3);
        size_t cur_in_chl[] = {out_chl / 3, out_chl / 3, out_chl - out_chl / 3 * 2};
        size_t cur_out_chl[] = {oc / 3, oc / 3, oc - oc / 3 * 2};
        for (int i = 0; i < 3; ++i) {
            gen_.std(sqrt(2.0 / (cur_in_chl[i] * h * w)));
            auto host_kern = gen({cur_out_chl[i], cur_in_chl[i], h, w});
            auto dev_kern = std::make_shared<DeviceTensorND>();
            dev_kern->copy_from(*host_kern);
            params.emplace_back(
                    opr::SharedDeviceTensor::make(*graph, dev_kern));
            auto f = opr::Convolution::make(
                prev[i], params.back().rename(ssprintf("param%zu:%d", layer_count, i)), {});
            if(i == 2)
                for(size_t j = 0; j < 10; ++ j)
                    f = opr::relu(f);
            cur_out[i] = f;
        }
        ++layer_count;
        out_chl = oc;
        out = opr::Concat::make(cur_out, 1);
    };

    add_layer(6, 3, 3);

    auto loss = opr::Dot::make(out.flatten(), out.flatten());
    std::vector<HostTensorND> grad_params_get(params.size());
    ComputingGraph::OutputSpec out_spec;
    for (size_t i = 0; i < params.size(); ++i) {
        out_spec.emplace_back(make_callback_copy(cg::grad(loss, params[i]),
                                                 grad_params_get[i]));
    }

    std::vector<HostTensorND> grad_params_expect(grad_params_get.size());
    for (bool sublinear : {false, true}) {
        graph->options().enable_sublinear_memory_opt = sublinear;
        auto func = graph->compile(out_spec);
        func->execute();
        if (!sublinear) {
            for (size_t i = 0; i < grad_params_get.size(); ++i)
                grad_params_expect[i].copy_from(grad_params_get[i]);
        }
    }

    for (size_t i = 0; i < grad_params_get.size(); ++i)
        MGB_ASSERT_TENSOR_NEAR(grad_params_get[i], grad_params_expect[i], 1e-3);
}

TEST(TestSublinearMemory, LongChain) {
    CHECK_REQ;

    HostTensorGenerator<> gen_;
    auto gen = [&](const TensorShape& shp) { return gen_(shp, "gpu0"); };
    constexpr size_t N = 32, C = 3, H = 224, W = 224;
    auto host_data = gen({N, C, H, W});

    auto graph = ComputingGraph::make();
    SymbolVarArray params;

    auto data = opr::Host2DeviceCopy::make(*graph, host_data).rename("data"),
         out = data;
    size_t out_chl = host_data->shape(1), layer_count = 0;
    opr::Convolution::Param conv_param;
    conv_param.pad_h = 1;
    conv_param.pad_w = 1;
    auto add_layer = [&](size_t oc, size_t h, size_t w) {
        gen_.std(sqrt(2.0 / (out_chl * h * w)));
        auto host_kern = gen({oc, out_chl, h, w});
        auto dev_kern = std::make_shared<DeviceTensorND>();
        dev_kern->copy_from(*host_kern);
        params.emplace_back(opr::SharedDeviceTensor::make(*graph, dev_kern));
        out = opr::relu(opr::Convolution::make(
                out, params.back().rename(ssprintf("param%zu", layer_count)),
                conv_param));
        out.rename(ssprintf("out%zu", layer_count));
        ++layer_count;
        out_chl = oc;
    };

    int OC[] = {1, 1, 1, 12, 1, 1, 1, 1, 15, 1};
    for (int i = 1; i <= 10; ++i) {
        for (int j = 0; j < 10; j++)
            add_layer(OC[j], 3, 3);
    }

    auto loss = opr::Dot::make(out.flatten(), out.flatten());
    std::vector<HostTensorND> grad_params_get(params.size());
    ComputingGraph::OutputSpec out_spec;

    for (int i = params.size() - 1; i >= 0; --i) {
        out_spec.emplace_back(make_callback_copy(cg::grad(loss, params[i]),
                                                 grad_params_get[i]));
    }

    std::vector<HostTensorND> grad_params_expect(grad_params_get.size());
    for (bool sublinear : {false, true}) {
        graph->options().enable_sublinear_memory_opt = sublinear;
        auto func = graph->compile(out_spec);
        func->execute();
        func->to_json()->writeto_fpath(output_file(
                ssprintf("TestSublinearMemory.LongChain%d.json", sublinear)));
        if (!sublinear) {
            for (size_t i = 0; i < grad_params_get.size(); ++i)
                grad_params_expect[i].copy_from(grad_params_get[i]);
        }
    }

    for (size_t i = 0; i < grad_params_get.size(); ++i)
        MGB_ASSERT_TENSOR_NEAR(grad_params_get[i], grad_params_expect[i], 1e-4);
}
#endif  // MGB_CUDA

TEST(TestSublinearMemory, MultiReuse) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    constexpr size_t N = 1024, NS = N * sizeof(dt_float32);
    auto host_x = gen({N}), host_y0 = gen({N * 2}), host_y1 = gen({N * 2}),
         host_z = gen({N});
    auto call_check = [&](SymbolVar val, const HostTensorND& expected) {
        auto cb = [expected](const DeviceTensorND& val) {
            HostTensorND get;
            get.copy_from(val).sync();
            MGB_ASSERT_TENSOR_EQ(expected, get);
        };
        return opr::CallbackInjector::make(val, {true, cb});
    };
    // x0 should be discarded after x2 finishes
    auto x0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
         z0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_z),
         z1 = call_check(z0, *host_z), x1 = call_check(x0, *host_x),
         x2 = call_check(x0, *host_x),
         y0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_y0),
         y01 = call_check(y0, *host_y0),
         y1 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_y1),
         y11 = call_check(y1, *host_y1), x3 = call_check(x0, *host_x);
    SymbolVar vars[] = {x0, z0, z1, x1, x2, y0, y01, y1, y11, x3};
    ComputingGraph::OutputSpec out_spec;
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        set_priority(vars[i], i);
        out_spec.push_back({vars[i], {}});
    }

    size_t alloc_size = 0;
    auto alloc_size_hdl =
            graph->event().register_receiver<cg::event::StaticMemAlloc>(
                    [&](const cg::event::StaticMemAlloc& s) {
                        if (s.comp_node.valid()) {
                            alloc_size = s.alloc_size;
                        }
                    });

    graph->options().enable_sublinear_memory_opt = true;
    auto func = graph->compile(out_spec);
    func->execute();
    ASSERT_GT(alloc_size, 0u);
    ASSERT_LT(alloc_size, NS * 2 + (NS / 2));
}

TEST(TestSublinearMemory, DynamicShape) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    constexpr size_t N = 1024, NS = N * sizeof(dt_float32);
    auto host_x = gen({N}), host_p = gen({N}), host_t = gen({N / 2 + 1, 2});
    auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x).rename("x"),
         y0 = (x + 1.f).rename("y0"), y1 = (y0 + .4f).rename("y1"),
         p = opr::Host2DeviceCopy::make_no_fwd(*graph, host_p).rename("p"),
         po0 = (p + .5f).rename("po0"), po1 = (p + .4f).rename("po1"),
         po = (po0 + po1).rename("po"), xt = (x + .5f).rename("xt"),
         xdyn = opr::MarkDynamicVar::make(xt),
         t1_shp = (opr::GetVarShape::make(xdyn) + 2).rename("t0"),
         t0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_t),
         t1 = t0.reshape(t1_shp);
    set_priority(y0, 1);
    set_priority(y1, 1);
    set_priority(p, 2);
    set_priority(po, 2);
    set_priority(xt, 3);
    set_priority(xdyn, 4);
    set_priority(t0, 5);
    HostTensorND host_y1, host_t1;

    size_t alloc_size = 0;
    auto alloc_size_hdl =
            graph->event().register_receiver<cg::event::StaticMemAlloc>(
                    [&](const cg::event::StaticMemAlloc& s) {
                        if (s.comp_node.valid()) {
                            alloc_size = s.alloc_size;
                        }
                    });

    graph->options().graph_opt_level = 0;
    graph->options().enable_sublinear_memory_opt = true;
    auto func = graph->compile({make_callback_copy(y1, host_y1),
                                {po, {}},
                                make_callback_copy(t1, host_t1)});
    func->execute().to_json()->writeto_fpath(
            output_file("TestSublinearMemory.DynamicShape.json"));
    ASSERT_GT(alloc_size, 0u);
    ASSERT_LT(alloc_size, NS * 2 + NS / 2);

    auto px = host_x->ptr<float>(), py = host_y1.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + 1.4f, py[i]);
    }
    host_t->resize({N + 2});
    MGB_ASSERT_TENSOR_EQ(*host_t, host_t1);
}

TEST(TestSublinearMemory, EmptyGraph) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().enable_sublinear_memory_opt = true;
    auto x = opr::SharedDeviceTensor::make(*graph, *gen({1}));
    auto func = graph->compile({{x, {}}});
    func->execute();
}

TEST(TestSublinearMemory, DepsInTopoSort) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    constexpr size_t N = 1024;
    auto host_x0 = gen({N}), host_x1 = gen({N}), host_x2 = gen({N}),
         host_x3 = gen({N});
    auto x0 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x0),
         x1 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x1),
         x2 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x2),
         x3 = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x3),
         x4 = opr::SharedDeviceTensor::make(*graph, *host_x0), y0 = x3 + x4,
         y1 = y0 + x2, y2 = y1 + x1, y3 = y2 + x0,
         y4 = opr::AddUpdate::make(x4, y3);
    SymbolVar vars[] = {x0, x1, x2, x3, x4, y0, y1, y2, y3, y4};
    ComputingGraph::OutputSpec out_spec;
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        set_priority(vars[i], i);
        out_spec.push_back({vars[i], {}});
    }
    graph->options().graph_opt_level = 0;
    for (bool enable_sublinear : {false, true}) {
        graph->options().enable_sublinear_memory_opt = enable_sublinear;
        auto func = graph->compile(out_spec);
        ASSERT_EQ(1u, y4.node()->owner_opr()->node_prop().dep_map().count(
                              y0.node()));
    }
}

TEST(TestSublinearMemory, BadOpr) {
   HostTensorGenerator<> gen;
   auto cn = CompNode::load("xpu0");
   constexpr size_t N = 1024, Scale = 2;
   auto host_x = gen({N}, cn);
   for (bool bad : {false, true}) {
       auto graph = ComputingGraph::make();
       auto x = opr::Host2DeviceCopy::make_no_fwd(*graph, host_x),
           bad_var = SublinearBadOpr::make(x, bad, Scale),
           y0 = opr::reduce_sum(bad_var, x.make_scalar_dt(1)),
           y1 = SublinearBadOpr::make(y0, false, N * Scale),
           y = y1 + 1,
           z = opr::reduce_max(bad_var, x.make_scalar_dt(1));
       set_priority(y0, 0);
       set_priority(y1, 1);
       set_priority(y, 2);
       set_priority(z, 3);
       graph->options().graph_opt_level = 0;
       graph->options().enable_sublinear_memory_opt = 1;
       graph->options().sublinear_mem_config.genetic_nr_iter = 50;
       auto func = graph->compile({{y, {}}, {z, {}}});
       auto&& results = static_cast<cg::ComputingGraphImpl*>(graph.get())
           ->seq_modifier_for_sublinear_memory().prev_min_bottleneck();
       // bottleneck:
       //  if bad : y = y1 + 1, bad_var should be saved to calculate
       //      z later, total memory usage is
       //      N * sclae * 2(bad_var and y1) + 1 (immutable tensor 1)
       //  else : bad_var = BadOpr(x), total memory usage is
       //      N(x) + N * scale(bad_var), bad_var would be recomputed
       //      when calculate z = reduce(bad_var)
       size_t expect = bad ? N * Scale * 2 + 1 : N * Scale + N;
       ASSERT_EQ(results.at(cn), expect * host_x->dtype().size());
       size_t nr_bad_opr = 0;
       auto count_up = [&nr_bad_opr](cg::OperatorNodeBase* op) {
           if (op->dyn_typeinfo() == SublinearBadOpr::typeinfo()) {
               ++ nr_bad_opr;
           }
           return true;
       };
       func->iter_opr_seq(count_up);
       ASSERT_EQ(nr_bad_opr, bad ? 2 : 3);
   }
}

#else
#pragma message "tests are disabled as Sublinear is not enabled."
#endif  // MGB_ENABLE_SUBLINEAR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
