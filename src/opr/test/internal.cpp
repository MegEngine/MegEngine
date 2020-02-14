/**
 * \file src/opr/test/internal.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/opr/io.h"

#include "../impl/internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using opr::intl::WorkspaceLimitGetter;

namespace {

//! forward unchanged value and set m_infer_called flag
MGB_DEFINE_OPR_CLASS(WorkspaceLimitGetterOpr,
        cg::SingleCNOperatorNodeBase) // {
    public:
        using InferShapeCallback = thin_function<void()>;

        WorkspaceLimitGetterOpr(VarNode *inp, const InferShapeCallback &cb):
            Super{inp->owner_graph(), {}, "workspace_limit_getter_opr", {inp}},
            m_infer_shape_callback{cb}
        {
            add_input({inp});
            add_output(None);
        }

        static SymbolVar make(SymbolVar inp, const InferShapeCallback &cb) {
            return inp.insert_single_output_opr<WorkspaceLimitGetterOpr>(
                    inp.node(), cb);
        }

    private:
        InferShapeCallback m_infer_shape_callback;

        void scn_do_execute() override {
            output(0)->dev_tensor().copy_from_fixlayout(input(0)->dev_tensor());
        }

        void init_output_static_infer_desc() override {
            using namespace cg::static_infer;
            auto infer_shp = [this](TensorShape &dest, const InpVal &inp) {
                dest = inp.val.at(0).shape();
                m_infer_shape_callback();
                return true;
            };
            auto &&mgr = owner_graph()->static_infer_manager();
            auto ivar = input(0), ovar = output(0);
            auto wk_var = WorkspaceLimitGetter::register_to_graph(
                    owner_graph());
            ASSERT_NE(nullptr, wk_var);
            mgr.register_shape_infer(
                    ovar,
                    {SourceType::DEP, {
                        {ivar, DepType::SHAPE},
                        {wk_var, DepType::VALUE}
                    }, infer_shp});
        }

};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(WorkspaceLimitGetterOpr);

void run_test(bool dynamic) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    if (dynamic) {
        graph->options().force_dynamic_alloc = true;
    }

    auto x = opr::SharedDeviceTensor::make(*graph, *gen({23}));

    int infer_shape_nr_call = 0;
    auto infer_shape_callback = [&]() {
        ++ infer_shape_nr_call;
        if (infer_shape_nr_call < 3) {
            ASSERT_TRUE(WorkspaceLimitGetter::is_prealloc_run(graph.get()));
        } else {
            ASSERT_FALSE(
                    WorkspaceLimitGetter::is_prealloc_run(graph.get()));
            auto wk = WorkspaceLimitGetter::get_workspace_limit(
                    graph.get(), x.node()->comp_node(), 123);
            ASSERT_GT(wk, 0u);
            ASSERT_LE(wk, 123u);
            return;
        }
    };

    auto y = WorkspaceLimitGetterOpr::make(x, infer_shape_callback);
    ASSERT_EQ(1, infer_shape_nr_call);

    graph->compile({{x, {}}})->execute();
    ASSERT_EQ(1, infer_shape_nr_call);

    auto func1 = graph->compile({{y, {}}});
    ASSERT_EQ(1, infer_shape_nr_call);
    func1->execute();
    ASSERT_EQ(3, infer_shape_nr_call);

    func1->execute();
    ASSERT_EQ(3, infer_shape_nr_call);
}

}

TEST(TestOprInternal, WorkspaceLimitGetter) {
    run_test(false);
}

TEST(TestOprInternal, WorkspaceLimitGetterDynamic) {
    run_test(true);
}

TEST(TestOprInternal, WorkspaceLimitGetterWithoutOpt) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().seq_opt.enable_mem_reuse_alloc = false;

    auto x = opr::SharedDeviceTensor::make(*graph, *gen({23}));

    int infer_shape_nr_call = 0;
    auto infer_shape_callback = [&]() {
        ++ infer_shape_nr_call;
        ASSERT_FALSE(WorkspaceLimitGetter::is_prealloc_run(graph.get()));
        auto wk = WorkspaceLimitGetter::get_workspace_limit(
                graph.get(), x.node()->comp_node(), 123);
        ASSERT_GT(wk, 0u);
        ASSERT_LE(wk, 123u);
    };

    auto y = WorkspaceLimitGetterOpr::make(x, infer_shape_callback);
    ASSERT_EQ(1, infer_shape_nr_call);

    graph->compile({{x, {}}})->execute();
    ASSERT_EQ(1, infer_shape_nr_call);

    auto func1 = graph->compile({{y, {}}});
    ASSERT_EQ(1, infer_shape_nr_call);
    func1->execute();
    ASSERT_EQ(2, infer_shape_nr_call);

    func1->execute();
    ASSERT_EQ(2, infer_shape_nr_call);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

