/**
 * \file src/jit/test/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

#if MGB_JIT

#include "megbrain/gopt/framework.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/rand.h"
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;
using namespace jit;

void jit::set_backend(Backend backend) {
    switch (backend) {
        case Backend::NONE:
            setenv("MGB_JIT_BACKEND", "non_exist", 1);
            return;
        case Backend::HALIDE:
            setenv("MGB_JIT_BACKEND", "HALIDE", 1);
            return;
        case Backend::NVRTC:
            setenv("MGB_JIT_BACKEND", "NVRTC", 1);
            return;
        case Backend::MLIR:
            setenv("MGB_JIT_BACKEND", "MLIR", 1);
            return;
        default:
            mgb_assert(0);
    }
}

std::vector<cg::OperatorNodeBase*> jit::get_rev_topo_order(
        SymbolVar nd, ThinHashSet<VarNode*> endpoints_set) {
    std::vector<cg::OperatorNodeBase*> topo;
    thin_function<void(VarNode*)> dfs;
    dfs = [&](VarNode* p) {
        if (endpoints_set.count(p)) {
            return;
        }
        endpoints_set.insert(p);
        for (auto i : p->owner_opr()->input()) {
            dfs(i);
        }
        topo.push_back(p->owner_opr());
    };
    dfs(nd.node());
    std::reverse(topo.begin(), topo.end());
    return topo;
}

FusionChecker& FusionChecker::disable_inp_grad() {
    for (size_t i = 0; i < m_nr_input; ++i) {
        m_disable_inp_grad.insert(i);
    }
    return *this;
}

void FusionChecker::ensure_init_graph() {
    if (m_jit_y.node())
        return;

    m_graph = ComputingGraph::make();
    SymbolVarArray inputs(m_nr_input);
    for (size_t i = 0; i < m_nr_input; ++i) {
        inputs[i] = opr::Host2DeviceCopy::make(*m_graph, m_inputs_val[i])
                            .rename(ssprintf("inp%zu", i));

        auto dt = m_idx2dtype.find(i);
        if (dt != m_idx2dtype.end()) {
            inputs[i] = opr::TypeCvt::make(inputs[i], dt->second);
        }
    }
    m_truth_y = m_exp_func(inputs);

    SymbolVar jit_y;
    if (m_direct_build) {
        auto ig_gen = std::make_unique<InternalGraphGenerator>(
                m_truth_y.node()->owner_opr());
        ThinHashSet<VarNode*> endpoints_set;
        for (size_t i = 0; i < m_nr_input; ++i) {
            endpoints_set.insert(inputs[i].node());
        }
        for (auto&& opr : get_rev_topo_order(m_truth_y, endpoints_set))
            ig_gen->add_opr(opr);
        jit_y = JITExecutor::make(ig_gen->generate(),
                                  cg::to_var_node_array(inputs));
    } else {
        ComputingGraph::Options opt;
        opt.graph_opt_level = 3;
        opt.graph_opt.jit = m_jit_level;
        unpack_vector(gopt::GraphOptimizer{}
                              .add_preset_passes(true, nullptr, &opt)
                              .apply({{m_truth_y}})
                              .endpoint_vars(),
                      jit_y);

        size_t nr_jit_opr = 0;
        cg::DepOprIter{[&nr_jit_opr, this](cg::OperatorNodeBase* opr) {
            if (opr->same_type<JITExecutor>()) {
                ++nr_jit_opr;
            } else {
                static const ThinHashSet<Typeinfo*> allowed_types{
                        opr::Host2DeviceCopy::typeinfo(),
                        opr::GetVarShape::typeinfo()};
                mgb_throw_if(m_check_opr_type &&
                                     !allowed_types.count(opr->dyn_typeinfo()),
                             InternalError,
                             "encountered non-JIT opr after fusion: %s{%s}",
                             opr->cname(), opr->dyn_typeinfo()->name);
            }
        }}
                .add(jit_y.node());
        mgb_assert(nr_jit_opr == 1);
    }

    SymbolVar loss_var0, loss_var1;
    SmallVector<std::tuple<size_t, SymbolVar, SymbolVar>> grad_vars;
    for (size_t i = 0; i < m_nr_input; ++i) {
        if (!m_disable_inp_grad.count(i)) {
            if (!loss_var1.node()) {
                auto y0 = m_truth_y.flatten(), y1 = jit_y.flatten(),
                     coeff = opr::TypeCvt::make(
                             opr::UniformRNG::make(y0.symshape()), y0.dtype());
                loss_var0 = opr::Dot::make(y0, coeff);
                loss_var1 = opr::Dot::make(y1, coeff);
            }
            grad_vars.emplace_back(i, cg::grad(loss_var0, inputs[i]),
                                   cg::grad(loss_var1, inputs[i]));
        }
    }

    m_outputs_val.resize(grad_vars.size() + 1);

    ComputingGraph::OutputSpec outspec(m_outputs_val.size() * 2);
    std::get<0>(m_outputs_val[0]) = -1;
    outspec[0] =
            make_callback_copy(m_truth_y, std::get<1>(m_outputs_val[0]), false);
    outspec[1] =
            make_callback_copy(jit_y, std::get<2>(m_outputs_val[0]), false);

    for (size_t i = 0; i < grad_vars.size(); ++i) {
        auto&& dst = m_outputs_val[i + 1];
        auto&& src = grad_vars[i];
        std::get<0>(dst) = std::get<0>(src);
        outspec[i * 2 + 2] =
                make_callback_copy(std::get<1>(src), std::get<1>(dst), false);
        outspec[i * 2 + 3] =
                make_callback_copy(std::get<2>(src), std::get<2>(dst), false);
    }

    m_func = m_graph->compile(outspec);
}

FusionChecker& FusionChecker::run(const TensorShapeArray& input_shapes) {
    if (::testing::Test::HasFailure()) {
        return *this;
    }
    mgb_assert(input_shapes.size() == m_nr_input);
    if (m_inputs_val.empty()) {
        m_inputs_val.resize(m_nr_input);
        for (size_t i = 0; i < m_nr_input; ++i) {
            m_inputs_val[i] = m_input_gen(input_shapes[i]);
        }
    } else {
        for (size_t i = 0; i < m_nr_input; ++i) {
            *m_inputs_val[i] = *m_input_gen(input_shapes[i]);
        }
    }

    ensure_init_graph();
    m_func->execute().wait();
    auto chk = [this]() {
        for (auto&& i : m_outputs_val) {
            MGB_ASSERT_TENSOR_NEAR(std::get<1>(i), std::get<2>(i), 1e-5)
                    << ssprintf("failed for input %zd", std::get<0>(i));
        }
    };
    chk();
    return *this;
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
