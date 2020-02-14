/**
 * \file src/plugin/impl/cpu_dispatch_checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/plugin/cpu_dispatch_checker.h"
#include "megbrain/graph.h"
#include "megbrain/graph/event.h"
#include "megbrain/comp_node_env.h"

using namespace mgb;

CPUDispatchChecker::CPUDispatchChecker(cg::ComputingGraph *graph):
    PluginBase(graph)
{
    auto on_exec_start = [this](const cg::event::OprExecKernelStart &event) {
        for (auto cn: cg::get_opr_comp_node_set(event.opr)) {
            if (cn.device_type() == CompNode::DeviceType::CPU) {
                auto callback = [this, cn]() {
                    record(cn);
                };
                event.env->dispatch_on_comp_node(cn, callback);
            }
        }
    };

    auto on_exec_finish = [this](const cg::event::OprExecKernelEnd &event) {
        for (auto cn: cg::get_opr_comp_node_set(event.opr)) {
            if (cn.device_type() == CompNode::DeviceType::CPU) {
                auto callback = [this, cn, opr=event.opr]() {
                    check(cn, opr);
                };
                event.env->dispatch_on_comp_node(cn, callback);
            }
        }
    };

    auto on_subgraph_associated = [this](
            const cg::event::SubgraphAssociated &event) {
        mgb_assert(event.par_graph == m_owner_graph);
        auto sub = std::make_unique<CPUDispatchChecker>(event.sub_graph);
        sub->m_failed_oprs = m_failed_oprs;
        sub->m_failed_oprs_mtx = m_failed_oprs_mtx;
        m_sub_graph_checkers.emplace_back(std::move(sub));
    };

    add_event_handler(graph->event().register_receiver<
        cg::event::OprExecKernelStart>(on_exec_start));
    add_event_handler(graph->event().register_receiver<
        cg::event::OprExecKernelEnd>(on_exec_finish));
    add_event_handler(graph->event().register_receiver<
        cg::event::SubgraphAssociated>(on_subgraph_associated));
}

void CPUDispatchChecker::record(CompNode cn) {
    auto num = CompNodeEnv::from_comp_node(
            cn).cpu_env().dispatcher->get_nr_dispatched_tasks();

    MGB_LOCK_GUARD(m_cn2nr_task_mtx);
    m_cn2nr_task[cn] = num;
}

void CPUDispatchChecker::check(CompNode cn, cg::OperatorNodeBase *opr) {
    size_t prev, now;
    {
        MGB_LOCK_GUARD(m_cn2nr_task_mtx);
        prev = m_cn2nr_task.at(cn);
    }
    now = CompNodeEnv::from_comp_node(
            cn).cpu_env().dispatcher->get_nr_dispatched_tasks();
    if (prev == now) {
        fprintf(stderr, "operator %s{%s} does not dispatch kernel on %s\n",
                opr->cname(), opr->dyn_typeinfo()->name,
                cn.to_string().c_str());
        {
            MGB_LOCK_GUARD(*m_failed_oprs_mtx);
            m_failed_oprs->insert(opr);
        }
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
