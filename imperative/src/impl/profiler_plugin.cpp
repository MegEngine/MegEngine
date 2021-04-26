/**
 * \file imperative/src/impl/profiler_plugin.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/profiler_plugin.h"

#include "megbrain/graph.h"
#include "megbrain/graph/event.h"

#include "./profiler/events.h"

namespace mgb::imperative::interpreter::intl {

ProfilerPlugin::ProfilerPlugin(cg::ComputingGraph* graph): PluginBase(graph) {
    using namespace cg;
    using namespace cg::event;
    using namespace profiler;
    auto on_seq_start = [this](CompSeqExecBeforeStart const& event) {
        // reset
        mgb_assert(!event.graph->options().imperative_proxy_graph);
        if (m_opr_dict.empty() && m_var_dict.empty()) {
            init_seq(event.exec);
        }
        Profiler::record<ScopeEvent>("DispatchOprs");
        event.exec->iter_opr_seq([this](OperatorNodeBase* opr) -> bool{
            auto& opr_info = get_opr_info(opr);
            SmallVector<uint64_t> inputs;
            for (auto input: opr->input()) {
                inputs.push_back(get_var_info(input).id);
            }
            SmallVector<uint64_t> outputs;
            for (auto output: opr->output()) {
                outputs.push_back(get_var_info(output).id);
            }
            auto opr_name = opr->dyn_typeinfo()->name;
            auto copy_params = [params = opr_info.params] { return *params; };
            Profiler::record<OpDispatchEvent>(opr_info.id, opr_name, copy_params, inputs, outputs);
            for (auto output: opr->output()) {
                auto var_id = get_var_info(output).id;
                Profiler::record<TensorDeclareEvent>(var_id);
            }
            return true;
        });
        Profiler::record<ScopeFinishEvent>("DispatchOprs");
        Profiler::record<ScopeEvent>("Constants");
        for (auto&& [var, var_info]: m_var_dict) {
            if (var_info->is_const) {
                bool valid = var->dev_tensor_valid();
                auto layout = valid ? var->layout() : TensorLayout();
                Profiler::record<TensorDeclareEvent>(var_info->id);
                Profiler::record<TensorProduceEvent>(var_info->id, layout, var->comp_node(), valid ? var->dev_tensor().raw_ptr() : nullptr);
            } else {
                var_info->rt_ref_cnt = var_info->ref_cnt;
            }
        }
        Profiler::record<ScopeFinishEvent>("Constants");
    };
    auto on_opr_start = [this](OprExecStart const& event) {
        OperatorNodeBase* opr = event.opr;
        auto& opr_info = get_opr_info(opr);
        auto comp_node = opr_info.comp_node;
        auto runner = [&opr_info] {
            Profiler::record<OpExecuteEvent>(opr_info.id);
        };
        event.env->dispatch_on_comp_node(comp_node, runner);
        auto inputs = opr->input();
        for (auto&& input: inputs) {
            auto& var_info = get_var_info(input);
            auto runner = [&var_info, input] {
                auto inp_id = var_info.id;
                Profiler::record<OpInputEvent>(inp_id, input->shape());
                Profiler::record<TensorUsageEvent>(inp_id);
                Profiler::record<OpInputFinishEvent>(inp_id, input->shape());
            };
            event.env->dispatch_on_comp_node(comp_node, runner);
        }
    };
    auto on_opr_finish = [this](OprExecKernelEnd const& event) {
        OperatorNodeBase* opr = event.opr;
        auto& opr_info = get_opr_info(opr);
        auto comp_node = opr_info.comp_node;
        auto inputs = opr->input();
        auto outputs = opr->output();
        for (auto input: inputs) {
            auto& var_info = get_var_info(input);
            auto runner = [&var_info] {
                if (!var_info.is_const) {
                    if (--var_info.rt_ref_cnt == 0) {
                        Profiler::record<TensorReleaseEvent>(var_info.id);
                    }
                }
            };
            event.env->dispatch_on_comp_node(comp_node, runner);
        }
        for (auto output: outputs) {
            auto& var_info = get_var_info(output);
            mgb_assert(comp_node == output->comp_node(), "opr comp_node mismatch");
            auto runner = [&var_info, output] {
                auto out_id = var_info.id;
                bool valid = output->dev_tensor_valid();
                auto layout = valid ? output->layout() : TensorLayout();
                Profiler::record<OpOutputEvent>(out_id, output->shape());
                Profiler::record<TensorProduceEvent>(out_id, layout, output->comp_node(), valid ? output->dev_tensor().raw_ptr() : nullptr);
                if (!var_info.ref_cnt) {
                    Profiler::record<TensorReleaseEvent>(var_info.id);
                }
                Profiler::record<OpOutputFinishEvent>(out_id, output->shape());
            };
            event.env->dispatch_on_comp_node(comp_node, runner);
        }
        auto runner = [&opr_info]() {
            Profiler::record<OpExecuteFinishEvent>(opr_info.id);
        };
        event.env->dispatch_on_comp_node(comp_node, runner);
    };
    auto on_before_kern = [this](BeforeKernel const& event) {
        OperatorNodeBase* opr = event.opr;
        Profiler::record<KernelExecuteEvent>(get_opr_info(opr).id, get_opr_info(opr).id, Timer::record_event(event.comp_node));
    };
    auto on_after_kern = [this](AfterKernel const& event) {
        OperatorNodeBase* opr = event.opr;
        Profiler::record<KernelExecuteFinishEvent>(get_opr_info(opr).id, get_opr_info(opr).id, Timer::record_event(event.comp_node));
    };
    auto on_graph_compile = [this](const CompSeqOrderDetermined&) {
        m_opr_dict.clear();
        m_var_dict.clear();
    };
    auto on_seq_finish = [this](CompSeqExecFinished const& event) {
        for (auto&& [var, var_info]: m_var_dict) {
            MGB_MARK_USED_VAR(var);
            if (var_info->is_const) {
                Profiler::record<TensorReleaseEvent>(var_info->id);
            }
            Profiler::record<TensorEraseEvent>(var_info->id, var_info->ref_cnt);
        }
    };
    add_event_handler(graph->event().register_receiver<CompSeqExecBeforeStart>(on_seq_start));
    add_event_handler(graph->event().register_receiver<OprExecStart>(on_opr_start));
    add_event_handler(graph->event().register_receiver<OprExecKernelEnd>(on_opr_finish));
    add_event_handler(graph->event().register_receiver<BeforeKernel>(on_before_kern));
    add_event_handler(graph->event().register_receiver<AfterKernel>(on_after_kern));
    add_event_handler(graph->event().register_receiver<CompSeqOrderDetermined>(on_graph_compile));
    add_event_handler(graph->event().register_receiver<CompSeqExecFinished>(on_seq_finish));
}

void ProfilerPlugin::init_seq(cg::AsyncExecutable *comp_seq) {
    mgb_assert(m_opr_dict.empty());
    mgb_assert(m_var_dict.empty());
    comp_seq->iter_opr_seq([this](cg::OperatorNodeBase* opr){
        auto comp_nodes = get_opr_comp_node_set(opr);
        mgb_assert(comp_nodes.size() == 1);
        register_opr(opr);
        for (auto&& input: opr->input()) {
            if (m_var_dict.count(input) == 0) {
                register_var(input).is_const = true;
            } else {
                get_var_info(input).ref_cnt++;
            }
        }
        for (auto&& output: opr->output()) {
            register_var(output).is_const = false;
        }
        //TODO: check ref_cnt
        return true;
    });
}

ProfilerPlugin::OprInfo& ProfilerPlugin::register_opr(cg::OperatorNodeBase *opr) {
    OprInfo info;
    info.id = Profiler::next_id();
    auto params = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto params_json = opr->to_json();
    for (auto&& [k, v]: params_json->cast_final<json::Object>().get_impl()) {
        params->insert({k.get_impl(), v->to_string()});
    }
    info.params = std::move(params);
    auto comp_nodes = cg::get_opr_comp_node_set(opr);
    mgb_assert(comp_nodes.size() == 1, "only support single comp_node opr");
    info.comp_node = *comp_nodes.begin();
    return m_opr_dict.insert({opr, info}).first->second;
}

ProfilerPlugin::VarInfo& ProfilerPlugin::register_var(cg::VarNode *var) {
    auto info = std::make_unique<VarInfo>();
    info->id = Profiler::next_id();
    info->is_const = false;
    info->ref_cnt = 0;
    info->rt_ref_cnt = 0;
    return *m_var_dict.insert({var, std::move(info)}).first->second;
}

ProfilerPlugin::OprInfo& ProfilerPlugin::get_opr_info(cg::OperatorNodeBase *opr) {
    return m_opr_dict.at(opr);
}

ProfilerPlugin::VarInfo& ProfilerPlugin::get_var_info(cg::VarNode *var) {
    return *m_var_dict.at(var);
}

}
