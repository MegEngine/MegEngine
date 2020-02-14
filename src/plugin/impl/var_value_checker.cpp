/**
 * \file src/plugin/impl/var_value_checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/plugin/var_value_checker.h"
#include "megbrain/opr/io.h"

using namespace mgb;

void VarValueChecker::Checker::reset() {
    m_func.reset();
}

void VarValueChecker::Checker::init(VarNode *var,
        const std::shared_ptr<DeviceTensorND> &expected) {
    if (!m_inp) {
        m_inp = std::make_shared<DeviceTensorND>();
    }
    setup_inp(var);
    auto graph = ComputingGraph::make();
    auto ex = opr::SharedDeviceTensor::make(*graph, expected, {"expected"}),
         get = opr::SharedDeviceTensor::make(*graph, m_inp, {
                 ssprintf("get:%s", cg::dump_var_info({var}).c_str())}),
         out = opr::AssertEqual::make(ex, get, {false});
    m_func = graph->compile({{out, {}}});
}

void VarValueChecker::Checker::check(VarNode *var) {
    setup_inp(var);
    m_func->clear_device_memory(); // because input dev tensor always changes
    m_func->execute().wait();
}

void VarValueChecker::Checker::setup_inp(VarNode *var) {
    auto &&val = var->dev_tensor();
    if (val.layout().is_contiguous()) {
        *m_inp = var->dev_tensor();
    } else {
        *m_inp = {};
        m_inp->copy_from(val);
    }
}

VarValueChecker::VarValueChecker(
        ComputingGraph *graph,
        size_t var_switch_interval, size_t init_var_idx):
    PluginBase(graph),
    m_init_var_idx{init_var_idx}, m_var_switch_interval{var_switch_interval}
{
    add_member_func_as_event_handler(
            &VarValueChecker::on_comp_seq_order_determined);
    add_member_func_as_event_handler(
            &VarValueChecker::on_opr_kern_end);
    add_member_func_as_event_handler(
            &VarValueChecker::on_comp_seq_exec_finished);
}

void VarValueChecker::on_comp_seq_order_determined(
        const cg::event::CompSeqOrderDetermined &event) {
    m_init_val_dumped = false;
    m_var2val.clear();
    m_vars.clear();
}

void VarValueChecker::on_opr_kern_end(
        const cg::event::OprExecKernelEnd &event) {
    if (event.opr->same_type<opr::AssertEqual>()) {
        // do not throw error from assertions in the graph, so error from
        // VarValueChecker can be observed
        event.opr->cast_final<opr::AssertEqual>().disable_throw_on_error();
    }
    for (VarNode *var: event.opr->output()) {
        if (!var->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            auto callback = [this, var]() {
                on_var_computed(var);
            };
            m_vars.push_back(var);
            event.env->dispatch_on_comp_node(var->comp_node(), callback);
        }
    }
}

void VarValueChecker::on_comp_seq_exec_finished(
        const cg::event::CompSeqExecFinished &) {
    if (!m_init_val_dumped) {
        m_init_val_dumped = true;
        mgb_assert(!m_vars.empty());
        m_cur_var_idx = m_init_var_idx;
        m_nr_exec = 0;
        m_checker.reset();
    } else {
        ++ m_nr_exec;
        if (m_nr_exec != m_var_switch_interval)
            return;

        m_nr_exec = 0;
        ++ m_cur_var_idx;
    }

    if (m_cur_var_idx >= m_vars.size()) {
        fprintf(stderr, "VarValueChecker: all check passed; "
                "start from beginning\n");
        m_cur_var_idx = 0;
    }
    auto var = m_vars[m_cur_var_idx];
    fprintf(stderr, "VarValueChecker: going to check #%zu: %s\n",
            m_cur_var_idx, cg::dump_var_info({var}).c_str());
    m_checker.reset();
}

void VarValueChecker::on_var_computed(VarNode *var) {
    if (!var->dev_tensor_valid()) {
        if (m_init_val_dumped && var == m_vars[m_cur_var_idx]) {
            // skip vars that are not on device
            on_comp_seq_exec_finished({});
        }
        return;
    }

    if (!m_init_val_dumped) {
        m_var2val_mtx.lock();
        auto &&val = m_var2val[var];
        m_var2val_mtx.unlock();

        mgb_assert(!val);
        val = std::make_shared<DeviceTensorND>();
        val->copy_from(var->dev_tensor());
        return;
    }

    if (var != m_vars[m_cur_var_idx])
        return;

    if (!m_checker.valid()) {
        m_checker.init(var, m_var2val.at(var));
    }
    m_checker.check(var);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
