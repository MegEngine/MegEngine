/**
 * \file src/plugin/impl/var_sanity_check.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/plugin/var_sanity_check.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/execution_mask.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/io.h"

using namespace mgb;

#define LOG_DETAILS_ENV_VAR_NAME "MGB_DEBUG_VAR_SANITY_CHECK_LOG"

VarSanityCheck::VarSanityCheck(cg::ComputingGraph* graph) : PluginBase(graph) {
    auto on_exec_start = [this](const cg::event::OprExecKernelStart& event) {
        setup_input_checker(true, event.opr, *event.env,
                            &VarSanityCheck::on_var_received);
    };

    auto on_exec_finish = [this](const cg::event::OprExecKernelEnd& event) {
        for (VarNode* var : event.opr->output()) {
            auto&& recv =
                    var->owner_graph()->var_receiver_in_current_comp_seq(var);
            auto check_var_basic = [ var, recv = &recv ]() {
                check_var_after_exec(var, *recv);
            };
            event.env->dispatch_on_comp_node(var->comp_node(), check_var_basic);

            // skip unused vars
            if (!recv.dev_value ||
                var->contain_flag(VarNode::Flag::VOLATILE_CONTENT))
                continue;

            m_debug_log.add_producer(var);
            auto callback = [this, var]() { on_var_produced(var); };
            event.env->dispatch_on_comp_node(var->comp_node(), callback);
        }

        setup_input_checker(false, event.opr, *event.env,
                            &VarSanityCheck::check_input_unmodified);
    };

    add_event_handler(
            graph->event().register_receiver<cg::event::OprExecKernelStart>(
                    on_exec_start));
    add_event_handler(
            graph->event().register_receiver<cg::event::OprExecKernelEnd>(
                    on_exec_finish));
}

std::string VarSanityCheck::str(const ChecksumResult& chk) {
    return ssprintf("{checksum:0x%x, last_int:%d, last_float:%g}", chk.checksum,
                    chk.last_val.iv, chk.last_val.fv);
}

void VarSanityCheck::check_var_after_exec(
        VarNode* var, const ComputingGraph::VarReceiverInfo& recv) {
    bool is_empty = !var->shape().ndim ||
                    (var->dev_tensor_valid() && var->dev_tensor().empty());

    if (is_empty) {
        auto allow = var->contain_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE),
             no_alloc = var->contain_flag(VarNode::Flag::NO_ALLOC_IF_UNUSED);
        mgb_throw_if(!(allow || (no_alloc && recv.empty())) ||
                             !recv.is_empty_allowed(),
                     GraphError,
                     "empty output var after node execution: %s "
                     "(allow=%d receiver=%s)",
                     cg::dump_var_info({var}).c_str(), allow,
                     recv.to_string().c_str());
    }
}

/* =================  DebugLog =================  */
#if MGB_ENABLE_GETENV && MGB_ENABLE_JSON

VarSanityCheck::DebugLog::DebugLog(VarSanityCheck* checker)
        : m_checker(checker) {
    auto idstr = MGB_GETENV(LOG_DETAILS_ENV_VAR_NAME);
    if (!idstr)
        return;
    m_enable = true;
    sscanf(idstr, "%d", &m_var_id);
    mgb_log_warn(LOG_DETAILS_ENV_VAR_NAME
                 " is set to %d; "
                 "details of var address and checksum would be logged",
                 m_var_id);
}

void VarSanityCheck::DebugLog::add_producer(VarNode* var) {
    if (static_cast<int>(var->id()) == m_var_id) {
        m_readcnt_init = 0;
        m_var = var;
    }
}

void VarSanityCheck::DebugLog::add_receiver(VarNode* var) {
    if (static_cast<int>(var->id()) == m_var_id) {
        mgb_assert(var == m_var);
        ++m_readcnt_init;
    }
}

void VarSanityCheck::DebugLog::on_var_produced(VarSanityCheck* checker,
                                               VarNode* var,
                                               ChecksumResult checksum) {
    if (!m_enable)
        return;

    mgb_log("var %s: addr=%p checksum=%s", cg::dump_var_info({var}).c_str(),
            var->dev_tensor().raw_ptr(), str(checksum).c_str());

    if (m_readcnt) {
        auto checksum = checker->calc_checksum(m_var);
        mgb_log("recheck var%zu after var%zu finished: addr=%p checksum=%s",
                m_var->id(), var->id(), m_var->dev_tensor().raw_ptr(),
                str(checksum).c_str());
        ChecksumResult checksum_expect;
        {
            MGB_LOCK_GUARD(m_checker->m_id2chksum_mtx);
            checksum_expect = m_checker->m_var2chksum.at(m_var);
        }
        if (checksum != checksum_expect) {
            var->owner_graph()->current_comp_seq()->to_json()->writeto_fpath(
                    "/tmp/mgb-graph-sanity-check-failed.json");
            mgb_throw(cg::OperatorNodeExcExtraInfo::ExcMaker{m_var->owner_opr()}
                              .make<VarSanityCheck::Error>,
                      "error in recheck");
        }
    }
    if (var == m_var) {
        mgb_assert(!m_readcnt);
        m_readcnt = m_readcnt_init;
    }
}

void VarSanityCheck::DebugLog::on_var_received(VarNode* var) {
    if (var == m_var) {
        auto nr = m_readcnt.fetch_sub(1);
        mgb_assert(nr);
        if (nr == 1)
            mgb_log("var %zu out of scope, stop tracking", var->id());
    }
}
#else
VarSanityCheck::DebugLog::DebugLog(VarSanityCheck* checker)
        : m_checker(checker) {}

void VarSanityCheck::DebugLog::add_producer(VarNode*) {}

void VarSanityCheck::DebugLog::add_receiver(VarNode*) {}

void VarSanityCheck::DebugLog::on_var_produced(VarSanityCheck*, VarNode*,
                                               ChecksumResult) {}

void VarSanityCheck::DebugLog::on_var_received(VarNode*) {}
#endif  // MGB_ENABLE_GETENV && MGB_ENABLE_JSON

/* ================= VarSanityCheck =================  */

VarSanityCheck::ChecksumResult VarSanityCheck::calc_checksum(VarNode* var) {
    // SharedDeviceTensor may be modified in callback, also return zero
    if (var->owner_opr()->same_type<opr::VolatileSharedDeviceTensor>() ||
        var->owner_opr()->same_type<opr::SharedDeviceTensor>())
        return ChecksumResult{0, {0}};

    auto&& dt = var->dev_tensor();
    if (!dt.layout().total_nr_elems()) {
        static ChecksumResult empty_checksum;
        return empty_checksum;
    }

    auto span = dt.layout().span();
    megdnn::TensorND tensor;
    tensor.raw_ptr = dt.raw_ptr() + span.low_byte;
    tensor.layout.init_contiguous_stride({span.dist_byte()});
    tensor.layout.dtype = dtype::Byte();

    DeviceTensorStorage* workspace;
    {
        MGB_LOCK_GUARD(m_workspace_mtx);
        workspace = &m_workspace[std::this_thread::get_id()]
                             .storage[var->comp_node()];
    }
    auto comp_node = var->comp_node();
    comp_node.activate();
    auto opr = opr::intl::get_megdnn_global_opr<megdnn::Checksum>(comp_node);
    auto workspace_reqsize = opr->get_workspace_in_bytes(tensor.layout);
    workspace->comp_node(var->comp_node()).ensure_size(workspace_reqsize);

    megdnn::Workspace mwk;
    if (workspace_reqsize)
        mwk = {workspace->ptr(), workspace_reqsize};
    return opr->exec(tensor, mwk);
}

void VarSanityCheck::on_var_produced(VarNode* var) {
    if (!var->shape().ndim)
        return;

    auto checksum = calc_checksum(var);
    m_debug_log.on_var_produced(this, var, checksum);

    {
        MGB_LOCK_GUARD(m_id2chksum_mtx);
        auto rst = m_var2chksum.emplace(var, checksum);
        mgb_assert(
                rst.second ||
                        (var->contain_flag(
                                 VarNode::Flag::PERSISTENT_DEVICE_VALUE) &&
                         (m_modified_vars.count(var) ||
                          rst.first->second == checksum)),
                "var recorded before produced: %s; checksum: record=%s calc=%s",
                cg::dump_var_info({var}).c_str(),
                str(rst.first->second).c_str(), str(checksum).c_str());
    }
}

void VarSanityCheck::on_var_received(cg::OperatorNodeBase* recv_opr,
                                     VarNode* var) {
    check_single_input(true, recv_opr, var);
}

void VarSanityCheck::check_input_unmodified(cg::OperatorNodeBase* recv_opr,
                                            VarNode* var) {
    auto ptr = var->dev_tensor().raw_ptr();
    for (auto i : recv_opr->output()) {
        if (i->dev_tensor_valid() && i->dev_tensor().raw_ptr() == ptr)
            return;
    }
    check_single_input(false, recv_opr, var);
}

void VarSanityCheck::check_single_input(bool add_debug_log,
                                        cg::OperatorNodeBase* recv_opr,
                                        VarNode* var) {
    if (var->contain_flag(VarNode::Flag::DISALLOW_VAR_SANITY_CHECK))
        return;

    auto checksum = calc_checksum(var);

    ChecksumResult checksum_expect;
    {
        MGB_LOCK_GUARD(m_id2chksum_mtx);
        auto&& node_prop = recv_opr->node_prop();
        if (node_prop.contain(cg::OperatorNodeBase::NodeProp::Flag::
                                      FORCE_UPDATE_INPUT_VAR)) {
            m_modified_vars.insert(var);
        }
        auto chk_iter = m_var2chksum.find(var);
        if (chk_iter == m_var2chksum.end()) {
            // PERSISTENT_DEVICE_VALUE vars can be read by oprs on other comp
            // nodes without being waited on
            mgb_assert(
                    var->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE),
                    "var checksum uncomputed, and its owner opr is not "
                    "PERSISTENT_DEVICE_VALUE: %s",
                    cg::dump_var_info({var}).c_str());
            auto rst = m_var2chksum.emplace(var, checksum);
            mgb_assert(rst.second);
            return;
        } else {
            checksum_expect = chk_iter->second;
        }
    }

    if (add_debug_log) {
        m_debug_log.on_var_received(var);
    }

    if (checksum != checksum_expect) {
        mgb_throw(Error,
                  "var sanity check failed: var: %s"
                  " (checksum: expect=%s got=%s); receiver: %s{%s}(%zu);"
                  " you can set " LOG_DETAILS_ENV_VAR_NAME
                  "=%zu to get more details; pass=%d",
                  cg::dump_var_info({var}).c_str(),
                  str(checksum_expect).c_str(), str(checksum).c_str(),
                  recv_opr->cname(), recv_opr->dyn_typeinfo()->name,
                  recv_opr->id(), var->id(), !add_debug_log);
    }
}

void VarSanityCheck::setup_input_checker(bool add_debug_log,
                                         cg::OperatorNodeBase* opr,
                                         cg::GraphExecutable::ExecEnv& env,
                                         input_checker_fn checker) {
    for (auto&& dep_entry : opr->node_prop().dep_map()) {
        if (!cg::OperatorNodeBase::NodeProp::is_device_value_dep(
                    dep_entry.second)) {
            continue;
        }

        auto var = dep_entry.first;
        if (add_debug_log) {
            m_debug_log.add_receiver(var);
        }

        // dispatch on output comp node, to check value before input var
        // is reclaimed

        CompNode cn;

        // prefer var comp node if opr executes on it
        for (auto i : opr->output()) {
            if (i->comp_node() == var->comp_node()) {
                cn = i->comp_node();
                break;
            }
        }
        if (!cn.valid()) {
            // find a comp node that waits var
            for (auto&& i : opr->input_waiting_spec()) {
                for (auto j : i.dev_ready) {
                    if (j == var) {
                        cn = i.comp_node;
                        break;
                    }
                }
                if (cn.valid())
                    break;
            }
            if (!cn.valid()) {
                // input waiting spec may be purged; now we just use any
                // comp node of this opr
                cn = opr->output(0)->comp_node();
            }
        }

        auto callback = [this, opr, var, checker]() {
#if MGB_ENABLE_COND_EXEC
            if (auto mask = cg::ExecutionMask::get_from_opr(var->owner_opr())) {
                if (!mask->enabled()) {
                    mgb_assert(!m_var2chksum.count(var),
                               "disabled opr has computed output: %s",
                               cg::dump_var_info({var}).c_str());
                    return;
                }
            }
#endif
            (this->*checker)(opr, var);
        };
        env.dispatch_on_comp_node(cn, callback);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
