/**
 * \file src/core/impl/graph/eager_eval.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./eager_eval.h"
#include "./cg_impl.h"

#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/helper.h"
#include "megbrain/utils/thread.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace cg;

#if !MGB_BUILD_SLIM_SERVING

constexpr size_t INF = std::numeric_limits<size_t>::max() >> 2;

namespace {

bool is_opr_mutable(OperatorNodeBase* opr) {
    for (auto &&i : opr->input()) {
        if (!is_const_var_value(i)) {
            return true;
        }
    }
    for (auto &&i : opr->output()) {
        if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT) &&
            !is_const_var_value(i)) {
            return true;
        }
    }
    return false;
}

}

/* ======================== EagerExecEnv ======================== */
class EagerEvalManager::EagerExecEnv final : public GraphExecutable::ExecEnv {
public:
    void dispatch_on_comp_node(CompNode, Task&& task) override {
        // note: we do not use different dispatch queues for different comp
        // nodes, even for CUDA. Using multiple threads here would complicate
        // things significantly since VarNodeMemManager and VarNode are not
        // thread-safe and we have to be very careful about locking and
        // synchronization.
        task();
    }

    void dispatch_on_comp_node_with_mask(CompNode, Task&& task,
                                         ExecutionMask* mask) override {
        mgb_throw_if(mask, GraphError,
                     "ExecutionMask not supported in eager mode");
        task();
    }

    void pause_exec() override {}

    void resume_exec() override {}
};

/* ======================== EagerEvalManager ======================== */
EagerEvalManager::EagerEvalManager(ComputingGraph* graph)
        : m_owner_graph{graph}, m_exec_env{new EagerExecEnv} {}

EagerEvalManager::~EagerEvalManager() noexcept {
    if (m_first_opr_enable_status == 1) {
        m_var_sync_mgr_pool.disable_freelist();
        for (auto&& i :
             ComputingGraphImpl::downcast(m_owner_graph)->all_oprs()) {
            for (auto var : i->output()) {
                auto mgr = VarNodeMemManager::var_node_cn_sync_manager(var);
                if (mgr) {
                    m_var_sync_mgr_pool.free(mgr);
                }
            }
        }
        m_version_trait_pool.disable_freelist();
        for (auto&& i: m_opr2version) {
            m_version_trait_pool.free(i.second);
        }
    }
}

void EagerEvalManager::init_waiting_spec(OperatorNodeBase* opr) {
    CompNode::UnorderedSet cur_used_cn;
    CompNode::UnorderedMap<ThinHashSet<VarNode*>> vars_to_wait;
    using NodeProp = OperatorNodeBase::NodeProp;

    OperatorNodeBase::InputWaitingSpec waiting_spec;
    for (auto ovar : opr->output()) {
        auto cn = ovar->comp_node();
        if (!cur_used_cn.insert(cn).second)
            continue;

        vars_to_wait.clear();

        for (auto&& i : opr->node_prop().dep_map()) {
            if (i.first->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
                // do not wait on PERSISTENT_DEVICE_VALUE vars
                continue;
            }
            if (NodeProp::is_device_comp_order_dep(i.second) &&
                i.first->comp_node() != cn) {
                vars_to_wait[i.first->comp_node()].insert(i.first);
            }
        }

        if (!vars_to_wait.empty()) {
            waiting_spec.emplace_back();
            waiting_spec.back().comp_node = cn;
            for (auto&& i : vars_to_wait) {
                for (auto j : i.second) {
                    waiting_spec.back().dev_ready.push_back(j);
                }
            }
        }
    }

    opr->input_waiting_spec(std::move(waiting_spec));
}

void EagerEvalManager::on_opr_insert(OperatorNodeBase* opr) {
    if (m_first_opr_enable_status == -1) {
        m_first_opr_enable_status = enabled();
    }
    mgb_assert(enabled() == m_first_opr_enable_status,
               "can not enable/disable eager eval after opr has been inserted");

    if (enabled()) {
        MGB_TRY { do_on_opr_insert(opr); }
        MGB_CATCH(MegBrainError & exc, {
            if (!exc.extra_info()) {
                OperatorNodeExcExtraInfo::record(opr, exc);
            }
            throw;
        });
    }
}

int EagerEvalManager::check_version(OperatorNodeBase* opr) {
    auto&& trait = m_opr2version[opr];
    if (!trait) {
        trait = m_version_trait_pool.alloc();
        using F = VersionTrait::Flag;
        if (is_opr_mutable(opr)) {
            if (opr->input().size()) {
                for (auto &&i : opr->input()) {
                    auto &&trait_i = m_opr2version.at(i->owner_opr());
                    trait_i->readers.push_back(trait);
                    if (trait_i->flag & F::MUTABLE_SOURCE) {
                        trait->flag = F::MUTABLE;
                    }
                }
            } else {
                trait->flag = static_cast<F>(F::MUTABLE | F::MUTABLE_SOURCE);
            }
        } else {
            trait->flag = F::CONST;
        }
        trait->need_reeval = true;
        return -1;
    }
    if (!trait->need_reeval) {
        // need following check since user could invalidate output
        // tensors explicitly (e.g. calling clear_device_memory())
        for (auto&& i : opr->output()) {
            if (!i->dev_tensor_valid()) {
                trait->need_reeval = true;
                break;
            }
        }
    }
    return trait->need_reeval;
}

void EagerEvalManager::prepare_for_exec(OperatorNodeBase* opr) {
    // validate inputs
    opr->add_input_layout_constraint();
    for (auto&& i : opr->node_prop().dep_map()) {
        using NodeProp = OperatorNodeBase::NodeProp;
        bool is_empty = !i.first->shape().ndim;
        if (NodeProp::is_device_value_dep(i.second)) {
            mgb_assert(i.first->dev_tensor_valid(),
                       "var value not valid, but required for opr input: "
                       "var=%s reader=%s{%s}",
                       cg::dump_var_info({i.first}).c_str(), opr->cname(),
                       opr->dyn_typeinfo()->name);
            if (i.first->dev_tensor().empty()) {
                is_empty = true;
            } else {
                ensure_input_layout(i.first);
            }
        }
        if (is_empty) {
            mgb_assert(i.second & NodeProp::DepType::VALUE_ALLOW_EMPTY,
                       "var value is empty but the reader opr does not allow "
                       "this: var=%s reader=%s{%s}",
                       cg::dump_var_info({i.first}).c_str(), opr->cname(),
                       opr->dyn_typeinfo()->name);
        }
    }

    // add input ready events
    for (auto&& i : opr->input_waiting_spec()) {
        for (auto j : i.dev_ready) {
            auto mgr = VarNodeMemManager::var_node_cn_sync_manager(j);
            if (!mgr->m_ready_event) {
                mgr->m_ready_event = mgr->m_comp_node.create_event();
                mgr->m_ready_event->record();
            }
        }
    }
}

void EagerEvalManager::update_static_infer_result(OperatorNodeBase* opr) {
    auto&& mgr = ComputingGraphImpl::downcast(m_owner_graph)
                         ->static_infer_manager_impl();
    auto sync_missing_trait =
            [&](static_infer::StaticInferManagerImpl::TagHandler* handler) {
                auto&& missing = mgr.get_missing_inp(handler);
                for (auto i : missing) {
                    i->sync_from_var();
                }
            };

    // set missing shapes/values for output shape infer
    using InferType = static_infer::InferType;
    for (auto var : opr->output()) {
        auto type = mgr.get_infer_type(var);
        if (type.shape & InferType::MISSING_INP) {
            sync_missing_trait(mgr.get_tag_handler_for_shape(var));
        }
    }

    // force udpate mutable src
    for (auto &&i : opr->output()) {
        if (i->contain_flag(VarNode::Flag::VOLATILE_CONTENT))
            continue;
        mgr.update_mutable_src_shape(i);
    }

    // set missing shapes/values for input value infer
    for (auto&& dep : opr->node_prop().dep_map()) {
        using Type = OperatorNodeBase::NodeProp::DepType;
        if ((dep.second & Type::HOST_VALUE) &&
            !is_static_var_value(dep.first)) {
            sync_missing_trait(mgr.get_tag_handler_for_value(dep.first));
        }
    }
}

void EagerEvalManager::ensure_input_layout(VarNode* var) {
    auto&& mem_mgr = ComputingGraphImpl::downcast(var->owner_graph())
                             ->var_node_mem_manager();

    auto trait = mem_mgr.get_var_node_mem_trait_nullable(var);
    if (!trait || trait->check_layout(var->layout())) {
        return;
    }
    DeviceTensorND val_contig;
    val_contig.copy_from(var->dev_tensor());

    auto&& chk = var->m_mem_plan.reset_from_owner_var().chunk();
    var->assign_dev_tensor_from_tensor(val_contig);
    chk.mem_alloc_status.set_from_owner_var();
}

void EagerEvalManager::alloc_output_mem(OperatorNodeBase* opr) {
    size_t nr_disallow_dynamic = 0, nr_readable_out = 0;
    for (auto i : opr->output()) {
        if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            ++nr_readable_out;
            if (i->contain_flag(
                        VarNode::Flag::DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC)) {
                ++nr_disallow_dynamic;
            }
        }
    }

    auto&& mgr = ComputingGraphImpl::downcast(m_owner_graph)
                         ->var_node_mem_manager();
    OprNodeArray opr_seq{opr};

    auto&& options = m_owner_graph->options();
    auto old_setting =
            std::make_pair(options.force_dynamic_alloc, options.log_level);
    MGB_TRY {
        options.log_level = 0;
        options.force_dynamic_alloc = true;
        mgr.reset_opr_seq_no_clear_dyn_alloc_info(m_comp_seq_extra_info,
                                                  &opr_seq, &m_run_id);
        mgr.alloc_var_node_mem_static();
    }
    MGB_FINALLY({
        std::tie(options.force_dynamic_alloc, options.log_level) = old_setting;
    });

    if (nr_disallow_dynamic) {
        mgb_assert(nr_disallow_dynamic == nr_readable_out,
                   "opr %s{%s} has %zu non-dynamic outputs, but total number "
                   "of readable outputs is %zu",
                   opr->cname(), opr->dyn_typeinfo()->name, nr_disallow_dynamic,
                   nr_readable_out);
    }

    // assume all outputs need to be waited
    for (auto i : opr->output()) {
        // note that we have set need_ready_event to false here to avoid an
        // abundant number of event objects; if an event is needed later, it
        // would be handled in prepare_for_exec() for that opr. This causes
        // unnecessary synchronization latency between comp nodes when ready
        // event is needed, but it is general good since this is the rare case
        // for eager evaluation (I assume usually only one comp node is involved
        // in eager evaluation).
        if (!i->m_cn_sync_manager) {
            auto mgr = m_var_sync_mgr_pool.alloc(i->comp_node());
            mgr->add_waiter_record(false, INF);
            VarNodeMemManager::set_var_node_cn_sync_manager(i, mgr);
        }
#if MGB_HAVE_THREAD
        // FIXME: cn_sync_manager would check if all readers of the previous
        // execution result has been finished. Here we use a trick, which set
        // nr_ready to zero, to bypass this check because eager eavaluation
        // works on single thread
        i->m_cn_sync_manager->m_nr_ready.store(0);
#endif
    }
}

void EagerEvalManager::do_on_opr_insert(OperatorNodeBase* opr) {
    if (!m_record_mode) {
        int status = check_version(opr);
        if (status < 0) {
            // initialize on first insertion
            init_waiting_spec(opr);
            prepare_for_exec(opr);
        }
        if (status) {
            update_static_infer_result(opr);
            alloc_output_mem(opr);
            auto&& mgr = ComputingGraphImpl::downcast(m_owner_graph)
                             ->var_node_mem_manager();
            mgr.on_graph_compile_finished();
            opr->execute(*m_exec_env);
            m_opr2version.at(opr)->update_version();
        }
    } else {
        m_record_oprs.insert(opr);
    }
}

const ComputingGraph::VarReceiverInfo* EagerEvalManager::var_receiver_info(
        const VarNode* var) const {
    if (enabled()) {
        // a fake info that requires the value and also allows empty shape
        static ComputingGraph::VarReceiverInfo ret = {
                .nr_direct_comp_req = 1,
                .dev_value = 1,
                .last_dev_value_reader = nullptr,
                .shape = 1,
                .host_value = 1,
                .allow_empty_value = 2};
        return &ret;
    }
    return nullptr;
}

GraphExecutable::ExecEnv* EagerEvalManager::exec_env() {
    if (enabled()) {
        return m_exec_env.get();
    }
    return nullptr;
}

size_t EagerEvalManager::get_var_nr_readers(VarNode* var) const {
    if (m_var2nr_readers.count(var)) {
        return m_var2nr_readers.at(var);
    } else {
        return REFCNT_INF;
    }
}


void EagerEvalManager::flush_record_oprs(
        const VarNodeArray &dest_vars) {
    if (!enabled()) {
        mgb_assert(m_record_oprs.empty());
        return;
    }
    m_record_mode = false;
    using NodeProp = OperatorNodeBase::NodeProp;
    ThinHashSet<OperatorNodeBase* > need_exec_oprs;
    ThinHashSet<OperatorNodeBase* > dest_oprs;
    std::function<void(OperatorNodeBase*)> visit = [&](OperatorNodeBase* opr) {
        if(!m_record_oprs.count(opr) || need_exec_oprs.count(opr))
            return;
        need_exec_oprs.insert(opr);
        for (auto inp: opr->input()) {
            visit(inp->owner_opr());
        }
    };
    for (auto var: dest_vars) {
        dest_oprs.insert(var->owner_opr());
        visit(var->owner_opr());
    }
    for (auto opr: need_exec_oprs) {
        auto&& node_prop = opr->node_prop();
        for (auto&& pair : node_prop.dep_map()) {
            if (NodeProp::is_device_value_dep(pair.second) &&
                    need_exec_oprs.count(pair.first->owner_opr())) {
                if (!dest_oprs.count(pair.first->owner_opr()))
                    m_var2nr_readers[pair.first] += opr->output().size();
            }
        }
    }
    SmallVector<std::pair<OperatorNodeBase*, size_t>> stack;
    ThinHashSet<OperatorNodeBase*> instack;
    auto push_stack = [&](OperatorNodeBase *opr) {
        if (need_exec_oprs.erase(opr)) {
            stack.push_back({opr, 0});
            instack.insert(opr);
        } else {
            mgb_assert(instack.count(opr) || m_opr2version.count(opr));
        }
    };
    for (auto &&var: dest_vars) {
        push_stack(var->owner_opr());
    }
    while (!stack.empty()) {
        auto &&frame = stack.back();
        auto opr = frame.first;
        if (frame.second < opr->input().size()) {
            auto var = opr->input()[frame.second++];
            push_stack(var->owner_opr());
        } else {
            MGB_TRY { do_on_opr_insert(opr); }
            MGB_CATCH(MegBrainError & exc, {
                if (!exc.extra_info()) {
                    OperatorNodeExcExtraInfo::record(opr, exc);
                }
                throw;
            });
            stack.pop_back();
        }
    }
    m_record_oprs.clear();
    m_var2nr_readers.clear();
}

#endif  // MGB_BUILD_SLIM_SERVING

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
