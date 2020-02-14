/**
 * \file src/plugin/impl/infkern_finder.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/plugin/infkern_finder.h"

using namespace mgb;
using namespace cg;

struct InfkernFinder::GlobalState {
    const bool record_input_value;
    std::unordered_map<VarNode*, InputValueRecord> var_value;

    //! constant marker value for event finish
    CompNode::UnorderedMap<DeviceTensorND> cn2marker_dev;

    std::unordered_map<size_t, OperatorNodeBase*> oprid2ptr;

    GlobalState(bool rec):
        record_input_value{rec}
    {}
};

class InfkernFinder::OprState {
    struct Event {
        static constexpr dt_int32
            STATE_UNRECORD = 0, STATE_RECORDED = 1, STATE_FINISHED = 2;
        size_t record_id = 0;
        HostTensorND host_buf;

        void init(CompNode cn) {
            host_buf.dtype(dtype::Int32()).
                comp_node(cn).
                resize({1}).
                ptr<dt_int32>()[0] = STATE_UNRECORD;
        }

        dt_int32& state() {
            return host_buf.ptr<dt_int32>()[0];
        }

        dt_int32 state() const {
            return host_buf.ptr<dt_int32>()[0];
        }
    };

    InfkernFinder * const m_par_finder = nullptr;
    OperatorNodeBase * const m_opr = nullptr;
    CompNode::UnorderedMap<Event> m_before_exec, m_after_wait;
    std::unordered_map<VarNode*, Event> m_output_done;

    void record_event(Event &ev) {
        ++ ev.record_id;
        ev.state() = Event::STATE_RECORDED;
        ev.host_buf.copy_from_fixlayout(
                m_par_finder->m_global_state->cn2marker_dev.at(
                    ev.host_buf.comp_node()));
    }

    void record_event_in_exec_env(GraphExecutable::ExecEnv *env, Event &ev) {
        auto cb = [this, e=&ev] {
            if (m_par_finder->m_global_state_storage &&
                    !m_par_finder->m_cg_start_log_printed.test_and_set()) {
                mgb_log("InfkernFinder: computing graph %p started",
                        m_par_finder->m_owner_graph);
            }
            record_event(*e);
        };
        env->dispatch_on_comp_node(ev.host_buf.comp_node(), cb);
    }

    public:
        static constexpr auto EVENT_STATE_MARKER_VALUE = Event::STATE_FINISHED;

        OprState() = default;

        OprState(InfkernFinder *par_finder,
                cg::OperatorNodeBase *opr, ComputingGraph *graph):
            m_par_finder{par_finder}, m_opr(opr)
        {
            for (auto &&i: cg::get_opr_comp_node_set(opr))
                m_before_exec[i].init(i);
            for (auto i: opr->input())
                m_before_exec[i->comp_node()].init(i->comp_node());
            for (auto &&i: opr->input_waiting_spec())
                m_after_wait[i.comp_node].init(i.comp_node);
            for (auto i: opr->output()) {
                auto &&recv = graph->var_receiver_in_current_comp_seq(i);
                if (recv.dev_value || recv.nr_direct_comp_req)
                    m_output_done[i].init(i->comp_node());
            }
        }

        OperatorNodeBase* opr() const {
            return m_opr;
        }

        void check_event_finished() {
            auto chk = [this](Event &ev, const char *type) {
                MGB_MARK_USED_VAR(this);
                mgb_assert(
                        ev.state() == Event::STATE_FINISHED,
                        "event %s not finished for operator %s{%s}",
                        type, m_opr->dyn_typeinfo()->name, m_opr->cname());
                ev.state() = Event::STATE_UNRECORD;
            };
            for (auto &&i: m_before_exec)
                chk(i.second, "before exec");
            for (auto &&i: m_after_wait)
                chk(i.second, "after wait");
            for (auto &&i: m_output_done)
                chk(i.second, "output done");
        }

        void on_opr_start(GraphExecutable::ExecEnv *env) {
            for (auto &&i: m_before_exec)
                record_event_in_exec_env(env, i.second);
        }

        void on_waiting_finished() {
            for (auto &&i: m_after_wait)
                record_event(i.second);
        }

        void record_output_values(GraphExecutable::ExecEnv *env) {
            for (auto &&i: m_output_done) {
                auto cb = [this, ovar=i.first]() {
                    auto &&dest = m_par_finder->m_global_state->var_value[ovar];
                    dest.run_id = m_before_exec.at(ovar->comp_node()).record_id;
                    if (dest.val.shape().ndim) {
                        // clean original data
                        memset(dest.val.raw_ptr(), -1,
                                dest.val.layout().span().high_byte);
                    }
                    if (ovar->dev_tensor_valid()) {
                        dest.val.copy_from(ovar->dev_tensor());
                    } else {
                        dest.val = {};
                    }
                };

                if (!i.first->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    env->dispatch_on_comp_node(i.first->comp_node(), cb);
                }
            }
        }

        void on_opr_finish(GraphExecutable::ExecEnv *env) {
            for (auto &&i: m_output_done)
                record_event_in_exec_env(env, i.second);
        }

        bool write_status(FILE *fout, size_t seq_step_num) {
            bool succ = true;
            auto rec = [&](const Event &ev) {
                auto state = ev.state();
                if (state == Event::STATE_UNRECORD) {
                    succ = false;
                    return "__unrec__";
                }
                if (state == Event::STATE_FINISHED) {
                    return "done";
                }
                mgb_assert(state == Event::STATE_RECORDED,
                        "bad event state: %d", state);
                succ = false;
                return "__waiting__";
            };
            fprintf(fout, "#%zu: opr %s{%s} id=%zu\n"_fmt,
                    seq_step_num, m_opr->dyn_typeinfo()->name, m_opr->cname(),
                    m_opr->id());

            fprintf(fout, " before exec:\n ");
            for (auto &&i: m_before_exec) {
                fprintf(fout, " %s:%s,run_id=%zu"_fmt,
                        i.first.to_string().c_str(),
                        rec(i.second), i.second.record_id);
            }
            if (!m_after_wait.empty()) {
                fprintf(fout, "\n after wait:\n ");
                for (auto &&i: m_after_wait)
                    fprintf(fout, " %s:%s", i.first.to_string().c_str(),
                            rec(i.second));
            }
            fprintf(fout, "\n used outputs:\n");
            for (auto &&i: m_output_done) {
                fprintf(fout, "  %s %s\n"_fmt,
                        rec(i.second),
                        cg::dump_var_info({i.first}).c_str());
            }
            return succ;
        }
};

InfkernFinder::InfkernFinder(ComputingGraph *graph, bool record_input_value):
    PluginBase{graph},
    m_global_state_storage{std::make_unique<GlobalState>(record_input_value)},
    m_global_state{m_global_state_storage.get()}
{
    init();
}

InfkernFinder::InfkernFinder(ComputingGraph *graph, GlobalState *global_state):
    PluginBase{graph},
    m_global_state_storage{},
    m_global_state{global_state}
{
    init();
}

InfkernFinder::~InfkernFinder() noexcept = default;

void InfkernFinder::init() {
    add_member_func_as_event_handler(&InfkernFinder::on_comp_seq_determined);
    add_member_func_as_event_handler(&InfkernFinder::on_comp_seq_finished);
    add_member_func_as_event_handler(&InfkernFinder::on_opr_start);
    add_member_func_as_event_handler(&InfkernFinder::on_waiting_finished);
    add_member_func_as_event_handler(&InfkernFinder::on_opr_kern_finish);
    add_member_func_as_event_handler(&InfkernFinder::on_opr_finish);
    add_member_func_as_event_handler(&InfkernFinder::on_subgraph_associated);
}


void InfkernFinder::on_comp_seq_determined(
        const cg::event::CompSeqOrderDetermined &ev) {
    m_current_comp_seq = ev.exec;
    m_opr_seq.clear();
    m_opr2state.clear();
    auto cb = [&](OperatorNodeBase *opr) {
        m_opr_seq.emplace_back(this, opr, ev.graph);
        m_global_state->oprid2ptr[opr->id()] = opr;
        for (auto &&i: get_opr_comp_node_set(opr))
            m_global_state->cn2marker_dev[i];
        return true;
    };
    ev.exec->iter_opr_seq(cb);
    for (auto &&i: m_opr_seq)
        m_opr2state[i.opr()] = &i;

    for (auto &&i: m_global_state->cn2marker_dev) {
        if (i.second.shape().ndim)
            continue;
        HostTensorND hv{i.first, dtype::Int32()};
        hv.resize({1}).ptr<dt_int32>()[0] = OprState::EVENT_STATE_MARKER_VALUE;
        i.second.copy_from(hv).sync();
    }
}

void InfkernFinder::on_comp_seq_finished(
        const cg::event::CompSeqExecFinished &ev) {
    mgb_assert(ev.exec == m_current_comp_seq);
    if (!ev.device_actually_finished)
        return;

    if (m_global_state_storage) {
        mgb_log("InfkernFinder: computing graph %p finished", m_owner_graph);
        m_cg_start_log_printed.clear();
    }
    for (auto &&i: m_opr_seq)
        i.check_event_finished();
    m_prev_succ_comp_seq_run_id = m_current_comp_seq->get_run_id();
}

void InfkernFinder::on_opr_start(const cg::event::OprExecStart &ev) {
    m_opr2state.at(ev.opr)->on_opr_start(ev.env);
}

void InfkernFinder::on_waiting_finished(const cg::event::AfterWait &ev) {
    m_opr2state.at(ev.opr)->on_waiting_finished();
}

void InfkernFinder::on_opr_kern_finish(const cg::event::OprExecKernelEnd &ev) {
    if (m_global_state->record_input_value) {
        m_opr2state.at(ev.opr)->record_output_values(ev.env);
    }
}

void InfkernFinder::on_opr_finish(const cg::event::OprExecFinished &ev) {
    m_opr2state.at(ev.opr)->on_opr_finish(ev.env);
}

void InfkernFinder::on_subgraph_associated(
        const cg::event::SubgraphAssociated &ev) {
    mgb_assert(ev.par_graph == m_owner_graph);
    m_sub_graph_finders.emplace_back(std::make_unique<InfkernFinder>(
                ev.sub_graph, m_global_state));
}

cg::OperatorNodeBase* InfkernFinder::write_to_file_opronly(FILE *fout) {
    size_t idx = 0;
    cg::OperatorNodeBase *bad_opr = nullptr;

    for (auto &&i: m_opr_seq) {
        if (!i.write_status(fout, idx ++) && !bad_opr)
            bad_opr = i.opr();
    }

    return bad_opr;
}

cg::OperatorNodeBase* InfkernFinder::write_to_file(const char *fpath) {
    FILE *fout = fopen(fpath, "w");
    mgb_assert(fout, "failed to open %s", fpath);

    size_t subg_idx = 0;
    cg::OperatorNodeBase *bad_opr = nullptr;
    for (auto &&i: m_sub_graph_finders) {
        fprintf(fout, "======== subgraph %zu ========\n"_fmt, subg_idx ++);
        auto o = i->write_to_file_opronly(fout);
        if (!bad_opr && o)
            bad_opr = o;
    }

    if (!m_sub_graph_finders.empty())
        fprintf(fout, "======== parent graph ========\n");

    auto self_bad_opr = write_to_file_opronly(fout);
    if (!bad_opr)
        bad_opr = self_bad_opr;

    fclose(fout);

    if (m_prev_succ_comp_seq_run_id == m_current_comp_seq->get_run_id())
        return nullptr;

    return bad_opr;
}

InfkernFinder::InputValueRecord::FullRecord
InfkernFinder::get_input_values(size_t opr_id) {
    mgb_assert(m_global_state->record_input_value);
    auto iter = m_global_state->oprid2ptr.find(opr_id);
    mgb_assert(iter != m_global_state->oprid2ptr.end(),
            "operator with ID %zu not found", opr_id);
    InputValueRecord::FullRecord rec;
    for (auto i: iter->second->input()) {
        auto iter = m_global_state->var_value.find(i);
        if (iter != m_global_state->var_value.end()) {
            rec.emplace_back(i, iter->second);
        } else {
            rec.push_back({i, {}});
        }
    }
    return rec;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
