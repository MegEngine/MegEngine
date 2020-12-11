/**
 * \file src/core/impl/graph/cg_impl_seq.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./cg_impl.h"
#include "./normal_exec_env.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/plugin/var_sanity_check.h"
#include "megbrain/utils/arith_helper.h"

namespace mgb {
namespace cg {

class ComputingGraphImpl::ComputingSequence final : public AsyncExecutable {
    const std::shared_ptr<ComputingGraph> m_owner_graph_refkeep;
    ComputingGraphImpl* const m_owner_graph;
    const bool m_have_parent_graph = true;
    bool m_wait_finished = true, m_first_exec = true,
         m_enable_comp_node_seq_recorder = false;
    size_t m_run_id = 0;
    size_t m_cg_event_version = 0;
    mutable Maybe<double> m_prev_exec_time;
    std::unique_ptr<VarSanityCheck> m_var_sanity_check;
    std::unique_ptr<CompNodeSeqRecorder> m_comp_node_seq_recorder;

    NormalExecEnv m_exec_env;

    const OprNodeArray* m_opr_seq = nullptr;
    ThinHashMap<OperatorNodeBase*, size_t> m_opr2stepnum;

    CompNode::UnorderedSet m_used_comp_node;

    using EventArray = CompNode::UnorderedMap<std::unique_ptr<CompNode::Event>>;
    EventArray m_event_start, m_event_end;

    class ExecContext;

    std::unique_ptr<MegBrainError> m_async_exc;
    std::mutex m_async_exc_mutex;

    /*!
     * \brief check whether recording comp seq is enabled
     *
     * m_enable_comp_node_seq_recorder would be setup and a temp recorderd would
     * be returned
     *
     * This is called from init_for_exec when m_first_exec is true
     */
    std::unique_ptr<CompNodeSeqRecorder> check_enable_comp_node_seq_recorder();

    void record_all_event(const EventArray& arr) {
        for (auto&& i : arr) {
            auto runner = [ev = i.second.get()]() { ev->record(); };
            m_exec_env.dispatch_on_comp_node(i.first, runner);
        }
    }

    void init_for_exec();

    //! called from init_for_exec() when m_first_exec is true
    void on_first_exec();

    /*!
     * \brief implements wait()
     * \param explicit_user_wait see event::CompSeqExecFinished
     */
    void do_wait(bool explicit_user_wait);

    void cleanup();

    /*!
     * This is used by both execute() and as_recorded_seq()
     * \param dtor_check if not null, it would be enabled after fake exec; used
     *      by as_recorded_seq()
     */
    void do_execute(MegDNNDtorCheck* dtor_check);

    /*!
     * This function does Memory allocation, ExecEnv initialization,
     * creation of events for profiling/sync and dispatching all tasks.
     * This method should only called from ExecContext's constructor.
     *
     * \param ctx pass all useful flags to given ExecContext.
     */
    void preprocess(ExecContext* ctx);

    std::shared_ptr<void> on_comp_node_finalize() override;

    ComputingGraph* owner_graph() const override { return m_owner_graph; }

public:
    ComputingSequence(const std::shared_ptr<ComputingGraph>& graph)
            : m_owner_graph_refkeep{graph},
              m_owner_graph{ComputingGraphImpl::downcast(graph.get())},
              m_have_parent_graph{
                      static_cast<bool>(m_owner_graph->m_parent_graph)} {}

    GraphExecutable::ExecEnv& exec_env() { return m_exec_env; }

    void assert_latest_comp_seq() const;

    void attach_to_graph();

    ~ComputingSequence();

    void setup_opr_seq(const OprNodeArray* seq) {
        mgb_assert(!m_opr_seq && seq);
        m_opr_seq = seq;
        for (size_t i = 0; i < seq->size(); ++i) {
            auto ins = m_opr2stepnum.emplace((*seq)[i], i);
            mgb_assert(ins.second);
        }
    }

    const static_infer::DepVal& get_rt_static_source_deps() override {
        return extra_info.rt_static_infer_src;
    }

    AsyncExecutable& execute() override;

    AsyncExecutable& wait() override;

    double get_prev_exec_time() const override;

    AsyncExecutable& iter_opr_seq(
            thin_function<bool(OperatorNodeBase*)> cb) override;

#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> to_json() const override;
#endif

    size_t nr_step() const { return m_opr_seq->size(); }

    Maybe<size_t> opr2stepnum(OperatorNodeBase* opr) {
        auto iter = m_opr2stepnum.find(opr);
        if (iter == m_opr2stepnum.end())
            return None;
        return iter->second;
    }

    CompSeqExtraInfo extra_info;

    size_t get_run_id() const override { return m_run_id; }

    //! get the pointer to the run id, so it can be accessed anytime
    const size_t* get_run_id_ptr() const { return &m_run_id; }

    virtual const CompNode::UnorderedMap<size_t>&
    update_static_alloc_plan_and_get_size() override;

    void clear_device_memory() override;

    void set_async_error(std::unique_ptr<MegBrainError> async_exc) {
        // all computing graphs executed concurrently can call this function
        // to set async error, so this function should be thread safe
        MGB_LOCK_GUARD(m_async_exc_mutex);
        if (!m_async_exc) {
            m_async_exc = std::move(async_exc);
        }
    }

    std::unique_ptr<RecordedComputingSequence> as_recorded_seq();
};

class ComputingGraphImpl::MegDNNDtorCheck : public NonCopyableObj {
    bool m_enabled = false;
    megdnn::Handle* const m_handle;
    CompNodeEnv* const m_env;
    thin_function<void(megdnn::OperatorBase*)> m_orig_dnn_cb;
    CompNodeEnv::MemEventHandler m_orig_mem_cb;
    GraphExecutable::ExecDependencyArray m_safe_dtor_objs;

    //! associated computing sequence; its on_graph_destroy() would be
    //! called in the dtor
    RecordedComputingSequence* m_comp_seq = nullptr;

public:
    explicit MegDNNDtorCheck(CompNode cn,
                             RecordedComputingSequence* comp_seq = nullptr)
            : m_handle{MegDNNHandle::get(CompNodeEnv::from_comp_node(cn))
                               .handle()},
              m_env{const_cast<CompNodeEnv*>(&CompNodeEnv::from_comp_node(cn))},
              m_comp_seq{comp_seq} {}

    ~MegDNNDtorCheck();

    void enable();

    //! called from dtor of RecordedComputingSequence
    void on_comp_seq_destroy(RecordedComputingSequence* ptr) {
        // the graph should only be compiled once, so comp seq can not have
        // other value
        mgb_assert(ptr == m_comp_seq);
        m_comp_seq = nullptr;
    }

    /*!
     * \brief exec deps to be associated with this checker that can be safely
     * destructed
     *
     * So objects in this array can be safely destructed without triggering
     * error
     */
    GraphExecutable::ExecDependencyArray& safe_dtor_objs() {
        return m_safe_dtor_objs;
    }
};

class ComputingGraphImpl::RecordedComputingSequence final
        : public AsyncExecutable {
    friend class ComputingGraphImpl::ComputingSequence;

    bool m_wait_finished = true;
    GraphExecutable::ExecDependencyArray m_exec_deps;
    std::vector<GraphExecutable::ExecDependency*> m_runtime_checks;
    UserDataContainer m_graph_user_data;
    DeviceTensorStorage m_static_mem;
    std::unique_ptr<CompNodeSeqRecorder> m_recorder;
    std::unique_ptr<CompNode::Event> m_event_start, m_event_end;
    //! valid if owner graph is not destroyed
    ComputingGraphImpl* m_owner_graph;
    mutable Maybe<double> m_prev_exec_time;

    std::shared_ptr<void> on_comp_node_finalize() override {
        clear_device_memory();
        m_exec_deps.clear();
        m_runtime_checks.clear();
        m_graph_user_data.clear_all_user_data();
        return {};
    }

    [[noreturn]] static void on_not_support(const char* name) {
        mgb_throw(MegBrainError, "%s unsupported on RecordedComputingSequence",
                  name);
    }

public:
    explicit RecordedComputingSequence(ComputingGraphImpl* owner_graph)
            : m_owner_graph{owner_graph} {}

    ~RecordedComputingSequence() {
        if (m_owner_graph) {
            m_owner_graph->m_recorded_seq_level2_dtor_chk->on_comp_seq_destroy(
                    this);
        }
    }

    AsyncExecutable& execute() override;

    AsyncExecutable& wait() override;

    double get_prev_exec_time() const override;

    /*!
     * \brief iterate over operator sequence
     * \param cb callback function, return false to stop iterating
     */
    AsyncExecutable& iter_opr_seq(
            thin_function<bool(OperatorNodeBase*)>) override {
        on_not_support(mgb_cstr_log("iter_opr_seq"));
    }

    const SmallVector<static_infer::DepElement>& get_rt_static_source_deps()
            override {
        on_not_support(mgb_cstr_log("get_rt_static_source_deps"));
    }

    size_t get_run_id() const override {
        on_not_support(mgb_cstr_log("get_run_id"));
    }

    virtual const CompNode::UnorderedMap<size_t>&
    update_static_alloc_plan_and_get_size() override {
        on_not_support(mgb_cstr_log("update_static_alloc_plan_and_get_size"));
    }

    void clear_device_memory() override {
        m_static_mem = {};
        m_recorder.reset();  // so it could not be executed again
    }

    //! called from MegDNNDtorCheck dtor
    void on_graph_destroy() { m_owner_graph = nullptr; }

    ComputingGraph* owner_graph() const override { return m_owner_graph; }

#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> to_json() const override {
        on_not_support(mgb_cstr_log("to_json"));
    }
#endif
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
