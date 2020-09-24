/**
 * \file src/core/impl/graph/cg_impl_seq.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl_seq.h"
#include "megbrain/graph/exc_extra_info.h"

using namespace mgb;
using namespace cg;

/* ========================== ExecContext ========================== */
/*!
 * \brief context for a single execution
 *
 * This class is a helper for implementing exec() and should only be constructed
 * on the stack
 */
class ComputingGraphImpl::ComputingSequence::ExecContext {
    // A context which contains some useful states for execution.

    ComputingSequence* const m_comp_seq;
    ComputingGraphImpl* const m_owner_graph;

    //! whether memory is re-alllocated in this execution
    bool m_mem_reallocated = false;

    //! whether we need to do the work (i.e. whether fake exec is not enabled)
    bool m_need_perform = true;

    //! whether comp seq recorder is initialized (computed at first exec)
    bool m_enable_comp_node_seq_recorder;

    //! states forwarded from the owner computing sequence
    const bool m_first_exec, m_fake_next_exec, m_have_parent_graph;

    CleanupCallback m_cleanup_callback;

    //! new recorder in current execution, which would be moved in
    //! stop_and_move_recorder()
    std::unique_ptr<CompNodeSeqRecorder> m_recorder;

    bool has_var_sanity_check() const {
        return static_cast<bool>(m_comp_seq->m_var_sanity_check);
    }

    void try_reset_recorder() {
        if (m_mem_reallocated) {
            // clear recorded sequence because memory has been reallocated
            m_comp_seq->m_comp_node_seq_recorder.reset();
        }
        if (m_comp_seq->m_comp_node_seq_recorder) {
            return;
        }
        // get first comp node to be used with recorder
        auto comp_node = *(m_comp_seq->m_used_comp_node.begin());
        // note: if m_first_exec or m_mem_reallocated is true, we can not record
        // because there might be dynamic memory allocations for temp storage in
        // the operators
        bool tmp_storage_warmup =
                (has_var_sanity_check() || m_first_exec || m_mem_reallocated) &&
                (comp_node.contain_flag(
                        CompNode::Flag::RECORDER_SUPPORT_DYNAMIC_ALLOC));
        if (m_fake_next_exec || !tmp_storage_warmup) {
            // all the asserts should have been checked in
            // check_enable_comp_node_seq_recorder()
            mgb_assert(m_comp_seq->m_used_comp_node.size() == 1);
            m_recorder = comp_node.create_seq_recorder(m_owner_graph);
            mgb_assert(m_recorder);
        }
    }

    void warmup_for_fake_exec_with_recorder() {
        // Rerun recorder to ensure that all internal caches stabilize
        auto comp_node = *(m_comp_seq->m_used_comp_node.begin());
        m_recorder->enter_fake_exec(comp_node);
        m_comp_seq->m_exec_env.start_exec();
        m_comp_seq->m_exec_env.wait_all();
        m_recorder->exit_fake_exec(comp_node);
    }

    void stop_and_move_recorder() {
        auto comp_node = *(m_comp_seq->m_used_comp_node.begin());
        m_recorder->stop(comp_node);
        if (m_fake_next_exec) {
            m_owner_graph->options().fake_next_exec = false;
        } else {
            m_recorder->replay();
        }
        // only move to m_comp_node_seq_recorder after all oprs succeeds
        m_comp_seq->m_comp_node_seq_recorder = std::move(m_recorder);
    }

    void after_fake_exec() {
        mgb_assert(!m_have_parent_graph,
                   "m_fake_next_exec should only be set on root graph");
        m_owner_graph->options().fake_next_exec = false;
        m_owner_graph->var_node_mem_manager()
                .static_device_memory_manager()
                ->prefault();
    }

    friend void ComputingSequence::preprocess(ExecContext* ctx);

public:
    inline ExecContext(ComputingSequence* comp_seq);
    inline ~ExecContext() noexcept;

    // call this method to run all tasks in the given ExecEnv,
    // this function can be called multiple times for partial-execution
    inline void perform(NormalExecEnv* env);
};

ComputingGraphImpl::ComputingSequence::ExecContext::ExecContext(
        ComputingSequence* comp_seq)
        : m_comp_seq{comp_seq},
          m_owner_graph{comp_seq->m_owner_graph},
          m_first_exec{comp_seq->m_first_exec},
          m_fake_next_exec{comp_seq->m_owner_graph->options().fake_next_exec},
          m_have_parent_graph{comp_seq->m_have_parent_graph} {
    {
        // lock the device memory manager to detect concurrent usage
        auto dev_mem_mgr = m_comp_seq->m_owner_graph->var_node_mem_manager()
                                   .static_device_memory_manager()
                                   .get();
        dev_mem_mgr->exec_enter();
        m_cleanup_callback.add([dev_mem_mgr]() { dev_mem_mgr->exec_exit(); });
    }

    if (!m_have_parent_graph) {
        // preprocess() would re-init static var mem plan and reallocate static
        // memory if needed, but async var mem release depends on chunk refcnt
        // which would be reset to one if mem plan is initialized, so we wait
        // for previous run (including async var mem release) to finish before
        // calling preprocess()
        m_comp_seq->do_wait(false);
    }

    m_comp_seq->preprocess(this);

    if (m_fake_next_exec) {
        if (!m_enable_comp_node_seq_recorder) {
            // fake exec is just for graph init; it has finished now
            m_need_perform = false;
            return;
        }
        // if m_enable_comp_node_seq_recorder and m_fake_next_exec are both
        // true, we warm up by directly recording into CompNodeSeqRecorder; this
        // is for best achievable efficiency. This requires var sanity check to
        // be disabled and this should also be the first exec
        mgb_assert(
                !has_var_sanity_check() &&
                        (m_first_exec ||
                         m_owner_graph->options().comp_node_seq_record_level >=
                                 2),
                "if m_fake_next_exec and m_enable_comp_node_seq_recorder are "
                "both set, they can only be set at the first run and var "
                "sanity check should be disabled");
    }

    if (m_enable_comp_node_seq_recorder) {
        // reset m_comp_node_seq_recorder and create new recorder if needed
        try_reset_recorder();
    }

    if (m_fake_next_exec) {
        // m_fake_next_exec without comp seq recorders has been handled
        // above; so reaching here means both m_fake_next_exec and comp
        // seq recorder are enabled.
        warmup_for_fake_exec_with_recorder();
    }
}

void ComputingGraphImpl::ComputingSequence::ExecContext::perform(
        NormalExecEnv* env) {
    if (!m_need_perform) {
        // no need for performing
        return;
    } else if (m_comp_seq->m_comp_node_seq_recorder) {
        // replay recorder
        m_comp_seq->m_comp_node_seq_recorder->replay();
    } else {
        // normal execute, execute with recorder and partial execution
        env->start_exec();
        // wait for all operations to be issued
        env->wait_all();
    }
}

ComputingGraphImpl::ComputingSequence::ExecContext::~ExecContext() noexcept {
    if (has_uncaught_exception()) {
        m_owner_graph->event().signal_inplace<event::CompSeqExecError>(
                m_owner_graph, m_comp_seq);
        return;
    }

    if (!m_need_perform) {
        after_fake_exec();
        return;
    }

    if (m_recorder) {
        // stop recorder and move it to m_comp_node_seq_recorder
        stop_and_move_recorder();
    }

    if (m_have_parent_graph) {
        m_owner_graph->event().signal_inplace<event::CompSeqExecFinished>(
                false, false, m_owner_graph, m_comp_seq);
    }

    // set m_wait_finished at last, so we would not wait if there are exceptions
    m_comp_seq->m_wait_finished = false;
}

/* ========================== ComputingSequence ========================== */

std::unique_ptr<CompNodeSeqRecorder>
ComputingGraphImpl::ComputingSequence::check_enable_comp_node_seq_recorder() {
    if (!m_owner_graph->options().comp_node_seq_record_level)
        return {};
    if (m_used_comp_node.size() != 1) {
        mgb_log_error(
                "can not enable CompNodeSeqRecorder because more than one comp "
                "nodes are involved: %zu",
                m_used_comp_node.size());
        return {};
    }
    if (m_owner_graph->options().force_dynamic_alloc) {
        mgb_log_error(
                "can not enable CompNodeSeqRecorder due to "
                "force_dynamic_alloc");
        return {};
    }
    if (m_owner_graph->m_parent_graph) {
        mgb_log_error(
                "can not enable CompNodeSeqRecorder because it has parent "
                "graph.");
        return {};
    }
    for (auto i : *m_opr_seq) {
        for (auto j : i->output()) {
            if (!is_static_var_storage(j)) {
                mgb_log_error(
                        "can not enable CompNodeSeqRecorder because var "
                        "storage not static: %s",
                        dump_var_info({j}).c_str());
                return {};
            }
        }
    }
    auto cn = *m_used_comp_node.begin();
    auto rec = cn.create_seq_recorder(m_owner_graph);
    if (!rec) {
        mgb_log_error(
                "can not enable CompNodeSeqRecorder on unsupported comp node "
                "%s",
                cn.to_string().c_str());
        return {};
    }
    m_enable_comp_node_seq_recorder = true;
    return rec;
}

void ComputingGraphImpl::ComputingSequence::do_execute(
        MegDNNDtorCheck* dtor_check) {
    ExecContext exec_ctx{this};

    if (dtor_check) {
        dtor_check->enable();
    }

    exec_ctx.perform(&m_exec_env);
}

void ComputingGraphImpl::ComputingSequence::preprocess(ExecContext* ctx) {
    assert_latest_comp_seq();
    ++m_run_id;
    m_prev_exec_time = None;

    ctx->m_mem_reallocated =
            m_owner_graph->var_node_mem_manager().alloc_var_node_mem_static();

    bool first_exec = m_first_exec;
    if (!first_exec) {
        // var sanity check only for first run
        m_var_sanity_check.reset();
    }

    m_owner_graph->event().signal_inplace<event::CompSeqExecBeforeStart>(
            m_owner_graph, this, &ctx->m_cleanup_callback, &m_used_comp_node,
            m_owner_graph->event().version());

    if (first_exec || m_cg_event_version != m_owner_graph->event().version()) {
        init_for_exec();
    }
    ctx->m_enable_comp_node_seq_recorder = m_enable_comp_node_seq_recorder;
}

std::shared_ptr<void>
ComputingGraphImpl::ComputingSequence::on_comp_node_finalize() {
    cleanup();
    m_exec_env.clear();
    m_comp_node_seq_recorder.reset();
    m_opr2stepnum.clear();
    return {};
}

void ComputingGraphImpl::ComputingSequence::assert_latest_comp_seq() const {
    mgb_throw_if(m_owner_graph->m_current_comp_seq != this, GraphError,
                 "only the latest compiled function could be used");
}

void ComputingGraphImpl::ComputingSequence::attach_to_graph() {
    auto gimpl = m_owner_graph;
    if (gimpl->m_current_comp_seq) {
        // remove previous handlers
        auto prev_seq =
                static_cast<ComputingSequence*>(gimpl->m_current_comp_seq);
        prev_seq->cleanup();
    }
    if (gimpl->options().var_sanity_check_first_run) {
        m_var_sanity_check = std::make_unique<VarSanityCheck>(gimpl);
    }
    gimpl->m_current_comp_seq = this;
}

ComputingGraphImpl::ComputingSequence::~ComputingSequence() {
    MGB_TRY { cleanup(); }
    MGB_HANDLE_EXCEPTION_DTOR("ComputingSequence dtor");
    if (!is_finalized()) {
        // always wait on comp node because the do_wait() impl only waits for
        // events, whose callback may have not been fully finished
        for (auto&& i : m_used_comp_node) {
            if (i.contain_flag(CompNode::Flag::EVENT_DTOR_UNSAFE)) {
                i.sync();
            }
        }
    }
}

void ComputingGraphImpl::ComputingSequence::do_wait(bool explicit_user_wait) {
    if (m_wait_finished)
        return;

    check_not_finalized();

    for (auto i : m_owner_graph->m_subgraphs) {
        if (i->m_current_comp_seq) {
            auto seq = static_cast<ComputingSequence*>(i->m_current_comp_seq);
            seq->do_wait(explicit_user_wait);
        }
    }

    for (auto cn : m_used_comp_node) {
        m_event_end.at(cn)->host_wait();
    }
    m_wait_finished = true;
#if MGB_NEED_MEGDNN_ASYNC_ERROR
    // FIXME: It CAN NOT work well if more than one ComputingSequnces has been
    // executed on the same compnode and got AsyncError concurrently, because
    // only the first async error on each comp_node would be recorded.
    for (auto&& cn : m_used_comp_node) {
        auto error = cn.check_async_error();
        if (error) {
            static_cast<const OperatorNodeExcExtraInfo*>(error->extra_info())
                    ->opr()
                    ->owner_graph()
                    ->record_async_error(std::move(error));
        }
    }
#endif
    m_owner_graph->event().signal_inplace<event::CompSeqExecFinished>(
            explicit_user_wait, true, m_owner_graph, this);

    if (m_async_exc) {
        auto tmp_async_exc = std::move(m_async_exc);
        mgb_throw_raw(*tmp_async_exc);
    }
}

void ComputingGraphImpl::ComputingSequence::cleanup() {
    m_var_sanity_check.reset();
    if (has_uncaught_exception()) {
        mgb_log_warn(
                "fallback to simple graph waiting in dtor due to uncaught "
                "exceptions");
        if (!m_wait_finished) {
            MGB_TRY {
                for (auto&& i : m_used_comp_node) {
                    i.sync();
                }
            }
            MGB_CATCH(..., {})
        }
    } else {
        wait();
    }
    if (m_owner_graph->m_current_comp_seq == this) {
        m_owner_graph->m_current_comp_seq = nullptr;
        MGB_TRY { m_owner_graph->clear_device_memory(); }
        MGB_CATCH(std::exception & exc, {
            mgb_log_error("failed to clear device memory: %s", exc.what());
        });
    }

    // ensure clear user data before destructing m_owner_graph
    user_data().clear_all_user_data();
}

void ComputingGraphImpl::ComputingSequence::init_for_exec() {
    if (m_first_exec) {
        on_first_exec();
    }

    // add all tasks into exec env
    m_exec_env.clear();
    if (!m_have_parent_graph) {
        record_all_event(m_event_start);
    }
    for (auto i : *m_opr_seq) {
        m_exec_env.set_active_opr(i);
        i->execute(m_exec_env);
    }
    m_exec_env.set_active_opr(nullptr);
    record_all_event(m_event_end);

    m_cg_event_version = m_owner_graph->event().version();
}

void ComputingGraphImpl::ComputingSequence::on_first_exec() {
    mgb_assert(m_first_exec);
    for (auto i : *m_opr_seq) {
        for (auto j : i->output())
            m_used_comp_node.insert(j->comp_node());
    }

    // we maintain a recorder because events may depend on whether recorder
    // is enabled
    auto recorder = check_enable_comp_node_seq_recorder();
    auto&& options = m_owner_graph->options();
    //! The recorder in comp_node is thread_local, so the create thread should
    //! the same as the execute thread, so set the Synchronize mode
    if (m_enable_comp_node_seq_recorder) {
        m_exec_env.set_async_level(0);
    } else {
        m_exec_env.set_async_level(options.async_exec_level);
    }
    if (options.async_exec_level) {
        for (auto i : m_used_comp_node)
            m_exec_env.add_comp_node(i);
    }

    // create events for timing and sync
    for (auto&& i : m_used_comp_node) {
        size_t flag = 0;
        if (!m_have_parent_graph) {
            flag = CompNode::Event::NEED_TIMER;
            m_event_start[i] = i.create_event(flag);
        }
        m_event_end[i] = i.create_event(flag);
    }
    m_first_exec = false;
}

AsyncExecutable& ComputingGraphImpl::ComputingSequence::execute() {
    check_not_finalized();
    do_execute(nullptr);
    return *this;
}

AsyncExecutable& ComputingGraphImpl::ComputingSequence::wait() {
    do_wait(true);
    return *this;
}

double ComputingGraphImpl::ComputingSequence::get_prev_exec_time() const {
    check_not_finalized();
    mgb_assert(m_wait_finished);
    if (m_prev_exec_time.valid()) {
        return m_prev_exec_time.val();
    }
    if (!m_have_parent_graph) {
        double max_time = 0;
        for (auto cn : m_used_comp_node) {
            update_max(max_time, m_event_start.at(cn)->elapsed_time_until(
                                         *m_event_end.at(cn)));
        }
        m_prev_exec_time = max_time;
        return max_time;
    }
    return 0;
}

AsyncExecutable& ComputingGraphImpl::ComputingSequence::iter_opr_seq(
        thin_function<bool(OperatorNodeBase*)> cb) {
    for (auto i : *m_opr_seq) {
        if (!cb(i))
            return *this;
    }
    return *this;
}

void ComputingGraphImpl::ComputingSequence::clear_device_memory() {
    check_not_finalized();
    if (m_owner_graph->current_comp_seq() == this) {
        m_owner_graph->clear_device_memory();
    }
}

const CompNode::UnorderedMap<size_t>&
ComputingGraphImpl::ComputingSequence::update_static_alloc_plan_and_get_size() {
    assert_latest_comp_seq();
    // waiting for previous execution or some tensor storage may be freed after
    // calling update_static_alloc_plan, which would cause use-after-free.
    do_wait(false);
    auto&& mgr = m_owner_graph->var_node_mem_manager();
    mgr.update_static_alloc_plan();
    return mgr.get_static_alloc_size();
}

#if MGB_ENABLE_JSON
std::shared_ptr<json::Value> ComputingGraphImpl::ComputingSequence::to_json()
        const {
    ThinHashSet<MemAllocPlan::Chunk*> all_mem_chunk;
    VarNodeSet all_var_node;
    ThinHashSet<OperatorNodeBase*> all_opr_node;

    auto comp_seq = json::Array::make();

    for (auto i : *m_opr_seq) {
        all_opr_node.insert(i);
        for (auto j : i->output()) {
            all_var_node.insert(j);
            if (j->mem_plan().valid())
                all_mem_chunk.insert(&j->mem_plan().chunk());
        }
        comp_seq->add(json::String::make(i->id_str()));
    }

    // expand opr and var nodes that do not appear in comp seq,
    // also expand var nodes which are only used in static infer
    {
        VarNodeArray new_var_node;
        auto&& mgr = m_owner_graph->static_infer_manager_impl();
        auto check_opr_input = [&](OperatorNodeBase* opr) {
            auto update = [&](VarNode* var) {
                if (!(all_var_node.count(var))) {
                    all_var_node.insert(var);
                    new_var_node.push_back(var);
                }
            };
            for (auto i : opr->input()) {
                update(i);
            }
            for (auto &&out : opr->output()) {
                using DepType = static_infer::DepType;
                for (auto&& i : mgr.get_deps({out, DepType::SHAPE})) {
                    update(i.dest);
                }
                for (auto&& i : mgr.get_deps({out, DepType::VALUE})) {
                    update(i.dest);
                }
            }
        };
        for (auto i : all_opr_node)
            check_opr_input(i);
        while (!new_var_node.empty()) {
            auto opr = new_var_node.back()->owner_opr();
            new_var_node.pop_back();
            all_opr_node.insert(opr);
            for (auto i : opr->output()) {
                all_var_node.insert(i);
            }
            check_opr_input(opr);
        }
    }

    auto dump_node_coll = [](auto&& collection) {
        auto objptr = json::Object::make();
        auto&& obj = *objptr;
        for (auto&& i : collection)
            obj[i->id_str()] = i->to_json();
        return objptr;
    };

    return json::Object::make({{"operator", dump_node_coll(all_opr_node)},
                               {"var", dump_node_coll(all_var_node)},
                               {"mem_chunk", dump_node_coll(all_mem_chunk)},
                               {"comp_seq", comp_seq}});
}
#endif

/* ========================== MegDNNDtorCheck ========================== */
void ComputingGraphImpl::MegDNNDtorCheck::enable() {
    mgb_assert(!m_enabled);
    m_enabled = true;
    auto cb_dnn = [](megdnn::OperatorBase* opr) {
        mgb_log_error("unexpected destruction of megdnn opr %p", opr);
        mgb_trap();
    };
    auto cb_mem = [](size_t alloc_size, bool, void* ptr) {
        if (!alloc_size) {
            mgb_log_error("unexpected mem release %p", ptr);
            mgb_trap();
        }
    };
    m_orig_dnn_cb = cb_dnn;
    m_orig_mem_cb = cb_mem;
    m_handle->set_opr_destruct_callback(m_orig_dnn_cb);
    m_env->mem_event_handler(m_orig_mem_cb);
    mgb_assert(!m_orig_dnn_cb && !m_orig_mem_cb);
}

ComputingGraphImpl::MegDNNDtorCheck::~MegDNNDtorCheck() {
    if (m_enabled) {
        m_handle->set_opr_destruct_callback(m_orig_dnn_cb);
        m_env->mem_event_handler(m_orig_mem_cb);
    }
    if (m_comp_seq) {
        m_comp_seq->on_graph_destroy();
    }
}

/* ======================= RecordedComputingSequence ======================= */

std::unique_ptr<ComputingGraphImpl::RecordedComputingSequence>
ComputingGraphImpl::ComputingSequence::as_recorded_seq() {
    on_first_exec();
    mgb_assert(m_enable_comp_node_seq_recorder,
               "can not enable comp_node_seq_record_level=2; more details are "
               "included in previous log messages");

    mgb_assert(m_used_comp_node.size() == 1);
    auto comp_node = *m_used_comp_node.begin();
    MegDNNDtorCheck megdnn_dtor_check{comp_node};

    // execute to get recorded comp seq
    mgb_assert(!m_owner_graph->options().fake_next_exec);
    m_owner_graph->options().fake_next_exec = true;
    do_execute(&megdnn_dtor_check);
    // to avoid wait at graph dtor which causes segfault because the events
    // would have been moved away from this seq
    m_wait_finished = true;

    auto ret = std::make_unique<RecordedComputingSequence>(m_owner_graph);
    m_owner_graph->m_recorded_seq_level2_dtor_chk.reset(
            new MegDNNDtorCheck{comp_node, ret.get()});

    // record opr dependencies
    ThinHashSet<OperatorNodeBase*> used_oprs;
    for (auto&& i : *m_opr_seq) {
        i->record_execute_deps(ret->m_exec_deps);
        used_oprs.insert(i);
    }
    for (auto&& i : ret->m_exec_deps) {
        if (i->has_runtime_check()) {
            ret->m_runtime_checks.push_back(i.get());
        }
    }

    // also record unused oprs so the MegDNNDtorCheck would not fail
    auto&& unused_deps =
            m_owner_graph->m_recorded_seq_level2_dtor_chk->safe_dtor_objs();
    for (auto&& i : m_owner_graph->m_opr_refkeeper) {
        if (!used_oprs.count(i.get())) {
            i->record_execute_deps(unused_deps);
        }
    }

    // graph user data main contain ref holders for tmp variables
    ret->m_graph_user_data.swap(m_owner_graph->options().user_data);

    // move other dependencies
    unpack_vector(m_owner_graph->var_node_mem_manager()
                          .static_device_memory_refholder(),
                  ret->m_static_mem);
    mgb_assert(m_event_start.size() == 1 && m_event_end.size() == 1);
    ret->m_event_start = std::move(m_event_start.begin()->second);
    ret->m_event_end = std::move(m_event_end.begin()->second);
    ret->user_data().swap(user_data());
    ret->m_recorder = std::move(m_comp_node_seq_recorder);

    return ret;
}

AsyncExecutable& ComputingGraphImpl::RecordedComputingSequence::execute() {
    check_not_finalized();

    mgb_assert(!m_owner_graph,
               "owner graph should be destroyed before using AsyncExecutable "
               "compiled with comp_node_seq_record_level=2");
    mgb_assert(m_recorder, "graph memory already cleared");
    m_prev_exec_time = None;
    if (!m_wait_finished) {
        wait();
    }
    m_wait_finished = false;
    for (auto i : m_runtime_checks) {
        i->do_runtime_check();
    }
    m_recorder->replay();
    return *this;
}

AsyncExecutable& ComputingGraphImpl::RecordedComputingSequence::wait() {
    check_not_finalized();

    if (!m_wait_finished) {
        m_event_end->host_wait();
        m_wait_finished = true;
    }
    return *this;
}

double ComputingGraphImpl::RecordedComputingSequence::get_prev_exec_time()
        const {
    mgb_assert(m_wait_finished);
    if (!m_prev_exec_time.valid()) {
        m_prev_exec_time = m_event_start->elapsed_time_until(*m_event_end);
    }
    return m_prev_exec_time.val();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
