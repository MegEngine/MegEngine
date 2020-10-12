/**
 * \file src/core/impl/graph/var_node_mem_mgr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl.h"
#include "./cg_impl_seq.h"

#include "var_node_mem_mgr.h"
#include "megbrain/comp_node_env.h"

#include "megbrain/graph/cg.h"
#include "megbrain/graph/helper.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/event.h"

#include "megbrain/system.h"
#include "megbrain/utils/timer.h"
#include "megbrain/utils/arith_helper.h"

#include "megbrain/opr/io.h"

#include <chrono>

using namespace mgb;
using namespace cg;

namespace {

void call_mem_status_changed(cg::OperatorNodeBase* opr) {
    auto cb = opr->get_opr_event_callback();
    if (cb.on_mem_status_changed.valid())
        cb.on_mem_status_changed.val()();
}
}  // namespace

/* ==================== StaticDeviceMemoryManager ==================== */

StaticDeviceMemoryManager::StaticDeviceMemoryManager()
        : m_allocator{std::make_shared<DeviceMemoryAllocator>()} {}

void StaticDeviceMemoryManager::exec_enter() {
    auto flag = m_in_exec.test_and_set();
    mgb_assert(!flag, "double-lock on StaticDeviceMemoryManager");
}

void StaticDeviceMemoryManager::exec_exit() {
    mgb_assert(m_in_exec.test_and_set());
    m_in_exec.clear();
}

const DeviceTensorStorage& StaticDeviceMemoryManager::alloc(
        ComputingGraph* graph, CompNode cn, size_t size, size_t cur_version) {
    if (cur_version != m_version) {
        m_storage.clear();
        m_version = cur_version;
    }
    auto&& storage = m_storage[cn];
    if (size > storage.size()) {
        if (!storage.comp_node_valid()) {
            storage.comp_node(cn);
        }
        m_allocator->alloc_static(graph, storage, size);
        auto ptr = storage.ptr();
        MGB_MARK_USED_VAR(ptr);
        mgb_assert(storage.size() >= size);
        mgb_log_debug(
                "static storage on %s: size=%.2fMiB addr_range=[%p, %p). ",
                cn.to_string().c_str(), storage.size() / 1024.0 / 1024.0, ptr,
                ptr + storage.size());
    }
    return storage;
}

void StaticDeviceMemoryManager::prefault() {
    for (auto&& i : m_storage) {
        if (i.first.device_type() == CompNode::DeviceType::CPU) {
            auto set = [ ptr = i.second.ptr(), size = i.second.size() ]() {
                memset(ptr, 0, size);
            };
            CompNodeEnv::from_comp_node(i.first).cpu_env().dispatch(set);
            i.first.sync();
        }
    }
}

#if MGB_THREAD_SAFE
size_t StaticDeviceMemoryManager::clear_all() {
    size_t ret = 0;
    for (auto&& i : m_storage) {
        update_max(ret, i.second.use_count());
    }
    m_storage.clear();
    return ret;
}

std::shared_ptr<StaticDeviceMemoryManager>
StaticDeviceMemoryManager::make_default_impl() {
    return std::make_shared<StaticDeviceMemoryManager>();
}
#else
size_t StaticDeviceMemoryManager::clear_all() {
    // do not actually clear so memory can be shared and reused by other graphs
    // (since all graphs share the same StaticDeviceMemoryManager, releasing
    // memory here would require memory allocation on next execution, so in the
    // worst case n graphs would have n different static buffers even if they
    // have the same static memory size)
    return 1;
}

std::shared_ptr<StaticDeviceMemoryManager>
StaticDeviceMemoryManager::make_default_impl() {
    // use a global instance to share memory across all graphs. It is safe
    // because we need no thread safety
    static StaticDeviceMemoryManager inst;

    return {std::shared_ptr<StaticDeviceMemoryManager>{}, &inst};
}
#endif  // MGB_THREAD_SAFE

/* ==================== AsyncVarReleaser ==================== */
#if MGB_CUDA || MGB_ATLAS
class VarNodeMemManager::AsyncVarReleaser {
    struct WaiterParam {
        CompNode cn;
        CompNode::Event *event;
        VarNode *var;
    };

    class Waiter final: public AsyncQueueSC<WaiterParam, Waiter> {
        AsyncVarReleaser *m_par_releaser;

        public:
            Waiter(AsyncVarReleaser *releaser):
                m_par_releaser(releaser)
            {
            }

            void process_one_task(const WaiterParam &param) {
                if (param.event->finished()) {
                    VarNodeMemManager::decr_var_mem_refcnt_sync(param.var);
                    MGB_LOCK_GUARD(m_par_releaser->m_event_pool_lock);
                    m_par_releaser->m_event_pool.at(param.cn).free(param.event);
                    return;
                }

                using namespace std::literals;
                std::this_thread::sleep_for(1us);
                add_task(param);
            }
    };
    Waiter m_waiter{this};
    CompNode::UnorderedMap<CompNode::EventPool> m_event_pool;
    Spinlock m_event_pool_lock;

    public:
        ~AsyncVarReleaser() {
            wait_release_finish();
        }

        void add(CompNode cn, VarNode *var) {
            CompNode::EventPool *pool;
            {
                MGB_LOCK_GUARD(m_event_pool_lock);
                auto iter = m_event_pool.find(cn);
                if (iter == m_event_pool.end()) {
                    iter = m_event_pool.emplace(
                            std::piecewise_construct,
                            std::forward_as_tuple(cn),
                            std::forward_as_tuple(cn)).first;
                }
                pool = &iter->second;
            }
            auto event = pool->alloc();
            event->record();
            m_waiter.add_task({cn, event, var});
        }

        void wait_release_finish() {
            m_waiter.wait_task_queue_empty();
            for (auto &&i: m_event_pool)
                i.second.assert_all_freed();
        }
};
#endif

/* ==================== ImpureMemPlanManager ==================== */
void VarNodeMemManager::ImpureMemPlanManager::record_ptr_changed(
        VarNodeMemManager* mgr, VarNode* var) {
    if (m_during_check) {
        if (var->m_mem_plan.next_readonly_fwd_reader()) {
            m_ptr_changed_mplans.emplace_back(&var->m_mem_plan);
        }
        if (auto dst = mgr->get_var_node_mem_trait_at(var).seq_force_update_dest) {
            if (mgr->m_sys_alloc_static_vars.count(dst)) {
                m_force_update_pairs.emplace_back(var, dst);
            }
        }
    }
}

bool VarNodeMemManager::ImpureMemPlanManager::check_need_realloc() {
    m_during_check = true;
    m_layout_changed = false;
    m_ptr_changed_mplans.clear();
    m_force_update_pairs.clear();

    for (auto opr : m_oprs) {
        opr->init_output_mem_plan(false);
        if (m_layout_changed) {
            m_during_check = false;
            return true;
        }
    }

    m_during_check = false;

    // update seq-force-update-dest
    for (auto &&pi : m_force_update_pairs) {
        auto src = pi.first, dst = pi.second;
        auto&& chk = src->m_mem_plan.chunk();
        auto&& storage = chk.owner_var->dev_tensor().storage();
        mgb_assert(chk.mem_alloc_status.is_from_owner_var());
        make_dev_tensor_from_mem_plan_single(dst, storage);
        m_ptr_changed_mplans.emplace_back(&dst->m_mem_plan);
    }

    for (auto mp : m_ptr_changed_mplans) {
        auto&& chk = mp->chunk();
        mgb_assert(chk.mem_alloc_status.is_from_owner_var());
        auto&& storage = chk.owner_var->dev_tensor().storage();
        for (auto reader = mp->next_readonly_fwd_reader(); reader;
             reader = reader->next_readonly_fwd_reader()) {
            mgb_assert(&reader->chunk() == &chk);
            make_dev_tensor_from_mem_plan_single(reader->owner_var(), storage);
        }
    }
    return false;
}

/* ==================== VarNodeMemManager ==================== */
VarNodeMemManager::VarNodeMemManager(ComputingGraphImpl *graph):
    m_owner_graph(graph),
    m_seq_mem_opt(graph)
#if MGB_CUDA || MGB_ATLAS
    ,m_asyn_var_releaser(new AsyncVarReleaser)
#endif
{
    auto on_comp_seq_finish = [this](const event::CompSeqExecFinished& ev) {
        MGB_MARK_USED_VAR(ev);
        // async release is only used for sync between multiple comp nodes, and
        // does not wait for device to finish
#if MGB_CUDA || MGB_ATLAS
        m_asyn_var_releaser->wait_release_finish();
#endif
        m_cpu_async_release_barrier.wait_zero();
    };

    auto on_comp_seq_error = [this](const event::CompSeqExecError&) {
        // release vars on error due to:
        // 1. refcnt is checked in on_var_node_device_comp_finish()
        // 2. some var may be unused in next run due to conditional execution,
        //    so memory would leak if we do not release them now
        mgb_log_error(
                "error occurred in computing sequence; synchronizing all comp "
                "nodes and releasing vars now ...");
        MGB_TRY {
            CompNode::sync_all();
            for (auto i : m_need_post_exec_action_vars) {
                if (!is_inf_refcnt_init(i) && i->m_refcnt) {
                    i->m_refcnt = 1;
                    decr_var_mem_refcnt_sync(i);
                }
            }
        }
        MGB_CATCH(const std::exception& exc, {
            mgb_log_error(
                    "caught exception during cleanup: %s; ignored due to "
                    "nested exception",
                    exc.what());
        })
        MGB_CATCH(..., {
            mgb_log_error(
                    "caught unknown exception during cleanup; ignored due to "
                    "nested exception");
        })
    };

    graph->event().register_receiver_permanent<event::CompSeqExecFinished>(
            on_comp_seq_finish);
    graph->event().register_receiver_permanent<event::CompSeqExecError>(
            on_comp_seq_error);

#if MGB_ENABLE_VAR_DEV_MEM_DEFRAGMENTER && (MGB_CUDA || MGB_ATLAS)
    auto on_mem_defrag_start = [this](const event::BeforeMemDefrag&) {
        m_asyn_var_releaser->wait_release_finish();
    };
    graph->event().register_receiver_permanent<event::BeforeMemDefrag>(
            on_mem_defrag_start);
#endif
}

VarNodeMemManager::~VarNodeMemManager() noexcept = default;

void VarNodeMemManager::VarNodeMemTrait::clear_opt_status() {
    readonly_src = nullptr;
}

bool VarNodeMemManager::DynamicAllocOprInfo::check_if_mem_status_change() {
    bool same = true;
    if (prev_dev_val_input.size() != dev_val_input.size()) {
        same = false;
        prev_dev_val_input.resize(dev_val_input.size());
    }
    for (size_t i = 0; i < dev_val_input.size(); i ++) {
        auto &&t = prev_dev_val_input[i];
        auto s = dev_val_input[i]->dev_tensor().as_megdnn();
        if (t.raw_ptr != s.raw_ptr || !t.layout.eq_layout(s.layout)) {
            same = false;
            t = s;
        }
    }
    for (auto &&i: static_infer_inp) {
        auto new_v = i.first->update_infer_result_version();
        same &= i.second == new_v;
        i.second = new_v;
    }
    return !same;
}

VarNodeMemManager::DynamicAllocOprInfo::DynamicAllocOprInfo(
        OperatorNodeBase *opr) {
    alloc_comp_seq_exec_id = -1;
    prev_dev_val_input.clear();
    static_infer_inp.clear();
    dev_val_input.clear();
    auto &&mgr = ComputingGraphImpl::downcast(opr->owner_graph())->
        static_infer_manager_impl();

    CompNode single_cn;
    {
        auto &&cnset = cg::get_opr_comp_node_set(opr);
        if (cnset.size() == 1)
            single_cn = *cnset.begin();
    }

    for (auto &&i: opr->node_prop().dep_map()) {
        using DT = OperatorNodeBase::NodeProp::DepType;

        if (i.second & DT::DEV_VALUE)
            dev_val_input.push_back(i.first);

        if (i.second & DT::HOST_VALUE)
            static_infer_inp.push_back({
                    mgr.get_tag_handler_for_value(i.first), 0});

        if (i.second & DT::SHAPE)
            static_infer_inp.push_back({
                    mgr.get_tag_handler_for_shape(i.first), 0});
    }

    has_dynamic_storage_input = !is_all_input_static_storage(opr);
}

bool VarNodeMemManager::alloc_var_node_mem_static() {
    RealTimer timer;

    if (!update_static_alloc_plan()) {
        // mem plan unchanged, just do the actual allocation
        return make_static_var_tensor_from_alloc_plan();
    }

    auto time0 = timer.get_msecs();
    make_static_var_tensor_from_alloc_plan();

    MGB_MARK_USED_VAR(time0);
    if (m_owner_graph->options().log_level) {
        auto time1 = timer.get_msecs();
        MGB_MARK_USED_VAR(time1);
        mgb_log_debug("static memory allocation: nr_opr=%zu nr_var=%zu realtime=%.2fmsec"
                " (plan%.2f alloc%.2f)",
                m_sys_alloc_static_oprs.size(), m_sys_alloc_static_vars.size(),
                time1, time0, time1 - time0);
    }

    return true;
}

bool VarNodeMemManager::update_static_alloc_plan() {
    // check whether unchanged
    bool free_no_need_memory = free_combine_memory_no_need_var();
    if (!m_owner_graph->static_infer_comp_seq_manager()
                 .update_static_check_shape_change() &&
        !m_first_static_plan_run &&
        !m_impure_mem_plan_mgr.check_need_realloc()) {
        return false || free_no_need_memory;
    }

    if (m_first_static_plan_run)
        init_dynamic_alloc_opr_info();

    // repeat allocating until plan_chunk_allocation() returns false
    for (;;) {
        // kill profiling worker since shape is known
        sys::TimedFuncInvoker::ins().kill_worker();

        for (auto opr: *m_opr_seq) {
            for (auto &&var: opr->output()) {
                if (auto trait = get_var_node_mem_trait_nullable(var)) {
                    trait->clear_opt_status();
                }
            }
        }

        for (auto var : m_sys_alloc_static_vars)
            var->m_mem_plan.reset_to_uninitialized();

        for (auto opr : m_sys_alloc_static_oprs)
            init_opr_outputs_mem_plan(opr, false);

        m_seq_mem_opt.optimize_mem_plan();

        if (!m_seq_mem_opt.plan_chunk_allocation()) {
            break;
        }

        m_owner_graph->static_infer_comp_seq_manager()
                .update_static_check_shape_change();
    }
    m_first_static_plan_run = false;
    // ensure that next call to make_static_var_tensor_from_alloc_plan() would
    // be effective
    m_static_mem_refholder_dev_mem_mgr_version =
            DeviceMemoryAllocator::VERSION_INVALID;
    return true;
}

bool VarNodeMemManager::make_static_var_tensor_from_alloc_plan() {
    auto&& cn2usage = m_seq_mem_opt.static_mem_usage();
    auto cur_version = m_static_dev_mem_mgr->version(m_owner_graph);
    mgb_assert(cur_version != DeviceMemoryAllocator::VERSION_INVALID);
    if (cur_version == m_static_mem_refholder_dev_mem_mgr_version) {
        return false;
    }

    m_static_mem_refholder.clear();

    auto&& dev_mem_mgr = *m_static_dev_mem_mgr;
    CompNode::UnorderedMap<DeviceTensorStorage> cn2storage;

    for (auto&& i : cn2usage) {
        DeviceTensorStorage storage = dev_mem_mgr.alloc(m_owner_graph, i.first,
                                                        i.second, cur_version);
        m_static_mem_refholder.emplace_back(storage);
        // the reference has been kept in m_static_mem_refholder, and we drop
        // the ref now so clear_static_device_memory() can be easily implemented
        using S = DeviceTensorStorage::RawStorage;
        S ptr(S{}, storage.ptr());
        storage.reset(storage.comp_node(), storage.size(), std::move(ptr));
        cn2storage[i.first] = std::move(storage);
    }

    for (auto opr : m_sys_alloc_static_oprs) {
        for (VarNode* var : opr->output()) {
            if (m_sys_alloc_static_vars.count(var)) {
                auto&& chunk = var->m_mem_plan.chunk();
                if (!chunk.size()) {
                    // empty chunks need no allocation
                    make_dev_tensor_from_mem_plan_single(var, {});
                } else if (chunk.mem_alloc_status.is_static_offset()) {
                    make_dev_tensor_from_mem_plan_single(
                            var, cn2storage.at(var->comp_node()),
                            chunk.mem_alloc_status.static_offset());
                } else {
                    // allocated by opr during init_output_mem_plan()
                    mgb_assert(chunk.mem_alloc_status.is_from_owner_var());
                    if (chunk.owner_var != var) {
                        make_dev_tensor_from_mem_plan_single(
                                var, chunk.owner_var->dev_tensor().storage());
                    }
                }
            }
        }
        if (m_sys_alloc_static_oprs_need_mem_status_changed_cb.count(opr)) {
            opr->get_opr_event_callback().on_mem_status_changed.val()();
        }
    }

    m_static_mem_refholder_dev_mem_mgr_version = cur_version;
    return true;
}

bool VarNodeMemManager::free_combine_memory_no_need_var() {
    if (!m_owner_graph->options().graph_opt.weight_preprocess ||
        m_already_free_no_need_mem) {
        return false;
    }
    bool reordered = false;
    //! free no need storage
    for (auto opr : *m_opr_seq) {
        if (opr->try_cast_final<opr::SharedDeviceTensor>() ||
            opr->try_cast_final<opr::SharedDeviceTensorWithFormat>()) {
            auto opr_base = static_cast<opr::intl::SharedDeviceTensorBase*>(opr);
            auto var = opr_base->output(0);
            if (var->contain_flag(VarNode::Flag::MEMORY_NO_NEED) &&
                var->dev_tensor_valid() && !var->dev_tensor().empty()) {
                //! Only the tensor share count is 1, it can be free
                if (opr_base->dev_data().use_count() == 1) {
                    auto layout = var->layout();
                    var->m_dev_tensor.reset(
                            DeviceTensorStorage{var->comp_node()}, layout);
                    opr_base->free_dev_data();
                    mgb_log_debug(
                            "preprocessed weight is freed, var name = %s, "
                            "var layout = %s",
                            var->name().c_str(), layout.to_string().c_str());
                }
                m_already_free_no_need_mem = true;
            }
        }
        bool memory_need_reorder = false;
        if (opr->try_cast_final<opr::MultipleDeviceTensorHolder>() ||
            opr->try_cast_final<opr::MultipleDeviceTensorWithFormatHolder>()) {
            auto opr_base =
                    static_cast<opr::intl::MultipleDeviceTensorHolderBase*>(
                            opr);
            for (size_t index = 0; index < opr_base->output().size(); index++) {
                auto var = opr_base->output(index);
                if (var->contain_flag(VarNode::Flag::MEMORY_NO_NEED) &&
                    var->dev_tensor_valid() && !var->dev_tensor().empty()) {
                    //! Only the tensor share count is 1, it can be free
                    if (opr_base->values()[index].use_count() == 1) {
                        auto layout = var->layout();
                        var->m_dev_tensor.reset(
                                DeviceTensorStorage{var->comp_node()}, layout);
                        opr_base->mutable_values()[index]->reset(
                                DeviceTensorStorage{var->comp_node()}, layout);
                        memory_need_reorder = true;
                        mgb_log_debug(
                                "preprocessed weight is freed, var name "
                                "= %s, var layout = %s",
                                var->name().c_str(),
                                layout.to_string().c_str());
                    }
                    m_already_free_no_need_mem = true;
                }
            }
        }
        //! recorder the other needed outputs, because they share the
        //! same chunk of mem in device with no needed var, see
        //! BatchedDeviceValueLoader
        if (memory_need_reorder) {
            auto opr_base =
                    static_cast<opr::intl::MultipleDeviceTensorHolderBase*>(
                            opr);
            auto comp_node = opr_base->output(0)->comp_node();
            bool is_device_opr =
                    comp_node.mem_node() != CompNode::default_cpu().mem_node();
            if (memory_need_reorder && is_device_opr) {
                for (size_t index = 0; index < opr_base->output().size();
                     index++) {
                    auto var = opr_base->output(index);
                    if (!var->contain_flag(VarNode::Flag::MEMORY_NO_NEED)) {
                        DeviceTensorStorage storage(var->comp_node());
                        size_t size = var->layout().span().dist_byte();
                        storage.ensure_size(size);
                        storage.copy_from(var->m_dev_tensor.storage(), size);

                        var->m_dev_tensor.reset(storage, var->layout());
                        opr_base->mutable_values()[index]->reset(storage,
                                                                var->layout());
                        reordered = true;
                    }
                }
                //! sync to make sure memcopy is finished
                comp_node.sync();
            }
        }
    }
    return reordered;
}

void VarNodeMemManager::init_dynamic_alloc_opr_info() {
    mgb_assert(m_first_static_plan_run);
    m_need_post_exec_action_vars.clear();
    m_var_dev_mem_defragmenter.clear_all();
    m_var_dev_mem_defragmenter.set_enable(
            m_owner_graph->options().enable_var_mem_defragment);
    bool is_eager_eval = m_owner_graph->eager_eval_manager().enabled();
    for (auto opr: *m_opr_seq) {
        auto info = m_dynamic_alloc_opr_info.alloc(opr);

        bool input_need_refcnt = false;
        for (auto&& pair : opr->node_prop().dep_map()) {
            if (OperatorNodeBase::NodeProp::is_device_value_dep(pair.second)) {
                pair.first->m_refcnt_init += opr->output().size();
                if (!is_inf_refcnt_init(pair.first)) {
                    input_need_refcnt = true;
                }
            }
        }
        for (auto&& i : opr->input_waiting_spec()) {
            for (auto j : i.dev_ready) {
                m_need_post_exec_action_vars.insert(j);
            }
        }

        bool all_sys_alloc = true;

        for (auto i: opr->output()) {
            m_node_mem_trait[i];
            bool static_alloc = m_sys_alloc_static_vars.count(i);

            if (static_alloc || is_eager_eval ||
                    i->contain_flag(VarNode::Flag::NO_MEM_RECLAIM)) {
                i->m_refcnt_init = m_owner_graph->eager_eval_manager().get_var_nr_readers(i);
            } else {
                i->m_refcnt_init = 0;
            }

            if (input_need_refcnt || !i->m_refcnt_init) {
                m_need_post_exec_action_vars.insert(i);
            }

            if (i->m_should_sys_alloc) {
                if (!static_alloc) {
                    info->dynamic_alloc_output.push_back(i);
                }
            } else {
                all_sys_alloc = false;
            }
        }

        if (all_sys_alloc) {
            // the outputs of oprs whose all output vars are managed by the
            // system can be safely moved
            for (auto i: info->dynamic_alloc_output) {
                m_var_dev_mem_defragmenter.register_var(i);
            }
        }

        if (info->has_dyn_input_or_output()) {
            m_dynamic_alloc_opr_info.set(opr, std::move(info));
        } else {
            if (opr->get_opr_event_callback().on_mem_status_changed.valid()) {
                m_sys_alloc_static_oprs_need_mem_status_changed_cb.insert(opr);
            }
        }
    }
}

void VarNodeMemManager::alloc_var_node_mem_dynamic(
        GraphExecutable::ExecEnv &env, OperatorNodeBase *opr) {
    auto info = m_dynamic_alloc_opr_info.get(opr);

    if (!info || !info->has_dyn_input_or_output())
        return;

    auto cnset = cg::get_opr_comp_node_set(opr);

    if (info->dynamic_alloc_output.empty()) {
        // has dynamic input storage but static output storage;
        // we only need to check mem status change in such case

        auto cbspec = opr->get_opr_event_callback();
        if (!cbspec.on_mem_status_changed.valid())
            return;

        auto check_mem_status = [info,
                 cb=cbspec.on_mem_status_changed.val()]() {

            auto &&opr_mtx = info->mtx;
            MGB_LOCK_GUARD(opr_mtx);
            if (info->check_if_mem_status_change())
                cb();
        };

        for (auto &&cn: cnset)
            env.dispatch_on_comp_node(cn, check_mem_status);
        return;
    }

    auto alloc = [this, opr, info]() {
        MGB_LOCK_GUARD(info->mtx);

        size_t cur_run_id = *m_run_id_ptr;
        if (info->alloc_comp_seq_exec_id == cur_run_id)
            return;

        auto &&mgr = m_owner_graph->static_infer_manager_impl();
        for (auto i: info->dynamic_alloc_output) {
            m_node_mem_trait.at(i).clear_opt_status();
            i->shape(mgr.infer_shape(i));
            i->m_mem_plan.reset_to_uninitialized();
        }

        init_opr_outputs_mem_plan(opr, true);
        {
            MGB_LOCK_GUARD(m_dynamic_alloc_mtx);
            m_seq_mem_opt.optimize_mem_plan_dynamic(opr);
        }

        for (auto i: info->dynamic_alloc_output) {
            if (!m_node_mem_trait.at(i).has_dynamic_mem_fwd_from_other()) {
                var_alloc_with_shape(i, i->shape());
            } else {
                // dynamic forwarding from another var
                auto span = i->m_mem_plan.layout().span();
                span.low_byte += i->m_mem_plan.offset_in_chunk_byte();
                span.high_byte += i->m_mem_plan.offset_in_chunk_byte();
                auto src_var = i->m_mem_plan.chunk().owner_var;
                mgb_assert(src_var != i);
                auto&& storage = src_var->m_dev_tensor.storage();
                mgb_assert(storage.valid_span(span));
                mgb_assert(i->m_mem_plan.layout().eq_shape(i->shape()));
                make_dev_tensor_from_mem_plan_single(i, storage);
            }
        }

        call_mem_status_changed(opr);
        info->alloc_comp_seq_exec_id = cur_run_id;
    };

    // note that alloc() is dispatched to all comp nodes, and only the first one
    // that executes alloc() is effective
    for (auto &&cn: cnset)
        env.dispatch_on_comp_node(cn, alloc);
}

void VarNodeMemManager::init_opr_outputs_mem_plan(
        OperatorNodeBase *opr, bool dynamic) {

    opr->init_output_mem_plan(dynamic);

    // check output shape valid
    for (auto i: opr->output()) {

        if (!i->m_should_sys_alloc)
            continue;

        if (!dynamic && !m_sys_alloc_static_vars.count(i))
            continue;

        mgb_throw_if(!i->mem_plan().valid(), GraphError,
                "invalid mem plan for var %s", cg::dump_var_info({i}).c_str());

        if (!i->mem_plan().chunk().size() || !i->shape().ndim) {
            // shape is known so we can check for empty mem plan here
            bool allow_empty = i->contain_flag(
                    VarNode::Flag::ALLOW_EMPTY_SHAPE);

            auto &&recv = opr->owner_graph()->
                var_receiver_in_current_comp_seq(i);
            mgb_throw_if(!allow_empty || !recv.is_empty_allowed(),
                    GraphError,
                    "var %s has empty memplan, but allowed=%d receiver=%s",
                    cg::dump_var_info({i}).c_str(),
                    allow_empty, recv.to_string().c_str());
        }
    }
}

void VarNodeMemManager::reset_opr_seq(CompSeqExtraInfo& extra_info,
                                      const OprNodeArray* seq) {
    auto run_id_ptr = static_cast<ComputingGraphImpl::ComputingSequence*>(
                              m_owner_graph->current_comp_seq())
                              ->get_run_id_ptr();
    m_dynamic_alloc_opr_info.clear();
    reset_opr_seq_no_clear_dyn_alloc_info(extra_info, seq, run_id_ptr);
}

void VarNodeMemManager::reset_opr_seq_no_clear_dyn_alloc_info(
        CompSeqExtraInfo& extra_info, const OprNodeArray* seq,
        const size_t* run_id_ptr) {
    bool eager = m_owner_graph->eager_eval_manager().enabled();
    m_first_static_plan_run = true;
    m_run_id_ptr = run_id_ptr;
    m_opr_seq = seq;
    m_sys_alloc_static_vars.clear();
    m_sys_alloc_static_oprs.clear();
    m_sys_alloc_static_oprs_need_mem_status_changed_cb.clear();
    m_impure_mem_plan_mgr.clear_tracked_oprs();

    m_optimize_started = false;
    init_var_force_dynamic_alloc_flag();

    m_optimize_started = true;
    if (!eager) {
        this->init_layout_constraint();
    }
    init_sys_alloc_info(extra_info);
    init_var_seq_force_update_dest();

    if (m_owner_graph->options().log_level) {
        print_seq_info_log();
    }
}

void VarNodeMemManager::print_seq_info_log() {
#if MGB_ENABLE_LOGGING
    auto seq = m_opr_seq;
    VarNode *first_dyn_shp_var = nullptr;
    size_t nr_static = 0, nr_dynamic_shape = 0, nr_dynamic_storage = 0,
           nr_no_sys_alloc = 0;
    for (auto i: *seq) {
        for (auto j: i->output()) {
            if (!j->m_should_sys_alloc) {
                ++ nr_no_sys_alloc;
            }
            if (!is_static_var_shape(j)) {
                ++ nr_dynamic_shape;
                if (!first_dyn_shp_var)
                    first_dyn_shp_var = j;
            } else if (!is_static_var_storage(j)) {
                ++ nr_dynamic_storage;
            } else {
                ++ nr_static;
            }
        }
    }
    mgb_log_debug("opr seq of length %zu: "
            "var_static=%zu var_dynamic_shape=%zu var_dynamic_storage=%zu "
            "no_sys_alloc=%zu",
            seq->size(), nr_static, nr_dynamic_shape, nr_dynamic_storage,
            nr_no_sys_alloc);
    if (nr_dynamic_shape) {
        mgb_log_debug(
                "there are %zu vars with dynamic shape; if this is not"
                " expected please contact the authors to implement more"
                " static inference (var: %s; owner_inputs: %s)",
                nr_dynamic_shape,
                cg::dump_var_info({first_dyn_shp_var}).c_str(),
                cg::dump_var_info(
                    first_dyn_shp_var->owner_opr()->input()).c_str());
    }
#endif
}

void VarNodeMemManager::init_var_force_dynamic_alloc_flag() {
    using Flag = VarNode::Flag;

    auto add_flag_single_var = [](VarNode *var) {
        if (!var->contain_flag(Flag::DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC)) {
            if (!var->contain_flag(Flag::NO_SYS_MEM_ALLOC)) {
                var->add_flag(Flag::RT_FORCE_DYNAMIC_MEM_ALLOC);
            }
            return true;
        }
        return false;
    };

    if (m_owner_graph->options().force_dynamic_alloc) {
        for (auto opr: *m_opr_seq)
            for (auto i: opr->output()) {
                add_flag_single_var(i);
            }
        return;
    }

    // clear previous flags
    for (auto i: *m_opr_seq) {
        for (auto j: i->output()) {
            j->m_flag = j->m_flag & ~Flag::RT_FORCE_DYNAMIC_MEM_ALLOC;
        }
    }

    VarNodeSet modified_vars;
    VarNodeArray to_modify_stack;

    // add flag for sub graph defined by init_var and
    // VarNode::m_rt_force_dynamic_mem_alloc_imply_chain
    auto add_flag_subgraph = [&](VarNode *init_var) {
        if (!modified_vars.insert(init_var).second)
            return;
        to_modify_stack.push_back(init_var);

        while (!to_modify_stack.empty()) {
            auto var = to_modify_stack.back();
            to_modify_stack.pop_back();

            if (!add_flag_single_var(var))
                continue;

            for (auto i: var->m_rt_force_dynamic_mem_alloc_imply_chain) {
                if(modified_vars.insert(i).second) {
                    to_modify_stack.push_back(i);
                }
            }
        }
    };

    auto &&infer_mgr = m_owner_graph->static_infer_manager();
    for (auto opr: *m_opr_seq) {

        bool single_opr_cn = true;
        CompNode opr_cn;

        bool need_add_rt_dyn = false;
        for (auto &&i: opr->node_prop().dep_map()) {
            using DT = OperatorNodeBase::NodeProp::DepType;
            if ((i.second & DT::HOST_VALUE_DYNOUT) &&
                    infer_mgr.get_infer_type(i.first).value !=
                    static_infer::InferType::CONST) {
                need_add_rt_dyn = true;
                break;
            }
        }

        for (auto i: opr->output()) {
            if (!opr_cn.valid()) {
                opr_cn = i->comp_node();
            } else if (opr_cn != i->comp_node())
                single_opr_cn = false;

            if (need_add_rt_dyn || !cg::is_static_var_storage(i))
                add_flag_subgraph(i);
        }

        if (!single_opr_cn)
            opr_cn = {};

        // when an input var is read by this opr on other cn, it must be
        // allocated dynamically; this condition is equivalent to (input_cn !=
        // opr_cn) in either of the following two cases:
        // when opr works on multiple cn, we have single_opr_cn == false and
        // opr_cn being invaid; when input and output are on differrent cns, we
        // have opr_cn == output_cn

        auto &&dep_map = opr->node_prop().dep_map();
        using NP = OperatorNodeBase::NodeProp;
        // force dynamic alloc for vars that are read by other comp nodes
        for (auto &&dep_entry: dep_map) {
            if (NP::is_device_value_dep(dep_entry.second) &&
                    dep_entry.first->comp_node() != opr_cn) {
                add_flag_subgraph(dep_entry.first);
            }
        }
    }
}

void VarNodeMemManager::init_layout_constraint() {
    for (auto &&i: m_node_mem_trait) {
        i.second.layout_constraint.level = LayoutConstraintLevel::NONE;
        i.second.layout_constraint.custom.clear();
    }

    {
        OperatorNodeBase *opr = nullptr;
        MGB_MARK_USED_VAR(opr);
        MGB_TRY {
            for (auto i: *m_opr_seq) {
                opr = i;
                i->add_input_layout_constraint();
            }
        } MGB_CATCH(MegBrainError &exc, {
            if (!exc.extra_info() && opr)
                OperatorNodeExcExtraInfo::record(opr, exc);
            throw;
        })
    }
}

void VarNodeMemManager::init_sys_alloc_info(CompSeqExtraInfo &extra_info) {
    auto &&infer_mgr = m_owner_graph->static_infer_manager_impl();

    auto init_var_should_sys_alloc = [&infer_mgr, graph=m_owner_graph](
            VarNode *var) {
        using F = VarNode::Flag;
        if (var->contain_flag(F::NO_SYS_MEM_ALLOC))
            return false;
        mgb_assert(infer_mgr.get_infer_type(var).shape !=
                static_infer::InferType::NO_DESC,
                "variable infer desc has not been set, but it does not have"
                " NO_SYS_MEM_ALLOC flag (var: %s)",
                cg::dump_var_info({var}).c_str());

        if (var->contain_flag(F::NO_ALLOC_IF_UNUSED)) {
            if (!graph->var_receiver_in_current_comp_seq(var).value_needed())
                return false;
        }
        return true;
    };

    CompNode::UnorderedSet all_comp_nodes;
    for (auto opr: *m_opr_seq) {
        bool has_static_out = false;
        for (auto var: opr->output()) {
            all_comp_nodes.insert(var->comp_node());
            var->m_should_sys_alloc = init_var_should_sys_alloc(var);
            extra_info.infer_dest.insert(
                    infer_mgr.get_tag_handler_for_shape(var));

            if (is_static_var_storage(var) && var->m_should_sys_alloc) {
                has_static_out = true;
                m_sys_alloc_static_vars.insert(var);

            }
        }

        if (has_static_out) {
            m_sys_alloc_static_oprs.push_back(opr);
            constexpr auto impure =
                    OperatorNodeBase::NodeProp::Flag::IMPURE_OUTPUT_MEM_PLAN;
            if (opr->node_prop().contain(impure)) {
                for (auto i : opr->output()) {
                    mgb_throw_if(!m_sys_alloc_static_vars.count(i), GraphError,
                                 "oprs with IMPURE_OUTPUT_MEM_PLAN should have "
                                 "all outputs as static; bad opr: %s{%s}",
                                 opr->cname(), opr->dyn_typeinfo()->name);
                }
                m_impure_mem_plan_mgr.add_opr_to_track(opr);
            }
        }
    }
    m_seq_mem_opt.reset_opr_seq(m_opr_seq, &m_sys_alloc_static_oprs,
                                &m_sys_alloc_static_vars,
                                {all_comp_nodes.begin(), all_comp_nodes.end()});
}

void VarNodeMemManager::init_var_seq_force_update_dest() {
    bool eager = m_owner_graph->eager_eval_manager().enabled();
    if (!eager) {
        for (auto &&i : m_node_mem_trait) {
            i.second.seq_force_update_dest = nullptr;
        }
    }

    for (auto opr: *m_opr_seq) {
        for (auto i: opr->output()) {
            auto src = m_node_mem_trait[i].force_update_src;
            if (src) {
                auto &&src_trait = m_node_mem_trait[src];
                mgb_assert(!src_trait.seq_force_update_dest || eager,
                        "multiple force update dests in a single comp seq: %s",
                        cg::dump_var_info({src}).c_str());
                src_trait.seq_force_update_dest = i;
            }
        }
    }
}

/* ============= implementation for methods in VarNode ============= */

bool VarNodeMemManager::fwd_in2out_readonly(
        VarNode *src, const SubTensorSpec &sub, VarNode *dest) {
    /*
     * readonly forward is implemented by sharing memory chunk and setting
     * layout/offset
     */

    if (!dest->m_mem_plan.valid()) {
        // fwd from static storage to dynamic storage, with statically
        // inferable shape
        mgb_assert(src->m_mem_plan.valid() &&
                is_static_var_storage(src) && !is_static_var_storage(dest));
        return false;
    }

    mgb_assert(
            src != dest &&
            src->comp_node().mem_node() == dest->comp_node().mem_node() &&
            dest->m_mem_plan.valid() && src->m_mem_plan.valid() &&
            dest->m_mem_plan.layout().eq_shape(sub.layout()) &&
            dest->m_mem_plan.layout().dtype.size() == sub.layout().dtype.size()
            );
    assert_in_mem_opt_phase(
            SeqMemOptimizer::Status::ALLOW_FWD_IN2OUT_READONLY);

    if (!m_owner_graph->options().seq_opt.enable_mem_plan_opt)
        return false;

    auto &&src_spec = m_node_mem_trait.at(src);

    if (src->comp_node() != dest->comp_node()) {
        if (src->comp_node().mem_node() != dest->comp_node().mem_node()) {
            return false;
        }
        if (is_static_var_storage(src) || is_static_var_storage(dest)) {
            if (!src->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
                // forwarding between comp nodes requires asynchronous memory
                // reclaiming, which is impossible in the static storage case
                return false;
            }
            // src is persistent, so we allow forwarding to dest regardless of
            // whether it is static or dynamic
        }
        if (src_spec.seq_force_update_dest) {
            // we simply disallow readonly fwd with force update; otherwise we
            // need to ensure dest finishes before seq_force_update_dest, and
            // three different comp noes can be involved in the most complicated
            // case: src, seq_force_update_dest and dest are all on different
            // comp nodes
            return false;
        }
    }

    auto &&dest_spec = m_node_mem_trait.at(dest);
    if (dest_spec.readonly_src) {
        // multiple calls may happen when an opr has multiple outputs containing
        // both static and dynamic storage, and it tries to forward static
        // output var in both static/dynamic memory alloc passes.
        // see TestTensorManip.SplitPreAllocatedMultiCN for a concrete example
        mgb_assert(
                dest_spec.readonly_src == src &&
                dest->m_mem_plan.layout().eq_layout(sub.layout()) &&
                &dest->m_mem_plan.chunk() == &src->m_mem_plan.chunk() &&
                dest->m_mem_plan.offset_in_chunk_byte() ==
                    static_cast<size_t>(
                        src->m_mem_plan.offset_in_chunk_byte() +
                        sub.offset_byte()),
                "inconsistent multiple calls to fwd_in2out_readonly");
        return true;
    }

    bool eager = m_owner_graph->eager_eval_manager().enabled();
    if (src_spec.seq_force_update_dest && !eager) {
        auto fu_dst = src_spec.seq_force_update_dest;
        auto og = m_owner_graph;
        auto fu_dst_step = og->opr_step_num_in_cur_comp_seq(
                fu_dst->owner_opr()).val(),
             min_safe_step = og->opr_step_num_in_cur_comp_seq(
                     dest->owner_opr()).val();
        if (auto last_opr = og->var_receiver_in_current_comp_seq(
                    dest).last_dev_value_reader) {
            auto step = og->opr_step_num_in_cur_comp_seq(last_opr).val();
            mgb_assert(step > min_safe_step);
            min_safe_step = step;
        }
        if (fu_dst_step <= min_safe_step)
            return false;
        if(fu_dst->comp_node() != src->comp_node()) {
            auto &&cnopt = static_cast<SeqCompNodeOptimizerImpl&>(
                    og->seq_comp_node_optimizer());
            auto s = cnopt.get_opr_other_cn_nr_finish(
                    fu_dst->comp_node(), fu_dst_step, src->comp_node());
            if (s <= min_safe_step)
                return false;
        }
    }

    if (!dest_spec.check_layout(sub.layout()))
        return false;

    if (is_static_var_storage(src) != is_static_var_storage(dest)) {
        if (!src->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
            // dyn to static: fail because output has been allocated
            // static to dyn: fail because input may be reused
            return false;
        }
        // if src is a persistent value, then it has static storage and dest has
        // dynamic storage, and we allow forwarding in this case since src and
        // dest are on the same computing node, and no writable forwarding is
        // allowed on persistent values
    }

    dest_spec.readonly_src = src;
    dest_spec.seq_force_update_dest = src_spec.seq_force_update_dest;
    dest->m_mem_plan.assign_for_forward(src->m_mem_plan, sub);

    return true;
}

void VarNodeMemManager::assert_in_mem_opt_phase(size_t status) {
    mgb_assert(m_seq_mem_opt.status() & status,
            "call mem opt function outside of mem opt phase; "
            "wrong node implementation?");
}

bool VarNodeMemManager::VarNodeMemTrait::check_layout(
        const TensorLayout &layout) const {
    switch (layout_constraint.level) {
        case LayoutConstraintLevel::CONTIG:
            return layout.is_contiguous();
        case LayoutConstraintLevel::MONOTONE:
            if (!layout.is_abs_monotonous_allow_brdcst()) {
                return false;
            }
            break;
        case LayoutConstraintLevel::NONE:
            break;
        default:
            mgb_throw(InternalError, "invalid layout_constraint_level");
    }

    for (auto &&i: layout_constraint.custom)
        if (!i(layout))
            return false;
    return true;
}

void VarNodeMemManager::fwd_in2out_writable(VarNode *src, VarNode *dest) {
    /*
     * if writable forward is applicable, tell seq mem optimizer to handle it
     */

    mgb_assert(dest != src &&
            dest->comp_node().mem_node() == src->comp_node().mem_node());

    if (is_static_var_storage(src) != is_static_var_storage(dest))
        return;

    if (m_node_mem_trait.at(src).seq_force_update_dest)
        return;

    mgb_assert(dest->m_mem_plan.layout().eq_shape(src->m_mem_plan.layout()));
    if (!m_owner_graph->options().seq_opt.enable_mem_plan_opt)
        return;
    assert_in_mem_opt_phase(SeqMemOptimizer::Status::ALLOW_FWD_IN2OUT_WRITABLE);
    auto &&dest_spec = m_node_mem_trait.at(dest);
    mgb_assert(!dest_spec.readonly_src,
            "already readonly forwarded from other var");

    MemAllocPlan* plan0 = &src->m_mem_plan;

    // do not allow non-contiguous writable fwd, because speed gain is by
    // inplace opr is negligible, but non-contiguous output may further affect
    // other oprs
    if (!plan0->layout().is_contiguous() ||
            !dest_spec.check_layout(plan0->layout()))
        return;

    m_seq_mem_opt.add_writable_fwd_mem_plan_pair(
            plan0, &dest->m_mem_plan);
}

void VarNodeMemManager::fwd_in2out_writable_force(VarNode *src, VarNode *dest) {
    /*
     * this functin must be called during operator init, and actual forwarding
     * is handled by init_single_var_mem_plan and
     * make_dev_tensor_from_mem_plan_single
     */

    mgb_assert(!m_optimize_started,
            "set_fwd_in2out_writable_force must be "
            "called during initialization");
    m_node_mem_trait[src]; // to avoid resizing causing dangling pointer
    mgb_assert(dest->owner_opr()->node_prop().contain(
                OperatorNodeBase::NodeProp::Flag::FORCE_UPDATE_INPUT_VAR));
    auto &&dest_spec = m_node_mem_trait[dest];
    mgb_assert(!dest_spec.force_update_src,
            "force update can only be set to one src(%s)",
            dest->cname());
    dest_spec.force_update_src = src;
}

void VarNodeMemManager::add_layout_constraint(VarNode *dest,
        VarNode::LayoutConstraintCallback callback) {
    auto &&trait = m_node_mem_trait[dest].layout_constraint;
    if (trait.level != LayoutConstraintLevel::CONTIG) {
        trait.custom.emplace_back(std::move(callback));
    }
}

void VarNodeMemManager::add_layout_constraint_level(
        VarNode* dest, LayoutConstraintLevel level) {
    auto&& trait = m_node_mem_trait[dest].layout_constraint;
    if (level > trait.level) {
        trait.level = level;
        if (level == LayoutConstraintLevel::CONTIG && !trait.custom.empty()) {
            // delete all custom callbacks and clear memory
            decltype(trait.custom) tmp;
            tmp.swap(trait.custom);
        }
    }
}

void VarNodeMemManager::init_single_var_mem_plan(
        VarNode* var, const DeviceTensorND* fixed_alloc) {
    if (fixed_alloc && var->m_mem_plan.valid() && var->dev_tensor_valid()) {
        if (var->m_dev_tensor.layout().eq_layout(fixed_alloc->layout())) {
            if (var->m_dev_tensor.raw_ptr() == fixed_alloc->raw_ptr()) {
                // for fixed alloc, it is likely to use the same tensor
                // repeatedly, so we add a quick return here
                auto&& chk = var->m_mem_plan.chunk();
                mgb_assert(chk.owner_var == var &&
                           chk.mem_alloc_status.is_from_owner_var());
                return;
            }
            m_impure_mem_plan_mgr.record_ptr_changed(this, var);
        } else {
            m_impure_mem_plan_mgr.record_layout_changed(&var->m_mem_plan);
        }
    }

    auto&& spec = m_node_mem_trait.at(var);
    if (spec.force_update_src) {
        var->m_mem_plan.assign(spec.force_update_src->m_mem_plan);
        mgb_assert(!fixed_alloc);
    } else {
        var->m_mem_plan.reset_from_owner_var();
    }

    if (fixed_alloc) {
        auto&& chk = var->m_mem_plan.layout(fixed_alloc->layout()).chunk();
        chk.mem_alloc_status.set_from_owner_var();
        chk.update_size_for_dynamic_alloc(fixed_alloc->storage().size());
        var->m_dev_tensor = *fixed_alloc;
        var->m_dev_tensor.comp_node(var->comp_node());
        var->m_prev_dev_ptr = fixed_alloc->raw_ptr();
    } else {
        var->m_dev_tensor.storage(DeviceTensorStorage{});
    }
}

void VarNodeMemManager::make_dev_tensor_from_mem_plan_single(
        VarNode* var, const DeviceTensorStorage& given_storage,
        size_t offset_in_given_storage) {
    auto&& plan = var->m_mem_plan;
    auto&& chunk = plan.chunk();
    mgb_assert(chunk.size() ||
               var->contain_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE));
    DeviceTensorStorage storage;
    if (auto offset = offset_in_given_storage + plan.offset_in_chunk_byte()) {
        storage = given_storage.sub(offset);
    } else {
        storage = given_storage;
    }
    if (given_storage.comp_node_allow_invalid() != var->comp_node()) {
        // comp node stream may change
        storage.comp_node(var->comp_node(), false);
    }
    var->m_prev_dev_ptr = storage.ptr();
    var->m_dev_tensor.reset(std::move(storage), plan.layout());
}

void VarNodeMemManager::var_alloc_with_shape(VarNode* var,
                                             const TensorShape& shape,
                                             size_t size_req) {
    mgb_assert(var->format().is_default(),
               "dynamic shape is currently only supported for var with "
               "default format; got %s",
               var->format().to_string().c_str());
    var->shape(shape);
    if (size_req != 0) {
        mgb_assert(var->dtype().size(shape.total_nr_elems()) <= size_req);
    } else {
        size_req = var->dtype().size(shape.total_nr_elems());
    }

    auto&& mplan = var->m_mem_plan;
    if (!mplan.valid() || mplan.chunk().owner_var != var)
        init_single_var_mem_plan(var);

    auto&& chk = mplan.chunk();
    DeviceTensorStorage storage;
    if (chk.owner_var == var) {
        storage = var->m_dev_tensor.storage();
        if (storage.size() < size_req) {
            // clear storage ref in var
            var->m_dev_tensor.storage(DeviceTensorStorage{});
            m_var_dev_mem_defragmenter.alloc_var_storage(var, storage,
                                                         size_req);
            auto addr = reinterpret_cast<size_t>(storage.ptr());
            auto alignment = var->comp_node().get_mem_addr_alignment();
            mgb_assert(
                    addr && !(addr & (alignment - 1)),
                    "address unaligned: 0x%zx (alignment: 0x%zx); size_req=%zu",
                    addr, alignment, size_req);
        }
        chk.update_size_for_dynamic_alloc(size_req);
        chk.mem_alloc_status.set_from_owner_var();
    } else {
        // this branch is only possible for force update
        storage = chk.owner_var->m_dev_tensor.storage();
        mgb_assert(chk.owner_var->m_dev_tensor.shape().eq_shape(shape));
    }
    mplan.layout({shape, var->dtype()}, true);
    mgb_assert(!mplan.offset_in_chunk_byte());
    make_dev_tensor_from_mem_plan_single(var, storage);
}

bool VarNodeMemManager::on_var_node_device_comp_finish_needed(
        VarNode* var) const {
    mgb_assert(!m_first_static_plan_run);
    return m_owner_graph->eager_eval_manager().enabled() ||
           m_need_post_exec_action_vars.count(var);
}

void VarNodeMemManager::on_var_node_device_comp_finish(VarNode* var,
                                                       bool compute_enabled) {
    if (!is_inf_refcnt_init(var)) {
        size_t old_refcnt = var->m_refcnt.exchange(var->m_refcnt_init);
        mgb_assert(!old_refcnt, "refcnt non-zero for new var: var=%s cnt=%zu",
                   var->cname(), old_refcnt);
        if (!compute_enabled) {
            var->mem_plan().reset_as_invalid_cond_exec();
        }
    }
    if (auto mgr = var->m_cn_sync_manager) {
        mgr->set_ready();
    }
    auto var_cn = var->comp_node();

    auto&& node_prop = var->owner_opr()->node_prop();
    using NodeProp = OperatorNodeBase::NodeProp;
    for (auto&& pair : node_prop.dep_map()) {
        if (NodeProp::is_device_value_dep(pair.second)) {
            if (!is_inf_refcnt_init(pair.first)) {
                decr_var_mem_refcnt(pair.first, var_cn);
            }
        }
    }

    if (!var->m_refcnt_init && var->dev_tensor_valid()) {
        // handle vars that are not accessed
        var->m_refcnt = 1;
        decr_var_mem_refcnt(var, var_cn);
    }
}

void VarNodeMemManager::decr_var_mem_refcnt(
        VarNode *var, CompNode dispatch_cn) {
    if (MGB_IF_COND_EXEC(var->mem_plan().is_invalid_cond_exec() ||)
                var->mem_plan()
                        .chunk()
                        .owner_var->comp_node() == dispatch_cn) {
        decr_var_mem_refcnt_sync(var);
        return;
    }

    using DT = CompNode::DeviceType;
    switch (dispatch_cn.device_type()) {
        case DT::CPU:
            {
                auto task = [this, var]() {
                    decr_var_mem_refcnt_sync(var);
                    m_cpu_async_release_barrier.incr(-1);
                };
                m_cpu_async_release_barrier.incr(1);
                CompNodeEnv::from_comp_node(dispatch_cn).cpu_env().dispatch(
                        task);
                break;
            }
#if MGB_CUDA
        case DT::CUDA:
            m_asyn_var_releaser->add(dispatch_cn, var);
            break;
#endif
#if MGB_ATLAS
        case DT::ATLAS:
            {
                m_asyn_var_releaser->add(dispatch_cn, var);
                break;
            }
#endif
        default:
            mgb_throw(MegBrainError,
                      "unsupported comp node in dynamic var shape: %s",
                      dispatch_cn.to_string().c_str());
    }
}

void VarNodeMemManager::decr_var_mem_refcnt_sync(VarNode *var) {
    if (! -- var->m_refcnt) {
        if (var->m_mem_plan.chunk().owner_var != var) {
            // var is forwarded from another var (or an invalid cond exec), so
            // we can release the device tensor; otherwise the device tensor
            // should be released in release_chunk()
            var->m_dev_tensor.storage({});
        }
        var->m_mem_plan.release_chunk();
    }
}

bool VarNodeMemManager::is_inf_refcnt_init(VarNode* var) {
    return (var->m_refcnt_init & REFCNT_INF) != 0;
}

size_t VarNodeMemManager::clear_static_device_memory() {
    m_static_mem_refholder.clear();
    m_static_mem_refholder_dev_mem_mgr_version =
            DeviceMemoryAllocator::VERSION_INVALID;
    return m_static_dev_mem_mgr->clear_all();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
