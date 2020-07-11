/**
 * \file src/core/impl/graph/var_node_mem_mgr/defrag.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./defrag.h"
#include "../cg_impl.h"
#include "megbrain/utils/arith_helper.h"

using namespace mgb;
using namespace cg;

void VarDevMemDefragmenter::alloc_direct(VarNode* var,
                                         DeviceTensorStorage& storage,
                                         size_t size) {
    if (!storage.comp_node_valid())
        storage.comp_node(var->comp_node());
    if (size > storage.size()) {
        m_mem_mgr->static_device_memory_manager()->allocator().alloc_dynamic(
                var, storage, size);
        storage.ptr();  // apply lazy alloc
    }
    mgb_assert(storage.size() >= size);
}

#if MGB_ENABLE_VAR_DEV_MEM_DEFRAGMENTER

struct VarDevMemDefragmenter::ChunkInfo {
    VarNodeArray readers;
    HostTensorStorage value;  // stored tensor value
};

void VarDevMemDefragmenter::register_var(VarNode* var) {
    m_cninfo_map[var->comp_node()].vars.insert(var);
    m_move_safe_oprs.insert(var->owner_opr());
}

void VarDevMemDefragmenter::clear_all() {
    m_cninfo_map.clear();
}

void VarDevMemDefragmenter::alloc_with_defrag(VarNode* var,
                                              DeviceTensorStorage& storage,
                                              size_t size) {
    CompNodeInfo* cninfo_ptr;
    {
        MGB_LOCK_GUARD(m_mtx);
        cninfo_ptr = &m_cninfo_map[var->comp_node()];
    }
    MGB_LOCK_GUARD(cninfo_ptr->mtx);

    if (!storage.comp_node_valid()) {
        storage.comp_node(var->comp_node());
        cninfo_ptr->vars.insert(var);
    }

    MGB_TRY { alloc_direct(var, storage, size); }
    MGB_CATCH(MemAllocError&, {
        mgb_log_warn("memory allocation failed for var %s; try defragmenting",
                     var->cname());
        defrag(var, *cninfo_ptr, size);
        alloc_direct(var, storage, size);
    });
}

void VarDevMemDefragmenter::defrag(VarNode* req_var,
                                   const CompNodeInfo& cn_info,
                                   size_t extra_size) {
    // pause all other comp nodes before calling defrag_impl()
    auto exec_env = ComputingGraphImpl::downcast(req_var->owner_graph())
                            ->current_exec_env();
    mgb_assert(exec_env);
    exec_env->pause_exec();
    m_mem_mgr->owner_graph()->event().signal_inplace<event::BeforeMemDefrag>();
    MGB_TRY { defrag_impl(req_var, cn_info, extra_size); }
    MGB_FINALLY(exec_env->resume_exec(););
}

void VarDevMemDefragmenter::defrag_impl(VarNode* req_var,
                                        const CompNodeInfo& cn_info,
                                        size_t extra_size) {
    ThinHashMap<MemAllocPlan::Chunk*, ChunkInfo> chunkinfo;
    VarNodeSet non_movable_vars;
    if (!m_move_safe_oprs.count(req_var->owner_opr())) {
        // input and output vars of current opr can not be moved
        auto opr = req_var->owner_opr();
        for (auto i : opr->node_prop().dep_map()) {
            if (OperatorNodeBase::NodeProp::is_device_value_dep(i.second)) {
                non_movable_vars.insert(i.first);
            }
        }
        for (auto i : opr->output()) {
            non_movable_vars.insert(i);
        }
    }
    for (auto i : cn_info.vars) {
        if (i->dev_tensor_valid() && !non_movable_vars.count(i)) {
            auto chk = &i->mem_plan().chunk();
            chunkinfo[chk].readers.push_back(i);
        }
    }

    auto cn = req_var->comp_node();

    // here we do not need to handle exceptions and restore vars, since
    // allocation failure requires the whole graph to be re-executed and all
    // vars would be re-allocated

    // release all memory
    size_t tot_size = extra_size, nr_refcnt_mismatch = 0, nr_var = 0;
    auto alignment = cn.get_mem_addr_alignment();
    for (decltype(chunkinfo.begin()) iter = chunkinfo.begin(), inext;
         iter != chunkinfo.end(); iter = inext) {
        inext = iter;
        ++inext;

        auto refcnt = iter->first->m_refcnt.load(std::memory_order_relaxed);
        if (refcnt == iter->second.readers.size()) {
            tot_size += get_aligned_power2(iter->first->size(), alignment);
            nr_var += iter->second.readers.size();
            auto owner_var = iter->first->owner_var;
            auto&& tensor = owner_var->m_dev_tensor;
            iter->second.value.comp_node(cn)
                    .ensure_size(iter->first->size())
                    .copy_from(tensor.storage(), iter->first->size());

            // release memory of all readers
            for (auto var : iter->second.readers) {
                const_cast<DeviceTensorND&>(var->dev_tensor()).storage({});
            }
            // release memory of owner_var
            auto&& mem_plan = owner_var->mem_plan();
            if (!mem_plan.valid()) {
                // mem_plan of owner_var was invalid here if all reader oprs
                // of owner_var have already been executed, but its tensor
                // storage should not be released until the refcnt of chunk
                // decreasing to zero (see release_chunk() for more details)
                mgb_assert(tensor.storage().comp_node_valid() &&
                    tensor.layout().eq_layout(mem_plan.layout()));
                tensor.storage({});
            }
        } else {
            mgb_assert(refcnt > iter->second.readers.size());
            ++nr_refcnt_mismatch;
            chunkinfo.erase(iter);
        }
    }

    // wait all other comp nodes to avoid moved var being read; note that
    // ExecEnv has been paused, so no new task would not be dispatched
    CompNode::sync_all();

    CompNode::try_coalesce_all_free_memory();
    mgb_log_debug("var defragment: vars=%zu chunks=%zu tot_size=%.3fMiB "
            "refcnt_mismatch=%zu current_free=%.3fMiB",
            nr_var, chunkinfo.size(), tot_size / 1024.0 / 1024,
            nr_refcnt_mismatch,
            cn.get_mem_status_bytes().second / 1024.0 / 1024);

    auto&& allocator = m_mem_mgr->static_device_memory_manager()->allocator();
    allocator.defrag_prealloc_contig(m_mem_mgr->owner_graph(), cn, tot_size);

    // allocate for each storage
    size_t offset = 0;
    for (auto&& i : chunkinfo) {
        DeviceTensorStorage storage{cn};
        allocator.alloc_dynamic(i.second.readers.at(0), storage,
                                i.first->size());
        storage.copy_from(i.second.value, i.first->size());
        offset += get_aligned_power2(i.first->size(), alignment);
        for (auto var : i.second.readers) {
            auto&& mplan = var->mem_plan();
            if (auto sub_off = mplan.offset_in_chunk_byte()) {
                var->m_dev_tensor.reset(storage.sub(sub_off), mplan.layout());
            } else {
                var->m_dev_tensor.reset(storage, mplan.layout());
            }
            mgb_assert(var->dev_tensor_valid());
        }
        auto owner_var = i.first->owner_var;
        if (!owner_var->mem_plan().valid()) {
            owner_var->m_dev_tensor.reset(
                storage, owner_var->mem_plan().layout());
        }
    }
    mgb_assert(offset + extra_size == tot_size);
    cn.sync();  // wait copy finish before destructing host values
}

#endif  // MGB_ENABLE_VAR_DEV_MEM_DEFRAGMENTER

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

