/**
 * \file src/core/impl/graph/var_node_mem_mgr.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./impl_common.h"
#include "./var_node_mem_mgr/seq_mem_opt.h"
#include "./var_node_mem_mgr/defrag.h"
#include "megbrain/graph/event.h"

#include "megbrain/utils/thread.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/thin/nullable_hash_map.h"

namespace mgb {
namespace cg {

// use a large but not the max value to leave space for incr
constexpr size_t REFCNT_INF = std::numeric_limits<size_t>::max() / 2 + 1;
// first bit is 1 and all others 0, so less-than test can be easy
static_assert(!(REFCNT_INF & (REFCNT_INF >> 1)), "bad value");

class ComputingGraphImpl;

/*!
 * \brief device memory manager for static memory allocation
 *
 * Static device memory is pre-allocated and unchanged between consecutive graph
 * executions as long as no var shape changes, so subsequent graph executions
 * can be faster.
 *
 * An instance of StaticDeviceMemoryManager can be shared by multiple
 * AsyncExecutable objects, so they can share device memory.
 *
 */
class StaticDeviceMemoryManager {
    std::atomic_flag m_in_exec = ATOMIC_FLAG_INIT;
    size_t m_version = 0;
    CompNode::UnorderedMap<DeviceTensorStorage> m_storage;
    std::shared_ptr<DeviceMemoryAllocator> m_allocator;

public:
    StaticDeviceMemoryManager();

    //! set the underlying allocator to be used
    void set_allocator(std::shared_ptr<DeviceMemoryAllocator> allocator) {
        m_allocator = std::move(allocator);
    }

    //! get current allocator
    DeviceMemoryAllocator& allocator() const { return *m_allocator; }

    /*!
     * \brief mark the start of graph execution
     *
     * Error would be raised if exec_enter() is called twice before calling
     * exec_exit().
     */
    void exec_enter();

    //! mark the end of graph execution
    void exec_exit();

    //! allocate storage for a comp node
    const DeviceTensorStorage& alloc(ComputingGraph* graph, CompNode cn,
                                     size_t size, size_t cur_version);

    //! get currently allocated size on a comp node; return 0 if nothing on
    //! given comp node
    size_t get_size(CompNode cn) const {
        auto iter = m_storage.find(cn);
        return iter == m_storage.end() ? 0 : iter->second.size();
    }

    //! prefault the pages for fast initial access
    void prefault();

    /*!
     * \brief clear all cached storage
     * \return max refcnt
     */
    size_t clear_all();

    size_t version(ComputingGraph* graph) const {
        return m_allocator->static_alloc_version(graph);
    }

    //! make a default implementation using system allocator
    static std::shared_ptr<StaticDeviceMemoryManager> make_default_impl();
};

/*!
 * \brief var node memory manager
 *
 * This class manages memory forward requests, static memory allocation and also
 * dynamic memory allocation.
 */
class VarNodeMemManager {
    public:
        enum class LayoutConstraintLevel {
            NONE = 0,   //!< use custom callback
            MONOTONE = 1,   //!< required to be monotonous (i.e. cache friendly)
            CONTIG = 2,     //!< required to be contiguous
        };

        struct VarNodeMemTrait {
            struct LayoutConstraint {
                LayoutConstraintLevel level;
                std::vector<VarNode::LayoutConstraintCallback> custom;
            };
            VarNode *readonly_src = nullptr;

            /*!
             * if b claims to forcely update a (
             * i.e. b->set_fwd_in2out_writable_force(a) * is called), then we
             * have b->force_update_src == a.
             *
             * When computing sequence is determined, at most only one force
             * update can take place, and seq_force_update_dest would be set
             * accordingly. They would share underlying mem plan.
             *
             * Note that seq_force_update_dest is only used for marking force
             * updates, and when a is readonly forwarded as b,
             * a->seq_force_update_dest would also be assigned to b's.
             */
            VarNode *force_update_src = nullptr,
                    *seq_force_update_dest = nullptr;

            LayoutConstraint layout_constraint;

            bool check_layout(const TensorLayout &layout) const;

            //! clear optimization status; called before opt process starts
            void clear_opt_status();

            bool has_dynamic_mem_fwd_from_other() const {
                return readonly_src;
            }
        };

        VarNodeMemManager(ComputingGraphImpl *graph);
        ~VarNodeMemManager() noexcept;

        /*!
         * \brief reset active operator sequence
         * \param[out] extra_info output param that would be filled with extra
         *      info for the comp seq
         */
        void reset_opr_seq(
                CompSeqExtraInfo& extra_info, const OprNodeArray *seq);

        /*!
         * Like reset_opr_seq() but do not clear m_dynamic_alloc_opr_info; this
         * is used in eager eval mode.
         */
        void reset_opr_seq_no_clear_dyn_alloc_info(CompSeqExtraInfo& extra_info,
                                                   const OprNodeArray* seq,
                                                   const size_t* run_id_ptr);

        /*!
         * \brief allocate static var node memory; should be called before graph
         *      execution
         *
         * \return whether memory is reallocated
         */
        bool alloc_var_node_mem_static();

        /*!
         * \brief free the memory of var with MEMORY_NO_NEED flag
         *
         * \return whether memory of MEMORY_NO_NEED var or related other var
         * memory changed
         */
        bool free_combine_memory_no_need_var();

        /*!
         * \brief initialize static memory allocation plan
         *
         * This can be used with custom StaticDeviceMemoryAllocator so static
         * memory storage can be controled.
         *
         * \return whether allocation plan changes
         */
        bool update_static_alloc_plan();

        /*!
         * \brief get static memory usage on each comp node
         *
         * This is only valid after calling update_static_alloc_plan()
         */
        const CompNode::UnorderedMap<size_t>& get_static_alloc_size() const {
            return m_seq_mem_opt.static_mem_usage();
        }

        /*!
         * \brief allocate dynamic output var node memory for operator; should
         * be called before operator execution
         */
        void alloc_var_node_mem_dynamic(GraphExecutable::ExecEnv &env,
                OperatorNodeBase *opr);

        //! get underlying device memory manager
        const std::shared_ptr<StaticDeviceMemoryManager>&
        static_device_memory_manager() const {
            return m_static_dev_mem_mgr;
        }

        //! set underlying device memory manager
        void static_device_memory_manager(
                std::shared_ptr<StaticDeviceMemoryManager> mgr) {
            m_static_dev_mem_mgr = std::move(mgr);
        }

        ComputingGraphImpl* owner_graph() const { return m_owner_graph; }

        /*!
         * \brief set the CompNodeSyncManager associated with a VarNode
         *
         * This is invoked by SeqCompNodeOptimizer. mgr->set_ready() would be
         * called when this var finishes computing.
         */
        static void set_var_node_cn_sync_manager(VarNode* var,
                                                 CompNodeSyncManager* mgr) {
            var->m_cn_sync_manager = mgr;
        }

        //! get the CompNodeSyncManager associated with a VarNode
        static CompNodeSyncManager* var_node_cn_sync_manager(VarNode* var) {
            return var->m_cn_sync_manager;
        }

        //! whether calling on_var_node_device_comp_finish is needed
        bool on_var_node_device_comp_finish_needed(VarNode *var) const;

        /*!
         * \brief called by operators when computing of a var is finished
         *
         * Set ready, init output refcnt, and decr input refcnt
         *
         * This method only needs to be called for vars which
         * on_var_node_device_comp_finish_needed() returns true.
         *
         * Note: this function shoould be dispatched regardless of the operator
         * execution mask. The system manages var refcnt even if an operator is
         * not executed, so vars can be correctly reclaimed in dynamic execution
         * case.
         *
         * \param compute_enabled whether the owner opr is actually executed
         *      (i.e. whether its ExecutionMask is enabled); if this is false,
         *      then only deref of input vars would be performed.
         */
        void on_var_node_device_comp_finish(VarNode *var, bool compute_enabled);

        /*!
         * \brief release static device memory storage
         *
         * Note: dev tensors in var nodes would not be touched, but their
         * content pointers would become dangling after calling this method.
         * This behavior is kind of dangerous, but it is designed so for best
         * performance.
         *
         * \return use count of device memory before clear; a value of 1
         *      indicates the memory would be actually released
         */
        size_t clear_static_device_memory();

        //! get the reference to the static device memory
        const SmallVector<DeviceTensorStorage>& static_device_memory_refholder()
                const {
            return m_static_mem_refholder;
        }

        /* ============= implementation for methods in VarNode ============= */

        /*!
         * \brief see VarNode::set_fwd_in2out_readonly
         */
        bool fwd_in2out_readonly(
                VarNode *src, const SubTensorSpec &sub, VarNode *dest);

        /*!
         * \brief see VarNode::set_fwd_in2out_writable
         */
        void fwd_in2out_writable(VarNode *src, VarNode *dest);

        /*!
         * \brief see VarNode::set_fwd_in2out_writable_force
         */
        void fwd_in2out_writable_force(VarNode *src, VarNode *dest);

        void add_layout_constraint(VarNode *dest,
                VarNode::LayoutConstraintCallback callback);

        void add_layout_constraint_level(
                VarNode *dest, LayoutConstraintLevel level);

        /**
         * \brief alloc var memory with shape.
         *
         * Alloc memory of size_seq if size_req != 0.
         */
        void var_alloc_with_shape(VarNode* var, const TensorShape& shape,
                                  size_t size_req = 0);

        /*!
         * \brief initialize mem plan for a single var
         *
         * This would check if force update is set, and act accordingly; note
         * that \p fixed_alloc must be NULL in this case.
         */
        void init_single_var_mem_plan(
                VarNode* var,
                const DeviceTensorND* fixed_alloc = nullptr);

        /* ============= misc methods ============= */
        VarNodeMemTrait& get_var_node_mem_trait(const VarNode *var) {
            return m_node_mem_trait[const_cast<VarNode*>(var)];
        }

        VarNodeMemTrait& get_var_node_mem_trait_at(const VarNode *var) {
            return m_node_mem_trait.at(const_cast<VarNode*>(var));
        }

        //! get VarNodeMemTrait, or nullptr if trait does not exist
        VarNodeMemTrait* get_var_node_mem_trait_nullable(const VarNode *var) {
            auto iter = m_node_mem_trait.find(const_cast<VarNode*>(var));
            return iter == m_node_mem_trait.end() ? nullptr : &iter->second;
        }

        void remove_var_node_mem_trait(VarNode *var) {
            m_node_mem_trait.erase(var);
        }

        bool optimize_started() const {
            return m_optimize_started;
        }

        void on_graph_compile_finished() {
            m_optimize_started = false;
        }

    private:
        /*!
         * \brief mem alloc info in dynamic alloc mode for oprs with static
         *      shape
         */
        struct DynamicAllocOprInfo {
            bool has_dynamic_storage_input;

            //! comp seq execution ID for recently finished alloc, so multiple
            //! outputs of a single opr is alloated only once
            size_t alloc_comp_seq_exec_id = -1;

            //! previously synced layout and address of dev_val_input
            megdnn::TensorNDArray prev_dev_val_input;

            //! static infer handler and previously synched version
            std::vector<std::pair<
                static_infer::StaticInferManagerImpl::TagHandler*, size_t>>
                static_infer_inp;

            VarNodeArray dev_val_input, dynamic_alloc_output;
            Spinlock mtx;

            DynamicAllocOprInfo(OperatorNodeBase *opr);

            //! whether any input or output is dynamc
            bool has_dyn_input_or_output() const {
                return has_dynamic_storage_input ||
                    !dynamic_alloc_output.empty();
            }

            //! whether current input vars are different from prev input
            bool check_if_mem_status_change();
        };

        class ImpureMemPlanManager {
            bool m_layout_changed = false, m_during_check = false;
            OprNodeArray m_oprs;  //!< only oprs with IMPURE_OUTPUT_MEM_PLAN
            SmallVector<MemAllocPlan*> m_ptr_changed_mplans;
            SmallVector<std::pair<VarNode*, VarNode*>> m_force_update_pairs;

        public:
            void clear_tracked_oprs() { m_oprs.clear(); }

            void add_opr_to_track(OperatorNodeBase* opr) {
                m_oprs.emplace_back(opr);
            }

            //! called from init_single_var_mem_plan() when fixed alloc causes
            //! layout change
            void record_layout_changed(MemAllocPlan*) {
                m_layout_changed = true;
            }

            //! called from init_single_var_mem_plan() when fixed alloc causes
            //! ptr change
            inline void record_ptr_changed(VarNodeMemManager* mgr, VarNode* var);

            /*!
             * \brief check if static memory allocation is needed (i.e. if any
             *      layout changes)
             *
             * Note: readonly-fwd readers and force update dest of vars with
             * only ptr change would be updated if this function returns false.
             */
            bool check_need_realloc();
        };

        bool m_first_static_plan_run = true, m_optimize_started = false,
             m_already_free_no_need_mem = false;
        ComputingGraphImpl *m_owner_graph;
        ThinHashMap<VarNode*, VarNodeMemTrait> m_node_mem_trait;
        NullableHashMap<OperatorNodeBase*, DynamicAllocOprInfo>
            m_dynamic_alloc_opr_info;
        const OprNodeArray* m_opr_seq;

        //! vars that should be statically allocated
        VarNodeSet m_sys_alloc_static_vars;

        //! vars on which on_var_node_device_comp_finish() should be called
        VarNodeSet m_need_post_exec_action_vars;

        //! oprs that have at least one outputs in m_sys_alloc_static_vars
        OprNodeArray m_sys_alloc_static_oprs;
        //! oprs in m_sys_alloc_static_oprs that need on_mem_status_changed()
        //! callback; initialized in init_dynamic_alloc_opr_info()
        ThinHashSet<OperatorNodeBase*>
            m_sys_alloc_static_oprs_need_mem_status_changed_cb;

        SeqMemOptimizer m_seq_mem_opt;

        ImpureMemPlanManager m_impure_mem_plan_mgr;

        std::mutex m_dynamic_alloc_mtx;
        const size_t* m_run_id_ptr = nullptr;

        SyncableCounter m_cpu_async_release_barrier;


#if MGB_CUDA || MGB_ATLAS
        //! release dynamic var on after compnode event finishes
        class AsyncVarReleaser;
        std::unique_ptr<AsyncVarReleaser> m_asyn_var_releaser;
#endif

        VarDevMemDefragmenter m_var_dev_mem_defragmenter{this};

        std::shared_ptr<StaticDeviceMemoryManager> m_static_dev_mem_mgr =
                StaticDeviceMemoryManager::make_default_impl();
        SmallVector<DeviceTensorStorage> m_static_mem_refholder;
        size_t m_static_mem_refholder_dev_mem_mgr_version = 0;

        void assert_in_mem_opt_phase(size_t status);

        //! init dynamic allocation info, refcnt and m_need_exec_callback_vars
        void init_dynamic_alloc_opr_info();

        /*!
         * \brief set RT_FORCE_DYNAMIC_MEM_ALLOC for vars that are read by other
         *      comp nodes
         */
        void init_var_force_dynamic_alloc_flag();

        //! call add_layout_constraint for all oprs
        void init_layout_constraint();

        /*!
         * \brief init m_sys_alloc_static_vars, m_sys_alloc_static_oprs and
         *      m_should_sys_alloc
         */
        void init_sys_alloc_info(CompSeqExtraInfo &extra_info);

        //! init VarNodeMemTrait::seq_force_update_dest for all vars
        void init_var_seq_force_update_dest();

        //! initialize dev_tensor from the mem plan and given storage
        static void make_dev_tensor_from_mem_plan_single(
                VarNode* var, const DeviceTensorStorage& given_storage,
                size_t offset_in_given_storage = 0);

        //! initialize mem plans for output vars of a single operator
        void init_opr_outputs_mem_plan(
                OperatorNodeBase *opr, bool dynamic);

        /*!
         * \brief decrease refcnt and release memory if refcnt drops to zero
         *
         * Note that the refcnt is decreased asynchronously, which is controlled
         * by \p dispatch_cn
         *
         * \param dispatch_cn refcnt would be decreased after tasks on
         *      dispatch_cn finishes
         */
        void decr_var_mem_refcnt(VarNode *var, CompNode dispatch_cn);

        //! like decr_var_mem_refcnt, but decr refcnt immediately
        static void decr_var_mem_refcnt_sync(VarNode *var);

        //! print allocation statistics in reset_opr_seq
        void print_seq_info_log();

        static inline bool is_inf_refcnt_init(VarNode* var);

        /*!
         * \brief initialize var dev_tensor for static allocation vars from
         *      current mem plan
         *
         * This should only be called by alloc_var_node_mem_static()
         *
         * \return whether memory is reallocated
         */
        bool make_static_var_tensor_from_alloc_plan();
};

}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
