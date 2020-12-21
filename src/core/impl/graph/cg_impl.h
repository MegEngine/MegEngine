/**
 * \file src/core/impl/graph/cg_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./eager_eval.h"
#include "./grad_manager.h"
#include "./graph_opt.h"
#include "./seq_comp_node_opt_impl.h"
#include "./seq_sublinear_memory.h"
#include "./static_infer_impl.h"
#include "./swap/memory_swap.h"
#include "./topo_sort.h"
#include "./var_node_mem_mgr.h"

#include "megbrain/utils/mempool.h"

namespace mgb {
namespace cg {

class ComputingGraphImpl final : public ComputingGraph {
    class CallbackCaller;
    class RecordedComputingSequence;
    class MegDNNDtorCheck;
    class MultiPartCompiler;
    friend class GradManager;

    //! temporary state in compiling
    struct CompileState {
        //! extra info that must be set in the ComputingSequence
        CompSeqExtraInfo extra_info;
        const OprNodeArray* opr_seq = nullptr;
    };

    struct CallbackCallerKey {
        OperatorNodeBase* opr;
        CompNode comp_node;

        bool operator==(const CallbackCallerKey& rhs) const {
            return opr == rhs.opr && comp_node == rhs.comp_node;
        }

        struct Hash {
            size_t operator()(const CallbackCallerKey& b) const {
                return hash_pair_combine(mgb::hash(b.opr),
                                         mgb::hash(b.comp_node));
            }
        };
    };

    struct CallbackCallerVal {
        SmallVector<VarNode*> vars;
        //! indexs of vars in out_spec.
        SmallVector<SmallVector<size_t>> indexs;
    };

    /*!
     * Components for implementing algorithms on a computing graph.
     *
     * They are put in a separate struct because they need to be destructed in
     * on_comp_node_finalize(), before ~ComputingGraphImpl().
     */
    struct Components : public NonCopyableObj {
        TopoSorter topo_sorter;
        VarNodeMemManager var_node_mem_manager;
        SeqCompNodeOptimizerImpl seq_comp_node_opt;
        static_infer::StaticInferManagerImpl static_infer_manager;
        static_infer::CompSeqManager static_infer_comp_seq_manager;
        GradManager grad_manager;
        GraphOptimizer graph_optimizer;
#if MGB_ENABLE_SUBLINEAR
        SeqModifierForSublinearMemory seq_modifier_for_sublinear_memory;
#endif
#if MGB_ENABLE_MEMORY_SWAP
        swap::MemorySwap memory_swap_support;
#endif
        EagerEvalManager eager_eval_manager;

        explicit Components(ComputingGraphImpl* owner);
    };

    //! valid if graph has been compiled with comp_node_seq_record_level == 2
    //! must be placed first so it can be destructed last
    std::unique_ptr<MegDNNDtorCheck> m_recorded_seq_level2_dtor_chk;

    MemPool<VarNode> m_var_node_pool;

    //! if not null, this graph is set as subgraph of it by set_as_subgraph()
    ComputingGraphImpl* m_parent_graph = nullptr;
    std::vector<ComputingGraphImpl*> m_subgraphs;

    AsyncExecutable* m_current_comp_seq = nullptr;

    std::shared_ptr<size_t> m_node_id_counter = std::make_shared<size_t>();

    std::vector<std::unique_ptr<OperatorNodeBase>> m_opr_refkeeper;

    /*!
     * list of operator nodes that take some var as one of the inputs; each
     * output var would be in this map, even with an empty operator set
     */
    ThinHashMap<VarNode*, OprNodeArray> m_var_receiver;

    std::aligned_storage_t<sizeof(Components), alignof(Components)>
            m_components_storage;

    /*!
     * \brief get dest vars and add extra_vardeps from OutputSpec
     * \param[out] has_virtual_grad whether there are VirtualGrad oprs that
     *      need to be expanded
     */
    VarNodeArray get_dest_vars_from_out_spec(const OutputSpec& spec,
                                             SpecialOprStat& sopr_stat);

    void cleanup();

    std::shared_ptr<void> on_comp_node_finalize() override;

    Components& components() {
        return reinterpret_cast<Components&>(m_components_storage);
    }

    const Components& components() const {
        return reinterpret_cast<const Components&>(m_components_storage);
    }

    //! prepare computing sequence and initialize opr sequence
    CompileState compile_prepare(const OutputSpec& out_spec);

    //! finalize the computing sequence for compiling
    std::unique_ptr<AsyncExecutable> compile_commit(CompileState state);

public:
    class ComputingSequence;

    ComputingGraphImpl();
    ~ComputingGraphImpl();

    template<typename T> static ComputingGraphImpl* downcast(T* ptr) = delete;

    inline static ComputingGraphImpl* downcast(ComputingGraph* graph) {
        mgb_assert(!graph->options().imperative_proxy_graph);
        return static_cast<ComputingGraphImpl*>(graph);
    }

    friend struct ComputingGraph::Options;

    std::unique_ptr<AsyncExecutable> compile(
            const OutputSpec& out_spec) override;

    SmallVector<std::unique_ptr<AsyncExecutable>> compile_multi_part(
            const SmallVector<OutputSpec>& out_specs) override;

    OperatorNodeBase* insert_opr(
            std::unique_ptr<OperatorNodeBase> opr) override;

    void* alloc_varnode_storage() override;

    void free_varnode_storage(void *ptr) override;

    const VarReceiverInfo& var_receiver_in_current_comp_seq(
            const VarNode* var) const override;

    /*!
     * \brief get the nodes in opr_set that directly depend on var (i.e.
     *      those oprs that take var as one input)
     *
     * Guaranteed no duplication in this list, and the order is stable
     */
    const OprNodeArray& var_receiver(VarNode* var) const {
        return m_var_receiver.at(var);
    }

    std::string get_mem_allocation_info() const override;

    VarNode* find_var_by_id(size_t id) const override;

    TopoSorter& topo_sorter() { return components().topo_sorter; }

    size_t next_node_id() override { return (*m_node_id_counter)++; }

    VarNodeMemManager& var_node_mem_manager() {
        return components().var_node_mem_manager;
    }

    SeqCompNodeOptimizer& seq_comp_node_optimizer() override {
        return components().seq_comp_node_opt;
    }

    static_infer::StaticInferManager& static_infer_manager() override {
        return components().static_infer_manager;
    }

    static_infer::StaticInferManagerImpl& static_infer_manager_impl() {
        return components().static_infer_manager;
    }

    static_infer::CompSeqManager& static_infer_comp_seq_manager() {
        return components().static_infer_comp_seq_manager;
    }

    GraphOptimizer& graph_optimizer() { return components().graph_optimizer; }

    EagerEvalManager& eager_eval_manager() {
        return components().eager_eval_manager;
    }

#if MGB_ENABLE_SUBLINEAR
    SeqModifierForSublinearMemory& seq_modifier_for_sublinear_memory();
#endif

    void share_device_memory_with(ComputingGraph& other) override;

    void set_device_memory_allocator(
            std::shared_ptr<DeviceMemoryAllocator> allocator) override;

    size_t get_device_memory_size(CompNode cn) override;

    size_t clear_device_memory() override;

    void set_as_subgraph(ComputingGraph& par_graph) override;

    void record_async_error(std::unique_ptr<MegBrainError> async_exc) override;

    /*!
     * \brief latest computing sequence from this graph
     *
     * Since new computing sequence would invalidate memory layout of older
     * ones and operators may have class-level states, only the last
     * computing sequence could be used
     *
     * \return current comp seq, or nullptr if no alive comp seq
     */
    AsyncExecutable* current_comp_seq() override {
        return static_cast<AsyncExecutable*>(m_current_comp_seq);
    }

    /*!
     * \brief get current ExecEnv
     * \return ExecEnv if there is a compiled computing sequence, or
     *      nullptr if not compiled yet
     */
    GraphExecutable::ExecEnv* current_exec_env();

    /*!
     * \brief get step number of an operator in current computing sequence
     * \return step number; None if opr not in seq
     */
    Maybe<size_t> opr_step_num_in_cur_comp_seq(OperatorNodeBase* opr);

    /*!
     * \brief get extra info for current computing sequence
     */
    const CompSeqExtraInfo& current_comp_seq_extra_info();

    /*!
     * \brief get associated grad manager
     */
    GradManager& grad_manager() { return components().grad_manager; }

    //! get all the operators in this graph
    auto&& all_oprs() const { return m_opr_refkeeper; }

    size_t nr_oprs_in_graph() const override { return m_opr_refkeeper.size(); }

    //! memory pool for the var nodes; used by OperatorNodeBase
    auto&& var_node_pool() { return m_var_node_pool; }
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
