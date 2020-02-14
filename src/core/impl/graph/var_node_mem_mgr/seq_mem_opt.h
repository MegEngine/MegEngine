/**
 * \file src/core/impl/graph/var_node_mem_mgr/seq_mem_opt.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "../impl_common.h"

namespace mgb {
namespace cg {

/*!
 * \brief Computing sequence memory optimizer.
 *
 * Optimize mem plans (mostly for statically allocated vars)
 */
class SeqMemOptimizer {
    class StaticMemAllocLogger;

    /*!
     * \brief life interval for a memory chunk
     */
    struct MemChunkLifeInterval {
        size_t begin = 0, end = 0;
        MemAllocPlan::Chunk *chunk = nullptr;
        CompNode comp_node;
    };

    using CompNode2Chunkset = CompNode::UnorderedMap<
        ThinHashSet<MemAllocPlan::Chunk*>>;

    ComputingGraphImpl *m_graph;
    const OprNodeArray *m_cur_seq_full;
    const OprNodeArray *m_cur_seq_sys_alloc;
    const VarNodeSet *m_cur_static_alloc_var;
    ThinHashSet<OperatorNodeBase*> m_cur_seq_sys_alloc_set;
    Maybe<CompNode::UnorderedMap<size_t>> m_static_mem_usage;
    SmallVector<CompNode> m_all_comp_nodes;

    size_t m_status = 0;
    std::vector<std::pair<MemAllocPlan*, MemAllocPlan*>>
        m_writable_fwd_mem_plans;

    bool should_static_alloc_var(VarNode *var);

    bool in_sys_alloc(OperatorNodeBase *opr) const {
        return m_cur_seq_sys_alloc_set.count(opr);
    }

    //! return as alloc_mem_chunk_storage
    bool run_static_mem_alloc();

    //! return as alloc_mem_chunk_storage
    bool run_static_mem_alloc_on_comp_node(CompNode cn,
            const std::vector<MemChunkLifeInterval> &chunks,
            StaticMemAllocLogger &static_mem_alloc_logger);

    public:
        SeqMemOptimizer(ComputingGraphImpl *graph):
            m_graph(graph)
        {}

        /*!
         * \brief reset the operator sequence to be optimized
         *
         * This function should be called by VarNodeMemManager::reset_opr_seq
         *
         * \param all_comp_nodes all the involved comp nodes, including thoses
         *      not involved in static memory allocation
         */
        void reset_opr_seq(const OprNodeArray *seq,
                const OprNodeArray *seq_sys_alloc,
                const VarNodeSet *static_alloc_var,
                SmallVector<CompNode> all_comp_nodes);

        /*!
         * \brief add a request that a MemAllocPlan should be forwarded to
         *      another MemAllocPlan in a writable way
         *
         * this is used to implement mem_plan_fwd_in2out_writable, and this
         * records would not be cleared by reset_opr_seq
         */
        void add_writable_fwd_mem_plan_pair(
                MemAllocPlan *from, MemAllocPlan *to);

        /*!
         * \brief optimize mem_plan for var nodes by performing
         *      readonly/writable forwarding
         */
        void optimize_mem_plan();

        /*!
         * \brief compute static memory allocation plan
         *
         * This initiates m_static_mem_usage and
         * stores the offsets in Chunk::static_offset_in_device_storage
         *
         * \return whether re-allocation is needed
         */
        bool plan_chunk_allocation();

        /*!
         * \brief get static memory usage on each comp node
         *
         * This is only valid after calling alloc_mem_chunk_storage_plan()
         */
        const CompNode::UnorderedMap<size_t>& static_mem_usage() const {
            return m_static_mem_usage.val();
        }

        void optimize_mem_plan_dynamic(OperatorNodeBase *opr);

        /*!
         * \brief bitmask for status
         */
        struct Status {
            static constexpr size_t
                ALLOW_FWD_IN2OUT_READONLY = 1,
                ALLOW_FWD_IN2OUT_WRITABLE = 2;
        };

        /*!
         * \brief get current allocation status, to determine whether
         *      mem_plan_fwd_in2out_* calls are legal
         */
        size_t status() const {
            return m_status;
        }
};

} // namespace cg
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

