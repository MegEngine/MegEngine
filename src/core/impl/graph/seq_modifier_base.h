/**
 * \file src/core/impl/graph/seq_modifier_base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./memory_optimizer.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/cg.h"
#include "megbrain/plugin/opr_footprint.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/system.h"
#include "megbrain/utils/arith_helper.h"
#include "megbrain/utils/mempool.h"
#include "megbrain/utils/timer.h"

#if MGB_ENABLE_SUBLINEAR || MGB_ENABLE_DTR
namespace mgb {
namespace cg {

/*!
 * \brief modifying computing sequence, with basically the same idea of Training
 *      Deep Nets with Sublinear Memory Cost and Dynamic Tensor Rematerialization
 */
class SeqModifierBase {
public:
    /*!
     * describes modifications that should be applied to an operator sequnce:
     * maps from an opr to the oprs that should be duplicated and inserted
     * before it.
     */
    using SeqModifyAction = std::unordered_map<OperatorNodeBase*, OprNodeArray>;

    struct Var;
    struct Opr;

    class ModifyActionPlannerBase {
        const SeqModifierBase* const m_par_modifier;
        const OprNodeArray* m_orig_opr_seq;

        MemPool<Var> m_var_mempool;
        MemPool<Opr> m_opr_mempool;
        std::vector<MemPool<Var>::UniquePtr> m_var_storage;
        std::vector<MemPool<Opr>::UniquePtr> m_seq;
        size_t m_nr_endpoint_oprs = 0;

    public:
        //! special creation time used for oprs duplicated from others
        static constexpr size_t DUPOPR_TIME =
                std::numeric_limits<size_t>::max() - 1;

        auto& par_modifier() {
            return m_par_modifier;
        }

        auto& orig_opr_seq() {
            return m_orig_opr_seq;
        }

        MemPool<Var>& var_mempool() {
            return m_var_mempool;
        }

        MemPool<Opr>& opr_mempool() {
            return m_opr_mempool;
        }

        std::vector<MemPool<Var>::UniquePtr>& var_storage() {
            return m_var_storage;
        }

        std::vector<MemPool<Opr>::UniquePtr>& seq() {
            return m_seq;
        }

        size_t& nr_endpoint_oprs() {
            return m_nr_endpoint_oprs;
        }

        ModifyActionPlannerBase(SeqModifierBase* par)
                : m_par_modifier{par} {}

        ~ModifyActionPlannerBase() noexcept {
            m_opr_mempool.disable_freelist();
            m_var_mempool.disable_freelist();
        }

        //! init m_orig_opr_seq from opr_seq, should be called first.
        void init_seq(const OprNodeArray& opr_seq, bool remove_unused_output=true);
    };

    SeqModifierBase(ComputingGraphImpl* owner) : m_mem_opt(owner), m_owner_graph(owner) {}

    MemoryOptimizerHelper& mem_opt() {
        return m_mem_opt;
    }

    auto& owner_graph() {
        return m_owner_graph;
    }

    ThinHashMap<VarNode*, VarNode*>& var_map() {
        return m_var_map;
    }

    /*!
     * \brief copy opr and set inputs to m_new_inputs, and add outputs in
     *     m_var_map
     * \return new operator
     */
    OperatorNodeBase* copy_opr_from_new_inputs(OperatorNodeBase* opr, bool recomp, size_t recomp_cnt=0);

    /*!
     * \brief replace input vars according to m_var_map, and store results in
     *      m_new_inputs;
     * \return whether any var is changed
     */
    bool replace_vars(const VarNodeArray& inputs);
    
    //! see memory_optimizer set_priority_before_opt
    void set_priority_before_opt(const VarNodeArray& endpoints) {
        m_mem_opt.set_priority_before_opt(endpoints);
    }
    
    //! see memory_optimizer restore_graph_option
    void restore_graph_option() {
        m_mem_opt.restore_graph_option();
    }

private:
    MemoryOptimizerHelper m_mem_opt;
    
    ComputingGraphImpl* const m_owner_graph = nullptr;

    //! map from original var to replaced var
    ThinHashMap<VarNode*, VarNode*> m_var_map;
    
    VarNodeArray m_new_inputs;  //!< setup by replace_vars
};

struct SeqModifierBase::Opr {
    OperatorNodeBase* const orig_opr;
    std::vector<Var*> input, output;
    const size_t time;  //!< index in opr sequence
    const bool is_endpoint;
    double estimate_compute_time = 1;

    //! input vars that have been discarded and need to be recomputed before
    //! this opr; for internal use by apply_discard_plan()
    std::vector<Var*> inputs_to_recompute;

    //! new oprs to be inserted before this opr; setup by apply_discard_plan()
    std::vector<MemPool<Opr>::UniquePtr> oprs_insert_before;

    //! [begin, end) interval of *time* for oprs belonging to this block; setup
    //! by make_discard_plan()
    size_t block_begin_time = 0, block_end_time = 0;

    std::vector<ptrdiff_t> inputs_size;

    Opr(OperatorNodeBase* opr, size_t t)
            : orig_opr{opr},
              time{t},
              is_endpoint{opr->owner_graph()
                                  ->options()
                                  .opr_attribute.get_sublinear_memory_endpoint(
                                          opr)} {}
};

struct SeqModifierBase::Var {
    VarNode* const orig_var;
    size_t size;  //!< memory usage in bytes of this var
    size_t recomp_id = 0;

    //! write or read access of a var
    struct AccessRecord {
        Opr* const opr;
        const size_t time;
        size_t stride;

        explicit AccessRecord(Opr* o = nullptr)
                : opr{o}, time{o->time}, stride{0} {}
    };

    //! access_rec[0] is the creation opr, and others are reader oprs
    std::vector<AccessRecord> access_rec;

    /*!
     * An index in access_rec
     *
     * if valid, then the var should be discarded after
     * discard_tailing_access->opr finishes
     *
     * setup by make_discard_plan
     */
    Maybe<size_t> discard_tailing_access;

    /*!
     * An index in access_rec
     * maintained during make_discard_plan(), for the next access relative to
     * current operator
     */
    Maybe<size_t> next_access;

    AccessRecord* visit_discard_tailing_access() {
        return discard_tailing_access.valid()
                       ? &access_rec.at(discard_tailing_access.val())
                       : nullptr;
    }

    AccessRecord* visit_next_access() {
        return next_access.valid() ? &access_rec.at(next_access.val())
                                   : nullptr;
    }

    auto owner_opr() const { return access_rec[0].opr; }

    auto last_access_opr() const { return access_rec.back().opr; }

    Var(VarNode* var, size_t s, Opr* opr) : orig_var{var}, size{s} {
        access_rec.emplace_back(opr);
    }
};

}   // namespace cg
}   // namespace mgb

#endif  //  MGB_ENABLE_SUBLINEAR || MGB_ENABLE_DTR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
