/**
 * \file src/opr/impl/loop/grad.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./impl.h"
#include "megbrain/opr/loop.h"
#include "megbrain/opr/internal/identical_fwd.h"

namespace mgb {
namespace opr {
namespace intl {

//! compute loop grad by loop
MGB_DEFINE_OPR_CLASS(LoopGrad, LoopImpl) // {
    friend class LoopGradSerializer;

    class GradProxy;
    class GradDesc;
    class AssignorGradOpr;
    class GraphModifier;

    Loop* const m_fwd_loop_opr;

    //! whether each output var has been bound in owner var
    std::vector<std::pair<bool, VarNode*>> m_grad_result_cache;

    //! index in input() for each var in outgrad given in make(); used for
    //! shallow copy; 0 for nullptr
    std::vector<size_t> m_orig_outgrad_idx_in_input;

    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    NodeProp *do_make_node_prop() const override;
    void add_input_layout_constraint() override;

    public:
        LoopGrad(Loop *loop_opr, std::unique_ptr<GradDesc> desc,
                const OperatorNodeConfig &config);

        /*!
         * \param outgrad output grad in owner graph, only including the output
         *      vars added by user
         */
        static LoopGrad* make(Loop *loop_opr, const VarNodeArray &outgrad,
                const OperatorNodeConfig &config = {});

        cg::OperatorNodeBase* shallow_copy(
                const VarNodeArray &inputs,
                const OperatorNodeConfig &config) const;

        /*!
         * \brief get grad var for given input
         */
        VarNode* get_grad_var(size_t inp_idx);
};

/*!
 * \brief add assignor grads to assignee grads
 *
 * When we have I[n+1] := O[n], grad for assignor O[n] must be added by grad
 * for assignee I[n+1].
 *
 * This operator is given original grad of O[n], and outputs modified grad.
 *
 * Note: we define assignee := assinor in loop fwd update.
 */
MGB_DEFINE_OPR_CLASS(LoopGrad::AssignorGradOpr,
        intl::ReadonlyFwdHelper<cg::SingleCNOperatorNodeBase>) // {

    struct State: public std::enable_shared_from_this<State>,
                  public NonCopyableObj {
        DepTensorUpdator::AccumulatorState accum_state;
        //! sum of grads of assignees in previous run
        DeviceTensorND prev_gsum;

        State() {
            accum_state.dest = &prev_gsum;
        }

        auto accum_state_shared() {
            return std::shared_ptr<DepTensorUpdator::AccumulatorState>{
                shared_from_this(), &accum_state};
        }
    };

    VarNode * const m_assignor;
    VarNode * const m_assignor_grad;     //!< original assignor grad
    std::shared_ptr<State> const m_state;

    bool m_assignee_grads_init = false, m_assignee_grads_empty = false,
         m_assignee_grads_buf_init = false;
    VarNodeArray m_assignee_grads;

    inline bool should_fwd() const;
    void mem_plan_fwd_in2out_readonly() override;
    void mem_plan_fwd_in2out_writable() override;
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp *do_make_node_prop() const override;

    public:

        AssignorGradOpr(VarNode *assignor_grad, VarNode *assignor,
                const std::shared_ptr<State> &state,
                const OperatorNodeConfig &config):
            Super{assignor->owner_graph(), config, "assignor_grad", {assignor}},
            m_assignor{assignor},
            m_assignor_grad{assignor_grad},
            m_state{state}
        {
            mgb_assert(assignor);
            if (assignor_grad) {
                add_input({assignor_grad});
            }
            add_input({assignor});
            add_output(None)->dtype(assignor->dtype());
            add_equivalence_component<ScalarHash<void*>>(m_state.get());
        }

        static SymbolVar make(SymbolVar assignor_grad, SymbolVar assignor,
                const std::shared_ptr<State> &state = std::make_shared<State>(),
                const OperatorNodeConfig &config = {}) {
            return assignor.insert_single_output_opr<AssignorGradOpr>(
                    assignor_grad.node(), assignor.node(), state, config);
        }

        void init_assignee_info(const VarNodeArray &assignees, SymbolVar loss);

        //! shallow copy this opr
        cg::OperatorNodeBase* shallow_copy(
                const VarNodeArray &inputs,
                const OperatorNodeConfig &config) const;

        //! called when grad loop finishes
        void on_grad_exec_finish() {
            m_state->prev_gsum = {};
            m_state->accum_state.reset();
        }

        VarNode* assignor() const {
            return m_assignor;
        }

        //! add extra compile output specs needed by this AssignorGradOpr
        void add_extra_compile_output_spec(ComputingGraph::OutputSpec &spec);
};

} // namespace intl
} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

