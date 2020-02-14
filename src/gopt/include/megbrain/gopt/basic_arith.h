/**
 * \file src/gopt/include/megbrain/gopt/basic_arith.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/internal/indexing_helper.h"

#include "megbrain/gopt/framework.h"

namespace mgb {
namespace gopt {

    /*!
     * \brief try to perform inplace optimization for elemwise arith expressions
     *
     * Inplace optimization can only replace the to-be-created operator but not
     * existing operators. For example, f(a, b) may be replaced by g(a, b), but
     * the vars a and b must not be removed since it may be required by the
     * user.
     *
     * This optimization is performed during graph building.
     *
     * \return node if current expr replaced successfully; nullptr if not
     */
    VarNode* optimize_elemwise_expr_inplace(
            const VarNodeArrayView& inputs,
            opr::Elemwise::Param param,
            const OperatorNodeConfig &config);

    /*!
     * \brief whether opr type and param may have inplace optimization on some
     *      inputs
     *
     * Used for checking inplace copy error
     */
    bool has_inplace_basic_arith_opt(const cg::OperatorNodeBase &opr);

    /*!
     * \brief optimize by modifying the terms in Elemwise::sum_grad_list
     * \param[in] grads grad array that would be modified in-place; its output
     *      value has no specific meaning and should not be used further
     * \param[out] mid_results intermediate sum results (i.e. all newly created
     *      oprs)
     */
    class GradSumListOptimizer {
        VarNode *m_wrt, *m_sum, *m_brdcast_sum_wrt_shp = nullptr;
        VarNodeArray& m_grads;
        std::vector<cg::OperatorNodeBase*> m_incr_subtensor_oprs;

        /*!
         * \brief remove null items in m_grads while keeping order
         * \return number of items removed
         */
        size_t remove_null_grads();

        //! check whether var is GetVarShape(m_wrt)
        bool check_is_shapeof_wrt(VarNode *var);

        /*!
         * \brief modify m_grads to find broadcast(x, wrt.shape()) terms and
         *      move the broadcast to outest
         */
        void remove_broadcast();

        //! merge incr_subtensor(zeros_like(wrt), x)
        void merge_incr_subtensor();

        void calc_sum(VarNodeArray &mid_results);

        public:
            GradSumListOptimizer(VarNode *wrt,
                    VarNodeArray &grads,
                    VarNodeArray &mid_results);

            VarNode* get_sum() {
                return m_sum;
            }
    };

    /*!
     * \brief check whether an operator is incr_sub(0, ...) type
     * \param require_brdcst whether to require input(0) from Broadcast
     */
    bool check_is_incr_subtensor_zero(
            cg::OperatorNodeBase *opr, bool require_brdcst = false);

    /*!
     * \brief create a new incr_sub opr by replacing inputs
     *
     * \param new_data new data node to be used as input(0) for new opr
     */
    VarNode* remake_incr_subtensor_zero(cg::OperatorNodeBase *orig_opr,
            VarNode *new_data = nullptr,
            const opr::intl::FancyIndexingHelper::InputTensorReplacer&
            input_tensor_replacer = {});

    /*!
     * \brief expand fused arith oprs to normal oprs
     *
     * For example, FMA3(a, b, c) would be changed to a * b + c
     */
    class ExpandFusedArithPass final: public Pass {
        public:
            const char* name() const override;
            void apply(OptState &opt) const override;
    };

    /*!
     * \brief normalize arithmetic expression chains
     *
     * Transform add/sub or mul/div arith expr chains to add-only or mul-only
     * chains, and add corresponding inverse operator at leaf nodes. For
     * example, (a - (b - c + d)) would be changed to (a + (-b) + c + (-d)).
     *
     * Identical ADD/MUL terms would be collapsed into MUL/POW. PowC would be
     * canonized to POW.
     */
    class NormalizeArithChainPass final: public Pass {
        class Impl;

        public:
            const char* name() const override;
            void apply(OptState &opt) const override;
    };

    /*!
     * \brief distribute mul over add to speed up computation
     *
     * This pass is usually used after NormalizeArithChainPass, and it trys to
     * distribute mul over add to move group operands with small shapes to speed
     * up computation.
     */
    class ArithMulDistributePass final: public Pass {
        class Impl;

        public:
            const char* name() const override;
            void apply(OptState &opt) const override;
    };

    /*!
     * \brief fuse arith expressions
     *
     * This pass is usually used after ArithMulDistributePass and it fuses oprs
     * like a * x + b => fma3(a, x, b)
     */
    class ArithFusePass final: public Pass {
        class Impl;

        public:
            const char* name() const override;
            void apply(OptState &opt) const override;
    };


    /*!
     * \brief reorder terms in arithmetic expression chains
     *
     * This pass is usually used after ArithFusePass, and it would
     * reorder chains consisting of ADD or MUL oprs.
     */
    class ReorderArithChainPass final: public Pass {
        ConstVarType m_const_var_type;
        class Impl;

        public:
            ReorderArithChainPass(ConstVarType const_var_type):
                m_const_var_type{const_var_type}
            {
            }

            const char* name() const override;
            void apply(OptState &opt) const override;
    };

    /*!
     * \brief final arithmetic transformations
     *
     * This pass is usually used as the last pass it perform miscellaneous
     * transformations like a + (-b) => a - b and replacing scalar POW with
     * PowC.
     */
    class FinalArithTransformPass final: public Pass {
        class Impl;

        public:
            const char* name() const override;
            void apply(OptState &opt) const override;
    };

} // namespace gopt
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

