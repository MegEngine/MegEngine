/**
 * \file src/core/impl/graph/graph_opt.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/operator_node.h"

namespace mgb {
namespace cg {

/*!
 * \brief computing graph optimizer
 *
 * The optimization takes place during graph construction; currently two
 * optimizations are implemented:
 * 1. common subexpression elimination
 * 2. swap type_cvt followed by broadcast when value is constant
 * 3. merge multiple broadcasts when value is constant
 * 4. constant folding
 */
class GraphOptimizer {
    //! group operator nodes by hash value, for CSE
    ThinHashMap<size_t, std::vector<OperatorNodeBase*>> m_opr_hash_list;

    //! map from const inferable var node to its ImmutableTensor opr
    ThinHashMap<VarNode*, OperatorNodeBase*> m_const_map;

    /*!
     * \brief try to replace multiple broadcasts into one
     *
     * \return nullptr if failed to replace; otherwise it returns the new
     *      Broadcast opr
     */
    OperatorNodeBase* merge_bcast(VarNode* var);

    /*!
     * \brief try to swap a TypeCvt followed by a Broadcast
     *
     * \return nullptr if failed to swap; otherwise it returns the swapped
     *      oprs
     */
    OperatorNodeBase* swap_typecvt_and_bcast(VarNode* var);

    /*!
     * \brief try to replace a var by an ImmutableTensor
     *
     * \return nullptr if failed to replace; otherwise it returns the new
     *      ImmutableTensor opr
     */
    OperatorNodeBase* replace_const_var(VarNode *var);

    public:

        /*!
         * \brief called at beginning of inserting opr to graph
         *
         * This method should be first quried when inserting an operator; if it
         * returns nullptr, normal insertion procedure continuous; otherwise the
         * returned opr should be used and new opr to be inserted should be
         * discarded.
         */
        OperatorNodeBase* insert_pre(OperatorNodeBase *opr);

        /*!
         * \brief called at end of inserting opr to graph
         *
         * This method should be quried after new operator is initialized and
         * stored; it would either return *opr*, or an optimized version of
         * *opr*.
         *
         * Currently it only replaces const values for single output operator.
         */
        OperatorNodeBase* insert_post(OperatorNodeBase *opr);
};

} // namespace cg
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
