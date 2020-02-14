/**
 * \file src/serialization/include/megbrain/serialization/opr_shallow_copy.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/graph.h"

namespace mgb {
namespace serialization {

/*!
 * \brief context for opr shallow copy
 * \param owner_graph specify the graph to store the copied operator; it is
 *      only useful when copying an operator with no inputs
 */
class OprShallowCopyContext {
    ComputingGraph* m_owner_graph;

public:
    OprShallowCopyContext(ComputingGraph* owner_graph = nullptr)
            : m_owner_graph{owner_graph} {}

    //! change default owner graph
    OprShallowCopyContext& owner_graph(ComputingGraph* graph) {
        m_owner_graph = graph;
        return *this;
    }

    //! get owner graph and check that it matches opr and inputs
    ComputingGraph* owner_graph(const cg::OperatorNodeBase& opr,
                                const VarNodeArray& inputs) const;
};

/*!
 * \brief copy a single operator by serializing and the then deserializing
 *      using new config and apply on new inputs
 */
cg::OperatorNodeBase* copy_opr_shallow(const cg::OperatorNodeBase& opr,
                                       const VarNodeArray& inputs,
                                       const OperatorNodeConfig& config = {},
                                       const OprShallowCopyContext& ctx = {});

namespace intl {

cg::OperatorNodeBase* copy_opr_shallow_default_impl(
        const OprShallowCopyContext& ctx, const cg::OperatorNodeBase& opr,
        const VarNodeArray& inputs, const OperatorNodeConfig& config);

}  // namespace intl

}  // namespace serialization
}  // namespace mgb
