/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/ops/opr_attr.h"

#include "./op_trait.h"

namespace mgb {
namespace imperative {

std::shared_ptr<OpDef> OpDef::make_from_op_node(
    cg::OperatorNodeBase* node) {
    OpTrait* trait;
    trait = OpTrait::find_by_typeinfo(node->dyn_typeinfo());
    if (!trait) {
        // TODO: register `make_from_op_node` for each OperatorNode
        // instead of forwarding to OprAttr
        trait = OpTrait::find_by_typeinfo(OprAttr::typeinfo());
    }
    mgb_assert(trait);
    return trait->make_from_op_node(node);
}

SmallVector<TensorPtr> OpDef::apply_on_physical_tensor(
    const OpDef& def,
    const SmallVector<TensorPtr>& inputs) {
    return def.trait()->apply_on_physical_tensor(def, inputs);
}

void OpDef::exec(
    const OpDef& def,
    const SmallVector<TensorPtr>& inputs,
    const SmallVector<TensorPtr>& outputs) {
    def.trait()->exec(def, inputs, outputs);
}

cg::OperatorNodeBase* OpDef::apply_on_var_node(
    const OpDef& def,
    const VarNodeArray& inputs) {
    return def.trait()->apply_on_var_node(def, inputs);
}

SmallVector<LogicalTensorDesc> OpDef::infer_output_attrs_fallible(
    const OpDef& def,
    const SmallVector<LogicalTensorDesc>& inputs) {
    return def.trait()->infer_output_attrs_fallible(def, inputs);
}

SmallVector<LogicalTensorDesc> OpDef::infer_output_attrs(
    const OpDef& def,
    const SmallVector<TensorPtr>& inputs) {
    return def.trait()->infer_output_attrs(def, inputs);
}

BackwardGraphResult OpDef::make_backward_graph(
    const OpDef& def,
    const SmallVector<LogicalTensorDesc>& inputs,
    const SmallVector<bool>& input_requires_grad,
    const SmallVector<bool>& output_has_grad) {
    return def.trait()->make_backward_graph(def, inputs, input_requires_grad, output_has_grad);
}

const OpTrait* OpDef::trait() const {
    if (!m_trait) {
        m_trait = OpTrait::find_by_typeinfo(dyn_typeinfo());
        mgb_throw_if(!m_trait, MegBrainError,
            "can not find op_trait by %s", dyn_typeinfo()->name);
    }
    return m_trait;
}

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
