/**
 * \file imperative/src/impl/op_def.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
    SmallVector<TensorPtr> inputs) {
    return def.trait()->apply_on_physical_tensor(def, std::move(inputs));
}

VarNodeArray OpDef::apply_on_var_node(
    const OpDef& def,
    const VarNodeArray& inputs) {
    return def.trait()->apply_on_var_node(def, inputs);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> OpDef::infer_output_attrs_fallible(
    const OpDef& def,
    const SmallVector<LogicalTensorDesc>& inputs) {
    return def.trait()->infer_output_attrs_fallible(def, inputs);
}

BackwardGraphResult OpDef::make_backward_graph(
    const OpDef& def,
    const SmallVector<LogicalTensorDesc>& inputs,
    const SmallVector<bool>& input_requires_grad,
    const SmallVector<bool>& output_has_grad) {
    return def.trait()->make_backward_graph(def, inputs, input_requires_grad, output_has_grad);
}

size_t OpDef::hash() const {
    return trait()->hash(*this);
}

bool OpDef::is_same_st(const Hashable& rhs) const {
    return trait()->is_same_st(*this, static_cast<const OpDef&>(rhs));
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
