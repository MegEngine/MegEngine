/**
 * \file imperative/src/impl/op_def.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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

DispatchMode OpDef::decide_dispatch_mode(
    const OpDef& def,
    const SmallVector<LogicalTensorDesc>& inputs) {
    return def.trait()->decide_dispatch_mode(def, inputs);
}

SmallVector<TensorPtr> OpDef::apply_on_physical_tensor(
    const OpDef& def,
    SmallVector<TensorPtr> inputs) {
    return def.trait()->apply_on_physical_tensor(def, std::move(inputs));
}

void OpDef::apply_on_device_tensornd(
    const OpDef& def,
    const SmallVector<DeviceTensorND>& inputs,
    SmallVector<DeviceTensorND>* outputs) {
    def.trait()->apply_on_device_tensornd(def, inputs, outputs);
    return;
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

std::vector<std::pair<const char*, std::string>> OpDef::props(
    const OpDef& def) {
    return def.trait()->props(def);
}

std::string OpDef::to_string() const {
    std::string builder = "{";
    for (auto&& [name, value]: props(*this)) {
        builder += name;
        builder += ": ";
        builder += value;
        builder += ",";
    }
    return builder + "}";
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

const std::string OpDef::scope() const {
    return m_scope;
}

void OpDef::set_scope(const std::string& scope) {
    m_scope = scope;
}

const std::string OpDef::make_name() const {
    if (m_scope.empty())
        return trait()->make_name(*this);
    return m_scope + "." + trait()->make_name(*this);
}

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
