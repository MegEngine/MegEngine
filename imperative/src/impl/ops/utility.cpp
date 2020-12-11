/**
 * \file imperative/src/impl/ops/utility.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/utility.h"
#include <string>
#include "megbrain/comp_node.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/opr/utility.h"
#include "../op_trait.h"

namespace mgb::imperative {
namespace {

cg::OperatorNodeBase* virtual_dep_apply_on_var_node(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& graph = inputs[0]->owner_graph();
    auto&& op = def.cast_final_safe<VirtualDep>();
    VarNodeArray inps(inputs.begin(), inputs.end());
    cg::OperatorNodeConfig config;
    if (op.device.length() > 0) {
        config.comp_node(CompNode::load(op.device));
    }
    cg::OperatorNodeBase* opr =
            graph->insert_opr(std::make_unique<mgb::opr::VirtualDep>(
                    inps, config));
    return opr;
}

OP_TRAIT_REG(VirtualDep, VirtualDep, mgb::opr::VirtualDep)
        .apply_on_var_node(virtual_dep_apply_on_var_node)
        .fallback();
} // namespace

MGB_DYN_TYPE_OBJ_FINAL_IMPL(VirtualDep);

} // namespace mgb::imperative
