/**
 * \file imperative/src/impl/ops/utility.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/opr/utility.h"
#include "../op_trait.h"

namespace mgb::imperative {

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GenericPyOp);

namespace { namespace fastpathcopy {
    auto apply_on_var_node(
            const OpDef& def,
            const VarNodeArray& inputs) {
        return inputs;
    }

OP_TRAIT_REG(FastpathCopy,FastpathCopy)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // fastpathcopy

namespace { namespace identity {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Identity>();
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::Identity::make(inputs[0], config);
}

auto apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    return SmallVector<TensorPtr>{inputs[0]};
}
OP_TRAIT_REG(Identity, Identity)
    .apply_on_var_node(apply_on_var_node)
    .apply_on_physical_tensor(apply_on_physical_tensor)
    .fallback();
}} // identity

} // namespace mgb::imperative
