/**
 * \file imperative/src/impl/ops/deformable_conv2d.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/dnn/convolution.h"

#include "../op_trait.h"

namespace mgb::imperative {

namespace { namespace deformableconv {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::DeformableConv>();
    return DeformableConv::make(node->param(), node->execution_policy());
}

auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& dcn = static_cast<const DeformableConv&>(def);
    mgb_assert(inputs.size() == 4);
    return opr::DeformableConv::make(inputs[0], inputs[1], inputs[2], inputs[3], dcn.param(), dcn.policy());
}

OP_TRAIT_REG(DeformableConv, DeformableConv, opr::DeformableConv)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // deformableconv

} // namespace mgb::imperative
