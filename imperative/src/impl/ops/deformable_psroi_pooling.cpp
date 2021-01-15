/**
 * \file imperative/src/impl/ops/deformable_psroi_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/dnn/roi_pooling.h"

#include "../op_trait.h"

namespace mgb::imperative {

namespace { namespace deformable_psroi_pooling {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    mgb_assert(inputs.size() == 3);
    auto&& op = static_cast<const DeformablePSROIPooling&>(def);
    return opr::DeformablePSROIPooling::make_all(inputs[0], inputs[1], inputs[2], op.param());
}

OP_TRAIT_REG(DeformablePSROIPooling, DeformablePSROIPooling)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // deformable_psroi_pooling

} // namespace mgb::imperative
