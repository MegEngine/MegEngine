/**
 * \file imperative/src/impl/ops/warp_affine.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

#include "megbrain/opr/imgproc.h"

namespace mgb::imperative {

namespace { namespace warp_affine {
    auto apply_on_var_node(
            const OpDef& def,
            const VarNodeArray& inputs) {
        mgb_assert(inputs.size() == 3);
        auto&& op = static_cast<const WarpAffine&>(def);
        OperatorNodeConfig config{op.make_name()};
        return opr::WarpAffine::make(inputs[0], inputs[1], inputs[2], op.param(), config);
    }

OP_TRAIT_REG(WarpAffine, WarpAffine)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // warp_affine

} // namespace mgb::imperative
