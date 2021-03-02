/**
 * \file imperative/src/impl/ops/vision.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/imgproc.h"

#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const CvtColor&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::CvtColor::make(inputs[0], op.param(), config);
}
OP_TRAIT_REG(CvtColor, CvtColor)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}
}
}
