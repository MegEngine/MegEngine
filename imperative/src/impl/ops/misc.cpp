/**
 * \file imperative/src/impl/ops/tensor_manip.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "../op_trait.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/misc.h"

namespace mgb {
namespace imperative {

namespace check_has_inf {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<CheckHasInf>();
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::CheckHasInf::make(inputs[0], {}, config);
}
OP_TRAIT_REG(CheckHasInf, CheckHasInf)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace check_has_inf

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
