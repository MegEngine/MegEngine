/**
 * \file imperative/src/impl/ops/nms.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "../op_trait.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/standalone/nms_opr.h"

namespace mgb {
namespace imperative {

using NMSKeepOpr = opr::standalone::NMSKeep;

namespace {
cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& nms_keep = def.cast_final_safe<NMSKeep>();

    NMSKeepOpr::Param param;
    param.iou_thresh = nms_keep.iou_thresh;
    param.max_output = nms_keep.max_output;

    return NMSKeepOpr::make(inputs[0], param).node()->owner_opr();
}

OP_TRAIT_REG(NMSKeep, NMSKeep, NMSKeepOpr)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
} // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
