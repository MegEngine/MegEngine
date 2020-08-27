/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */
#include "../op_trait.h"

#include "megbrain/imperative/ops/nms.h"
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

MGB_DYN_TYPE_OBJ_FINAL_IMPL(NMSKeep);

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
