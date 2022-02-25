#include "../op_trait.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/standalone/nms_opr.h"

namespace mgb {
namespace imperative {

using NMSKeepOpr = opr::standalone::NMSKeep;

namespace {
cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& nms_keep = def.cast_final_safe<NMSKeep>();

    NMSKeepOpr::Param param;
    param.iou_thresh = nms_keep.iou_thresh;
    param.max_output = nms_keep.max_output;

    OperatorNodeConfig config{nms_keep.make_name()};

    return NMSKeepOpr::make(inputs[0], param, config).node()->owner_opr();
}

OP_TRAIT_REG(NMSKeep, NMSKeep, NMSKeepOpr)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
