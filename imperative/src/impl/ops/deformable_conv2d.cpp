#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/dnn/convolution.h"

#include "../op_trait.h"

namespace mgb::imperative {

namespace {
namespace deformableconv {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::DeformableConv>();
    return DeformableConv::make(node->param(), node->execution_policy());
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& dcn = static_cast<const DeformableConv&>(def);
    mgb_assert(inputs.size() == 4);
    return opr::DeformableConv::make(
            inputs[0], inputs[1], inputs[2], inputs[3], dcn.param(), dcn.policy());
}

OP_TRAIT_REG(DeformableConv, DeformableConv, opr::DeformableConv)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace deformableconv
}  // namespace

}  // namespace mgb::imperative
