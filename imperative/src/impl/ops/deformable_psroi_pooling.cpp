#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/dnn/roi_pooling.h"

#include "../op_trait.h"

namespace mgb::imperative {

namespace {
namespace deformable_psroi_pooling {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    mgb_assert(inputs.size() == 3);
    auto&& op = static_cast<const DeformablePSROIPooling&>(def);
    return opr::DeformablePSROIPooling::make_all(
            inputs[0], inputs[1], inputs[2], op.param());
}

OP_TRAIT_REG(DeformablePSROIPooling, DeformablePSROIPooling)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace deformable_psroi_pooling
}  // namespace

}  // namespace mgb::imperative
