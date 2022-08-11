#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

#include "megbrain/opr/imgproc.h"

namespace mgb::imperative {

namespace {
namespace warp_perspective_backward_data {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    mgb_assert(inputs.size() == 3);
    auto&& op = static_cast<const WarpPerspectiveBackwardData&>(def);
    OperatorNodeConfig config{op.make_name()};
    return opr::WarpPerspectiveBackwardData::make(
            inputs[0], inputs[1], inputs[2], op.param(), config);
}

OP_TRAIT_REG(WarpPerspectiveBackwardData, WarpPerspectiveBackwardData)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace warp_perspective_backward_data

namespace warp_perspective_backward_mat {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    mgb_assert(inputs.size() == 3);
    auto&& op = static_cast<const WarpPerspectiveBackwardMat&>(def);
    OperatorNodeConfig config{op.make_name()};
    return opr::WarpPerspectiveBackwardMat::make(
            inputs[0], inputs[1], inputs[2], op.param(), config);
}

OP_TRAIT_REG(WarpPerspectiveBackwardMat, WarpPerspectiveBackwardMat)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace warp_perspective_backward_mat
}  // namespace

}  // namespace mgb::imperative
