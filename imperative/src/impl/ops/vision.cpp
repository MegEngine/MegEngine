#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/imgproc.h"

#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const CvtColor&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::CvtColor::make(inputs[0], op.param(), config);
}
OP_TRAIT_REG(CvtColor, CvtColor).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace
}  // namespace imperative
}  // namespace mgb
