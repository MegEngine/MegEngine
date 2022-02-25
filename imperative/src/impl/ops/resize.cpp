#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/imgproc.h"

#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Resize&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::Resize::make(inputs[0], inputs[1], op.param(), config);
}

OP_TRAIT_REG(Resize, Resize).apply_on_var_node(apply_on_var_node).fallback();
}  // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
