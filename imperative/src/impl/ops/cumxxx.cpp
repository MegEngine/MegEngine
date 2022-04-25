#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/misc.h"

namespace mgb::imperative {

namespace {
namespace cumsum {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Cumsum&>(def);
    OperatorNodeConfig config{op.make_name()};
    return opr::Cumsum::make(inputs[0], op.param(), config);
}

OP_TRAIT_REG(Cumsum, Cumsum).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace cumsum
}  // namespace

namespace {
namespace cumprod {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Cumprod>();
    OperatorNodeConfig config{op.make_name()};
    return opr::Cumprod::make(inputs[0], op.param(), config);
}

OP_TRAIT_REG(Cumprod, Cumprod).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace cumprod
}  // namespace

}  // namespace mgb::imperative
