#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

#if MGB_ATLAS
#include "megbrain/opr/atlas_runtime_op.h"
namespace mgb::imperative {

namespace {
namespace atlas_runtime {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const AtlasRuntime&>(def);
    SymbolVarArray symbol_var_inputs(inputs.begin(), inputs.end());
    OperatorNodeConfig config{op.make_name()};
    return opr::AtlasRuntimeOpr::make(
            op.buf.c_str(), op.buf_size, symbol_var_inputs, config);
}
OP_TRAIT_REG(AtlasRuntime, AtlasRuntime)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace atlas_runtime
}  // namespace

}  // namespace mgb::imperative
#endif
