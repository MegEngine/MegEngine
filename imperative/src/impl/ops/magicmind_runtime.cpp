#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

#if MGB_CAMBRICON
#include "megbrain/cambricon/magicmind_runtime_opr.h"
namespace mgb::imperative {

namespace {
namespace magicmind_runtime {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
#if CNRT_MAJOR_VERSION >= 5
    auto&& op = static_cast<const MagicMindRuntime&>(def);
    SymbolVarArray symbol_var_inputs(inputs.begin(), inputs.end());
    OperatorNodeConfig config{op.make_name()};
    return opr::MagicMindRuntimeOpr::make(
            op.buf.c_str(), op.buf_size, symbol_var_inputs, config);
#else
    mgb_assert(
            false,
            "Magicmind runtime opr is disabled at compile time, the reason of which is "
            "the version of cnrt runtime is lower than 5.0. Please check the version "
            "of your cambricon toolkit, and recompile megengine.");
    return SymbolVar{};
#endif
}
OP_TRAIT_REG(MagicMindRuntime, MagicMindRuntime)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace magicmind_runtime
}  // namespace

}  // namespace mgb::imperative
#endif
