#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_runtime_opr.h"
namespace mgb::imperative {

namespace {
namespace tensorrt_runtime {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const TensorRTRuntime&>(def);
    OperatorNodeConfig config{op.make_name()};
    SymbolVarArray sinputs(inputs.begin(), inputs.end());
    return opr::TensorRTRuntimeOpr::make(op.buf.c_str(), op.buf_size, sinputs, config);
}
OP_TRAIT_REG(TensorRTRuntime, TensorRTRuntime)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace tensorrt_runtime
}  // namespace

}  // namespace mgb::imperative
#endif  // MGB_ENABLE_TENSOR_RT
