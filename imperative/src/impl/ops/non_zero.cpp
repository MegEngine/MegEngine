#include <utility>

#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/misc.h"

using namespace megdnn;

namespace mgb::imperative {

namespace {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<NonZero>();

    OperatorNodeConfig config{op.make_name()};
    return opr::NonZero::make(inputs[0], {}, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    mgb_assert(inputs.size() == 1, "NonZero take 1 inputs, got %lu", inputs.size());

    auto&& condition = inputs[0];
    if (condition->layout().is_empty()) {
        // empty tensor
        return {Tensor::make(
                TensorLayout{{condition->layout().ndim, 0}, dtype::Int32()},
                condition->comp_node())};
    } else {
        megdnn::NonZero::Param param;
        DnnOprCaller<megdnn::NonZero> dnn_op(condition->comp_node(), param);
        auto&& [out] = dnn_op.exec_dynout<1>(condition);
        return {out};
    }
}

std::tuple<SmallVector<LogicalTensorDesc, 1>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    LogicalTensorDesc input_0 = inputs[0];
    auto cn = inputs[0].comp_node;
    return {{{TensorLayout(dtype::Int32()), cn}}, false};
}

OP_TRAIT_REG(NonZero, NonZero, opr::NonZero)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .fallback();

}  // namespace

}  // namespace mgb::imperative
