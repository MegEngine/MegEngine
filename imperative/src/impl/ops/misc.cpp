#include "../dnn_op_helper.h"
#include "../op_trait.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/misc.h"

namespace mgb {
namespace imperative {

namespace check_non_finite {
SymbolVarArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<CheckNonFinite>();
    OperatorNodeConfig config{op.make_name()};
    return opr::CheckNonFinite::make(inputs, op.param(), config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<CheckNonFinite>();
    auto comp_node = inputs[0]->comp_node();
    auto dest = Tensor::make(TensorLayout({1}, dtype::Int32()), comp_node);
    SmallVector<TensorPtr> outputs;
    outputs.reserve(inputs.size() + 1);
    for (auto&& input : inputs) {
        outputs.push_back(Tensor::make(input->layout(), comp_node));
        outputs.back()->dev_tensor().copy_from_fixlayout(input->dev_tensor());
    }
    DnnOprCaller<megdnn::CheckNonFinite> dnn_opr(comp_node, {op.scale});
    dnn_opr.exec_with_ws(outputs, dest);
    outputs.push_back(dest);
    return outputs;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    size_t size = inputs.size();
    SmallVector<LogicalTensorDesc> dests(size + 1);
    bool validated = true;
    for (size_t i = 0; i < size; ++i) {
        dests[i].comp_node = inputs[i].comp_node;
        dests[i].layout = inputs[i].layout;
        validated &= bool(dests[i].layout.ndim);
    }
    dests[size].comp_node = inputs[0].comp_node;
    dests[size].layout = TensorLayout({1}, dtype::Int32());
    return {dests, validated};
}

OP_TRAIT_REG(CheckNonFinite, CheckNonFinite)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .fallback();
}  // namespace check_non_finite

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
