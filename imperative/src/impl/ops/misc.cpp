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
    size_t size = inputs.size();
    auto&& op = def.cast_final_safe<CheckNonFinite>();
    SmallVector<TensorPtr> outputs(size + 1);
    outputs[size] = Tensor::make(
            TensorLayout(TensorShape({1}), dtype::Int32()), inputs[0]->comp_node());

    auto dest = outputs[size];
    auto cn = dest->comp_node();
    DnnOprCaller<megdnn::CheckNonFinite> dnn_opr(cn);
    SmallVector<megdnn::TensorND> srcs(size);
    // copy an outputs to the dnn for inplace
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = Tensor::make(inputs[i]->layout(), inputs[0]->comp_node());
        outputs[i]->dev_tensor().copy_from_fixlayout(inputs[i]->dev_tensor());
        srcs[i] = outputs[i]->dev_tensor().as_megdnn();
    }
    megdnn::CheckNonFinite::Param param({op.scale});
    dnn_opr.op->param() = param;
    size_t sz = dnn_opr.op->get_workspace_in_bytes(srcs, dest->layout());
    TensorLayout w_layout({sz}, dtype::Byte());
    auto dnn_wk = dnn_opr.create_workspace(w_layout);
    dnn_opr.op->exec(srcs, dest->dev_tensor().as_megdnn(), dnn_wk);
    return outputs;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    size_t size = inputs.size();
    SmallVector<LogicalTensorDesc> dests(size + 1);
    for (size_t i = 0; i < size; ++i) {
        dests[i].comp_node = inputs[i].comp_node;
        dests[i].layout = inputs[i].layout;
    }
    dests[size].comp_node = inputs[0].comp_node;
    dests[size].layout = TensorLayout(TensorShape({1}), dtype::Int32());
    return {dests, true};
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
