#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {
namespace subgraph_detail {

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated);

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs);

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad);

cg::VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs);

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad);

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs);

EncodedSubgraph make_backward_graph_from_forward(
        const EncodedSubgraph& forward, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad);

EncodedSubgraph make_from_computing_graph(
        const VarNodeArray& inputs, const VarNodeArray& outputs);

}  // namespace subgraph_detail
}  // namespace imperative
}  // namespace mgb
