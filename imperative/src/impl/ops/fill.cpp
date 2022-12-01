#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/graph/helper.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"

namespace mgb {
namespace imperative {

namespace fill {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Fill&>(def);
    mgb_assert(inputs.size() == 1);
    auto comp_node = inputs[0]->comp_node();
    auto name = op.make_name();
    DTypeScalar scalar(op.dtype);
    scalar.set_retain_dtype(op.value);
    auto graph = inputs[0]->owner_graph();
    auto scalar_shape = opr::ImmutableTensor::make(*graph, scalar, {name, comp_node});
    return opr::Broadcast::make(scalar_shape, inputs[0], {name});
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<Fill>();
    auto&& tshp = inputs[0];
    auto comp_node = inputs[0].comp_node;
    if (tshp.layout.ndim == 0 || tshp.value.empty()) {
        return {{{TensorLayout(op.dtype), comp_node}}, false};
    }
    TensorShape out_shape;
    out_shape.ndim = tshp.layout.shape[0];
    auto* ptr = tshp.value.ptr<dt_int32>();
    for (size_t i = 0; i < out_shape.ndim; ++i) {
        out_shape[i] = ptr[i];
    }
    return {{{TensorLayout(out_shape, op.dtype), comp_node}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<Fill>();
    auto comp_node = inputs[0]->comp_node();
    TensorShape tshp;
    cg::copy_tensor_value_to_shape(tshp, inputs[0]->get_value().proxy_to_default_cpu());
    TensorLayout oup_layout = TensorLayout{tshp, op.dtype};
    auto output = Tensor::make(oup_layout, comp_node);

    if (oup_layout.total_nr_elems() != 0) {  // empty tensor like Tensor([])
        DnnOprCaller<megdnn::Fill> caller(comp_node, megdnn::Fill::Param{op.value});
        caller.exec_with_ws(output);
    }
    return {output};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(Fill, Fill)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace fill

namespace fill_like {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const FillLike&>(def);
    mgb_assert(inputs.size() == 1);
    auto comp_node = inputs[0]->comp_node();
    megdnn::DType oup_dtype = inputs[0]->dtype();
    auto name = op.make_name();
    DTypeScalar scalar(oup_dtype);
    scalar.set_retain_dtype(op.value);
    auto graph = inputs[0]->owner_graph();
    auto scalar_shape = opr::ImmutableTensor::make(*graph, scalar, {name, comp_node});
    return opr::Broadcast::make(
            scalar_shape, opr::GetVarShape::make(inputs[0]), {name});
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(inputs.size() == 1);
    auto&& inp = inputs[0];
    if (inp.layout.ndim == 0) {
        return {{{TensorLayout{inp.layout.dtype}, inp.comp_node}}, false};
    }
    return {{{TensorLayout(inp.layout), inp.comp_node}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    mgb_assert(inputs.size() == 1);
    auto&& op = def.cast_final_safe<FillLike>();
    auto&& inp = inputs[0];

    TensorLayout oup_layout = inp->layout();
    CompNode oup_cn = inp->comp_node();

    auto output = Tensor::make(oup_layout, oup_cn);
    if (oup_layout.total_nr_elems() != 0) {  // empty tensor like Tensor([])
        DnnOprCaller<megdnn::Fill> caller(oup_cn, megdnn::Fill::Param{op.value});
        caller.exec_with_ws(output);
    }
    return {output};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(FillLike, FillLike)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();

}  // namespace fill_like

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}