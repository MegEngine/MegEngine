#include "../dnn_op_helper.h"
#include "megbrain/imperative/ops/autogen.h"

#include "../op_trait.h"

#include "megbrain/opr/misc.h"
#include "megdnn/oprs/general.h"

namespace mgb {
namespace imperative {

namespace {
namespace where {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    mgb_assert(input_descs.size() == 3, "Where expects three inputs");
    auto comp_node = input_descs[0].comp_node;
    TensorLayout mask = input_descs[0].layout, data1 = input_descs[1].layout,
                 data2 = input_descs[2].layout;

    mgb_assert(mask.dtype == dtype::Bool(), "mask dtype must be boolean");
    mgb_assert(
            data1.dtype == dtype::Float32() || data1.dtype == dtype::Int32() ||
                    data1.dtype == dtype::Bool(),
            "data1 dtype must be float32 or int32");
    mgb_assert(
            data2.dtype == dtype::Float32() || data2.dtype == dtype::Int32() ||
                    data2.dtype == dtype::Bool(),
            "data2 dtype must be float32 or int32");

    if (!mask.ndim || !data1.ndim || !data2.ndim) {
        return {{{TensorLayout{data1.dtype}, comp_node, {}}}, false};
    }

    if (!mask.is_empty())
        mgb_assert(mask.is_contiguous(), "mask should be contiguous");
    if (!data1.is_empty())
        mgb_assert(data1.is_contiguous(), "data1 should be contiguous");
    if (!data2.is_empty())
        mgb_assert(data2.is_contiguous(), "data2 should be contiguous");

    mgb_assert(mask.eq_shape(data1), "mask shape doesn't match data1");
    mgb_assert(mask.eq_shape(data2), "mask shape doesn't match data2");
    mgb_assert(data1.eq_layout(data2), "data1 layout doesn't match data2");

    TensorLayout dst = data1;
    dst.init_contiguous_stride();
    return {{{dst, comp_node}}, true};
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Where>();
    mgb_assert(inputs.size() == 3);
    OperatorNodeConfig config{op.make_name()};
    return opr::Where::make(inputs[0], inputs[1], inputs[2], {}, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validatad) {
    auto&& mask = inputs[0];
    auto&& data1 = inputs[1];
    auto&& data2 = inputs[2];
    auto&& mask_layout = mask->layout();
    auto&& data1_layout = data1->layout();
    auto&& data2_layout = data2->layout();
    DnnOprCaller<megdnn::Where> dnn_op(mask->comp_node());
    auto tlayout = dnn_op.deduce_layout(mask_layout, data1_layout, data2_layout);
    auto out = Tensor::make(tlayout, mask->comp_node());
    if (!mask_layout.is_empty())
        dnn_op.exec_with_ws(mask, data1, data2, out);
    return {out};
}

OP_TRAIT_REG(Where, Where)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace where

namespace where_backward {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    mgb_assert(input_descs.size() == 2, "WhereBackward expects two inputs");
    auto comp_node = input_descs[0].comp_node;
    TensorLayout diff = input_descs[0].layout, mask = input_descs[1].layout;

    mgb_assert(
            diff.dtype == dtype::Float32() || diff.dtype == dtype::Int32(),
            "diff dtype must be float32 or int32");
    mgb_assert(mask.dtype == dtype::Bool(), "mask dtype must be boolean");

    if (!diff.ndim || !mask.ndim) {
        return {{{diff, comp_node}}, false};
    }

    if (!diff.is_empty())
        mgb_assert(diff.is_contiguous(), "diff should be contiguous");
    if (!mask.is_empty())
        mgb_assert(mask.is_contiguous(), "mask should be contiguous");

    mgb_assert(diff.eq_shape(mask), "diff shape doesn't match mask");

    TensorLayout data1 = diff;
    data1.init_contiguous_stride();
    TensorLayout data2 = diff;
    data2.init_contiguous_stride();
    return {{{data1, comp_node}, {data2, comp_node}}, true};
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<WhereBackward>();
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::WhereBackward::make(inputs[0], inputs[1], {}, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& diff = inputs[0];
    auto&& mask = inputs[1];
    auto&& diff_layout = diff->layout();
    auto&& mask_layout = mask->layout();
    DnnOprCaller<megdnn::WhereBackward> dnn_op(diff->comp_node());
    auto tlayouts = dnn_op.deduce_layouts<2>(diff_layout, mask_layout);
    auto grad1 = Tensor::make(tlayouts.at(0), diff->comp_node());
    auto grad2 = Tensor::make(tlayouts.at(1), diff->comp_node());
    dnn_op.exec_with_ws(diff, mask, grad1, grad2);
    return {grad1, grad2};
}

OP_TRAIT_REG(WhereBackward, WhereBackward)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace where_backward

}  // anonymous namespace
}  // namespace imperative
}  // namespace mgb