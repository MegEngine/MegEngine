#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/dnn/roi_align.h"
#include "megbrain/opr/dnn/roi_pooling.h"
#include "megbrain/opr/imgproc.h"

#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"
namespace mgb {
namespace imperative {

namespace {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const CvtColor&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::CvtColor::make(inputs[0], op.param(), config);
}
OP_TRAIT_REG(CvtColor, CvtColor).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace

namespace {
namespace flip {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Flip&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::Flip::make(inputs[0], op.param(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<Flip>();
    DnnOprHelper<megdnn::Flip> dnn_opr(op.param());
    auto cn = inputs[0].comp_node;
    auto out_layout = dnn_opr.deduce_layout(inputs[0].layout);
    bool validated = out_layout.ndim == 0;
    return {{{out_layout, cn}}, validated};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<Flip>();
    auto cn = inputs[0]->comp_node();

    DnnOprCaller<megdnn::Flip> dnn_opr(cn, op.param());
    auto&& out_layout = [&]() -> TensorLayout {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_opr.deduce_layout(inputs[0]->layout());
        }
    }();

    auto out = Tensor::make(out_layout, cn);

    dnn_opr.exec_with_ws(inputs[0], out);
    return {out};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(Flip, Flip)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace flip
}  // namespace

namespace {
namespace rotate {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Rotate&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::Rotate::make(inputs[0], op.param(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<Rotate>();
    DnnOprHelper<megdnn::Rotate> dnn_opr(op.param());
    auto cn = inputs[0].comp_node;
    auto out_layout = dnn_opr.deduce_layout(inputs[0].layout);
    bool validated = out_layout.ndim == 0;
    return {{{out_layout, cn}}, validated};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<Rotate>();
    auto cn = inputs[0]->comp_node();

    DnnOprCaller<megdnn::Rotate> dnn_opr(cn, op.param());
    auto&& out_layout = [&]() -> TensorLayout {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_opr.deduce_layout(inputs[0]->layout());
        }
    }();

    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_with_ws(inputs[0], out);
    return {out};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(Rotate, Rotate)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace rotate
}  // namespace

namespace {
namespace gaussianblur {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const GaussianBlur&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::GaussianBlur::make(inputs[0], op.param(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<GaussianBlur>();
    DnnOprHelper<megdnn::GaussianBlur> dnn_opr(op.param());
    auto cn = inputs[0].comp_node;
    auto out_layout = dnn_opr.deduce_layout(inputs[0].layout);
    bool validated = out_layout.ndim == 0;
    return {{{out_layout, cn}}, validated};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<GaussianBlur>();
    auto cn = inputs[0]->comp_node();

    DnnOprCaller<megdnn::GaussianBlur> dnn_opr(cn, op.param());
    auto&& out_layout = [&]() -> TensorLayout {
        if (validated) {
            return output_descs[0].layout;
        } else {
            return dnn_opr.deduce_layout(inputs[0]->layout());
        }
    }();

    auto out = Tensor::make(out_layout, cn);
    dnn_opr.exec_with_ws(inputs[0], out);
    return {out};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(GaussianBlur, GaussianBlur)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();

}  // namespace gaussianblur
}  // namespace

namespace {
namespace roi_align {
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const ROIAlign&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    auto* opr = opr::ROIAlign::make(inputs[0], inputs[1], op.param(), config)
                        .node()
                        ->owner_opr();
    return {opr->output(0), opr->output(1)};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<ROIAlign>();
    DnnOprHelper<megdnn::ROIAlign> dnn_opr(op.param());
    auto cn = inputs[0].comp_node;
    auto&& [out_layout, ind_layout] =
            dnn_opr.deduce_layouts<2>(inputs[0].layout, inputs[1].layout);
    bool validated = out_layout.ndim == 0 && ind_layout.ndim == 0;
    return {{{out_layout, cn}, {ind_layout, cn}}, validated};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<ROIAlign>();
    auto cn = inputs[0]->comp_node();

    DnnOprCaller<megdnn::ROIAlign> dnn_opr(cn, op.param());
    auto&& [out_layout, ind_layout] = [&]() -> std::array<TensorLayout, 2> {
        if (validated) {
            return {output_descs[0].layout, output_descs[1].layout};
        } else {
            return dnn_opr.deduce_layouts<2>(inputs[0]->layout(), inputs[1]->layout());
        }
    }();

    auto out = Tensor::make(out_layout, cn);
    auto ind = Tensor::make(ind_layout, cn);

    if (out_layout.is_empty() || ind_layout.is_empty()) {
        return {out, ind};
    }

    dnn_opr.exec_with_ws(inputs[0], inputs[1], out, ind);
    return {out, ind};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = layout_checker[1] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(ROIAlign, ROIAlign)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace roi_align
}  // namespace

namespace {
namespace roi_pooling {
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const ROIPooling&>(def);
    mgb_assert(inputs.size() == 3);
    OperatorNodeConfig config{op.make_name()};
    auto* opr =
            opr::ROIPooling::make(inputs[0], inputs[1], inputs[2], op.param(), config)
                    .node()
                    ->owner_opr();
    return {opr->output(0), opr->output(1)};
}
OP_TRAIT_REG(ROIPooling, ROIPooling).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace roi_pooling
}  // namespace

}  // namespace imperative
}  // namespace mgb
