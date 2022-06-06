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
    auto&& op = static_cast<const ROIAlign&>(def);
    if (inputs[0].layout.is_empty() || inputs[1].layout.is_empty()) {
        return {{{TensorLayout(inputs[0].layout.dtype), inputs[0].comp_node},
                 {TensorLayout(dtype::Int32()), inputs[1].comp_node}},
                false};
    }

    SmallVector<LogicalTensorDesc> descs(2u);
    size_t n = inputs[1].layout[0];
    size_t c = inputs[0].layout[1];
    descs[0].layout = TensorLayout(
            {n, c, op.pooled_height, op.pooled_width}, inputs[0].layout.dtype);
    descs[0].layout.init_contiguous_stride();
    descs[0].comp_node = inputs[0].comp_node;

    descs[1].layout =
            TensorLayout({n, c, op.pooled_height, op.pooled_width}, dtype::Int32());
    descs[1].layout.init_contiguous_stride();
    descs[1].comp_node = descs[0].comp_node;

    return {descs, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = static_cast<const ROIAlign&>(def);
    CompNode cn = inputs[0]->comp_node();

    TensorLayout out_layout = output_descs[0].layout;
    TensorLayout ind_layout = output_descs[1].layout;
    if (!validated) {
        size_t n = inputs[1]->layout()[0];
        size_t c = inputs[0]->layout()[1];
        out_layout = TensorLayout(
                {n, c, op.pooled_height, op.pooled_width}, inputs[0]->layout().dtype);
        out_layout.init_contiguous_stride();
        ind_layout =
                TensorLayout({n, c, op.pooled_height, op.pooled_width}, dtype::Int32());
        ind_layout.init_contiguous_stride();
    }

    DeviceTensorND out =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, out_layout);
    DeviceTensorND inds =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, ind_layout);

    if (out_layout.is_empty() || ind_layout.is_empty()) {
        return {Tensor::make(out), Tensor::make(inds)};
    }

    DnnOprCaller<megdnn::ROIAlign> dnn_opr(cn);
    dnn_opr.op->param() = op.param();

    size_t sz = dnn_opr.op->get_workspace_in_bytes(
            inputs[0]->layout(), inputs[1]->layout(), out_layout, ind_layout);

    auto dnn_wk = dnn_opr.create_workspace(sz);

    dnn_opr.op->exec(
            inputs[0]->dnn_tensor(), inputs[1]->dnn_tensor(), out.as_megdnn(),
            inds.as_megdnn(), dnn_wk);
    return {Tensor::make(out), Tensor::make(inds)};
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
