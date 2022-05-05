#include "megbrain/opr/dnn/roi_align.h"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/utility.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

/* ==================== ROIAlignForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ROIAlignForward);
ROIAlignForward::ROIAlignForward(
        VarNode* src, VarNode* rois, const Param& param,
        const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "roi_align", {src, rois}} {
    init_megdnn_opr(*this, param);
    mgb_assert(src->dtype() == dtype::Float32());
    add_input({src, rois});
    output(0)->dtype(dtype::Float32());
    output(1)->dtype(dtype::Int32());
    output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    output(1)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

SymbolVar ROIAlignForward::make(
        SymbolVar src, SymbolVar rois, const Param& param,
        const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<ROIAlignForward>(
            src.node(), rois.node(), param, config);
}

ROIAlignForward::NodeProp* ROIAlignForward::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    ret->add_dep_type_existing_var(input(1), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

void ROIAlignForward::scn_do_execute() {
    auto src = input(0)->dev_tensor().as_megdnn(),
         rois = input(1)->dev_tensor().as_megdnn(),
         dst = output(0)->dev_tensor().as_megdnn(),
         index = output(1)->dev_tensor().as_megdnn();

    if ((src.layout.is_empty() || rois.layout.is_empty())) {
        return;
    }
    megdnn_opr()->exec(
            src, rois, dst, index, intl::get_megdnn_workspace_from_var(output(2)));
}

size_t ROIAlignForward::get_workspace_size_bytes(
        const TensorShapeArray& inp_shapes, const TensorShapeArray& out_shapes) const {
    TensorLayout inp{inp_shapes[0], input(0)->dtype(), input(0)->format()},
            rois{inp_shapes[1], input(1)->dtype(), input(1)->format()},
            out{out_shapes[0], output(0)->dtype(), output(0)->format()},
            index{out_shapes[1], output(1)->dtype(), output(1)->format()};
    return megdnn_opr()->get_workspace_in_bytes(inp, rois, index, out);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ROIAlignForward) {
    if (wrt_idx == 0) {
        // wrt src
        SymbolVar grad = ROIAlignBackward::make(
                out_grad[0], opr.input(0), opr.input(1), opr.output(1), opr.param());
        return grad.node();
    } else {
        mgb_assert(wrt_idx == 1);
        return nullptr;
    }
}
#endif

/* ==================== ROIAlignBackward ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ROIAlignBackward);
MEGDNN_OPR_INIT4(ROIAlignBackward, "roi_align_backward", 1, true);

void ROIAlignBackward::scn_do_execute() {
    megdnn_opr()->exec(
            input(0)->dev_tensor().as_megdnn(), input(2)->dev_tensor().as_megdnn(),
            input(3)->dev_tensor().as_megdnn(), output(0)->dev_tensor().as_megdnn(),
            intl::get_megdnn_workspace_from_var(output(1)));
}

size_t ROIAlignBackward::get_workspace_size_bytes(
        const TensorShapeArray& inp_shapes, const TensorShapeArray& out_shapes) const {
    TensorLayout diff{inp_shapes[0], input(0)->dtype(), input(0)->format()},
            rois{inp_shapes[2], input(2)->dtype(), input(2)->format()},
            index{inp_shapes[3], input(3)->dtype(), input(3)->format()},
            grad{out_shapes[0], output(0)->dtype(), output(0)->format()};
    return megdnn_opr()->get_workspace_in_bytes(diff, rois, index, grad);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
