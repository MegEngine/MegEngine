/**
 * \file src/opr/impl/dnn/roi_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/roi_pooling.h"

#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/utility.h"
#include "megbrain/graph/grad_impl.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

/* ==================== ROIPoolingForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ROIPoolingForward);
ROIPoolingForward::ROIPoolingForward(VarNode *src,
        VarNode *rois, VarNode *dst_shape,
        const Param &param,
        const OperatorNodeConfig &config):
    Super{src->owner_graph(), config, "roi_pooling",
        {src, rois, dst_shape}}
{
    init_megdnn_opr(*this, param);
    mgb_assert(src->dtype() == dtype::Float32());
    add_input({src, rois, dst_shape});
    output(0)->dtype(dtype::Float32());
    output(1)->dtype(dtype::Int32());
    outshape_by_symvar_enable(2, 2);
}

SymbolVar ROIPoolingForward::make(
        SymbolVar src, SymbolVar rois, SymbolVar dst_shape,
        const Param &param, const OperatorNodeConfig &config) {
    return src.insert_single_output_opr<ROIPoolingForward>(
            src.node(), rois.node(), dst_shape.node(), param, config);
}

void ROIPoolingForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void ROIPoolingForward::outshape_by_symvar_do_get_output_shape(
        TensorShape &dest, const ShapeInferInfo &shpinfo) {
    TensorShape oshp2d;
    cg::copy_tensor_value_to_shape(oshp2d, *shpinfo.shpval_inp_val.at(0));
    auto src = shpinfo.shape_inp_shp.at(0),
         rois = shpinfo.shape_inp_shp.at(1);
    mgb_assert(src.ndim == 4 && rois.ndim == 2 && oshp2d.ndim == 2 &&
            rois.shape[1] == 5,
            "shape mismatch for ROIPooling: src=%s, rois=%s, out2d=%s",
            src.to_string().c_str(),
            rois.to_string().c_str(),
            oshp2d.to_string().c_str());
    dest.ndim = 4;
    dest.shape[0] = rois.shape[0];
    dest.shape[1] = src.shape[1];
    dest.shape[2] = oshp2d.shape[0];
    dest.shape[3] = oshp2d.shape[1];
}

void ROIPoolingForward::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();

    using namespace cg::static_infer;
    owner_graph()->static_infer_manager().register_shape_infer(output(1),
            ShapeInferDesc::make_identity(output(0)));

    init_output_static_infer_desc_workspace(false);
}

size_t ROIPoolingForward::get_workspace_size_bytes(
        const TensorShapeArray &input_shapes,
        const TensorShapeArray &output_shapes) const {
    	return mixin_get_workspace_size_bytes_by_megdnn(*this,
            input_shapes, output_shapes);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ROIPoolingForward) {
    if (wrt_idx == 2) {
        return InvalidGrad::make(opr, wrt_idx);
    }
    if (wrt_idx == 0) {
        // wrt src
        SymbolVar grad = ROIPoolingBackward::make(out_grad[0],
                opr.input(0), opr.input(1), opr.output(1), opr.param());
        return grad.node();
    } else {
        mgb_assert(wrt_idx == 1);
        return nullptr;
    }
}
#endif

void ROIPoolingForward::scn_do_execute() {
    return intl::MegDNNOprMethInvoker<megdnn::ROIPoolingForward>::
        exec(megdnn_opr(), this);
}

void ROIPooling::record_execute_deps(ExecDependencyArray &deps) {
    record_megdnn_opr(deps);
}

/* ==================== ROIPoolingBackward ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ROIPoolingBackward);

MEGDNN_OPR_INIT4(ROIPoolingBackward, "roi_pooling_backward", 1, true);

/* ==================== DeformablePSROIPoolingForward  ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(DeformablePSROIPoolingForward);
DeformablePSROIPoolingForward::DeformablePSROIPoolingForward(
        VarNode* src, VarNode* rois, VarNode* trans, const Param& param,
        const OperatorNodeConfig& config)
        : Super{src->owner_graph(),
                config,
                "deformable_ps_roi_pooling",
                {src, rois, trans}} {
    init_megdnn_opr(*this, param);
    mgb_assert(src->dtype() == dtype::Float32());
    add_input({src, rois, trans});
    output(0)->dtype(dtype::Float32());
    output(1)->dtype(dtype::Float32());
}

SymbolVarArray DeformablePSROIPoolingForward::make_all(
        SymbolVar src, SymbolVar rois, SymbolVar trans, const Param& param,
        const OperatorNodeConfig& config) {
    auto graph = src.node()->owner_graph();
    auto node =
            graph->insert_opr(std::make_unique<DeformablePSROIPoolingForward>(
                    src.node(), rois.node(), trans.node(), param, config));
    return {node->output(0), node->output(1)};
}

SymbolVar DeformablePSROIPoolingForward::make(
        SymbolVar src, SymbolVar rois, SymbolVar trans, const Param& param,
        const OperatorNodeConfig& config) {
    auto all = make_all(src, rois, trans, param, config);
    return all[0];
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(DeformablePSROIPooling) {
    mgb_assert(wrt_idx <= 2);  // wrt_idx = 0 or 1 or 2

    auto no_trans = opr.param().no_trans;
    auto back_opr = DeformablePSROIPoolingBackward::make_all(
            opr.input(0), opr.input(1), opr.input(2), out_grad[0],
            opr.output(1), opr.param(), opr.config());

    switch (wrt_idx) {
        case 0:
            //! backward src
            return back_opr[0].node();
        case 1:
            return nullptr;
        case 2:
            //! backward trans if no_trans = false
            return no_trans ? nullptr : back_opr[1].node();
        default:
            mgb_assert(false);
    }
    return nullptr;
}
#endif

/* ==================== DeformablePSROIPoolingBackward ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(DeformablePSROIPoolingBackward);

DeformablePSROIPoolingBackward::DeformablePSROIPoolingBackward(
        VarNode* src, VarNode* rois, VarNode* trans, VarNode* out_diff,
        VarNode* out_count, const Param& param,
        const OperatorNodeConfig& config)
        : Super(src->owner_graph(), config,
                "deformable_ps_roi_pooling_backward",
                {src, rois, trans, out_diff, out_count}) {
    init_megdnn_opr(*this, param);
    mgb_assert(src->dtype() == dtype::Float32());
    add_input({src, rois, trans, out_diff, out_count});
}

SymbolVarArray DeformablePSROIPoolingBackward::make_all(
        SymbolVar src, SymbolVar rois, SymbolVar trans, SymbolVar out_diff,
        SymbolVar out_count, const Param& param,
        const OperatorNodeConfig& config) {
    auto graph = src.node()->owner_graph();
    auto node =
            graph->insert_opr(std::make_unique<DeformablePSROIPoolingBackward>(
                    src.node(), rois.node(), trans.node(), out_diff.node(),
                    out_count.node(), param, config));
    return {node->output(0), node->output(1)};
}

SymbolVar DeformablePSROIPoolingBackward::make(
        SymbolVar src, SymbolVar rois, SymbolVar trans, SymbolVar out_diff,
        SymbolVar out_count, const Param& param,
        const OperatorNodeConfig& config) {
    auto graph = src.node()->owner_graph();
    auto node =
            graph->insert_opr(std::make_unique<DeformablePSROIPoolingBackward>(
                    src.node(), rois.node(), trans.node(), out_diff.node(),
                    out_count.node(), param, config));
    return node->output(0);
}

void DeformablePSROIPoolingBackward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    bool no_trans = param().no_trans;
    TensorShape src_shp = inp_shape[0];
    TensorShape rois_shp = inp_shape[1];
    TensorShape trans_shp = inp_shape[2];

    mgb_assert(src_shp.ndim == 4, "invalid src shape: %s",
               src_shp.to_string().c_str());
    mgb_assert(rois_shp.ndim == 2 and rois_shp[1] == 5,
               "invalid rois shape: %s", rois_shp.to_string().c_str());
    mgb_assert(trans_shp.ndim == 4, "invalid trans shape: %s",
               trans_shp.to_string().c_str());

    if (!no_trans) {
        size_t pool_h = param().pooled_h;
        size_t pool_w = param().pooled_w;
        mgb_assert(trans_shp[1] == 2 and trans_shp[2] == pool_h and
                           trans_shp[3] == pool_w,
                   "invalid trans shape: %s, pooled_h: %zu, pooled_w: %zu",
                   trans_shp.to_string().c_str(), pool_h, pool_w);
    }

    mgb_assert(out_shape.size() == 2);
    out_shape[0] = src_shp;
    out_shape[1] = trans_shp;
}

size_t DeformablePSROIPoolingBackward::get_workspace_size_bytes(
        const TensorShapeArray& inp_shape,
        const TensorShapeArray& out_shape) const {
    return mixin_get_workspace_size_bytes_by_megdnn(*this, inp_shape,
                                                    out_shape);
}

void DeformablePSROIPoolingBackward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<
                    megdnn::DeformablePSROIPoolingBackward>::val);
}

void DeformablePSROIPoolingBackward::init_output_dtype() {
    DType output_dtype = config().output_dtype();
    mgb_assert(!output_dtype.valid() || output_dtype == dtype::Float32());
    output_dtype = dtype::Float32();
    output(0)->dtype(output_dtype);
    output(1)->dtype(output_dtype);
}

void DeformablePSROIPoolingBackward::init_output_format() {
    mgb_assert(output().size() == 3);
    output(0)->format(input(0)->format());
    output(1)->format(input(2)->format());
}

cg::OperatorNodeBase::NodeProp*
DeformablePSROIPoolingBackward::do_make_node_prop() const {
    auto prop = Super::Super::do_make_node_prop();
    using D = NodeProp::DepType;
    mgb_assert(input().size() == 5);
    prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_VALUE, D::DEV_VALUE,
                                   D::DEV_VALUE, D::DEV_VALUE});
    return prop;
}

void DeformablePSROIPoolingBackward::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),   // src
                       input(1)->dev_tensor().as_megdnn(),   // rois
                       input(2)->dev_tensor().as_megdnn(),   // trans
                       input(3)->dev_tensor().as_megdnn(),   // out_diff
                       input(4)->dev_tensor().as_megdnn(),   // out_count
                       output(0)->dev_tensor().as_megdnn(),  // src_diff
                       output(1)->dev_tensor().as_megdnn(),  // trans_diff
                       intl::get_megdnn_workspace_from_var(output(2)));
};

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
