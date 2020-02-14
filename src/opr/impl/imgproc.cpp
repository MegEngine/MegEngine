/**
 * \file src/opr/impl/imgproc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./internal/megdnn_opr_wrapper.inl"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/utility.h"
#include "megbrain/graph/grad_impl.h"

using namespace mgb;
using namespace opr;


/* ======================= WarpPerspectiveForward ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspectiveForward);

WarpPerspectiveForward::WarpPerspectiveForward(VarNode* src, VarNode* mat,
                                               VarNode* mat_idx,
                                               VarNode* out_shape,
                                               const Param& param,
                                               const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{
                  src->owner_graph(), config, "warp_perspective", {src, mat}}) {
    init_megdnn_opr(*this, param);
    if (mat_idx) {
        add_input({src, mat, mat_idx, out_shape});
    } else {
        add_input({src, mat, out_shape});
    }
    outshape_by_symvar_enable(input().size() - 1, input().size() - 1);
}

SymbolVar WarpPerspectiveForward::make(SymbolVar i0, SymbolVar i1, SymbolVar i2,
                                       SymbolVar i3, const Param& param,
                                       const OperatorNodeConfig& config) {
    return i0.insert_single_output_opr<WarpPerspectiveForward>(
            i0.node(), i1.node(), i2.node(), i3.node(), param, config);
}

void WarpPerspectiveForward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

void WarpPerspectiveForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void WarpPerspectiveForward::outshape_by_symvar_do_get_output_shape(
        TensorShape &dest, const ShapeInferInfo &shpinfo) {

    TensorShape oshp2d;
    cg::copy_tensor_value_to_shape(oshp2d, *shpinfo.shpval_inp_val.at(0));
    auto imgshp = shpinfo.shape_inp_shp.at(0),
         matshp = shpinfo.shape_inp_shp.at(1);
    mgb_assert((imgshp.ndim == 4 || imgshp.ndim == 5) && matshp.ndim == 3 &&
                       oshp2d.ndim == 2 && matshp.shape[1] == 3 &&
                       matshp.shape[2] == 3,
               "shape mismatch for WarpPerspectiveForward: img=%s mat=%s "
               "out2d=%s",
               imgshp.to_string().c_str(), matshp.to_string().c_str(),
               oshp2d.to_string().c_str());
    if (input().size() == 3) {
        mgb_assert(imgshp[0] == matshp[0],
                   "batchsize mismatch: img=%zu mat=%zu", imgshp[0], matshp[0]);
    } else {
        mgb_assert(input().size() == 4);
        auto mat_idx_shp = shpinfo.shape_inp_shp.at(2);
        mgb_assert(mat_idx_shp[0] == matshp[0] && mat_idx_shp.ndim == 1,
                   "invalid mat_idx shape: mat=%zu mat_idx=%s", matshp[0],
                   mat_idx_shp.to_string().c_str());
    }

    //! The index of height, e.g.,[b, h, w, c], the height_idx = 1
    size_t height_idx = 0;
    if (param().format == Param::Format::NCHW ||
        param().format == Param::Format::NCHW4) {
        height_idx = 2;
    } else {
        height_idx = 1;
    }

    dest = imgshp;
    dest[0] = matshp[0];
    if (param().format == Param::Format::NHWCD4) {
        dest.shape[height_idx] = oshp2d.shape[0];
        dest.shape[height_idx + 2] = oshp2d.shape[1];
    } else {
        for (int i = 0; i < 2; ++i)
            dest.shape[height_idx + i] = oshp2d.shape[i];
    }
}

void WarpPerspectiveForward::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    init_output_static_infer_desc_workspace(false);
}

void WarpPerspectiveForward::scn_do_execute() {
    if (input().size() == 3) {
        intl::_MegDNNOprMethInvoker<2, 1>::exec(megdnn_opr(), this);
    } else {
        intl::_MegDNNOprMethInvoker<3, 1>::exec(megdnn_opr(), this);
    }
}

size_t WarpPerspectiveForward::get_workspace_size_bytes(
        const TensorShapeArray &input_shapes,
        const TensorShapeArray &output_shapes) const {
    if (input().size() == 3) {
        return intl::_MegDNNOprMethInvoker<2, 1>::get_workspace_in_bytes(
                megdnn_opr(), this, input_shapes, output_shapes);
    } else {
        return intl::_MegDNNOprMethInvoker<3, 1>::get_workspace_in_bytes(
                megdnn_opr(), this, input_shapes, output_shapes);
    }
}

void WarpPerspectiveForward::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

MGB_IMPL_OPR_GRAD(WarpPerspectiveForward) {
    mgb_assert(opr.input().size() == 3,
            "backward with mat_idx is currently unsupported");
    if (wrt_idx == 0) {
        // wrt data
        SymbolVar grad = WarpPerspectiveBackwardData::make(
                opr.input(1), out_grad[0], opr.input(0),
                opr.param());
        return grad.node();
    } else if (wrt_idx == 1){
        // wrt mat
        SymbolVar grad = WarpPerspectiveBackwardMat::make(
                opr.input(0), opr.input(1), out_grad[0],
                opr.param());
        return grad.node();
    } else
        return InvalidGrad::make(opr, wrt_idx);
}

/* ====================== WarpPerspectiveBackwardData ====================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspectiveBackwardData);
MEGDNN_OPR_INIT3(WarpPerspectiveBackwardData, "warp_perspective_bwd_data",
        2, false);

/* ====================== WarpPerspectiveBackwardMat ====================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspectiveBackwardMat);
MEGDNN_OPR_INIT3(WarpPerspectiveBackwardMat, "warp_perspective_bwd_mat",
        1, true);

/* ====================== Cv operator ====================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RotateForward);
MEGDNN_OPR_INIT1(RotateForward, "rotate")

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CvtColorForward);
MEGDNN_OPR_INIT1(CvtColorForward, "cvt_color")

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GaussianBlurForward);
MEGDNN_OPR_INIT1(GaussianBlurForward, "gaussion_blur")

/* ======================= ResizeForward ======================= */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ResizeForward);
MEGDNN_OPR_INIT2(ResizeForward, "resize")

void ResizeForward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
    outshape_by_symvar_enable(1, 1);
}

void ResizeForward::add_input_layout_constraint() {
    if (param().format != Param::Format::NCHW) {
        input(0)->add_layout_constraint_contiguous();
    }
    input(1)->add_layout_constraint_contiguous();
}

void ResizeForward::outshape_by_symvar_do_get_output_shape(
        TensorShape &dest, const ShapeInferInfo &shpinfo) {

    TensorShape oshp2d;
    cg::copy_tensor_value_to_shape(oshp2d, *shpinfo.shpval_inp_val.at(0));
    auto imgshp = shpinfo.shape_inp_shp.at(0);
    mgb_assert((imgshp.ndim == 4 || imgshp.ndim == 5) && oshp2d.ndim == 2,
               "shape mismatch for ResizeForward: img=%s out2d=%s",
               imgshp.to_string().c_str(), oshp2d.to_string().c_str());

    //! The index of height, e.g.,[b, h, w, c], the height_idx = 1
    size_t height_idx = 0;
    if (param().format == Param::Format::NCHW ||
        param().format == Param::Format::NCHW4) {
        height_idx = 2;
    } else {
        height_idx = 1;
    }

    dest = imgshp;
    if (param().format == Param::Format::NHWCD4) {
        dest.shape[height_idx] = oshp2d.shape[0];
        dest.shape[height_idx + 2] = oshp2d.shape[1];
    } else {
        for (int i = 0; i < 2; ++i)
            dest.shape[height_idx + i] = oshp2d.shape[i];
    }
}

void ResizeForward::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    init_output_static_infer_desc_workspace(false);
}

void ResizeForward::scn_do_execute() {
    intl::MegDNNOprMethInvoker<megdnn::Resize>::exec(megdnn_opr(), this);
}

size_t ResizeForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return intl::MegDNNOprMethInvoker<megdnn::Resize>::get_workspace_in_bytes(
            megdnn_opr(), this, input_shapes, output_shapes);
}

void ResizeForward::record_execute_deps(ExecDependencyArray &deps) {
    record_megdnn_opr(deps);
}

MGB_IMPL_OPR_GRAD(ResizeForward) {
    mgb_assert(opr.input().size() == 2);
    if (wrt_idx == 0) {
        SymbolVar grad =
                ResizeBackward::make(out_grad[0], opr.input(0), opr.param());
        return grad.node();
    } else
        return InvalidGrad::make(opr, wrt_idx);
}

/* ====================== ResizeBackward ====================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ResizeBackward);
MEGDNN_OPR_INIT2(ResizeBackward, "resize_bwd", 1, false);

/* ======================= WarpAffineForward ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpAffineForward);
MEGDNN_OPR_INIT3(WarpAffineForward, "warp_affine")

void WarpAffineForward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
    outshape_by_symvar_enable(2, 2);
}

void WarpAffineForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void WarpAffineForward::outshape_by_symvar_do_get_output_shape(
        TensorShape &dest, const ShapeInferInfo &shpinfo) {

    TensorShape oshp2d;
    cg::copy_tensor_value_to_shape(oshp2d, *shpinfo.shpval_inp_val.at(0));
    auto imgshp = shpinfo.shape_inp_shp.at(0),
         matshp = shpinfo.shape_inp_shp.at(1);
    mgb_assert(
            (imgshp.ndim == 4 || imgshp.ndim == 5) && matshp.ndim == 3 && oshp2d.ndim == 2 &&
            matshp.shape[0] == imgshp.shape[0] &&
            matshp.shape[1] == 2 && matshp.shape[2] == 3,
            "shape mismatch for WarpAffineForward: img=%s mat=%s out2d=%s",
            imgshp.to_string().c_str(), matshp.to_string().c_str(),
            oshp2d.to_string().c_str());

    size_t height_idx = 0;
    if (param().format == Param::Format::NCHW) {
        height_idx = 2;
    } else {
        height_idx = 1;
    }

    dest = imgshp;
    if (param().format == Param::Format::NHWCD4) {
        dest.shape[height_idx] = oshp2d.shape[0];
        dest.shape[height_idx + 2] = oshp2d.shape[1];
    } else {
        for (int i = 0; i < 2; ++i)
            dest.shape[height_idx + i] = oshp2d.shape[i];
    }
}

void WarpAffineForward::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    init_output_static_infer_desc_workspace(false);
}

void WarpAffineForward::scn_do_execute() {
    intl::MegDNNOprMethInvoker<megdnn::WarpAffine>::
        exec(megdnn_opr(), this);
}

size_t WarpAffineForward::get_workspace_size_bytes(
        const TensorShapeArray &input_shapes,
        const TensorShapeArray &output_shapes) const {
    return intl::MegDNNOprMethInvoker<megdnn::WarpAffine>::
        get_workspace_in_bytes(megdnn_opr(), this, input_shapes, output_shapes);
}

void WarpAffineForward::record_execute_deps(ExecDependencyArray &deps) {
    record_megdnn_opr(deps);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
