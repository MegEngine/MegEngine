/**
 * \file src/opr/impl/imgproc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"

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
        TensorShape& dest, const ShapeInferInfo& shpinfo) {
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
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
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

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(WarpPerspectiveForward) {
    if (opr.input().size() == 4) {
        if (wrt_idx == 0) {
            // wrt data
            SymbolVar grad = WarpPerspectiveBackwardData::make(
                    opr.input(1), opr.input(2), out_grad[0], opr.input(0),
                    opr.param());
            return grad.node();
        } else if (wrt_idx == 1) {
            // wrt mat
            SymbolVar grad = WarpPerspectiveBackwardMat::make(
                    opr.input(0), opr.input(1), opr.input(2), out_grad[0],
                    opr.param());
            return grad.node();
        } else {
            return InvalidGrad::make(opr, wrt_idx);
        }
    }

    mgb_assert(opr.input().size() == 3);
    if (wrt_idx == 0) {
        // wrt data
        SymbolVar grad = WarpPerspectiveBackwardData::make(
                opr.input(1), out_grad[0], opr.input(0), opr.param());
        return grad.node();
    } else if (wrt_idx == 1) {
        // wrt mat
        SymbolVar grad = WarpPerspectiveBackwardMat::make(
                opr.input(0), opr.input(1), out_grad[0], opr.param());
        return grad.node();
    } else
        return InvalidGrad::make(opr, wrt_idx);
}
#endif

/* ====================== WarpPerspectiveBackwardData ====================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspectiveBackwardData);

WarpPerspectiveBackwardData::WarpPerspectiveBackwardData(
        VarNode* mat, VarNode* out_diff, VarNode* in_for_shape,
        const Param& param, const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{mat->owner_graph(),
                                          config,
                                          "warp_perspective_bwd_data",
                                          {mat}},
                2, false) {
    init_megdnn_opr(*this, param);
    add_input({mat, out_diff, in_for_shape});
    intl::MegDNNOprInitPostCtor<WarpPerspectiveBackwardData>::apply(*this);
}

WarpPerspectiveBackwardData::WarpPerspectiveBackwardData(
        VarNode* mat, VarNode* mat_idx, VarNode* out_diff,
        VarNode* in_for_shape, const Param& param,
        const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{mat->owner_graph(),
                                          config,
                                          "warp_perspective_bwd_data",
                                          {mat, mat_idx}},
                3, false) {
    init_megdnn_opr(*this, param);
    add_input({mat, mat_idx, out_diff, in_for_shape});
    intl::MegDNNOprInitPostCtor<WarpPerspectiveBackwardData>::apply(*this);
}

SymbolVar WarpPerspectiveBackwardData::make(SymbolVar i0, SymbolVar i1,
                                            SymbolVar i2, const Param& param,
                                            const OperatorNodeConfig& config) {
    intl::MegDNNOprInitInputsModifier<WarpPerspectiveBackwardData>::apply(
            param, {&i0, &i1, &i2});
    return i0.insert_single_output_opr<WarpPerspectiveBackwardData>(
            i0.node(), i1.node(), i2.node(), param, config);
}

SymbolVar WarpPerspectiveBackwardData::make(SymbolVar i0, SymbolVar i1,
                                            SymbolVar i2, SymbolVar i3,
                                            const Param& param,
                                            const OperatorNodeConfig& config) {
    intl::MegDNNOprInitInputsModifier<WarpPerspectiveBackwardData>::apply(
            param, {&i0, &i1, &i2, &i3});
    return i0.insert_single_output_opr<WarpPerspectiveBackwardData>(
            i0.node(), i1.node(), i2.node(), i3.node(), param, config);
}

void WarpPerspectiveBackwardData::scn_do_execute() {
    if (input().size() == 3) {
        megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                           input(1)->dev_tensor().as_megdnn(),
                           output(0)->dev_tensor().as_megdnn(),
                           intl::get_megdnn_workspace_from_var(output(1)));
    } else {
        mgb_assert(input().size() == 4);
        megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                           input(1)->dev_tensor().as_megdnn(),
                           input(2)->dev_tensor().as_megdnn(),
                           output(0)->dev_tensor().as_megdnn(),
                           intl::get_megdnn_workspace_from_var(output(1)));
    }
}

/* ====================== WarpPerspectiveBackwardMat ====================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WarpPerspectiveBackwardMat);

WarpPerspectiveBackwardMat::WarpPerspectiveBackwardMat(
        VarNode* src, VarNode* mat, VarNode* mat_idx, VarNode* out_diff,
        const Param& param, const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{src->owner_graph(),
                                          config,
                                          "warp_perspective_bwd_mat",
                                          {src, mat, mat_idx}},
                1, true) {
    init_megdnn_opr(*this, param);
    if (mat_idx) {
        add_input({src, mat, mat_idx, out_diff});
    } else {
        add_input({src, mat, out_diff});
    }
    intl::MegDNNOprInitPostCtor<WarpPerspectiveBackwardMat>::apply(*this);
}

void WarpPerspectiveBackwardMat::scn_do_execute() {
    if (input().size() == 3) {
        megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                           input(1)->dev_tensor().as_megdnn(),
                           input(2)->dev_tensor().as_megdnn(),
                           output(0)->dev_tensor().as_megdnn(),
                           intl::get_megdnn_workspace_from_var(output(1)));
    } else {
        mgb_assert(input().size() == 4);
        megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                           input(1)->dev_tensor().as_megdnn(),
                           input(2)->dev_tensor().as_megdnn(),
                           input(3)->dev_tensor().as_megdnn(),
                           output(0)->dev_tensor().as_megdnn(),
                           intl::get_megdnn_workspace_from_var(output(1)));
    }
}

SymbolVar WarpPerspectiveBackwardMat::make(SymbolVar i0, SymbolVar i1,
                                           SymbolVar i2, SymbolVar i3,
                                           const Param& param,
                                           const OperatorNodeConfig& config) {
    intl::MegDNNOprInitInputsModifier<WarpPerspectiveBackwardMat>::apply(
            param, {&i0, &i1, &i2, &i3});
    return i0.insert_single_output_opr<WarpPerspectiveBackwardMat>(
            i0.node(), i1.node(), i2.node(), i3.node(), param, config);
}

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
        TensorShape& dest, const ShapeInferInfo& shpinfo) {
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

void ResizeForward::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ResizeForward) {
    mgb_assert(opr.input().size() == 2);
    if (wrt_idx == 0) {
        SymbolVar grad =
                ResizeBackward::make(out_grad[0], opr.input(0), opr.param());
        return grad.node();
    } else
        return InvalidGrad::make(opr, wrt_idx);
}
#endif

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
        TensorShape& dest, const ShapeInferInfo& shpinfo) {
    TensorShape oshp2d;
    cg::copy_tensor_value_to_shape(oshp2d, *shpinfo.shpval_inp_val.at(0));
    auto imgshp = shpinfo.shape_inp_shp.at(0),
         matshp = shpinfo.shape_inp_shp.at(1);
    mgb_assert((imgshp.ndim == 4 || imgshp.ndim == 5) && matshp.ndim == 3 &&
                       oshp2d.ndim == 2 && matshp.shape[0] == imgshp.shape[0] &&
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
    intl::MegDNNOprMethInvoker<megdnn::WarpAffine>::exec(megdnn_opr(), this);
}

size_t WarpAffineForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return intl::MegDNNOprMethInvoker<
            megdnn::WarpAffine>::get_workspace_in_bytes(megdnn_opr(), this,
                                                        input_shapes,
                                                        output_shapes);
}

void WarpAffineForward::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

/* ======================= RemapForward ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemapForward);
MEGDNN_OPR_INIT2(RemapForward, "remap")

void RemapForward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(RemapForward) {
    mgb_assert(opr.input().size() == 2);
    if (wrt_idx == 0) {
        SymbolVar grad = RemapBackwardData::make(opr.input(1), out_grad[0],
                                                 opr.input(0), opr.param());
        return grad.node();
    } else if (wrt_idx == 1) {
        SymbolVar grad = RemapBackwardMat::make(opr.input(0), opr.input(1),
                                                out_grad[0], opr.param());
        return grad.node();
    } else
        return InvalidGrad::make(opr, wrt_idx);
}
#endif

/* ====================== RemapBackward ====================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemapBackwardData);
MEGDNN_OPR_INIT3(RemapBackwardData, "remap_bwd_data", 2, false);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemapBackwardMat);
MEGDNN_OPR_INIT3(RemapBackwardMat, "remap_bwd_mat", 1, true);

/* ======================= DctChannelSelectForward ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(DctChannelSelectForward);
namespace mgb {
namespace opr {
namespace intl {
template <>
struct MegDNNOprInitPostCtor<DctChannelSelectForward> {
    static void apply(cg::OperatorNodeBase& opr) {
        if (opr.config().output_dtype().valid()) {
            opr.output(0)->dtype(opr.config().output_dtype());
        } else {
            opr.output(0)->dtype(dtype::Float32());
        }
    }
};
}  // namespace intl
}  // namespace opr
}  // namespace mgb

void DctChannelSelectForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    auto mo = megdnn_opr();
    TensorLayout dst;
    dst.dtype = output(0)->dtype();
    if (inp_shape.size() == 1) {
        mo->deduce_layout({inp_shape[0], input(0)->dtype(), input(0)->format()},
                          {}, {}, dst);
    } else {
        mgb_assert(inp_shape.size() == 3, "no support input tensor num %zu",
                   inp_shape.size());
        mo->deduce_layout({inp_shape[0], input(0)->dtype(), input(0)->format()},
                          {inp_shape[1], input(1)->dtype(), input(1)->format()},
                          {inp_shape[2], input(2)->dtype(), input(2)->format()},
                          dst);
    }
    out_shape[0] = dst;
}

size_t DctChannelSelectForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    auto mo = megdnn_opr();

    return mo->get_workspace_in_bytes(
            {input_shapes[0], input(0)->dtype(), input(0)->format()}, {}, {},
            {output_shapes[0], output(0)->dtype(), output(0)->format()});
}

void DctChannelSelectForward::scn_do_execute() {
    auto&& inp = input();
    auto mo = megdnn_opr();
    if (inp.size() == 1) {
        mo->exec(inp[0]->dev_tensor().as_megdnn(), {}, {},
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));

    } else {
        mgb_assert(inp.size() == 3, "no support input tensor num %zu",
                   inp.size());
        mo->exec(inp[0]->dev_tensor().as_megdnn(),
                 inp[1]->dev_tensor().as_megdnn(),
                 inp[2]->dev_tensor().as_megdnn(),
                 output(0)->dev_tensor().as_megdnn(),
                 intl::get_megdnn_workspace_from_var(output().back()));
    }
}

void DctChannelSelectForward::valid_mask(const int* mask_offset, int mask_len,
                                         const int* mask_val, int mask_val_len,
                                         const Param& param) {
    if (mask_len <= 0)
        return;
    mgb_assert(mask_offset[0] == 0,
               "The first element of mask_offset must be zero, but got %d. For "
               "example mask offset [0, 15, 20] indicate there are 2 ic, and "
               "ic_0 will have (15 - 0) oc, ic_1 have (20 - 15) oc",
               mask_offset[0]);
    for (int i = 1; i < mask_len; ++i) {
        if (param.format == Param::Format::NCHW4) {
            mgb_assert(mask_offset[i] % 4 == 0,
                       "Invalid mask offset %d at %d, it should be times of "
                       "4 when using nchw4 format",
                       mask_offset[i], i);
        }
        mgb_assert(mask_offset[i] >= mask_offset[i - 1],
                   "The offset of mask must be increasing, but %d(%d) is less "
                   "than %d(%d)",
                   mask_offset[i], i, mask_offset[i - 1], i - 1);
    }
    const int max_mask = param.dct_block_size * param.dct_block_size;
    for (int i = 0; i < mask_val_len; ++i) {
        mgb_assert(0 <= mask_val[i] && mask_val[i] < max_mask,
                   "Invalid mask_val, assert 0 <= mask_val[%d] < %d, aka 0 <= "
                   "%d < %d",
                   i, max_mask, mask_val[i], max_mask);
    }
}

DctChannelSelectForward::DctChannelSelectForward(
        VarNode* src, VarNode* mask_offset, VarNode* mask_val,
        const Param& param, const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{
                  src->owner_graph(), config, "dct_channel_select", {src}}) {
    init_megdnn_opr(*this, param);
    add_input({src, mask_offset, mask_val});
    if (mask_offset != nullptr) {
        mgb_assert(mask_val,
                   "mask_val should not be null when mask_offset is not null");
        auto host_offset = mask_offset->owner_opr()
                                   ->cast_final_safe<opr::ImmutableTensor>()
                                   .host_value();
        auto host_val = mask_val->owner_opr()
                                ->cast_final_safe<opr::ImmutableTensor>()
                                .host_value();

        valid_mask(host_offset.ptr<int>(),
                   host_offset.layout().total_nr_elems(), host_val.ptr<int>(),
                   host_val.layout().total_nr_elems(), param);
    }
    intl::MegDNNOprInitPostCtor<DctChannelSelectForward>::apply(*this);
}

SymbolVar DctChannelSelectForward::make(SymbolVar src, SymbolVar mask_offset,
                                        SymbolVar mask_val, const Param& param,
                                        const OperatorNodeConfig& config) {
    intl::MegDNNOprInitInputsModifier<DctChannelSelectForward>::apply(
            param, {&src, &mask_offset, &mask_val});
    return src.insert_single_output_opr<DctChannelSelectForward>(
            src.node(), mask_offset.node(), mask_val.node(), param, config);
}

MEGDNN_OPR_INIT1(DctChannelSelectForward, "dct_channel_select")

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
