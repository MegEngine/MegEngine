/**
 * \file src/opr/include/megbrain/opr/imgproc.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"

#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

/*!
 * \brief apply perspective transformation to batched 2D images
 *
 * see
 * http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
 * for details on perspective transformations.
 *
 * Input data shape: batch, channel, height, width
 * Input mat shape: batch, 3, 3; note that the mat is used to translate output
 * coordinate onto input coordinate, so it is not inversed.
 *
 * Impl note: this operator might have 3 or 4 inputs depending on whether
 * \p mat_idx is given
 */
MGB_DEFINE_OPR_CLASS(
        WarpPerspectiveForward,
        intl::WorkspaceSizeInfer<
                intl::OutshapeBySymvarSCNOpr<mixin::MegDNNOprHolderImpl<
                        megdnn::WarpPerspectiveForward>>>)  // {
public:
WarpPerspectiveForward(VarNode* in_tensor, VarNode* mat, VarNode* mat_idx,
                       VarNode* out_shape, const Param& param,
                       const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar in_tensor, SymbolVar mat, SymbolVar mat_idx,
                      SymbolVar out_shape, const Param& param = {},
                      const OperatorNodeConfig& config = {});

static SymbolVar make(SymbolVar in_tensor, SymbolVar mat, SymbolVar out_shape,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {}) {
    return make(in_tensor, mat, SymbolVar{}, out_shape, param, config);
}

static SymbolVar make(SymbolVar in_tensor, SymbolVar mat,
                      const TensorShape& out_shape, const Param& param = {},
                      const OperatorNodeConfig& config = {}) {
    return make(in_tensor, mat, cg::var_from_tensor_shape(in_tensor, out_shape),
                param, config);
}

private:
void init_output_dtype() override;
void add_input_layout_constraint() override;
void init_output_static_infer_desc() override;
void outshape_by_symvar_do_get_output_shape(
        TensorShape& dest, const ShapeInferInfo& shpinfo) override;

void scn_do_execute() override;
size_t get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const override;

void record_execute_deps(ExecDependencyArray& deps) override;
};
using WarpPerspective = WarpPerspectiveForward;

MGB_DEFINE_OPR_CLASS(
        WarpPerspectiveBackwardData,
        intl::MegDNNOprWrapperBwd<megdnn::WarpPerspectiveBackwardData>)  // {
public:
WarpPerspectiveBackwardData(VarNode* mat, VarNode* out_diff,
                            VarNode* in_for_shape, const Param& param,
                            const OperatorNodeConfig& config);

WarpPerspectiveBackwardData(VarNode* mat, VarNode* mat_idx, VarNode* out_diff,
                            VarNode* in_for_shape, const Param& param,
                            const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar mat, SymbolVar out_diff, SymbolVar in_for_shape,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {});

static SymbolVar make(SymbolVar mat, SymbolVar mat_idx, SymbolVar out_diff,
                      SymbolVar in_for_shape, const Param& param = {},
                      const OperatorNodeConfig& config = {});

void scn_do_execute() override;
};

MGB_DEFINE_OPR_CLASS(
        WarpPerspectiveBackwardMat,
        intl::MegDNNOprWrapperBwd<megdnn::WarpPerspectiveBackwardMat>)  // {
public:
WarpPerspectiveBackwardMat(VarNode* src, VarNode* mat, VarNode* mat_idx,
                           VarNode* out_diff, const Param& param,
                           const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar src, SymbolVar mat, SymbolVar out_diff,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {}) {
    return make(src, mat, {}, out_diff, param, config);
}

static SymbolVar make(SymbolVar src, SymbolVar mat, SymbolVar mat_idx,
                      SymbolVar out_diff, const Param& param = {},
                      const OperatorNodeConfig& config = {});

void scn_do_execute() override;
};

/* ============================= shape infer ============================== */
//! param: src, dst
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD1(RotateForward);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD1(CvtColorForward);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD1(GaussianBlurForward);

using Rotate = RotateForward;
using CvtColor = CvtColorForward;
using GaussianBlur = GaussianBlurForward;

/* ============================= user set shape =========================== */
MGB_DEFINE_OPR_CLASS(
        ResizeForward,
        intl::WorkspaceSizeInfer<intl::OutshapeBySymvarSCNOpr<
                mixin::MegDNNOprHolderImpl<megdnn::ResizeForward>>>)  // {
public:
ResizeForward(VarNode* in_tensor, VarNode* out_shape, const Param& param,
              const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar in_tensor, SymbolVar out_shape,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {});

static SymbolVar make(SymbolVar in_tensor, const TensorShape& out_shape,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {}) {
    return make(in_tensor, cg::var_from_tensor_shape(in_tensor, out_shape),
                param, config);
}

private:
void init_output_dtype() override;
void add_input_layout_constraint() override;
void init_output_static_infer_desc() override;
void outshape_by_symvar_do_get_output_shape(
        TensorShape& dest, const ShapeInferInfo& shpinfo) override;

void scn_do_execute() override;
size_t get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const override;
void record_execute_deps(ExecDependencyArray& deps) override;
};
using Resize = ResizeForward;

MGB_DEFINE_OPR_CLASS(ResizeBackward,
                     intl::MegDNNOprWrapperBwd<megdnn::ResizeBackward>)  // {
public:
ResizeBackward(VarNode* out_diff, VarNode* in_for_shape, const Param& param,
               const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar out_diff, SymbolVar in_for_shape,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS(RemapForward,
                     intl::MegDNNOprWrapperFwd<megdnn::RemapForward>)  // {
public:
RemapForward(VarNode* in_tensor, VarNode* map, const Param& param,
             const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar in_tensor, SymbolVar map,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {});

private:
void init_output_dtype() override;
};
using Remap = RemapForward;

MGB_DEFINE_OPR_CLASS(RemapBackwardData,
        intl::MegDNNOprWrapperBwd<megdnn::RemapBackwardData>) // {
public:
RemapBackwardData(VarNode *map, VarNode *out_diff,
        VarNode *in_for_shape, const Param &param,
        const OperatorNodeConfig &config);

static SymbolVar make(SymbolVar map, SymbolVar out_diff,
        SymbolVar in_for_shape, const Param &param = {},
        const OperatorNodeConfig &config = {});
};

MGB_DEFINE_OPR_CLASS(RemapBackwardMat,
        intl::MegDNNOprWrapperBwd<megdnn::RemapBackwardMat>) // {
public:
RemapBackwardMat(VarNode *src, VarNode *map, VarNode *out_diff,
                 const Param &param, const OperatorNodeConfig &config);

static SymbolVar make(SymbolVar src, SymbolVar map, SymbolVar out_diff,
        const Param &param = {}, const OperatorNodeConfig &config = {});
};

/*!
 * \brief apply affine transformation to batched 2D images
 *
 * see
 * http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
 * for details on affine transformations.
 *
 * Input data shape: batch, height, width, channel
 * Input mat shape: batch, 2, 2; note that the mat is used to translate output
 * coordinate onto input coordinate, so it is not inversed.
 */
MGB_DEFINE_OPR_CLASS(
        WarpAffineForward,
        intl::WorkspaceSizeInfer<intl::OutshapeBySymvarSCNOpr<
                mixin::MegDNNOprHolderImpl<megdnn::WarpAffineForward>>>)  // {
public:
WarpAffineForward(VarNode* in_tensor, VarNode* mat, VarNode* out_shape,
                  const Param& param, const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar in_tensor, SymbolVar mat, SymbolVar out_shape,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {});

static SymbolVar make(SymbolVar in_tensor, SymbolVar mat,
                      const TensorShape& out_shape, const Param& param = {},
                      const OperatorNodeConfig& config = {}) {
    return make(in_tensor, mat, cg::var_from_tensor_shape(in_tensor, out_shape),
                param, config);
}

private:
void init_output_dtype() override;
void add_input_layout_constraint() override;
void init_output_static_infer_desc() override;
void outshape_by_symvar_do_get_output_shape(
        TensorShape& dest, const ShapeInferInfo& shpinfo) override;

void scn_do_execute() override;
size_t get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const override;
void record_execute_deps(ExecDependencyArray& deps) override;
};
using WarpAffine = WarpAffineForward;

}  // opr
}  // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
