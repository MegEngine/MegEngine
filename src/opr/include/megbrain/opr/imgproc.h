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
MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        WarpPerspectiveForward,
        intl::WorkspaceSizeInfer<intl::OutshapeBySymvarSCNOpr<
                mixin::MegDNNOprHolderImpl<megdnn::WarpPerspectiveForward>>>) // {
public:
    WarpPerspectiveForward(
            VarNode* in_tensor, VarNode* mat, VarNode* mat_idx, VarNode* out_shape,
            const Param& param, const OperatorNodeConfig& config);

    WarpPerspectiveForward(
            const VarNodeArrayView& in_tensor, VarNode* mat, VarNode* mat_idx,
            VarNode* out_shape, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar in_tensor, SymbolVar mat, SymbolVar mat_idx, SymbolVar out_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {});

    static SymbolVar make(
            SymbolVar in_tensor, SymbolVar mat, SymbolVar out_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {}) {
        return make(in_tensor, mat, SymbolVar{}, out_shape, param, config);
    }

    static SymbolVar make(
            SymbolVar in_tensor, SymbolVar mat, const TensorShape& out_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {}) {
        return make(
                in_tensor, mat, cg::var_from_tensor_shape(in_tensor, out_shape), param,
                config);
    }

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            const VarNodeArrayView& in_tensor, SymbolVar mat, SymbolVar mat_idx,
            SymbolVar out_shape, const Param& param = {},
            OperatorNodeConfig config = {});

    static SymbolVar make(
            const VarNodeArrayView& in_tensor, SymbolVar mat, SymbolVar out_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {}) {
        return make(in_tensor, mat, SymbolVar{}, out_shape, param, config);
    }

    static SymbolVar make(
            const VarNodeArrayView& in_tensor, SymbolVar mat,
            const TensorShape& out_shape, const Param& param = {},
            const OperatorNodeConfig& config = {}) {
        return make(
                in_tensor, mat, cg::var_from_tensor_shape(in_tensor[0], out_shape),
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
    bool m_is_multi_src = false;
    size_t m_srcs_size = 0;
};
using WarpPerspective = WarpPerspectiveForward;

MGB_DEFINE_OPR_CLASS(
        WarpPerspectiveBackwardData,
        intl::MegDNNOprWrapperBwd<megdnn::WarpPerspectiveBackwardData>) // {
public:
    WarpPerspectiveBackwardData(
            VarNode* mat, VarNode* out_diff, VarNode* in_for_shape, const Param& param,
            const OperatorNodeConfig& config);

    WarpPerspectiveBackwardData(
            VarNode* mat, VarNode* mat_idx, VarNode* out_diff, VarNode* in_for_shape,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar mat, SymbolVar out_diff, SymbolVar in_for_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {});

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar mat, SymbolVar mat_idx, SymbolVar out_diff,
            SymbolVar in_for_shape, const Param& param = {},
            const OperatorNodeConfig& config = {});

    void scn_do_execute() override;
};

MGB_DEFINE_OPR_CLASS(
        WarpPerspectiveBackwardMat,
        intl::MegDNNOprWrapperBwd<megdnn::WarpPerspectiveBackwardMat>) // {
public:
    WarpPerspectiveBackwardMat(
            VarNode* src, VarNode* mat, VarNode* mat_idx, VarNode* out_diff,
            const Param& param, const OperatorNodeConfig& config);

    static SymbolVar make(
            SymbolVar src, SymbolVar mat, SymbolVar out_diff, const Param& param = {},
            const OperatorNodeConfig& config = {}) {
        return make(src, mat, {}, out_diff, param, config);
    }

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar mat, SymbolVar mat_idx, SymbolVar out_diff,
            const Param& param = {}, const OperatorNodeConfig& config = {});

    void scn_do_execute() override;
};

/* ============================= shape infer ============================== */
//! param: src, dst
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD1(RotateForward);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD1(GaussianBlurForward);

// clang-format off
MGB_DEFINE_OPR_CLASS(
        CvtColorForward, intl::MegDNNOprWrapperFwd<megdnn::CvtColorForward>) // {
public:
    CvtColorForward(VarNode * p0, const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar p0, const Param& param = {},
            const OperatorNodeConfig& config = {});
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
};
// clang-format on

using Rotate = RotateForward;
using CvtColor = CvtColorForward;
using GaussianBlur = GaussianBlurForward;

/* ============================= user set shape =========================== */
MGB_DEFINE_OPR_CLASS(
        ResizeForward, intl::WorkspaceSizeInfer<intl::OutshapeBySymvarSCNOpr<
                               mixin::MegDNNOprHolderImpl<megdnn::ResizeForward>>>) // {
public:
    ResizeForward(
            VarNode* in_tensor, VarNode* out_shape, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar in_tensor, SymbolVar out_shape, const Param& param = {},
            const OperatorNodeConfig& config = {});

    static SymbolVar make(
            SymbolVar in_tensor, const TensorShape& out_shape, const Param& param = {},
            const OperatorNodeConfig& config = {}) {
        return make(
                in_tensor, cg::var_from_tensor_shape(in_tensor, out_shape), param,
                config);
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

MGB_DEFINE_OPR_CLASS(
        ResizeBackward, intl::MegDNNOprWrapperBwd<megdnn::ResizeBackward>) // {
public:
    ResizeBackward(
            VarNode* out_diff, VarNode* in_for_shape, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar out_diff, SymbolVar in_for_shape, const Param& param = {},
            const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS(
        RemapForward, intl::MegDNNOprWrapperFwd<megdnn::RemapForward>) // {
public:
    RemapForward(
            VarNode* in_tensor, VarNode* map, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar in_tensor, SymbolVar map, const Param& param = {},
            const OperatorNodeConfig& config = {});

private:
    void init_output_dtype() override;
};
using Remap = RemapForward;

MGB_DEFINE_OPR_CLASS(
        RemapBackwardData, intl::MegDNNOprWrapperBwd<megdnn::RemapBackwardData>) // {
public:
    RemapBackwardData(
            VarNode* map, VarNode* out_diff, VarNode* in_for_shape, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar map, SymbolVar out_diff, SymbolVar in_for_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS(
        RemapBackwardMat, intl::MegDNNOprWrapperBwd<megdnn::RemapBackwardMat>) // {
public:
    RemapBackwardMat(
            VarNode* src, VarNode* map, VarNode* out_diff, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar map, SymbolVar out_diff, const Param& param = {},
            const OperatorNodeConfig& config = {});
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
                mixin::MegDNNOprHolderImpl<megdnn::WarpAffineForward>>>) // {
public:
    WarpAffineForward(
            VarNode* in_tensor, VarNode* mat, VarNode* out_shape, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar in_tensor, SymbolVar mat, SymbolVar out_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {});

    static SymbolVar make(
            SymbolVar in_tensor, SymbolVar mat, const TensorShape& out_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {}) {
        return make(
                in_tensor, mat, cg::var_from_tensor_shape(in_tensor, out_shape), param,
                config);
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

/*!
 * \brief apply DCT transformation to batched 2D images
 */
MGB_DEFINE_OPR_CLASS(
        DctChannelSelectForward,
        intl::MegDNNOprWrapperFwd<megdnn::DctChannelSelectForward>) // {
public:
    DctChannelSelectForward(
            VarNode* src, VarNode* mask_offset, VarNode* mask_val, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar mask_offset, SymbolVar mask_val,
            const Param& param, const OperatorNodeConfig& config = {});

    MGE_WIN_DECLSPEC_FUC DctChannelSelectForward(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param, const OperatorNodeConfig& config = {});
    MGE_WIN_DECLSPEC_FUC void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override;

    MGE_WIN_DECLSPEC_FUC size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    void scn_do_execute() override;

    MGE_WIN_DECLSPEC_FUC void valid_mask(
            const int* mask_offset, int mask_len, const int* mask_val, int mask_val_len,
            const Param& param);
};

using DctChannelSelect = DctChannelSelectForward;

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
