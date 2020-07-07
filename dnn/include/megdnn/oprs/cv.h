/**
 * \file dnn/include/megdnn/oprs/cv.h
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
#include "megdnn/internal/opr_header_prologue.h"

namespace megdnn {

/**
 * \brief This file contains CV operators, The layout is NHWC
 */

class FlipBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(FlipBase, OperatorBase);
    DEF_OPR_PARAM(Flip);

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);
};

class FlipForward : public FlipBase {
    DEF_OPR_IMPL(FlipForward, FlipBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using Flip = FlipForward;

class RotateBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(RotateBase, OperatorBase);
    DEF_OPR_PARAM(Rotate);

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);
};

class RotateForward : public RotateBase {
    DEF_OPR_IMPL(RotateForward, RotateBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using Rotate = RotateForward;

class ROICopyBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(ROICopyBase, OperatorBase);
    DEF_OPR_PARAM(ROICopy);

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);
};

class ROICopyForward : public ROICopyBase {
    DEF_OPR_IMPL(ROICopyForward, ROICopyBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using ROICopy = ROICopyForward;

class CvtColorBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(CvtColorBase, OperatorBase);
    DEF_OPR_PARAM(CvtColor);

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);
};

class CvtColorForward : public CvtColorBase {
    DEF_OPR_IMPL(CvtColorForward, CvtColorBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using CvtColor = CvtColorForward;

/**
 * \brief Applices an affine transformation
 */
class WarpAffineBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(WarpAffineBase, OperatorBase);
    DEF_OPR_PARAM(WarpAffine);

public:
    using InterpolationMode = Param::InterpolationMode;
    using BorderMode = Param::BorderMode;

protected:
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& trans,
                          const TensorLayout& dst);
    std::string param_msg() const;
    int get_real_coord(int p, int len);
};

class WarpAffineForward : public WarpAffineBase {
    DEF_OPR_IMPL(WarpAffineForward, WarpAffineBase, 2, 1);

public:
    /**
     * \param[in] src input tensor
     * \param[in] trans transform matrix tensor
     * \param[in] dst output tensor
     *
     * \warning src, trans, border_value, dst should be contiguous
     * The size of trans is N * 2 * 3
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in trans,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& trans,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& trans,
                    const TensorLayout& dst, size_t workspace_in_bytes);
};
using WarpAffine = WarpAffineForward;

class GaussianBlurBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(GaussianBlurBase, OperatorBase);
    DEF_OPR_PARAM(GaussianBlur);

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);
};

class GaussianBlurForward : public GaussianBlurBase {
    DEF_OPR_IMPL(GaussianBlurForward, GaussianBlurBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using GaussianBlur = GaussianBlurForward;

/**
 * \brief Resize opr.
 */
class ResizeBase : public OperatorBase {
    DEF_OPR_PARAM(Resize);
    DEF_OPR_IMPL(ResizeBase, OperatorBase, 1, 1);

public:
    using InterpolationMode = Param::InterpolationMode;

protected:
    //! get origin coord
    std::pair<float, int> get_origin_coord(float scale, int size, int idx);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);
};

class ResizeForward : public ResizeBase {
    DEF_OPR_IMPL(ResizeForward, ResizeBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using Resize = ResizeForward;

class ResizeBackward : public ResizeBase {
    DEF_OPR_IMPL(ResizeBackward, ResizeBase, 1, 1);

public:
    virtual void exec(_megdnn_tensor_in diff, _megdnn_tensor_out grad,
                      _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorLayout& diff,
                                          const TensorLayout& mat) = 0;

protected:
    void check_exec(const TensorLayout& diff, const TensorLayout& mat,
                    size_t workspace_in_bytes);
};

/**
 * \brief Remap opr.
 */
class RemapBase : public OperatorBase {
    DEF_OPR_PARAM(Remap);
    DEF_OPR_IMPL(RemapBase, OperatorBase, 2, 1);

public:
    using InterpolationMode = Param::InterpolationMode;
    using BorderMode = Param::BorderMode;

protected:
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& map_xy,
                          const TensorLayout& dst);
    void deduce_layout_fwd(const TensorLayout& src, const TensorLayout& map_xy,
                           TensorLayout& dst);
};

class RemapForward : public RemapBase {
    DEF_OPR_IMPL(RemapForward, RemapBase, 2, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in map_xy,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;

    void deduce_layout(const TensorLayout& src, const TensorLayout& map_xy,
                       TensorLayout& dst);

    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& map_xy,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& map_xy,
                    const TensorLayout& dst, size_t workspace_in_bytes);
};
using Remap = RemapForward;

class RemapBackwardData : public RemapBase {
    DEF_OPR_IMPL(RemapBackwardData, RemapBase, 2, 1);

public:
    virtual void exec(_megdnn_tensor_in map_xy, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorLayout& map_xy,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& map_xy, const TensorLayout& diff,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class RemapBackwardMat : public RemapBase {
    DEF_OPR_IMPL(RemapBackwardMat, RemapBase, 3, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in map_xy,
                      _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                      _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& map_xy,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& map_xy,
                    const TensorLayout& diff, const TensorLayout& grad,
                    size_t workspace_in_bytes);
};

class SeparableFilterBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(SeparableFilterBase, OperatorBase);
    DEF_OPR_PARAM(SeparableFilter);

protected:
    void deduce_layout_fwd(const TensorLayout& src,
                           const TensorLayout& filter_x,
                           const TensorLayout& filter_y, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& filter_x,
                          const TensorLayout& filter_y,
                          const TensorLayout& dst);
};

class SeparableFilterForward : public SeparableFilterBase {
    DEF_OPR_IMPL(SeparableFilterForward, SeparableFilterBase, 3, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter_x,
                      _megdnn_tensor_in filter_y, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter_x,
                       const TensorLayout& filter_y, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter_x,
                                          const TensorLayout& filter_y,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& filter_x,
                    const TensorLayout& filter_y, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using SeparableFilter = SeparableFilterForward;

}  // namespace megdnn

#include "megdnn/internal/opr_header_epilogue.h"

// vim: syntax=cpp.doxygen
