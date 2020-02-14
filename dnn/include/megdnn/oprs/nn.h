/**
 * \file dnn/include/megdnn/oprs/nn.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/internal/opr_header_prologue.h"

namespace megdnn {

class SeparableConvBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(SeparableConvBase, OperatorBase);
    DEF_OPR_PARAM(SeparableConv);

public:
    using Mode = Param::Mode;

protected:
    void deduce_layout_fwd(const TensorLayout& src,
                           const TensorLayout& filter_x,
                           const TensorLayout& filter_y, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& filter_x,
                          const TensorLayout& filter_y,
                          const TensorLayout& dst);
};

class SeparableConvForward : public SeparableConvBase {
    DEF_OPR_IMPL(SeparableConvForward, SeparableConvBase, 3, 1);

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
using SeparableConv = SeparableConvForward;

/**
 * \brief base class for convolution operation
 *
 * This operator is supposed to perform convolution on arbitrary input
 * dimensions. The input/output format is N, C, dims..., and kernel format can
 * take two forms:
 *  1. OC, IC, dims..., for conventional dense convolution
 *  2. GROUP, OC_PER_GRP, IC_PER_GRP, dims... for sparse group convolution
 *
 * Currently, only 2D images are supported.
 */
template <typename Parameter>
class ConvolutionBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(ConvolutionBase, OperatorBase);
    using Param = Parameter;

public:
    Param& param() { return m_param; }
    const Param& param() const { return m_param; }

protected:
    Param m_param;

public:
    static constexpr size_t MAX_SPATIAL_DIM = 2;
    using Mode = typename Param::Mode;
    struct CanonizedFilterMeta {
        DType dtype;
        typename Param::Format format;

        uint32_t
                //! whether filter should be flipped (i.e. is CONVOLUTION)
                should_flip,
                group,  //!< number of groups
                icpg,   //!< input channels per group
                ocpg,   //!< output channels per group
                spatial_ndim, stride[MAX_SPATIAL_DIM], padding[MAX_SPATIAL_DIM],
                //! spatial dim
                spatial[MAX_SPATIAL_DIM], dilation[MAX_SPATIAL_DIM],
                //! spatial dim with dilation applied
                dilated_spatial[MAX_SPATIAL_DIM];

        //! T should be a ConvolutionBase<Z>::CanonizedFilterMeta
        template <typename T>
        void copy_from(const T& b) {
            dtype = b.dtype;
            format = b.format;
            should_flip = b.should_flip;
            group = b.group;
            icpg = b.icpg;
            ocpg = b.ocpg;
            spatial_ndim = b.spatial_ndim;
            memcpy(stride, b.stride, sizeof(stride));
            memcpy(padding, b.padding, sizeof(padding));
            memcpy(spatial, b.spatial, sizeof(spatial));
            memcpy(dilation, b.dilation, sizeof(dilation));
            memcpy(dilated_spatial, b.dilated_spatial, sizeof(dilated_spatial));
        }

        bool operator==(const CanonizedFilterMeta& b) const {
            bool flag = true;
            flag = flag && (format == b.format);
            flag = flag && (dtype == b.dtype);
            flag = flag && (should_flip == b.should_flip);
            flag = flag && (group == b.group);
            flag = flag && (icpg == b.icpg);
            flag = flag && (ocpg == b.ocpg);
            flag = flag && (spatial_ndim == b.spatial_ndim);
            if (flag) {
                for (uint32_t i = 0; i < spatial_ndim; ++i) {
                    flag = flag && (stride[i] == b.stride[i]);
                    flag = flag && (padding[i] == b.padding[i]);
                    flag = flag && (spatial[i] == b.spatial[i]);
                    flag = flag && (dilation[i] == b.dilation[i]);
                    flag = flag && (dilated_spatial[i] == b.dilated_spatial[i]);
                }
            }
            return flag;
        }
    };

protected:
    // Check or deduce output DType
    void check_or_deduce_dtype_fwd(DType src, DType filter, DType& dst) const;
    CanonizedFilterMeta deduce_layout_fwd(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          TensorLayout& dst) const;
    CanonizedFilterMeta check_layout_fwd(const TensorLayout& src,
                                         const TensorLayout& filter,
                                         const TensorLayout& dst) const;

    CanonizedFilterMeta make_canonized_filter_meta(
            size_t src_ndim, const TensorLayout& filter) const;
};

class MaskPropagate : public OperatorBase {
    DEF_OPR_IMPL(MaskPropagate, OperatorBase, 1, 1);
    DEF_OPR_PARAM(MaskPropagate);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
};

/**
 * \brief ConvolutionForward Operator with 0/1 Mask matrix
 */
class MaskConvForward : public ConvolutionBase<param::Convolution> {
    DEF_OPR_IMPL(MaskConvForward, ConvolutionBase, 3, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_in mask, _megdnn_tensor_out dst,
                      _megdnn_workspace worksapce) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& mask,
                                          const TensorLayout& dst) = 0;

    void deduce_dtype(DType src, DType filter, DType mask, DType& dst);
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       const TensorLayout& mask, TensorLayout& dst);

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& src,
                                   const TensorLayout& filter,
                                   const TensorLayout& mask,
                                   const TensorLayout& dst,
                                   size_t workspace_in_bytes);
};
using MaskConvolution = MaskConvForward;

/**
 * \brief ConvolutionForward operator.
 */
class ConvolutionForward : public ConvolutionBase<param::Convolution>,
                           public detail::MultiAlgoOpr<ConvolutionForward, 3> {
    DEF_OPR_IMPL(ConvolutionForward, ConvolutionBase, 2, 1);

public:
    /**
     * \param[in] src (n, ic, ih, iw)
     * \param[in] filter (oc, ic, fh, fw)
     * \param[out] dst (n, oc, oh, ow)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_dtype(DType src, DType filter, DType& dst);
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& dst) = 0;

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& src,
                                   const TensorLayout& filter,
                                   const TensorLayout& dst,
                                   size_t workspace_in_bytes);
};
using Convolution = ConvolutionForward;

/**
 * \brief ConvolutionBackwardData operator.
 *
 * Calculating the gradient wrt. convolution input data.
 */
class ConvolutionBackwardData
        : public ConvolutionBase<param::Convolution>,
          public detail::MultiAlgoOpr<ConvolutionBackwardData, 3> {
    DEF_OPR_IMPL(ConvolutionBackwardData, ConvolutionBase, 2, 1);

public:
    /**
     * \param[in] filter (oc, ic, fh, fw)
     * \param[in] diff (n, oc, oh, ow)
     * \param[out] grad (n, ic, ih, iw)
     */
    virtual void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& filter,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

    void deduce_dtype(DType filter, DType diff, DType& grad);
    void deduce_layout(const TensorLayout& filter, const TensorLayout& diff,
                       TensorLayout& grad);

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& filter,
                                   const TensorLayout& diff,
                                   const TensorLayout& grad,
                                   size_t workspace_in_bytes);
};

/**
 * \brief ConvolutionBackwardFilter operator.
 *
 * Calculating the gradient wrt. convolution filter.
 */
class ConvolutionBackwardFilter
        : public ConvolutionBase<param::Convolution>,
          public detail::MultiAlgoOpr<ConvolutionBackwardFilter, 3> {
    DEF_OPR_IMPL(ConvolutionBackwardFilter, ConvolutionBase, 2, 1);

public:
    /**
     * \param[in] src (n, ic, ih, iw)
     * \param[in] diff (n, oc, oh, ow)
     * \param[out] grad (oc, ic, fh, fw)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& src,
                                   const TensorLayout& diff,
                                   const TensorLayout& grad,
                                   size_t workspace_in_bytes);
};

/**
 * \brief ConvolutionBias operator
 */
class ConvBiasForward : public ConvolutionBase<param::ConvBias>,
                        public detail::MultiAlgoOpr<ConvBiasForward, 5> {
    DEF_OPR_IMPL(ConvBiasForward, ConvolutionBase, 4, 1);

public:
    /**
     * \param[in] src (n, ic, ih, iw) or (n, ih, iw, ic)
     * \param[in] filter (oc, ic, fh, fw) or (oc, fh, fw, ic) or (oc/4, fh, fw,
     * 4*ic) \param[in] bias (1, oc, 1, 1) \param[in] z same as dst \param[out]
     * dst (n, oc, oh, ow) or (n, oh, ow, oc)
     *
     * \note if the format is NCHW_WINOGRAD, the filter layout is (alphah,
     * alphaw, oc, ic)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_in bias, _megdnn_tensor_in z,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_dtype(DType src, DType filter, DType bias, DType z, DType& dst);
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       const TensorLayout& bias, const TensorLayout& z,
                       TensorLayout& dst);

    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& bias,
                                          const TensorLayout& z,
                                          const TensorLayout& dst) = 0;
    enum class BiasMode : uint32_t {
        NO_BIAS = 0,             //!< no bias
        BROADCAST_CHANNEL_BIAS,  //!< broadcast channel bias, [1, c, 1, 1]
        BIAS                     //!< [N, C, H, W]
    };

    //! param for winograd algos.
    struct WinogradParam {
        uint32_t channel_block_size;
        uint32_t output_block_size;
        uint32_t tile_size;
        bool operator==(const WinogradParam& rhs) const {
            return channel_block_size == rhs.channel_block_size &&
                   output_block_size == rhs.output_block_size &&
                   tile_size == rhs.tile_size;
        }

        std::string to_string() const;
    };
    static constexpr WinogradParam INVALID_WINOGRAD_PARAM = {0, 0, 0};

    struct DirectParam {
        std::string to_string() const { return ""; }
    };

    struct MatmulParam {
        std::string to_string() const { return ""; }
    };

    struct DefaultParam {
        std::string to_string() const { return ""; }
    };

    //! get algo name, the format is ParamTrait<T>::category:base:p.to_string()
    //! \warning: base must not contain :.
    template <typename T>
    static std::string algo_name(const std::string& base, const T& p);
    /*!
     * \brief parse algo_name and get WinogradParam from algo name.
     *
     * \param algo name string
     * \return WinogradParam parsed from algo name, use pattern
     * winograd:base:m:tile_size.
     *
     * \warning: INVALID_WINOGRAD_PARAM returns if the algo_name is not matched.
     */
    static WinogradParam parse_winograd_name(const std::string& algo_name);

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& src,
                                   const TensorLayout& filter,
                                   const TensorLayout& bias,
                                   const TensorLayout& z,
                                   const TensorLayout& dst,
                                   size_t workspace_in_bytes);
};
using ConvBias = ConvBiasForward;

/**
 * \brief base class for Conv - Nonline - Pooling
 */
class ConvPoolingBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(ConvPoolingBase, OperatorBase);

    /**
     *  \ Param::Method: Two methods to fetch the input data.
     *  Default methods is WITH_TEXTURE_OBJ.
     *  If you want to use WITH_SHARED_MEM mode,
     *  please make sure that the size of
     *   [ all of the fliter kernels + a channel
     *  of input data + a channel of output data]
     *  should be no large than 38KB.
     *  And the pooling mode should not be "MAX".
     */
    DEF_OPR_PARAM(ConvPooling);

protected:
    virtual void deduce_layout(const TensorLayout& src,
                               const TensorLayout& filter,
                               const TensorLayout& bias, TensorLayout& dst) = 0;
    virtual void check_layout(const TensorLayout& src,
                              const TensorLayout& filter,
                              const TensorLayout& bias, TensorLayout& dst,
                              size_t workspace_limit_in_bytes) = 0;
};

class ConvPoolingForward : public ConvPoolingBase {
    DEF_OPR_IMPL(ConvPoolingForward, ConvPoolingBase, 2, 1);

public:
    /**
     * \param[in] src input tensor
     * \param[out] dst output tensor
     */
    virtual void exec(const _megdnn_in TensorND src,
                      const _megdnn_in TensorND filter,
                      const _megdnn_in TensorND bias, _megdnn_out TensorND dst,
                      _megdnn_out Workspace workspace) = 0;
    virtual void deduce_layout(const TensorLayout& src,
                               const TensorLayout& filter,
                               const TensorLayout& bias, TensorLayout& dst) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& bias,
                                          const TensorLayout& dst) = 0;

protected:
    virtual void check_layout(const TensorLayout& src,
                              const TensorLayout& filter,
                              const TensorLayout& bias, TensorLayout& dst,
                              size_t workspace_limit_in_bytes) = 0;
};
using ConvPooling = ConvPoolingForward;

class GroupLocalBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(GroupLocalBase, OperatorBase);
    DEF_OPR_PARAM(Convolution);

public:
    using Mode = Param::Mode;

protected:
    void deduce_layout_fwd(const TensorLayout& src, const TensorLayout& filter,
                           TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& filter,
                          const TensorLayout& dst);
};

class GroupLocalForward : public GroupLocalBase {
    DEF_OPR_IMPL(GroupLocalForward, GroupLocalBase, 2, 1);

public:
    /**
     * \param[in] src (N, IC, IH, IW)
     * \param[in] filter (G, OH, OW, IC/G, FH, FW, OC/G)
     * \param[out] dst (N, OC, OH, OW)
     **/
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       TensorLayout& dst) {
        deduce_layout_fwd(src, filter, dst);
    }
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& filter,
                    const TensorLayout& dst, size_t workspace_in_bytes);
};
using GroupLocal = GroupLocalForward;

class GroupLocalBackwardData : public GroupLocalBase {
    DEF_OPR_IMPL(GroupLocalBackwardData, GroupLocalBase, 2, 1);

public:
    virtual void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& filter,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& filter, const TensorLayout& diff,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class GroupLocalBackwardFilter : public GroupLocalBase {
    DEF_OPR_IMPL(GroupLocalBackwardFilter, GroupLocalBase, 2, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& filter, const TensorLayout& diff,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class Images2NeibsBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(Images2NeibsBase, OperatorBase);
    DEF_OPR_PARAM(Images2Neibs);

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& filter, const TensorLayout& dst);
};

class Images2NeibsForward : public Images2NeibsBase {
    DEF_OPR_IMPL(Images2NeibsForward, Images2NeibsBase, 1, 1);

public:
    /**
     * \param[in] src (N, C, IH, IW)
     * \param[out] dst (N, C, OH, OW, window_h, window_w)
     *
     * \see
     * http://deeplearning.net/software/theano/library/tensor/nnet/neighbours.html
     *
     * \f$ dst_{n, c, oh, ow, wh, ww} = src_{n, c, ih+wh, iw+fw}\f$,
     * where \f$ ih=-pad_h+oh*stride_h, iw=-pad_w+ow*stride_w\f$.
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using Images2Neibs = Images2NeibsForward;

class Images2NeibsBackward : public Images2NeibsBase {
    DEF_OPR_IMPL(Images2NeibsBackward, Images2NeibsBase, 1, 1);

public:
    /**
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[out] grad the backpropagated gradient wrt. src
     */
    virtual void exec(_megdnn_tensor_in diff, _megdnn_tensor_out grad,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& diff, const TensorLayout& grad,
                    size_t workspace_in_bytes);
};

/**
 * \brief base class for Pooling
 */
class PoolingBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(PoolingBase, OperatorBase);
    DEF_OPR_PARAM(Pooling);

public:
    using Mode = Param::Mode;

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& dst);
};

class PoolingForward : public PoolingBase {
    DEF_OPR_IMPL(PoolingForward, PoolingBase, 1, 1);

public:
    /**
     * \param[in] src input tensor
     * \param[out] dst output tensor
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};

using Pooling = PoolingForward;

class PoolingBackward : public PoolingBase {
    DEF_OPR_IMPL(PoolingBackward, PoolingBase, 3, 1);

public:
    /**
     * \param[in] src the `src' parameter in PoolingForward::exec
     * \param[in] dst the `dst' parameter in PoolingForward::exec
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[out] grad the backpropagated gradient wrt. src
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    const TensorLayout& diff, const TensorLayout& grad,
                    size_t workspace_in_bytes);
};

/**
 * \brief base class for Local
 */
class LocalBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(LocalBase, OperatorBase);
    DEF_OPR_PARAM(Convolution);

public:
    using Mode = Param::Mode;

protected:
    void deduce_layout_fwd(const TensorLayout& src, const TensorLayout& filter,
                           TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& filter,
                          const TensorLayout& dst);
};

class LocalForward : public LocalBase {
    DEF_OPR_IMPL(LocalForward, LocalBase, 2, 1);

public:
    /**
     * \param[in] src (n, ic, ih, iw)
     * \param[in] filter (oh, ow, ic, fh, fw, oc)
     * \param[out] dst (n, oc, oh, ow)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    /**
     * \brief Deducing output tensor layouts from input tensor layouts.
     *
     * Be aware that the first and second dimension of `filter' are ignored
     * when deducing `dst' layout.
     */
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& filter,
                    const TensorLayout& dst, size_t workspace_in_bytes);
};
using Local = LocalForward;

class LocalBackwardData : public LocalBase {
    DEF_OPR_IMPL(LocalBackwardData, LocalBase, 2, 1);

public:
    /**
     * \param[in] filter (oh, ow, ic, fh, fw, oc)
     * \param[in] diff (n, oc, oh, ow)
     * \param[out] grad (n, ic, ih, iw)
     */
    virtual void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorLayout& filter,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& filter, const TensorLayout& diff,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class LocalBackwardFilter : public LocalBase {
    DEF_OPR_IMPL(LocalBackwardFilter, LocalBase, 2, 1);

public:
    /**
     * \param[in] src (n, ic, ih, iw)
     * \param[in] diff (n, oc, oh, ow)
     * \param[out] grad (oh, ow, ic, fh, fw, oc)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& diff,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class BNBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(BNBase, OperatorBase);
    DEF_OPR_PARAM(BN);

protected:
    void check_param();
};

class BNForward : public BNBase {
    DEF_OPR_IMPL(BNForward, BNBase, 6, 5);

public:
    /**
     * \dst[i] = gemma
     * *(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + beta \where
     * epsilon is a very small value to avoid a "divide by zero" error.
     * \param[in] src (n, c, h, w)
     * \param[out] dst (n, c, h, w)
     * \param[out] mean (see m_param.ParamDim) Global mean.
     * \param[out] variance (see m_param.ParamDim) Global variance.
     * \Param[out] batch_mean (see m_param.ParamDim)
     *   Optionally cached intermediate mean from forward pass
     * \Param[out] batch_inv_variance (see m_param.ParamDim)
     *   Optionally cached intermediate variance from forward pass
     * src and dst must have the same shape.
     * src and dst must be contiguous.
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
                      _megdnn_tensor_in bn_bias, _megdnn_tensor_inout mean,
                      _megdnn_tensor_inout variance,
                      _megdnn_tensor_out batch_mean,
                      _megdnn_tensor_out batch_inv_variance,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& bn_scale,
                       TensorLayout& bn_bias, TensorLayout& mean,
                       TensorLayout& variance, TensorLayout& batch_mean,
                       TensorLayout& batch_inv_variance, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& bn_scale,
            const TensorLayout& bn_bias, const TensorLayout& mean,
            const TensorLayout& variance, const TensorLayout& batch_mean,
            const TensorLayout& batch_inv_variance,
            const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& bn_scale,
                    const TensorLayout& bn_bias, const TensorLayout& mean,
                    const TensorLayout& variance,
                    const TensorLayout& batch_mean,
                    const TensorLayout& batch_inv_variance,
                    const TensorLayout& dst, size_t workspace_in_bytes);
};
using BN = BNForward;

class BNBackward : public BNBase {
    DEF_OPR_IMPL(BNBackward, BNBase, 5, 3);

public:
    /**
     * \param[in] input data of forwarding propagate.
     * \param[in] dy the backpropagated gradient of y.
     * \param[out] dx the backpropagated gradient of x.
     * \param[out] d_bn_scale, the backpropagated gradient of bn_scale.
     * \param[out] d_bn_bias, the backpropagated gradient of bn_bias.
     * Optionally cached intermediate results from forward pass
     * \param[in] saved_batch_mean mean of the input batch.
        Calculated in the forwardpropagation.
     * \param[in] saved_batch_variance of the input batch.
        Calculated in the forwardpropagation.
     */
    virtual void exec(_megdnn_tensor_in x, _megdnn_tensor_in dy,
                      _megdnn_tensor_in saved_batch_mean,
                      _megdnn_tensor_in saved_batch_variance,
                      _megdnn_tensor_in bn_scale, _megdnn_tensor_out d_bn_scale,
                      _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& x, const TensorLayout& dy,
            const TensorLayout& saved_batch_mean,
            const TensorLayout& saved_batch_variance,
            const TensorLayout& bn_scale, const TensorLayout& d_bn_scale,
            const TensorLayout& d_bn_bias, const TensorLayout& dx) = 0;

protected:
    void check_exec(const TensorLayout& x, const TensorLayout& dy,
                    const TensorLayout& saved_batch_mean,
                    const TensorLayout& saved_batch_variance,
                    const TensorLayout& bn_scale,
                    const TensorLayout& d_bn_scale,
                    const TensorLayout& d_bn_bias, const TensorLayout& dx,
                    size_t workspace_in_bytes);
};

class LRNBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(LRNBase, OperatorBase);
    DEF_OPR_PARAM(LRN);

protected:
    void check_param();
};

class LRNForward : public LRNBase {
    DEF_OPR_IMPL(LRNForward, LRNBase, 1, 1);

public:
    /**
     * \see ImageNet Classification with Deep Convolutional Neural Networks
     * \param[in] src (n, c, h, w)
     * \param[out] dst (n, c, h, w)
     *
     * src and dst must have the same shape.
     * src and dst must be contiguous.
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    size_t workspace_in_bytes);
};
using LRN = LRNForward;

class LRNBackward : public LRNBase {
    DEF_OPR_IMPL(LRNBackward, LRNBase, 3, 1);

public:
    /**
     * \param[in] src the `src' parameter in LRNForward::exec
     * \param[in] dst the `dst' parameter in LRNForward::exec
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[out] grad the backpropagated gradient wrt. src
     *
     * All tensors should be contiguous and of the same shape.
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& dst,
                    const TensorLayout& diff, const TensorLayout& grad,
                    size_t workspace_in_bytes);
};

class ROIPoolingBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(ROIPoolingBase, OperatorBase);
    DEF_OPR_PARAM(ROIPooling);

protected:
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& rois,
                          const TensorLayout& dst, const TensorLayout& index);
};

class ROIPoolingForward : public ROIPoolingBase {
    DEF_OPR_IMPL(ROIPoolingForward, ROIPoolingBase, 2, 2);

public:
    /**
     * \param[in] src (n, c, ih, iw)
     * \param[in] rois (m, 5)
     * \param[out] dst (m, c, oh, ow)
     * \param[out] index (m, c, oh, ow) if mode is MAX, (0) if mode is AVERAGE
     *
     * The internal implementation is akin to
     * https://github.com/rbgirshick/caffe-fast-rcnn .d
     * Note that rois(, 0) denotes the input image index. We store it as
     * a float, but it should be an integer instead.
     *
     * index is a temporary tensor to facilitate its backward operator.
     * It is used to store argmax indicex in MAX mode, and it is not used
     * in AVERAGE mode.
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in rois,
                      _megdnn_tensor_out dst, _megdnn_tensor_out index,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& rois,
                                          const TensorLayout& dst,
                                          const TensorLayout& index) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& rois,
                    const TensorLayout& dst, const TensorLayout& index,
                    size_t workspace_in_bytes);
};
using ROIPooling = ROIPoolingForward;

class ROIPoolingBackward : public ROIPoolingBase {
    DEF_OPR_IMPL(ROIPoolingBackward, ROIPoolingBase, 4, 1);

public:
    /**
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[in] src the `src' parameter in ROIPoolingForward::exec
     * \param[in] rois the `rois' parameter in ROIPoolingForward::exec
     * \param[in] index the `index' parameter in ROIPoolingForward::exec
     * \param[out] grad the backpropagated gradient wrt. src
     */
    virtual void exec(_megdnn_tensor_in diff, _megdnn_tensor_in src,
                      _megdnn_tensor_in rois, _megdnn_tensor_in index,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& diff,
                                          const TensorLayout& src,
                                          const TensorLayout& rois,
                                          const TensorLayout& index,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& diff, const TensorLayout& src,
                    const TensorLayout& rois, const TensorLayout& index,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class Convolution3DBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(Convolution3DBase, OperatorBase);
    DEF_OPR_PARAM(Convolution3D);

public:
    static constexpr size_t MAX_SPATIAL_DIM = 3;
    using Mode = Param::Mode;
    struct CanonizedFilterMeta {
        DTypeEnum dtype_enum;
        Param::Format format;
        uint32_t
                //! whether filter should be flipped (i.e. is CONVOLUTION)
                should_flip,
                group,  //!< number of groups
                icpg,   //!< input channels per group
                ocpg,   //!< output channels per group
                spatial_ndim, stride[MAX_SPATIAL_DIM], padding[MAX_SPATIAL_DIM],
                //! spatial dim
                spatial[MAX_SPATIAL_DIM], dilation[MAX_SPATIAL_DIM],
                //! spatial dim with dilation applied
                dilated_spatial[MAX_SPATIAL_DIM];
    } MEGDNN_PACKED;

protected:
    CanonizedFilterMeta deduce_layout_fwd(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          TensorLayout& dst) const;
    CanonizedFilterMeta check_layout_fwd(const TensorLayout& src,
                                         const TensorLayout& filter,
                                         const TensorLayout& dst) const;

    CanonizedFilterMeta make_canonized_filter_meta(
            size_t src_ndim, const TensorLayout& filter) const;
};

class Convolution3DForward
        : public Convolution3DBase,
          public detail::MultiAlgoOpr<Convolution3DForward, 3> {
    DEF_OPR_IMPL(Convolution3DForward, Convolution3DBase, 2, 1);

public:
    /**
     * \param[in] src (n, ic, id, ih, iw)
     * \param[in] filter (oc, ic, fd, fh, fw)
     * \param[out] dst (n, oc, od, oh, ow)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& dst) = 0;

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& src,
                                   const TensorLayout& filter,
                                   const TensorLayout& dst,
                                   size_t workspace_in_bytes);
};
using Convolution3D = Convolution3DForward;

class Convolution3DBackwardData
        : public Convolution3DBase,
          public detail::MultiAlgoOpr<Convolution3DBackwardData, 3> {
    DEF_OPR_IMPL(Convolution3DBackwardData, Convolution3DBase, 2, 1);

public:
    /**
     * \param[in] filter (oc, ic, fd, fh, fw)
     * \param[in] diff (n, oc, od, oh, ow)
     * \param[out] grad (n, ic, id, ih, iw)
     */
    virtual void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& filter,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

    void deduce_layout(const TensorLayout& filter, const TensorLayout& diff,
                       TensorLayout& grad);

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& filter,
                                   const TensorLayout& diff,
                                   const TensorLayout& grad,
                                   size_t workspace_in_bytes);
};

class Convolution3DBackwardFilter
        : public Convolution3DBase,
          public detail::MultiAlgoOpr<Convolution3DBackwardFilter, 3> {
    DEF_OPR_IMPL(Convolution3DBackwardFilter, Convolution3DBase, 2, 1);

public:
    /**
     * \param[in] src (n, ic, id, ih, iw)
     * \param[in] diff (n, oc, od, oh, ow)
     * \param[out] grad (oc, ic, fd, fh, fw)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& src,
                                   const TensorLayout& diff,
                                   const TensorLayout& grad,
                                   size_t workspace_in_bytes);
};

class LocalShareBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(LocalShareBase, OperatorBase);
    DEF_OPR_PARAM(LocalShare);

protected:
    void deduce_layout_fwd(const TensorLayout& src, const TensorLayout& filter,
                           TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& filter,
                          const TensorLayout& dst);
};

class LocalShareForward : public LocalShareBase,
                          public detail::MultiAlgoOpr<LocalShareForward, 3> {
    DEF_OPR_IMPL(LocalShareForward, LocalShareBase, 2, 1);

public:
    /**
     * \param[in] src (N, IC, IH, IW)
     * \param[in] filter (G, spatial_groups_h, spatial_groups_w, IC / G,
     * FH, FW, OC / G)
     * \param[out] dst (N, OC, OH, OW)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    /**
     * \brief deduce layout of the ouput tensor
     */
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& dst) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& filter,
                    const TensorLayout& dst, size_t workspace_in_bytes);
};
using LocalShare = LocalShareForward;

class LocalShareBackwardData
        : public LocalShareBase,
          public detail::MultiAlgoOpr<LocalShareBackwardData, 3> {
    DEF_OPR_IMPL(LocalShareBackwardData, LocalShareBase, 2, 1);

public:
    /**
     * \param[in] filter (G, spatial_groups_h, spatial_groups_w, IC / G,
     * FH, FW, OC / G)
     * \param[in] diff (N, OC, OH, OW)
     * \param[out] grad (N, IC, IH, IW)
     */
    virtual void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& filter,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;
    void deduce_layout(const TensorLayout& filter, const TensorLayout& diff,
                       TensorLayout& grad);

protected:
    void check_exec(const TensorLayout& filter, const TensorLayout& diff,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class LocalShareBackwardFilter
        : public LocalShareBase,
          public detail::MultiAlgoOpr<LocalShareBackwardFilter, 3> {
    DEF_OPR_IMPL(LocalShareBackwardFilter, LocalShareBase, 2, 1);

public:
    /**
     * \param[in] src (N, IC, IH, IW)
     * \param[in] diff (N, OC, OH, OW)
     * \param[out] grad (G, spatial_groups_h, spatial_groups_w, IC / G,
     * FH, FW, OC / G)
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
                      _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& diff,
                    const TensorLayout& grad, size_t workspace_in_bytes);
};

class ROIAlignBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(ROIAlignBase, OperatorBase);
    DEF_OPR_PARAM(ROIAlign);

protected:
    void deduce_layout_fwd(const TensorLayout& src, const TensorLayout& rois,
                           TensorLayout& dst, TensorLayout& index);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& rois,
                          const TensorLayout& dst, const TensorLayout& index);
};

class ROIAlignForward : public ROIAlignBase {
    DEF_OPR_IMPL(ROIAlignForward, ROIAlignBase, 2, 2);

public:
    /**
     * \param[in] src (n, c, ih, iw)
     * \param[in] rois (m, 5)
     * \param[out] dst (m, c, oh, ow)
     * \param[out] index (m, c, oh, ow) if mode is MAX, (0) if mode is AVERAGE
     *
     * Note that rois(, 0) denotes the input image index. We store it as
     * a float, but it should be an integer instead.
     *
     * index is a temporary tensor to facilitate its backward operator.
     * It is used to store argmax indicex in MAX mode, and it is not used
     * in AVERAGE mode.
     */
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in rois,
                      _megdnn_tensor_out dst, _megdnn_tensor_out index,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, const TensorLayout& rois,
                       TensorLayout& dst, TensorLayout& index);
    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& rois,
                                          const TensorLayout& dst,
                                          const TensorLayout& index) = 0;

protected:
    void check_exec(const TensorLayout& src, const TensorLayout& rois,
                    const TensorLayout& dst, const TensorLayout& index,
                    size_t workspace_in_bytes);
};
using ROIAlign = ROIAlignForward;

class ROIAlignBackward : public ROIAlignBase {
    DEF_OPR_IMPL(ROIAlignBackward, ROIAlignBase, 3, 1);

public:
    /**
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[in] rois the `rois' parameter in ROIAlignForward::exec
     * \param[in] index the `index' parameter in ROIAlignForward::exec
     * \param[out] grad the backpropagated gradient wrt. src
     */
    virtual void exec(_megdnn_tensor_in diff, _megdnn_tensor_in rois,
                      _megdnn_tensor_in index, _megdnn_tensor_out grad,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& diff,
                                          const TensorLayout& rois,
                                          const TensorLayout& index,
                                          const TensorLayout& grad) = 0;

protected:
    void check_exec(const TensorLayout& diff, const TensorLayout& rois,
                    const TensorLayout& index, const TensorLayout& grad,
                    size_t workspace_in_bytes);
};

class DeformableConvBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(DeformableConvBase, OperatorBase);
    DEF_OPR_PARAM(Convolution);

public:
    static constexpr size_t MAX_SPATIAL_DIM = 2;
    struct CanonizedFilterMeta : Convolution::CanonizedFilterMeta {
        uint32_t deformable_group;
    };

protected:
    CanonizedFilterMeta make_canonized_filter_meta(
            size_t src_ndim, const TensorLayout& filter,
            const TensorLayout& offset) const;
    void deduce_layout_fwd(const TensorLayout& im, const TensorLayout& filter,
                           const TensorLayout& mask, const TensorLayout& offset,
                           TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& src, const TensorLayout& filter,
                          const TensorLayout& mask, const TensorLayout& offset,
                          const TensorLayout& dst);
};

class DeformableConvForward
        : public DeformableConvBase,
          public detail::MultiAlgoOpr<DeformableConvForward, 5> {
    DEF_OPR_IMPL(DeformableConvForward, DeformableConvBase, 4, 1);

public:
    /**
     * \param[in] im (n, ic, ih, iw)
     * \param[in] filter (oc, ic, fh, fw)
     * \param[in] offset (dg, 2, fh, fw, oh, ow)
     * \param[in] mask (dg, fh, fw, oh, ow)
     * \param[out] dst (n, oc, oh, ow)
     */
    virtual void exec(_megdnn_tensor_in im, _megdnn_tensor_in filter,
                      _megdnn_tensor_in offset, _megdnn_tensor_in mask,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& im, const TensorLayout& filter,
                       const TensorLayout& offset, const TensorLayout& mask,
                       TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(const TensorLayout& im,
                                          const TensorLayout& filter,
                                          const TensorLayout& offset,
                                          const TensorLayout& mask,
                                          const TensorLayout& dst) = 0;

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& im,
                                   const TensorLayout& filter,
                                   const TensorLayout& offset,
                                   const TensorLayout& mask,
                                   const TensorLayout& dst,
                                   size_t workspace_in_bytes);
};
using DeformableConv = DeformableConvForward;

/**
 * \brief DeformableConvBackwardFilter operator.
 *
 * Calculating the gradient wrt. convolution filter.
 */
class DeformableConvBackwardFilter
        : public DeformableConvBase,
          public detail::MultiAlgoOpr<DeformableConvBackwardFilter, 5> {
    DEF_OPR_IMPL(DeformableConvBackwardFilter, DeformableConvBase, 4, 1);

public:
    /**
     * \param[in] im (oc, ic, fh, fw)
     * \param[in] offset (dg, 2, fh, fw, oh, ow)
     * \param[in] mask (dg, fh, fw, oh, ow)
     * \param[in] out_grad (n, oc, oh, ow)
     * \param[out] filter_grad (oc, ic, ih, iw)
     */
    virtual void exec(_megdnn_tensor_in im, _megdnn_tensor_in offset,
                      _megdnn_tensor_in mask, _megdnn_tensor_in out_grad,
                      _megdnn_tensor_out filter_grad,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& im,
                                          const TensorLayout& offset,
                                          const TensorLayout& mask,
                                          const TensorLayout& out_grad,
                                          const TensorLayout& filter_grad) = 0;
    void deduce_layout(const TensorLayout& im, const TensorLayout& offset,
                       const TensorLayout& mask, const TensorLayout& out_grad,
                       TensorLayout& filter_grad);

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& im,
                                   const TensorLayout& offset,
                                   const TensorLayout& mask,
                                   const TensorLayout& out_grad,
                                   const TensorLayout& filter_grad,
                                   size_t workspace_in_bytes);
};

/**
 * \brief DeformableConvBackwardData operator.
 *
 * Calculating the gradient wrt. convolution input data, offset and mask.
 */
class DeformableConvBackwardData
        : public DeformableConvBase,
          public detail::MultiAlgoOpr<DeformableConvBackwardData, 8> {
    DEF_OPR_IMPL(DeformableConvBackwardData, DeformableConvBase, 5, 3);

public:
    /**
     * \param[in] im (oc, ic, fh, fw)
     * \param[in] filter (oc, ic, fh, fw)
     * \param[in] offset (dg, 2, fh, fw, oh, ow)
     * \param[in] mask (dg, fh, fw, oh, ow)
     * \param[in] out_grad (n, oc, oh, ow)
     * \param[out] im_grad (n, ic, ih, iw)
     * \param[out] offset_grad (dg, 2, fh, fw, oh, ow)
     * \param[out] mask_grad (dg, fh, fw, oh, ow)
     */
    virtual void exec(_megdnn_tensor_in im, _megdnn_tensor_in filter,
                      _megdnn_tensor_in offset, _megdnn_tensor_in mask,
                      _megdnn_tensor_in out_grad, _megdnn_tensor_out im_grad,
                      _megdnn_tensor_out offset_grad,
                      _megdnn_tensor_out mask_grad,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad) = 0;
    void deduce_layout(const TensorLayout& im, const TensorLayout& filter,
                       const TensorLayout& offset, const TensorLayout& mask,
                       const TensorLayout& out_grad, TensorLayout& im_grad,
                       TensorLayout& offset_grad, TensorLayout& mask_grad);

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad,
            size_t workspace_in_bytes);
};

class DeformablePSROIPoolingBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(DeformablePSROIPoolingBase, OperatorBase);
    DEF_OPR_PARAM(DeformablePSROIPooling);

protected:
    void deduce_layout_fwd(const TensorLayout& data, const TensorLayout& trans,
                           const TensorLayout& rois, TensorLayout& out_data,
                           TensorLayout& out_count);

    void check_layout_fwd(const TensorLayout& data, const TensorLayout& trans,
                          const TensorLayout& rois,
                          const TensorLayout& out_data,
                          const TensorLayout& out_count,
                          size_t workspace_in_bytes);
};

class DeformablePSROIPoolingForward : public DeformablePSROIPoolingBase {
    DEF_OPR_IMPL(DeformablePSROIPoolingForward, DeformablePSROIPoolingBase, 3,
                 2);

public:
    /**
     * \param[in]  data       (oc, ic, ih, iw)
     * \param[in]  rois       (xx, xx, xx, xx)
     * \param[in]  trans      (oc, ic, fh, fw)
     * \param[out] out_data   ( n, ic, ih, iw)
     * \param[out] out_count  ( n, ic, ih, iw)
     */
    virtual size_t get_workspace_in_bytes(const TensorLayout& data,
                                          const TensorLayout& rois,
                                          const TensorLayout& trans,
                                          const TensorLayout& out_data,
                                          const TensorLayout& out_count) = 0;
    virtual void exec(_megdnn_tensor_in data, _megdnn_tensor_in rois,
                      _megdnn_tensor_in trans, _megdnn_tensor_out out_data,
                      _megdnn_tensor_out out_count,
                      _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& data, const TensorLayout& rois,
                       const TensorLayout& trans, TensorLayout& out_data,
                       TensorLayout& out_count);
    void check_exec(const TensorLayout& data, const TensorLayout& rois,
                    const TensorLayout& trans, const TensorLayout& out_data,
                    const TensorLayout& out_count, size_t workspace_in_bytes);
};

using DeformablePSROIPooling = DeformablePSROIPoolingForward;

class DeformablePSROIPoolingBackward : public DeformablePSROIPoolingBase {
    DEF_OPR_IMPL(DeformablePSROIPoolingBackward, DeformablePSROIPoolingBase, 5,
                 2);

public:
    /**
     * \param[in]  data        (oc, ic, ih, iw)
     * \param[in]  rois        (xx, xx, xx, xx)
     * \param[in]  trans       (oc, ic, fh, fw)
     * \param[in]  out_diff    (xx, xx, xx, xx)
     * \param[in]  out_count   (xx, xx, xx, xx)
     * \param[out] data_diff   ( n, ic, ih, iw)
     * \param[out] trans_diff  ( n, ic, ih, iw)
     */
    virtual void exec(_megdnn_tensor_in data, _megdnn_tensor_in rois,
                      _megdnn_tensor_in trans, _megdnn_tensor_in out_diff,
                      _megdnn_tensor_in out_count, _megdnn_tensor_out data_diff,
                      _megdnn_tensor_out trans_diff,
                      _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(const TensorLayout& data,
                                          const TensorLayout& rois,
                                          const TensorLayout& trans,
                                          const TensorLayout& out_diff,
                                          const TensorLayout& out_count,
                                          const TensorLayout& data_diff,
                                          const TensorLayout& trans_diff) = 0;

    void check_exec(const TensorLayout& data, const TensorLayout& rois,
                    const TensorLayout& trans, const TensorLayout& out_diff,
                    const TensorLayout& out_count,
                    const TensorLayout& data_diff,
                    const TensorLayout& trans_diff, size_t workspace_in_bytes);
};

class BatchConvBiasForward
        : public ConvolutionBase<param::BatchConvBias>,
          public detail::MultiAlgoOpr<BatchConvBiasForward, 5> {
    DEF_OPR_IMPL(BatchConvBiasForward, ConvolutionBase, 4, 1);

public:
    virtual void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                      _megdnn_tensor_in bias, _megdnn_tensor_in z,
                      _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;

    void deduce_dtype(DType src, DType filter, DType bias, DType z, DType& dst);
    void deduce_layout(const TensorLayout& src, const TensorLayout& filter,
                       const TensorLayout& bias, const TensorLayout& z,
                       TensorLayout& dst);

    virtual size_t get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& filter,
                                          const TensorLayout& bias,
                                          const TensorLayout& z,
                                          const TensorLayout& dst) = 0;

protected:
    CanonizedFilterMeta check_exec(const TensorLayout& src,
                                   const TensorLayout& filter,
                                   const TensorLayout& bias,
                                   const TensorLayout& z,
                                   const TensorLayout& dst,
                                   size_t workspace_in_bytes);
};
using BatchConvBias = BatchConvBiasForward;

}  // namespace megdnn
#include "megdnn/internal/opr_header_epilogue.h"

// vim: syntax=cpp.doxygen
