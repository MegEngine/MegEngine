#pragma once
#include "megdnn/internal/opr_header_prologue.h"

namespace megdnn {

class SeparableConvBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(SeparableConvBase, OperatorBase);
    DEF_OPR_PARAM(SeparableConv);

public:
    using Mode = Param::Mode;

protected:
    void deduce_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter_x,
            const TensorLayout& filter_y, TensorLayout& dst);
    void check_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter_x,
            const TensorLayout& filter_y, const TensorLayout& dst);
};

class SeparableConvForward : public SeparableConvBase {
    DEF_OPR_IMPL(SeparableConvForward, SeparableConvBase, 3, 1);

public:
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter_x,
            _megdnn_tensor_in filter_y, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter_x,
            const TensorLayout& filter_y, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter_x,
            const TensorLayout& filter_y, const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& filter_x,
            const TensorLayout& filter_y, const TensorLayout& dst,
            size_t workspace_in_bytes);
};
using SeparableConv = SeparableConvForward;

namespace detail {

struct PreprocessedFilter {
    //! user data; its lifetime should be bound to MegDNN Convolution
    //! operator
    void* algorithm_id;
    TensorNDArray tensors;
};

}  // namespace detail

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
    using PreprocessedFilter = detail::PreprocessedFilter;

protected:
    // Check or deduce output DType
    void check_or_deduce_dtype_fwd(DType src, DType filter, DType& dst) const;
    CanonizedFilterMeta deduce_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
            TensorLayout& dst) const;
    CanonizedFilterMeta check_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) const;

    CanonizedFilterMeta make_canonized_filter_meta(
            size_t src_ndim, const TensorLayout& filter) const;
};

class MaskPropagate : public OperatorBase {
    DEF_OPR_IMPL(MaskPropagate, OperatorBase, 1, 1);
    DEF_OPR_PARAM(MaskPropagate);

public:
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) = 0;

    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
};

/**
 * \brief ConvolutionForward Operator with 0/1 Mask matrix
 */
class MaskConvForward : public ConvolutionBase<param::Convolution> {
    DEF_OPR_IMPL(MaskConvForward, ConvolutionBase, 3, 1);

public:
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in mask,
            _megdnn_tensor_out dst, _megdnn_workspace worksapce) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& mask, const TensorLayout& dst) = 0;

    void deduce_dtype(DType src, DType filter, DType mask, DType& dst);
    void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& mask, TensorLayout& dst);

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& mask, const TensorLayout& dst,
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
     * \param[in] preprocessed_filter if weight no preprocessed it will be
     * nullptr, else the preprocessed weights store in the tensors of
     * preprocessed_filter.
     * \param[in] workspace if weight no preprocessed
     * (preprocessed_filter == nullptr), The size of the workspace satisfies the
     * situation that weights is not processed, other wise the size of workspace
     * satisfies the situation that weights is preprocessed
     * \param[out] dst (n, oc, oh, ow)
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            const PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace) = 0;

    MGE_WIN_DECLSPEC_FUC void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) {
        exec(src, filter, dst, nullptr, workspace);
    }
    /**
     * \brief execute weight preprocessing, read weights form filter and write
     * to preprocessed_filter after preprocessed.
     *
     * \praram[in] workspace the needed tmp workspace when exec_preprocess
     */
    virtual void exec_preprocess(
            const TensorLayout& src_layout, _megdnn_tensor_in filter,
            const TensorLayout& dst_layout, PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_dtype(DType src, DType filter, DType& dst);

    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst);

    /**
     * \brief query the workspace needed when executing the opr, if the weights
     * are preprocessed the preprocessed_filter will not be nullptr, else it
     * will be nullptr, the workspace size maybe different whether weights are
     * preprocessed
     *
     * \return the size of workspace needed when executing
     */
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst, const PreprocessedFilter* preprocessed_filter) = 0;

    /**
     * \brief deduce the preprocessed filter layouts according to the src,
     * filter and dst layout, the result may contain multi layouts when the
     * weights is not one
     *
     * \return SmallVector<TensorLayout> Derive the layouts of weight
     * preprocessing, return empty if preprocessing is not needed.
     */
    virtual SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) = 0;

    /**
     * \brief query the workspace needed when preprocessing the weights,
     * according to the return size, a _megdnn_workspace will be created and
     * passed through exec_preprocess
     *
     * \return the size of workspace needed when preprocessing
     */
    virtual size_t get_preprocess_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::CONVOLUTION_FORWARD;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst, size_t workspace_in_bytes,
            const PreprocessedFilter* preprocessed_filter);
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
    virtual void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

    MGE_WIN_DECLSPEC_FUC void deduce_dtype(DType filter, DType diff, DType& grad);
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& filter, const TensorLayout& diff, TensorLayout& grad);

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::CONVOLUTION_BACKWARD_DATA;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_in_bytes);
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::CONVOLUTION_BACKWARD_FILTER;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad,
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
     * 4 * ic)
     * \param[in] bias (1, oc, 1, 1)
     * \param[in] z same as dst
     * \param[in] preprocessed_filter if weight no preprocessed it will be
     * nullptr, else the preprocessed weights store in the tensors of
     * preprocessed_filter.
     * \param[in] workspace if weight no preprocessed
     * (preprocessed_filter == nullptr), The size of the workspace satisfies the
     * situation that weights is not processed, other wise the size of workspace
     * satisfies the situation that weights is preprocessed
     * \param[out] dst (n, oc, oh, ow) or (n, oh, ow, oc)
     *
     * \note if the format is NCHW_WINOGRAD, the filter layout is (alphah,
     * alphaw, oc, ic)
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
            _megdnn_tensor_in z, _megdnn_tensor_out dst,
            const PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace) = 0;

    MGE_WIN_DECLSPEC_FUC void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
            _megdnn_tensor_in z, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
        exec(src, filter, bias, z, dst, nullptr, workspace);
    }

    /**
     * \brief execute weight preprocessing, read weights form filter and bias,
     * write to preprocessed_filter after preprocessed.
     *
     * \praram[in] workspace the needed tmp workspace when exec_preprocess
     * running, the size is got by get_preprocess_workspace_in_bytes
     */
    virtual void exec_preprocess(
            const TensorLayout& src_layout, _megdnn_tensor_in filter,
            _megdnn_tensor_in bias, const TensorLayout& z_layout,
            const TensorLayout& dst_layout, PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_dtype(
            DType src, DType filter, DType bias, DType z, DType& dst);
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, TensorLayout& dst);

    /**
     * \brief query the workspace needed when executing the opr, if the weights
     * are preprocessed the preprocessed_filter will not be nullptr, else it
     * will be nullptr, the workspace size maybe different whether weights are
     * preprocessed
     *
     * \return the size of workspace needed when executing
     */
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
            const PreprocessedFilter* preprocessed_filter) = 0;

    /**
     * \brief query the workspace needed when pre-processing the weights,
     * according to the return size, a _megdnn_workspace will be created and
     * passed through exec_preprocess
     *
     * \return the size of workspace needed when pre-processing
     */
    virtual size_t get_preprocess_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) = 0;

    /**
     * \brief deduce the pre-processed filter layouts according to the src,
     * filter and dst layout, which may contain multi layouts when the weights
     * is not one
     *
     * \return SmallVector<TensorLayout> Derive the layouts of weight
     * preprocessing, return empty if preprocessing is not needed.
     */
    virtual SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
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
        uint32_t filter_size;
        bool operator==(const WinogradParam& rhs) const {
            return channel_block_size == rhs.channel_block_size &&
                   output_block_size == rhs.output_block_size &&
                   tile_size == rhs.tile_size && filter_size == rhs.filter_size;
        }

        std::string to_string() const;
    };
    static constexpr WinogradParam INVALID_WINOGRAD_PARAM = {0, 0, 0, 0};

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
    static std::string algo_name(
            const std::string& base, const T& p,
            param::ConvBias::Format format = param::ConvBias::Format::NCHW);
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

    /**
     * @brief find if there is nchw_nchwxx conv kernel optimized for argment,
     * nchw44 used for arm, nchw88 used for x86
     *
     * @param src_dtype  conv feature map data type
     * @param filter_dtype  conv filter or weight data type
     * @param dst_dtype output data type
     * @param fm filter meta param
     * @param bias_mode bias mode, no_bias or broadcast or bias
     * @param nonline_mode identity or relu or h_swish or sigmoid
     * @return true, found a kernel
     * @return false, can`t found any kernel
     */
    static bool is_nchw_nchwxx_optimized(
            const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
            const DTypeEnum dst_dtype,
            const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
            const ConvBiasForward::BiasMode bias_mode,
            const param::ConvBias::NonlineMode nonline_mode);

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::CONVBIAS_FORWARD;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
            size_t workspace_in_bytes, const PreprocessedFilter* preprocessed_filter);

    CanonizedFilterMeta check_exec_allow_noncontiguous(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
            size_t workspace_in_bytes, const PreprocessedFilter* preprocessed_filter);
};
using ConvBias = ConvBiasForward;

/**
 * \brief RegionRestrictedConvolutionForward operator.
 */
class RegionRestrictedConvolutionForward : public ConvolutionBase<param::Convolution> {
    DEF_OPR_IMPL(RegionRestrictedConvolutionForward, ConvolutionBase, 4, 1);

public:
    /**
     * \param[in] src (n, ic, ih, iw) or (n, g*icpg, ih, iw)
     * \param[in] filter (oc, ic, fh, fw) or (g, ocpg, icpg, fh, fw)
     * \param[in] rin (n, ih, iw)
     * \param[in] rout (n, oh, ow)
     * \param[out] dst (n, oc, oh, ow) or (n, g*ocpg, oh, ow)
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in rin,
            _megdnn_tensor_in rout, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;

    void deduce_dtype(DType src, DType filter, DType rin, DType rout, DType& dst);

    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& rin, const TensorLayout& rout, TensorLayout& dst);

    /**
     * \brief query the workspace needed when executing the opr
     * \return the size of workspace needed when executing
     */
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& rin, const TensorLayout& rout,
            const TensorLayout& dst) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::REGIONRESTRICTEDCONVOLUTION_FORWARD;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& rin, const TensorLayout& rout, const TensorLayout& dst,
            size_t workspace_in_bytes);
};
using RegionRestrictedConvolution = RegionRestrictedConvolutionForward;

/**
 * \brief RegionRestrictedConvolutionBackwardData operator.
 *
 * Calculating the gradient wrt. convolution input data.
 */
class RegionRestrictedConvolutionBackwardData
        : public ConvolutionBase<param::Convolution> {
    DEF_OPR_IMPL(RegionRestrictedConvolutionBackwardData, ConvolutionBase, 4, 1);

public:
    /**
     * \param[in] filter (oc, ic, fh, fw) or (g, ocpg, icpg, fh, fw)
     * \param[in] diff (n, oc, oh, ow) or (n, g*ocpg, oh, ow)
     * \param[in] rin (n, ih, iw)
     * \param[in] rout (n, oh, ow)
     * \param[out] grad (n, ic, ih, iw) or (n, g*icpg, ih, iw)
     */
    virtual void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
            _megdnn_tensor_in rout, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& rin, const TensorLayout& rout,
            const TensorLayout& grad) = 0;

    MGE_WIN_DECLSPEC_FUC void deduce_dtype(
            DType filter, DType diff, DType rin, DType rout, DType& grad);
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& rin, const TensorLayout& rout, TensorLayout& grad);

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::REGIONRESTRICTEDCONVOLUTION_BACKWARD_DATA;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& rin, const TensorLayout& rout, const TensorLayout& grad,
            size_t workspace_in_bytes);
};

/**
 * \brief RegionRestrictedConvolutionBackwardFilter operator.
 *
 * Calculating the gradient wrt. convolution filter.
 */
class RegionRestrictedConvolutionBackwardFilter
        : public ConvolutionBase<param::Convolution> {
    DEF_OPR_IMPL(RegionRestrictedConvolutionBackwardFilter, ConvolutionBase, 4, 1);

public:
    /**
     * \param[in] src (n, ic, ih, iw) or (n, g*icpg, ih, iw)
     * \param[in] diff (n, oc, oh, ow) or (n, g*ocpg, oh, ow)
     * \param[in] rin (n, ih, iw)
     * \param[in] rout (n, oh, ow)
     * \param[out] grad (oc, ic, fh, fw) or (g, ocpg, icpg, fh, fw)
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
            _megdnn_tensor_in rout, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff, const TensorLayout& rin,
            const TensorLayout& rout, const TensorLayout& grad) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::REGIONRESTRICTEDCONVOLUTION_BACKWARD_FILTER;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& diff, const TensorLayout& rin,
            const TensorLayout& rout, const TensorLayout& grad,
            size_t workspace_in_bytes);
};

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
    virtual void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, TensorLayout& dst) = 0;
    virtual void check_layout(
            const TensorLayout& src, const TensorLayout& filter,
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
    virtual void exec(
            const _megdnn_in TensorND src, const _megdnn_in TensorND filter,
            const _megdnn_in TensorND bias, _megdnn_out TensorND dst,
            _megdnn_out Workspace workspace) = 0;
    virtual void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, TensorLayout& dst) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& dst) = 0;

protected:
    virtual void check_layout(
            const TensorLayout& src, const TensorLayout& filter,
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
    void deduce_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst);
    void check_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst) {
        deduce_layout_fwd(src, filter, dst);
    }
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst, size_t workspace_in_bytes);
};
using GroupLocal = GroupLocalForward;

class GroupLocalBackwardData : public GroupLocalBase {
    DEF_OPR_IMPL(GroupLocalBackwardData, GroupLocalBase, 2, 1);

public:
    virtual void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_in_bytes);
};

class GroupLocalBackwardFilter : public GroupLocalBase {
    DEF_OPR_IMPL(GroupLocalBackwardFilter, GroupLocalBase, 2, 1);

public:
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& filter, const TensorLayout& diff,
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
     * where \f$ ih=-pad_h+oh*stride_h+(wh-1)*(dilation_h-1),
     * iw=-pad_w+ow*stride_w+(ww-1)*(dilation_w-1)\f$.
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& dst,
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
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& grad,
            size_t workspace_in_bytes);
};

class SlidingWindowTransposeBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(SlidingWindowTransposeBase, OperatorBase);
    DEF_OPR_PARAM(SlidingWindowTranspose);

protected:
    void deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst);
    void check_layout_fwd(const TensorLayout& filter, const TensorLayout& dst);
};

class SlidingWindowTransposeForward : public SlidingWindowTransposeBase {
    DEF_OPR_IMPL(SlidingWindowTransposeForward, SlidingWindowTransposeBase, 1, 1);

public:
    /**
     * \param[in] src (N, C, IH, IW, window_h, window_w)
     * \param[out] dst (N, C, OH, OW)
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_in_bytes);
};
using SlidingWindowTranspose = SlidingWindowTransposeForward;

class SlidingWindowTransposeBackward : public SlidingWindowTransposeBase {
    DEF_OPR_IMPL(SlidingWindowTransposeBackward, SlidingWindowTransposeBase, 1, 1);

public:
    /**
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[out] grad the backpropagated gradient wrt. src
     */
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& grad,
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

public:
    static void deduce_layout_impl(
            const TensorLayout& src, const Param& param, TensorLayout& dst);
};

class PoolingForward : public PoolingBase,
                       public detail::MultiAlgoOpr<PoolingForward, 2> {
    DEF_OPR_IMPL(PoolingForward, PoolingBase, 1, 1);

public:
    /**
     * \param[in] src input tensor
     * \param[out] dst output tensor
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::POOLING_FORWARD;
    }

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_in_bytes);
};

using Pooling = PoolingForward;

class PoolingBackward : public PoolingBase,
                        public detail::MultiAlgoOpr<PoolingBackward, 4> {
    DEF_OPR_IMPL(PoolingBackward, PoolingBase, 3, 1);

public:
    /**
     * \param[in] src the `src' parameter in PoolingForward::exec
     * \param[in] dst the `dst' parameter in PoolingForward::exec
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[out] grad the backpropagated gradient wrt. src
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::POOLING_BACKWARD;
    }

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_in_bytes);
};

/**
 * \brief base class for AdaptivePooling
 */
class AdaptivePoolingBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(AdaptivePoolingBase, OperatorBase);
    DEF_OPR_PARAM(AdaptivePooling);

protected:
    param::Pooling deduce_pooling_param(
            const TensorLayout& src, const TensorLayout& dst);
};

class AdaptivePoolingForward : public AdaptivePoolingBase {
    DEF_OPR_IMPL(AdaptivePoolingForward, AdaptivePoolingBase, 1, 1);

public:
    /**
     * \param[in] src input tensor
     * \param[out] dst output tensor
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) = 0;
};

using AdaptivePooling = AdaptivePoolingForward;

class AdaptivePoolingBackward : public AdaptivePoolingBase {
    DEF_OPR_IMPL(AdaptivePoolingBackward, AdaptivePoolingBase, 3, 1);

public:
    /**
     * \param[in] src the `src' parameter in AdaptivePoolingForward::exec
     * \param[in] dst the `dst' parameter in AdaptivePoolingForward::exec
     * \param[in] diff the backpropagated gradient wrt. dst
     * \param[out] grad the backpropagated gradient wrt. src
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad) = 0;
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
    void deduce_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst);
    void check_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    /**
     * \brief Deducing output tensor layouts from input tensor layouts.
     *
     * Be aware that the first and second dimension of `filter' are ignored
     * when deducing `dst' layout.
     */
    void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& filter,
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
    virtual void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& filter, const TensorLayout& diff,
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad,
            size_t workspace_in_bytes);
};

class BNBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(BNBase, OperatorBase);
    DEF_OPR_PARAM(BN);

protected:
    void check_param();
};

class BNForward : public BNBase {
    DEF_OPR_IMPL(BNForward, BNBase, 6, 6);

public:
    /**
     * \dst[i] = gemma
     * *(x[i]-estimatedMean[k])/sqrt(epsilon+estimatedVariance[k]) + beta \where
     * epsilon is a very small value to avoid a "divide by zero" error.
     * \param[in] src (n, c, h, w)
     * \param[out] dst (n, c, h, w)
     * \param[out] mean (see m_param.ParamDim) Global mean.
     * \param[out] variance (see m_param.ParamDim) Global variance.
     * \param[out] batch_mean (see m_param.ParamDim)
     *   Optionally cached intermediate mean from forward pass
     * \param[out] batch_inv_variance (see m_param.ParamDim)
     *   Optionally cached intermediate variance from forward pass
     * \param[out] reserve (see cudnnBatchNormalizationForwardTrainingEx)
     * src and dst must have the same shape.
     * src and dst must be contiguous.
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
            _megdnn_tensor_in bn_bias, _megdnn_tensor_inout mean,
            _megdnn_tensor_inout variance, _megdnn_tensor_out batch_mean,
            _megdnn_tensor_out batch_inv_variance, _megdnn_tensor_out reserve,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& src, const TensorLayout& bn_scale,
            const TensorLayout& bn_bias, TensorLayout& mean, TensorLayout& variance,
            TensorLayout& batch_mean, TensorLayout& batch_inv_variance,
            TensorLayout& reserve, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& bn_scale,
            const TensorLayout& bn_bias, const TensorLayout& mean,
            const TensorLayout& variance, const TensorLayout& batch_mean,
            const TensorLayout& batch_inv_variance, const TensorLayout& reserve,
            const TensorLayout& dst) = 0;
    virtual size_t get_reserve_in_bytes(const TensorLayout& src) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& bn_scale,
            const TensorLayout& bn_bias, const TensorLayout& mean,
            const TensorLayout& variance, const TensorLayout& batch_mean,
            const TensorLayout& batch_inv_variance, const TensorLayout& dst,
            size_t workspace_in_bytes, size_t reserve_in_bytes = 0);
};
using BN = BNForward;

class BNBackward : public BNBase {
    DEF_OPR_IMPL(BNBackward, BNBase, 6, 3);

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
     * \param[in] reserve (see cudnnBatchNormalizationBackwardEx)
     */
    virtual void exec(
            _megdnn_tensor_in x, _megdnn_tensor_in dy,
            _megdnn_tensor_in saved_batch_mean, _megdnn_tensor_in saved_batch_variance,
            _megdnn_tensor_in bn_scale, _megdnn_tensor_in reserve,
            _megdnn_tensor_out d_bn_scale, _megdnn_tensor_out d_bn_bias,
            _megdnn_tensor_out dx, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& x, const TensorLayout& dy,
            const TensorLayout& saved_batch_mean,
            const TensorLayout& saved_batch_variance, const TensorLayout& bn_scale,
            const TensorLayout& reserve, const TensorLayout& d_bn_scale,
            const TensorLayout& d_bn_bias, const TensorLayout& dx) = 0;
    virtual size_t get_reserve_in_bytes(const TensorLayout& src) = 0;

protected:
    virtual void check_exec(
            const TensorLayout& x, const TensorLayout& dy,
            const TensorLayout& saved_batch_mean,
            const TensorLayout& saved_batch_variance, const TensorLayout& bn_scale,
            const TensorLayout& d_bn_scale, const TensorLayout& d_bn_bias,
            const TensorLayout& dx, size_t workspace_in_bytes,
            size_t reserve_in_bytes = 0);
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& src, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& dst,
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_in_bytes);
};

class ROIPoolingBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(ROIPoolingBase, OperatorBase);
    DEF_OPR_PARAM(ROIPooling);

protected:
    void check_layout_fwd(
            const TensorLayout& src, const TensorLayout& rois, const TensorLayout& dst,
            const TensorLayout& index);
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in rois, _megdnn_tensor_out dst,
            _megdnn_tensor_out index, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& rois, const TensorLayout& dst,
            const TensorLayout& index) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& rois, const TensorLayout& dst,
            const TensorLayout& index, size_t workspace_in_bytes);
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
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in src, _megdnn_tensor_in rois,
            _megdnn_tensor_in index, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& src, const TensorLayout& rois,
            const TensorLayout& index, const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& src, const TensorLayout& rois,
            const TensorLayout& index, const TensorLayout& grad,
            size_t workspace_in_bytes);
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
    CanonizedFilterMeta deduce_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
            TensorLayout& dst) const;
    CanonizedFilterMeta check_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) const;

    static CanonizedFilterMeta make_canonized_filter_meta_impl(
            size_t src_ndim, const TensorLayout& filter, const Param& param);
    CanonizedFilterMeta make_canonized_filter_meta(
            size_t src_ndim, const TensorLayout& filter) const;
};

class Convolution3DForward : public Convolution3DBase,
                             public detail::MultiAlgoOpr<Convolution3DForward, 3> {
    DEF_OPR_IMPL(Convolution3DForward, Convolution3DBase, 2, 1);

public:
    /**
     * \param[in] src (n, ic, id, ih, iw)
     * \param[in] filter (oc, ic, fd, fh, fw)
     * \param[out] dst (n, oc, od, oh, ow)
     */
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::CONVOLUTION3D_FORWARD;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst, size_t workspace_in_bytes);
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
    static void deduce_layout_impl(
            const TensorLayout& filter, const TensorLayout& diff, const Param& param,
            TensorLayout& grad);
    virtual void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& filter, const TensorLayout& diff, TensorLayout& grad);

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::CONVOLUTION3D_BACKWARD_DATA;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_in_bytes);
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::CONVOLUTION3D_BACKWARD_FILTER;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad,
            size_t workspace_in_bytes);
};

class LocalShareBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(LocalShareBase, OperatorBase);
    DEF_OPR_PARAM(LocalShare);

protected:
    void deduce_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst);
    void check_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    /**
     * \brief deduce layout of the ouput tensor
     */
    void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::LOCAL_SHARE_FORWARD;
    }

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst, size_t workspace_in_bytes);
};
using LocalShare = LocalShareForward;

class LocalShareBackwardData : public LocalShareBase,
                               public detail::MultiAlgoOpr<LocalShareBackwardData, 3> {
    DEF_OPR_IMPL(LocalShareBackwardData, LocalShareBase, 2, 1);

public:
    /**
     * \param[in] filter (G, spatial_groups_h, spatial_groups_w, IC / G,
     * FH, FW, OC / G)
     * \param[in] diff (N, OC, OH, OW)
     * \param[out] grad (N, IC, IH, IW)
     */
    virtual void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) = 0;
    void deduce_layout(
            const TensorLayout& filter, const TensorLayout& diff, TensorLayout& grad);

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::LOCAL_SHARE_BACKWARD_DATA;
    }

protected:
    void check_exec(
            const TensorLayout& filter, const TensorLayout& diff,
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;

    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::LOCAL_SHARE_BACKWARD_FILTER;
    }

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad,
            size_t workspace_in_bytes);
};

class ROIAlignBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(ROIAlignBase, OperatorBase);
    DEF_OPR_PARAM(ROIAlign);

protected:
    void deduce_layout_fwd(
            const TensorLayout& src, const TensorLayout& rois, TensorLayout& dst,
            TensorLayout& index);
    void check_layout_fwd(
            const TensorLayout& src, const TensorLayout& rois, const TensorLayout& dst,
            const TensorLayout& index);
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
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in rois, _megdnn_tensor_out dst,
            _megdnn_tensor_out index, _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& src, const TensorLayout& rois, TensorLayout& dst,
            TensorLayout& index);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& rois, const TensorLayout& dst,
            const TensorLayout& index) = 0;

protected:
    void check_exec(
            const TensorLayout& src, const TensorLayout& rois, const TensorLayout& dst,
            const TensorLayout& index, size_t workspace_in_bytes);
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
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in rois, _megdnn_tensor_in index,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& rois,
            const TensorLayout& index, const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& rois,
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
    void deduce_layout_fwd(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& mask, const TensorLayout& offset, TensorLayout& dst);
    void check_layout_fwd(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& mask, const TensorLayout& offset,
            const TensorLayout& dst);
};

class DeformableConvForward : public DeformableConvBase,
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
    virtual void exec(
            _megdnn_tensor_in im, _megdnn_tensor_in filter, _megdnn_tensor_in offset,
            _megdnn_tensor_in mask, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask, TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& dst) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::DEFORMABLE_CONV_FORWARD;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& dst, size_t workspace_in_bytes);
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
    virtual void exec(
            _megdnn_tensor_in im, _megdnn_tensor_in offset, _megdnn_tensor_in mask,
            _megdnn_tensor_in out_grad, _megdnn_tensor_out filter_grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            const TensorLayout& filter_grad) = 0;
    void deduce_layout(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            TensorLayout& filter_grad);

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::DEFORMABLE_CONV_BACKWARD_FILTER;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            const TensorLayout& filter_grad, size_t workspace_in_bytes);
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
    virtual void exec(
            _megdnn_tensor_in im, _megdnn_tensor_in filter, _megdnn_tensor_in offset,
            _megdnn_tensor_in mask, _megdnn_tensor_in out_grad,
            _megdnn_tensor_out im_grad, _megdnn_tensor_out offset_grad,
            _megdnn_tensor_out mask_grad, _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad) = 0;
    void deduce_layout(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, TensorLayout& im_grad,
            TensorLayout& offset_grad, TensorLayout& mask_grad);

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::DEFORMABLE_CONV_BACKWARD_DATA;
    }

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
    void deduce_layout_fwd(
            const TensorLayout& data, const TensorLayout& trans,
            const TensorLayout& rois, TensorLayout& out_data, TensorLayout& out_count);

    void check_layout_fwd(
            const TensorLayout& data, const TensorLayout& trans,
            const TensorLayout& rois, const TensorLayout& out_data,
            const TensorLayout& out_count, size_t workspace_in_bytes);
};

class DeformablePSROIPoolingForward : public DeformablePSROIPoolingBase {
    DEF_OPR_IMPL(DeformablePSROIPoolingForward, DeformablePSROIPoolingBase, 3, 2);

public:
    /**
     * \param[in]  data       (oc, ic, ih, iw)
     * \param[in]  rois       (xx, xx, xx, xx)
     * \param[in]  trans      (oc, ic, fh, fw)
     * \param[out] out_data   ( n, ic, ih, iw)
     * \param[out] out_count  ( n, ic, ih, iw)
     */
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& data, const TensorLayout& rois,
            const TensorLayout& trans, const TensorLayout& out_data,
            const TensorLayout& out_count) = 0;
    virtual void exec(
            _megdnn_tensor_in data, _megdnn_tensor_in rois, _megdnn_tensor_in trans,
            _megdnn_tensor_out out_data, _megdnn_tensor_out out_count,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& data, const TensorLayout& rois,
            const TensorLayout& trans, TensorLayout& out_data, TensorLayout& out_count);
    void check_exec(
            const TensorLayout& data, const TensorLayout& rois,
            const TensorLayout& trans, const TensorLayout& out_data,
            const TensorLayout& out_count, size_t workspace_in_bytes);
};

using DeformablePSROIPooling = DeformablePSROIPoolingForward;

class DeformablePSROIPoolingBackward : public DeformablePSROIPoolingBase {
    DEF_OPR_IMPL(DeformablePSROIPoolingBackward, DeformablePSROIPoolingBase, 5, 2);

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
    virtual void exec(
            _megdnn_tensor_in data, _megdnn_tensor_in rois, _megdnn_tensor_in trans,
            _megdnn_tensor_in out_diff, _megdnn_tensor_in out_count,
            _megdnn_tensor_out data_diff, _megdnn_tensor_out trans_diff,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& data, const TensorLayout& rois,
            const TensorLayout& trans, const TensorLayout& out_diff,
            const TensorLayout& out_count, const TensorLayout& data_diff,
            const TensorLayout& trans_diff) = 0;

    void check_exec(
            const TensorLayout& data, const TensorLayout& rois,
            const TensorLayout& trans, const TensorLayout& out_diff,
            const TensorLayout& out_count, const TensorLayout& data_diff,
            const TensorLayout& trans_diff, size_t workspace_in_bytes);
};

class BatchConvBiasForward : public ConvolutionBase<param::BatchConvBias>,
                             public detail::MultiAlgoOpr<BatchConvBiasForward, 5> {
    DEF_OPR_IMPL(BatchConvBiasForward, ConvolutionBase, 4, 1);

public:
    virtual void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
            _megdnn_tensor_in z, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) = 0;

    void deduce_dtype(DType src, DType filter, DType bias, DType z, DType& dst);
    void deduce_layout(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, TensorLayout& dst);

    virtual size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) = 0;

    static Algorithm::OprType get_opr_type() {
        return Algorithm::OprType::BATCH_CONV_FORWARD;
    }

protected:
    CanonizedFilterMeta check_exec(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
            size_t workspace_in_bytes);
};
using BatchConvBias = BatchConvBiasForward;

class FakeQuantBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(FakeQuantBase, OperatorBase);
    DEF_OPR_PARAM(FakeQuant);

protected:
    void deduce_layout_fwd(const TensorLayout& input, TensorLayout& output);
    void check_layout_fwd(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, const TensorLayout& output);
};

class FakeQuantForward : public FakeQuantBase {
    DEF_OPR_IMPL(FakeQuantForward, FakeQuantBase, 3, 1);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out output,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, TensorLayout& output);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, const TensorLayout& output) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, const TensorLayout& output,
            size_t workspace_in_bytes);
};

using FakeQuant = FakeQuantForward;

class FakeQuantBackward : public FakeQuantBase {
    DEF_OPR_IMPL(FakeQuantBackward, FakeQuantBase, 4, 1);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& input,
            const TensorLayout& scale, const TensorLayout& zero_point,
            const TensorLayout& grad) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& input,
            const TensorLayout& scale, const TensorLayout& zero_point,
            const TensorLayout& grad, size_t workspace_in_bytes);
};

class TQTBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(TQTBase, OperatorBase);
    DEF_OPR_PARAM(TQT);

protected:
    void deduce_layout_fwd(const TensorLayout& input, TensorLayout& output);
    void check_layout_fwd(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& output);
};

class TQTForward : public TQTBase {
    DEF_OPR_IMPL(TQTForward, TQTBase, 2, 1);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale, _megdnn_tensor_out output,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& input, const TensorLayout& scale, TensorLayout& output);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& output) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& output, size_t workspace_in_bytes);
};
using TQT = TQTForward;

class TQTBackward : public TQTBase {
    DEF_OPR_IMPL(TQTBackward, TQTBase, 3, 2);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& input,
            const TensorLayout& scale, const TensorLayout& grad_x,
            const TensorLayout& grad_s) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& input,
            const TensorLayout& scale, const TensorLayout& grad_x,
            const TensorLayout& grad_s, size_t workspace_in_bytes);
};

class LSQBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(LSQBase, OperatorBase);
    DEF_OPR_PARAM(LSQ);

protected:
    void deduce_layout_fwd(const TensorLayout& input, TensorLayout& output);
    void check_layout_fwd(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, const TensorLayout& grad_scale,
            const TensorLayout& output);
};

class LSQForward : public LSQBase {
    DEF_OPR_IMPL(LSQForward, LSQBase, 4, 1);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out output, _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, const TensorLayout& grad_scale,
            TensorLayout& output);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, const TensorLayout& grad_scale,
            const TensorLayout& output) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& scale,
            const TensorLayout& zero_point, const TensorLayout& grad_scale,
            const TensorLayout& output, size_t workspace_in_bytes);
};
using LSQ = LSQForward;

class LSQBackward : public LSQBase {
    DEF_OPR_IMPL(LSQBackward, LSQBase, 5, 2);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in input, _megdnn_tensor_in scale,
            _megdnn_tensor_in zero_point, _megdnn_tensor_in grad_scale,
            _megdnn_tensor_out grad_x, _megdnn_tensor_out grad_s,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& input,
            const TensorLayout& scale, const TensorLayout& zero_point,
            const TensorLayout& grad_scale, const TensorLayout& grad_x,
            const TensorLayout& grad_s) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& input,
            const TensorLayout& scale, const TensorLayout& zero_point,
            const TensorLayout& grad_scale, const TensorLayout& grad_x,
            const TensorLayout& grad_s, size_t workspace_in_bytes);
};

class LayerNormBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(LayerNormBase, OperatorBase);
    DEF_OPR_PARAM(LayerNorm);

public:
    MGE_WIN_DECLSPEC_FUC static void deduce_layout_fwd_impl(
            const TensorLayout& data, const Param& p, TensorLayout& dst,
            TensorLayout& mean, TensorLayout& rstd);

protected:
    void deduce_layout_fwd(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, TensorLayout& dst, TensorLayout& mean,
            TensorLayout& rstd);
    void check_layout_fwd(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd);
};

class LayerNormForward : public LayerNormBase {
    DEF_OPR_IMPL(LayerNormForward, LayerNormBase, 3, 3);

public:
    virtual void exec(
            _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
            _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, TensorLayout& dst, TensorLayout& mean,
            TensorLayout& rstd);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd) = 0;

protected:
    void check_exec(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd, size_t workspace_in_bytes);
};
using LayerNorm = LayerNormForward;

class LayerNormBackward : public LayerNormBase {
    DEF_OPR_IMPL(LayerNormBackward, LayerNormBase, 5, 3);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
            _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
            _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, TensorLayout& ddata, TensorLayout& dweight,
            TensorLayout& dbias);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, const TensorLayout& ddata,
            const TensorLayout& dweight, const TensorLayout& dbias) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, const TensorLayout& ddata,
            const TensorLayout& dweight, const TensorLayout& dbias,
            size_t workspace_in_bytes);
};

class GeneralNormBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(GeneralNormBase, OperatorBase);
    DEF_OPR_PARAM(GeneralNorm);

public:
    MGE_WIN_DECLSPEC_FUC static void deduce_layout_fwd_impl(
            const TensorLayout& data, const Param& p, TensorLayout& dst,
            TensorLayout& mean, TensorLayout& rstd);

protected:
    void deduce_layout_fwd(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, TensorLayout& dst, TensorLayout& mean,
            TensorLayout& rstd);
    void check_layout_fwd(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd);
};

class GeneralNormForward : public GeneralNormBase {
    DEF_OPR_IMPL(GeneralNormForward, GeneralNormBase, 3, 3);

public:
    virtual void exec(
            _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
            _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, TensorLayout& dst, TensorLayout& mean,
            TensorLayout& rstd);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd) = 0;

protected:
    void check_exec(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd, size_t workspace_in_bytes);
};
using GeneralNorm = GeneralNormForward;

class GeneralNormBackward : public GeneralNormBase {
    DEF_OPR_IMPL(GeneralNormBackward, GeneralNormBase, 5, 3);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
            _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
            _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, TensorLayout& ddata, TensorLayout& dweight,
            TensorLayout& dbias);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, const TensorLayout& ddata,
            const TensorLayout& dweight, const TensorLayout& dbias) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, const TensorLayout& ddata,
            const TensorLayout& dweight, const TensorLayout& dbias,
            size_t workspace_in_bytes);
};

class DropoutBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(DropoutBase, OperatorBase);
    DEF_OPR_PARAM(Dropout);
};

class DropoutForward : public DropoutBase {
    DEF_OPR_IMPL(DropoutForward, DropoutBase, 1, 2);

public:
    void deduce_layout(const TensorLayout& inp, TensorLayout& oup, TensorLayout& mask);
    virtual void exec(
            _megdnn_tensor_in inp, _megdnn_tensor_out oup, _megdnn_tensor_out mask,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& inp, const TensorLayout& oup,
            const TensorLayout& mask) = 0;
    virtual size_t get_mask_size_in_bytes(const TensorLayout& inp) = 0;

protected:
    void check_exec(
            const TensorLayout& inp, const TensorLayout& oup, const TensorLayout& mask,
            size_t workspace_in_bytes);
};
using Dropout = DropoutForward;

class DropoutBackward : public DropoutBase {
    DEF_OPR_IMPL(DropoutBackward, DropoutBase, 2, 1);

public:
    void deduce_layout(
            const TensorLayout& doup, const TensorLayout& mask, TensorLayout& dinp);
    virtual void exec(
            _megdnn_tensor_in doup, _megdnn_tensor_in mask, _megdnn_tensor_out dinp,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& doup, const TensorLayout& mask,
            const TensorLayout& dinp) = 0;

protected:
    void check_exec(
            const TensorLayout& doup, const TensorLayout& mask,
            const TensorLayout& dinp, size_t workspace_in_bytes);
};
class SoftmaxBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(SoftmaxBase, OperatorBase);
    DEF_OPR_PARAM(Softmax);

protected:
    void deduce_layout_fwd(const TensorLayout& input, TensorLayout& output);
    void check_layout_fwd(const TensorLayout& input, const TensorLayout& output);
};

class SoftmaxForward : public SoftmaxBase {
    DEF_OPR_IMPL(SoftmaxForward, SoftmaxBase, 1, 1);

public:
    /**
     * \param[in] input input tensor
     * \param[out] output output tensor
     */
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_out output,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(const TensorLayout& input, TensorLayout& output);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& output) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& output,
            size_t workspace_in_bytes);
};
using Softmax = SoftmaxForward;

class SoftmaxBackward : public SoftmaxBase {
    DEF_OPR_IMPL(SoftmaxBackward, SoftmaxBase, 2, 1);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in diff, _megdnn_tensor_out grad_x,
            _megdnn_workspace workspace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& diff,
            const TensorLayout& grad_x) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& diff,
            const TensorLayout& grad_x, size_t workspace_in_bytes);
};

class RNNCellForward : public OperatorBase {
    DEF_OPR_PARAM(RNNCell);
    DEF_OPR_IMPL(RNNCellForward, OperatorBase, 6, 1);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
            _megdnn_tensor_in bias_ih, _megdnn_tensor_in hx,
            _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            TensorLayout& dst);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& dst) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& dst, size_t workspace_in_bytes);
};
using RNNCell = RNNCellForward;

class LSTMCellForward : public OperatorBase {
    // DEF_OPR_PARAM(LSTMCell);
    DEF_OPR_PARAM(Empty);
    DEF_OPR_IMPL(LSTMCellForward, OperatorBase, 7, 3);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in weight_ih,
            _megdnn_tensor_in bias_ih, _megdnn_tensor_in hx,
            _megdnn_tensor_in weight_hh, _megdnn_tensor_in bias_hh,
            _megdnn_tensor_in cx, _megdnn_tensor_out h_new, _megdnn_tensor_out c_new,
            _megdnn_tensor_out gates, _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& cx, TensorLayout& h_new, TensorLayout& c_new,
            TensorLayout& gates);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& cx, const TensorLayout& h_new,
            const TensorLayout& c_new, const TensorLayout& gates) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& weight_ih,
            const TensorLayout& bias_ih, const TensorLayout& hx,
            const TensorLayout& weight_hh, const TensorLayout& bias_hh,
            const TensorLayout& cx, const TensorLayout& h_new,
            const TensorLayout& c_new, const TensorLayout& gates,
            size_t workspace_in_bytes);
};
using LSTMCell = LSTMCellForward;

class RNNForward : public OperatorBase {
    DEF_OPR_PARAM(RNN);
    DEF_OPR_IMPL(RNNForward, OperatorBase, 3, 3);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in hx,
            _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
            _megdnn_tensor_out hy, _megdnn_tensor_out reserve_space,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& input, const TensorLayout& hx,
            const TensorLayout& flatten_weights, TensorLayout& output, TensorLayout& hy,
            TensorLayout& reserve_space);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& hx,
            const TensorLayout& flatten_weights, const TensorLayout& output,
            const TensorLayout& hy, const TensorLayout& reserve_space) = 0;
    virtual size_t get_reserve_size_in_bytes(const TensorLayout& input) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& hx,
            const TensorLayout& flatten_weights, const TensorLayout& output,
            const TensorLayout& hy, const TensorLayout& reserve_space,
            size_t workspace_in_bytes);
};
using RNN = RNNForward;

class RNNBackward : public OperatorBase {
    DEF_OPR_PARAM(RNN);
    DEF_OPR_IMPL(RNNBackward, OperatorBase, 7, 3);

public:
    virtual void exec(
            _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in hx,
            _megdnn_tensor_in dy, _megdnn_tensor_in dhy,
            _megdnn_tensor_in flatten_weights, _megdnn_tensor_in reserve_space,
            _megdnn_tensor_out dx, _megdnn_tensor_out dhx, _megdnn_tensor_out dw,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
            const TensorLayout& dy, const TensorLayout& dhy,
            const TensorLayout& flatten_weights, const TensorLayout& reserve_space,
            TensorLayout& dx, TensorLayout& dhx, TensorLayout& dw);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
            const TensorLayout& dy, const TensorLayout& dhy,
            const TensorLayout& flatten_weights, const TensorLayout& reserve_space,
            const TensorLayout& dx, const TensorLayout& dhx,
            const TensorLayout& dw) = 0;

protected:
    void check_exec(
            const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
            const TensorLayout& dy, const TensorLayout& dhy,
            const TensorLayout& flatten_weights, const TensorLayout& reserve_space,
            const TensorLayout& dx, const TensorLayout& dhx, const TensorLayout& dw,
            size_t workspace_in_bytes);
};

class LSTMForward : public OperatorBase {
    DEF_OPR_PARAM(LSTM);
    DEF_OPR_IMPL(LSTMForward, OperatorBase, 4, 4);

public:
    virtual void exec(
            _megdnn_tensor_in input, _megdnn_tensor_in hx, _megdnn_tensor_in cx,
            _megdnn_tensor_in flatten_weights, _megdnn_tensor_out output,
            _megdnn_tensor_out hy, _megdnn_tensor_out cy,
            _megdnn_tensor_out reserve_space, _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
            const TensorLayout& flatten_weights, TensorLayout& output, TensorLayout& hy,
            TensorLayout& cy, TensorLayout& reserve_space);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
            const TensorLayout& flatten_weights, const TensorLayout& output,
            const TensorLayout& hy, const TensorLayout& cy,
            const TensorLayout& reserve_space) = 0;
    virtual size_t get_reserve_size_in_bytes(const TensorLayout& input) = 0;

protected:
    void check_exec(
            const TensorLayout& input, const TensorLayout& hx, const TensorLayout& cx,
            const TensorLayout& flatten_weights, const TensorLayout& output,
            const TensorLayout& hy, const TensorLayout& cy,
            const TensorLayout& reserve_space, size_t workspace_in_bytes);
};
using LSTM = LSTMForward;

class LSTMBackward : public OperatorBase {
    DEF_OPR_PARAM(LSTM);
    DEF_OPR_IMPL(LSTMBackward, OperatorBase, 9, 4);

public:
    virtual void exec(
            _megdnn_tensor_in x, _megdnn_tensor_in y, _megdnn_tensor_in hx,
            _megdnn_tensor_in cx, _megdnn_tensor_in dy, _megdnn_tensor_in dhy,
            _megdnn_tensor_in dcy, _megdnn_tensor_in flatten_weights,
            _megdnn_tensor_in reserve_space, _megdnn_tensor_out dx,
            _megdnn_tensor_out dhx, _megdnn_tensor_out dcx, _megdnn_tensor_out dw,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
            const TensorLayout& cx, const TensorLayout& dy, const TensorLayout& dhy,
            const TensorLayout& dcy, const TensorLayout& flatten_weights,
            const TensorLayout& reserve_space, TensorLayout& dx, TensorLayout& dhx,
            TensorLayout& dcx, TensorLayout& dw);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
            const TensorLayout& cx, const TensorLayout& dy, const TensorLayout& dhy,
            const TensorLayout& dcy, const TensorLayout& flatten_weights,
            const TensorLayout& reserve_space, const TensorLayout& dx,
            const TensorLayout& dhx, const TensorLayout& dcx,
            const TensorLayout& dw) = 0;

protected:
    void check_exec(
            const TensorLayout& x, const TensorLayout& y, const TensorLayout& hx,
            const TensorLayout& cx, const TensorLayout& dy, const TensorLayout& dhy,
            const TensorLayout& dcy, const TensorLayout& flatten_weights,
            const TensorLayout& reserve_space, const TensorLayout& dx,
            const TensorLayout& dhx, const TensorLayout& dcx, const TensorLayout& dw,
            size_t workspace_in_bytes);
};

class GroupNormBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(GroupNormBase, OperatorBase);
    DEF_OPR_PARAM(GroupNorm);

protected:
    void deduce_layout_fwd(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, TensorLayout& dst, TensorLayout& mean,
            TensorLayout& rstd);
    void check_layout_fwd(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd);
};

class GroupNormForward : public GroupNormBase {
    DEF_OPR_IMPL(GroupNormForward, GroupNormBase, 3, 3);

public:
    virtual void exec(
            _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
            _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, TensorLayout& dst, TensorLayout& mean,
            TensorLayout& rstd);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd) = 0;

protected:
    void check_exec(
            const TensorLayout& data, const TensorLayout& weight,
            const TensorLayout& bias, const TensorLayout& dst, const TensorLayout& mean,
            const TensorLayout& rstd, size_t workspace_in_bytes);
};
using GroupNorm = GroupNormForward;

class GroupNormBackward : public GroupNormBase {
    DEF_OPR_IMPL(GroupNormBackward, GroupNormBase, 5, 3);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
            _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
            _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
            _megdnn_workspace workspace) = 0;
    void deduce_layout(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, TensorLayout& ddata, TensorLayout& dweight,
            TensorLayout& dbias);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, const TensorLayout& ddata,
            const TensorLayout& dweight, const TensorLayout& dbias) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& data,
            const TensorLayout& weight, const TensorLayout& mean,
            const TensorLayout& rstd, const TensorLayout& ddata,
            const TensorLayout& dweight, const TensorLayout& dbias,
            size_t workspace_in_bytes);
};

class MultiHeadAttnBase : public OperatorBase {
    DEF_OPR_IMPL_CTOR(MultiHeadAttnBase, OperatorBase);
    DEF_OPR_PARAM(MultiHeadAttn);
};

class MultiHeadAttnForward : public MultiHeadAttnBase {
    DEF_OPR_IMPL(MultiHeadAttnForward, MultiHeadAttnBase, 7, 4);

public:
    /**
     * \param[in] queries (N, L, E_q), where N is the batch size, L is the target
     * sequence length, and E_q is the query embedding dimension embed_dim.
     * \param[in] keys (N, S, E_k), where N is the batch size, S is the source
     * sequence length, and E_k is the key embedding dimension k_dim.
     * \param[in] values (N, S, E_v), where N is the batch size, S is the source
     * sequence length, and E_v is the value embedding dimension v_dim.
     * \param[in] qkvo_weight_bias, input/output projection weight/bias all in one.
     * The order of arrangement is: query weight, key weight, value weight,
     * out weight, query bias, key bias, value bias, out bias, the following parameters
     * in param will be used to indicate whether these items exist: qproj_size,
     * kproj_size, vproj_size, oproj_size, qbias, kbias, vbias, obias.
     * Note: Y=X@W+B is used here instead of Y=X@W^T+B in pytorch.
     * \param[in] attn_mask, (N*num_heads, L, S) or (L, S), where N is the batch size,
     * num_heads is the number of parallel attention heads, L is the target sequence
     * length, and S is the source sequence length. attention mask is obtained by
     * combining attn_mask, key_padding_mask, is_causal and maybe_cudnn_style_mask by
     * mge.functional._merge_masks.
     * \param[in] bias_k, (1, 1, kproj_size), where kproj_size is the projected
     * dimension of key weight, if kproj_size == 0, will be the key embedding dimension
     * k_dim.
     * Note: bias_k and bias_v are the bias of the K and V sequences to be added at
     * sequence dim, distinguished from kbias and vbias, bias_kv here is not kbias and
     * vbias in the linear layer, and bias_kv here will be added to the K and V at
     * sequence dimensions, where K and V are the matrices of key and value after
     * projection, and K and V will be used to calculate the attention matrix.
     * \param[in] bias_v, (1, 1, vproj_size), where vproj_size is the projected
     * dimension of value weight, if vproj_size == 0, will be the value embedding
     * dimension v_dim.
     * Note: see bias_k.
     * \param[out] out, (N, S, oproj_size), where N is
     * the batch size, S is the source sequence length, and oproj_size is the projected
     * dimension of output weight, if oproj_size == 0, will be the projected
     * dimension of value weight vproj_size, but if vproj_size == 0, will be the value
     * embedding dimension v_dim.
     * \param[out] attn_weight, (N * num_heads, L, S), where N is the batch size,
     * num_heads is the number of parallel attention heads, L is the target sequence
     * length, and S is the source sequence length.
     * Note: attn_weight is the output of softmax.
     * \param[out] mask_reservespace, when param.training=true, we need this output to
     * save the mask of attention dropout and output dropout.
     * \param[out] othr_reservespace, when param.training=true, we need this output to
     * save the intermediate calculation results.
     */
    virtual void exec(
            _megdnn_tensor_in queries, _megdnn_tensor_in keys, _megdnn_tensor_in values,
            _megdnn_tensor_in qkvo_weight_bias, _megdnn_tensor_in attn_mask,
            _megdnn_tensor_in bias_k, _megdnn_tensor_in bias_v, _megdnn_tensor_out out,
            _megdnn_tensor_out attn_weight, _megdnn_tensor_out mask_reservespace,
            _megdnn_tensor_out othr_reservespace, _megdnn_workspace workspace) = 0;
    virtual void deduce_layout(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, TensorLayout& out, TensorLayout& attn_weight,
            TensorLayout& mask_reservespace, TensorLayout& othr_reservespace) = 0;
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, const TensorLayout& out,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace) = 0;
    virtual size_t get_mask_reservespace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, const TensorLayout& out,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace) = 0;
    virtual size_t get_othr_reservespace_in_bytes(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, const TensorLayout& out,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace) = 0;

protected:
    void check_exec(
            const TensorLayout& queries, const TensorLayout& keys,
            const TensorLayout& values, const TensorLayout& qkvo_weight_bias,
            const TensorLayout& attn_mask, const TensorLayout& bias_k,
            const TensorLayout& bias_v, const TensorLayout& out,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace, size_t workspace_in_bytes);
};
using MultiHeadAttn = MultiHeadAttnForward;

class MultiHeadAttnBackward : public MultiHeadAttnBase {
    DEF_OPR_IMPL(MultiHeadAttnBackward, MultiHeadAttnBase, 9, 6);

public:
    virtual void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in queries, _megdnn_tensor_in keys,
            _megdnn_tensor_in values, _megdnn_tensor_in qkvo_weight_bias,
            _megdnn_tensor_in attn_mask, _megdnn_tensor_in attn_weight,
            _megdnn_tensor_in mask_reservespace, _megdnn_tensor_in othr_reservespace,
            _megdnn_tensor_out dqueries, _megdnn_tensor_out dkeys,
            _megdnn_tensor_out dvalues, _megdnn_tensor_out dqkvo_weight_bias,
            _megdnn_tensor_out dbias_k, _megdnn_tensor_out dbias_v,
            _megdnn_workspace workspace) = 0;
    MGE_WIN_DECLSPEC_FUC void deduce_layout(
            const TensorLayout& diff, const TensorLayout& queries,
            const TensorLayout& keys, const TensorLayout& values,
            const TensorLayout& qkvo_weight_bias, const TensorLayout& attn_mask,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace, TensorLayout& dqueries,
            TensorLayout& dkeys, TensorLayout& dvalues, TensorLayout& dqkvo_weight_bias,
            TensorLayout& dbias_k, TensorLayout& dbias_v);
    virtual size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& queries,
            const TensorLayout& keys, const TensorLayout& values,
            const TensorLayout& qkvo_weight_bias, const TensorLayout& attn_mask,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace, const TensorLayout& dqueries,
            const TensorLayout& dkeys, const TensorLayout& dvalues,
            const TensorLayout& dqkvo_weight_bias, const TensorLayout& dbias_k,
            const TensorLayout& dbias_v) = 0;

protected:
    void check_exec(
            const TensorLayout& diff, const TensorLayout& queries,
            const TensorLayout& keys, const TensorLayout& values,
            const TensorLayout& qkvo_weight_bias, const TensorLayout& attn_mask,
            const TensorLayout& attn_weight, const TensorLayout& mask_reservespace,
            const TensorLayout& othr_reservespace, const TensorLayout& dqueries,
            const TensorLayout& dkeys, const TensorLayout& dvalues,
            const TensorLayout& dqkvo_weight_bias, const TensorLayout& dbias_k,
            const TensorLayout& dbias_v, size_t workspace_in_bytes);
};
}  // namespace megdnn
#include "megdnn/internal/opr_header_epilogue.h"

// vim: syntax=cpp.doxygen
