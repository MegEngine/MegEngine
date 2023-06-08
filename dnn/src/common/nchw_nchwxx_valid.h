#pragma once
#include "megdnn/oprs.h"

#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
#include "src/fallback/conv_bias/opr_impl.h"
#endif

namespace megdnn {
#if MGE_BUILD_WITHOUT_NAIVE_EXEC
using NonlineMode = ConvBias::Param::NonlineMode;
using BiasMode = ConvBiasForward::BiasMode;
#endif
namespace {
enum NchwNchwxxType {
    NCHW44_FP32,
    NCHW44_INT8,
    NCHW44_INT8_INT8_INT16,
    NCHW44_INT8_DOT,
    NCHW88,
#if !MEGDNN_DISABLE_FLOAT16
    NCHW88_FP16,
#endif
};
template <NchwNchwxxType T>
static inline bool nchw_nchwxx_valid(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode nonline_mode);

template <>
inline bool nchw_nchwxx_valid<NCHW44_FP32>(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode nonline_mode) {
    bool ok_type =
            ((src_dtype == DTypeEnum::Float32 && filter_dtype == DTypeEnum::Float32 &&
              (dst_dtype == DTypeEnum::Float32))) &&
            (fm.format == param::Convolution::Format::NCHW44);
    bool ok_nonline = nonline_mode == param::ConvBias::NonlineMode::IDENTITY ||
                      nonline_mode == param::ConvBias::NonlineMode::RELU ||
                      nonline_mode == param::ConvBias::NonlineMode::H_SWISH;
    bool ok_src_dst =
            fm.icpg < 4 && (fm.ocpg % 4 == 0 && fm.ocpg >= 4) && fm.group == 1;

    bool ok_filter = fm.spatial_ndim == 2 && fm.spatial[0] == fm.spatial[1] &&
                     (fm.spatial[0] == 2 || fm.spatial[0] == 3 || fm.spatial[0] == 5 ||
                      fm.spatial[0] == 7);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 1 || fm.stride[1] == 2);
    bool ok_conv = !fm.should_flip && bias_mode != BiasMode::BIAS;
    bool avaible =
            ok_type && ok_nonline && ok_src_dst && ok_filter && ok_slide && ok_conv;
    return avaible;
}
template <>
inline bool nchw_nchwxx_valid<NCHW44_INT8>(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode nonline_mode) {
    bool ok_type = ((src_dtype == DTypeEnum::QuantizedS8 &&
                     filter_dtype == DTypeEnum::QuantizedS8 &&
                     (dst_dtype == DTypeEnum::QuantizedS8))) &&
                   (fm.format == param::Convolution::Format::NCHW44);
    bool ok_nonline = nonline_mode == param::ConvBias::NonlineMode::IDENTITY ||
                      nonline_mode == param::ConvBias::NonlineMode::RELU ||
                      nonline_mode == param::ConvBias::NonlineMode::H_SWISH;
    bool ok_src_dst =
            fm.icpg < 4 && (fm.ocpg % 4 == 0 && fm.ocpg >= 4) && fm.group == 1;
    bool ok_filter = fm.spatial_ndim == 2 && fm.spatial[0] == fm.spatial[1] &&
                     (fm.spatial[0] == 2 || fm.spatial[0] == 3 || fm.spatial[0] == 5 ||
                      fm.spatial[0] == 7 ||
                      (fm.spatial[0] == 1 && fm.stride[0] == 1 && fm.stride[1] == 1));
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 1 || fm.stride[1] == 2);
    bool ok_conv = !fm.should_flip && bias_mode != BiasMode::BIAS;
    bool avaible =
            ok_type && ok_nonline && ok_src_dst && ok_filter && ok_slide && ok_conv;
    return avaible;
}
template <>
inline bool nchw_nchwxx_valid<NCHW44_INT8_INT8_INT16>(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode nonline_mode) {
    bool ok_type = ((src_dtype == DTypeEnum::Int8 && filter_dtype == DTypeEnum::Int8 &&
                     (dst_dtype == DTypeEnum::Int16))) &&
                   (fm.format == param::Convolution::Format::NCHW44);
    bool ok_nonline = nonline_mode == param::ConvBias::NonlineMode::IDENTITY;
    bool ok_src_dst =
            fm.icpg < 4 && (fm.ocpg % 4 == 0 && fm.ocpg >= 4) && fm.group == 1;
    bool ok_filter = fm.spatial_ndim == 2 && fm.spatial[0] == fm.spatial[1] &&
                     (fm.spatial[0] == 2 || fm.spatial[0] == 3 || fm.spatial[0] == 5 ||
                      fm.spatial[0] == 7);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 2 || fm.stride[0] == 1);
    bool ok_conv = !fm.should_flip && bias_mode != BiasMode::BIAS;
    bool avaible =
            ok_type && ok_nonline && ok_src_dst && ok_filter && ok_slide && ok_conv;
    return avaible;
}
template <>
inline bool nchw_nchwxx_valid<NCHW44_INT8_DOT>(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode nonline_mode) {
    bool ok_type = ((src_dtype == DTypeEnum::QuantizedS8 &&
                     filter_dtype == DTypeEnum::QuantizedS8 &&
                     (dst_dtype == DTypeEnum::QuantizedS8))) &&
                   (fm.format == param::Convolution::Format::NCHW44_DOT);
    bool ok_nonline = nonline_mode == param::ConvBias::NonlineMode::IDENTITY ||
                      nonline_mode == param::ConvBias::NonlineMode::RELU ||
                      nonline_mode == param::ConvBias::NonlineMode::H_SWISH;
    bool ok_src_dst =
            fm.icpg < 4 && (fm.ocpg % 4 == 0 && fm.ocpg >= 4) && fm.group == 1;
    bool ok_filter = fm.spatial_ndim == 2 && fm.spatial[0] == fm.spatial[1] &&
                     (fm.spatial[0] == 2 || fm.spatial[0] == 3 || fm.spatial[0] == 5 ||
                      fm.spatial[0] == 7 ||
                      (fm.spatial[0] == 1 && fm.stride[0] == 1 && fm.stride[1] == 1));
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 1 || fm.stride[1] == 2);
    bool ok_conv = !fm.should_flip && bias_mode != BiasMode::BIAS;
    bool avaible =
            ok_type && ok_nonline && ok_src_dst && ok_filter && ok_slide && ok_conv;
    return avaible;
}

template <>
inline bool nchw_nchwxx_valid<NCHW88>(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode) {
    bool ok_type =
            ((src_dtype == DTypeEnum::Float32 && filter_dtype == DTypeEnum::Float32 &&
              (dst_dtype == DTypeEnum::Float32))) &&
            (fm.format == param::Convolution::Format::NCHW88);
    bool ok_src_dst =
            fm.icpg < 8 && (fm.ocpg % 8 == 0 && fm.ocpg >= 8) && fm.group == 1;
    bool ok_conv = !fm.should_flip && bias_mode != BiasMode::BIAS;
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1;
    bool avaible = ok_type && ok_src_dst && ok_slide && ok_conv;
    return avaible;
}
#if !MEGDNN_DISABLE_FLOAT16
template <>
inline bool nchw_nchwxx_valid<NCHW88_FP16>(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode nonline_mode) {
    bool ok_type =
            ((src_dtype == DTypeEnum::Float16 && filter_dtype == DTypeEnum::Float16 &&
              (dst_dtype == DTypeEnum::Float16))) &&
            (fm.format == param::Convolution::Format::NCHW88);
    bool ok_nonline = nonline_mode == param::ConvBias::NonlineMode::IDENTITY ||
                      nonline_mode == param::ConvBias::NonlineMode::RELU ||
                      nonline_mode == param::ConvBias::NonlineMode::H_SWISH;
    bool ok_src_dst =
            fm.icpg < 8 && (fm.ocpg % 8 == 0 && fm.ocpg >= 8) && fm.group == 1;

    bool ok_filter = fm.spatial_ndim == 2 && fm.spatial[0] == fm.spatial[1] &&
                     (fm.spatial[0] == 2 || fm.spatial[0] == 3 || fm.spatial[0] == 5 ||
                      fm.spatial[0] == 7);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 1 || fm.stride[1] == 2);
    bool ok_conv = !fm.should_flip && bias_mode != BiasMode::BIAS;
    bool avaible =
            ok_type && ok_nonline && ok_src_dst && ok_filter && ok_slide && ok_conv;
    return avaible;
}
#endif

}  // namespace
}  // namespace megdnn
