/**
 * \file dnn/src/x86/conv_bias/int8/algo_usable_preferred.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/x86/conv_bias/int8/algo_usable_preferred.h"
#include "src/x86/utils.h"

#if MEGDNN_X86_WITH_MKL_DNN
#include <mkldnn.hpp>
#endif

#include <cstring>

#if MEGDNN_X86_WITH_MKL_DNN
using namespace dnnl;
#endif
using namespace megdnn;
using namespace x86;

namespace megdnn {
namespace x86 {

bool chanwise_avx2_stride1_qint8_usable(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool aviliable =
            (param.bias_mode != BiasMode::BIAS) &&
            ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
             (((param.src_type.enumv() == DTypeEnum::Int8 &&
                param.filter_type.enumv() == DTypeEnum::Int8 &&
                param.dst_type.enumv() == DTypeEnum::Int32) ||
               (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                param.dst_type.enumv() == DTypeEnum::QuantizedS32)))) &&
            fm.format == ConvBiasImpl::Param::Format::NCHW &&
            fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 && (FH == 2 || FH == 3 || FH == 5 || FH == 7) &&
            fm.stride[0] == 1 && fm.stride[1] == 1 && (fm.icpg == 1) &&
            (fm.ocpg == 1) && is_supported(SIMDType::AVX2);
    return aviliable;
}

bool chanwise_avx2_stride1_qint8_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    MEGDNN_MARK_USED_VAR(param);
    return true;
}

bool chanwise_avx2_stride1_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return chanwise_avx2_stride1_qint8_usable(param) &&
           chanwise_avx2_stride1_qint8_preferred(param);
}

bool chanwise_avx2_stride2_qint8_usable(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool aviliable =
            (param.bias_mode != BiasMode::BIAS) &&
            ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
             (((param.src_type.enumv() == DTypeEnum::Int8 &&
                param.filter_type.enumv() == DTypeEnum::Int8 &&
                param.dst_type.enumv() == DTypeEnum::Int32) ||
               (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                param.dst_type.enumv() == DTypeEnum::QuantizedS32)))) &&
            fm.format == ConvBiasImpl::Param::Format::NCHW &&
            fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 && (FH == 2 || FH == 3 || FH == 5 || FH == 7) &&
            fm.stride[0] == 2 && fm.stride[1] == 2 && (fm.icpg == 1) &&
            (fm.ocpg == 1) && is_supported(SIMDType::AVX2);
    return aviliable;
}

bool chanwise_avx2_stride2_qint8_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    MEGDNN_MARK_USED_VAR(param);
    return true;
}

bool chanwise_avx2_stride2_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return chanwise_avx2_stride2_qint8_usable(param) &&
           chanwise_avx2_stride2_qint8_preferred(param);
}

bool direct_avx2_stride1_int8_usable(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool aviliable = ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                      (((param.src_type.enumv() == DTypeEnum::Int8 &&
                         param.filter_type.enumv() == DTypeEnum::Int8 &&
                         param.dst_type.enumv() == DTypeEnum::Int32) ||
                        (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.dst_type.enumv() == DTypeEnum::QuantizedS32)) &&
                       param.bias_mode == BiasMode::NO_BIAS &&
                       param.nonlineMode == NonlineMode::IDENTITY)) &&
                     fm.format == ConvBiasImpl::Param::Format::NCHW &&
                     fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
                     fm.dilation[1] == 1 &&
                     (FH == 2 || FH == 3 || FH == 5 || FH == 7) &&
                     fm.stride[0] == 1 && fm.stride[1] == 1 &&
                     is_supported(SIMDType::AVX2);
    return aviliable;
}

bool direct_avx2_stride1_int8_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto IC = fm.icpg;
    auto OC = fm.ocpg;
    auto is_preferred = true;
    if (IC > 128 && OC > 128)
        is_preferred = false;

    return is_preferred;
}

bool direct_avx2_stride1_int8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return direct_avx2_stride1_int8_usable(param) &&
           direct_avx2_stride1_int8_preferred(param);
}

bool direct_avx2_stride2_int8_usable(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool aviliable = ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                       param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                      (((param.src_type.enumv() == DTypeEnum::Int8 &&
                         param.filter_type.enumv() == DTypeEnum::Int8 &&
                         param.dst_type.enumv() == DTypeEnum::Int32) ||
                        (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                         param.dst_type.enumv() == DTypeEnum::QuantizedS32)) &&
                       param.bias_mode == BiasMode::NO_BIAS &&
                       param.nonlineMode == NonlineMode::IDENTITY)) &&
                     fm.format == ConvBiasImpl::Param::Format::NCHW &&
                     fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
                     fm.dilation[1] == 1 &&
                     (FH == 2 || FH == 3 || FH == 5 || FH == 7) &&
                     fm.stride[0] == 2 && fm.stride[1] == 2 &&
                     is_supported(SIMDType::AVX2);
    return aviliable;
}

bool direct_avx2_stride2_int8_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto IC = fm.icpg;
    auto OC = fm.ocpg;
    auto is_preferred = false;
    if (IC <= 31 && OC <= 31)
        is_preferred = true;

    return is_preferred;
}

bool direct_avx2_stride2_int8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return direct_avx2_stride2_int8_usable(param) &&
           direct_avx2_stride2_int8_preferred(param);
}

#if MEGDNN_X86_WITH_MKL_DNN
bool mkldnn_qint8_usable(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    return (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
            param.src_type.enumv() == DTypeEnum::Int8) &&
           (param.dst_type.enumv() == DTypeEnum::QuantizedS32 ||
            param.dst_type.enumv() == DTypeEnum::Int32) &&
           fm.format == param::ConvBias::Format::NCHW && fm.spatial_ndim == 2 &&
           fm.dilation[0] == 1 && fm.dilation[1] == 1 && !fm.should_flip &&
           param.bias_mode == BiasMode::NO_BIAS &&
           param.nonlineMode == NonlineMode::IDENTITY;
}

bool mkldnn_qint8_preferred(const ConvBiasImpl::NCBKernSizeParam& param) {
    MEGDNN_MARK_USED_VAR(param);
    return is_supported(SIMDType::VNNI);
}

bool mkldnn_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return mkldnn_qint8_usable(param) && mkldnn_qint8_preferred(param);
}

bool mkldnn_matmul_qint8_usable(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    return (param.src_type.enumv() == DTypeEnum::QuantizedS8 ||
            param.src_type.enumv() == DTypeEnum::Int8) &&
           (param.dst_type.enumv() == DTypeEnum::QuantizedS32 ||
            param.dst_type.enumv() == DTypeEnum::Int32) &&
           fm.format == param::ConvBias::Format::NCHW && fm.spatial_ndim == 2 &&
           fm.group == 1 && fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
           param.bias_mode == BiasMode::NO_BIAS &&
           param.nonlineMode == NonlineMode::IDENTITY &&
           //! The matmul opr is only used in single thread
           //! TODO:support the no pack matmul algo in fallback im2col + matmul
           param.nr_threads == 1_z;
}

bool mkldnn_matmul_qint8_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto is_preferred = true;
    auto&& fm = param.filter_meta;

    // single channel conv should never use matrix mul
    if (fm.ocpg == 1 || fm.icpg == 1)
        is_preferred = false;

    return is_preferred && is_supported(SIMDType::VNNI);
}

bool mkldnn_matmul_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return mkldnn_matmul_qint8_usable(param) &&
           mkldnn_matmul_qint8_preferred(param);
}
#endif

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
