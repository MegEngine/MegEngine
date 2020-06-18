/**
 * \file dnn/src/fallback/conv_bias/conv1x1/conv1x1_utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/conv1x1/conv1x1_utils.h"

namespace megdnn {
namespace fallback {
namespace conv1x1 {
namespace utils{

//! get_thread_bundle
WorkspaceBundle get_thread_bundle(const ConvBiasImpl::NCBKernSizeParam& param,
                                  size_t matmul_c_size, size_t oc_tile_size) {
    //! for some cases, matmul result need temp space to store
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    bool is_dst_8bit = (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                        param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                       (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                        param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
    size_t matmul_dst_bytes_per_thread =
            is_dst_8bit ? oc_tile_size * OH * OW * sizeof(param.bias_type) : 0;
    return WorkspaceBundle{nullptr,
                           {matmul_c_size, matmul_dst_bytes_per_thread}};
}

//! get_matmul_kern_param
MatrixMulImpl::KernSizeParam get_matmul_kern_param(
        const ConvBiasImpl::NCBKernSizeParam& param, size_t n, size_t m) {
    size_t M = m;
    size_t N = n;
    size_t K = param.filter_meta.icpg;  //! K = IC
    size_t LDA = K, LDB = N, LDC = N;
    bool is_dst_8bit = (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                        param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                       (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                        param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
    size_t pack_c_size = pack_size(param.filter_meta.format);
    auto format = param::MatrixMul::Format::DEFAULT;
    if (param.filter_meta.format == param::ConvBias::Format::NCHW44) {
        format = param::MatrixMul::Format::MK4;
    } else if (param.filter_meta.format ==
               param::ConvBias::Format::NCHW44_DOT) {
        format = param::MatrixMul::Format::MK4_DOT;
    }

    return {param.filter_type,
            param.src_type,
            is_dst_8bit ? param.bias_type : param.dst_type,
            M,
            N,
            K,
            LDA * pack_c_size,
            LDB * pack_c_size,
            LDC * pack_c_size,
            false,
            false,
            param::MatrixMul::ComputeMode::DEFAULT,
            format};
}

}  // namespace utils
}  // namespace conv1x1
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen