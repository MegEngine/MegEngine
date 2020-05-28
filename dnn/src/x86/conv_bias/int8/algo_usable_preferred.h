/**
 * \file dnn/src/x86/conv_bias/int8/algo_usable_preferred.h
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

#include "src/common/utils.h"
#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {

bool chanwise_avx2_stride1_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride1_qint8_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride1_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);

bool chanwise_avx2_stride2_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride2_qint8_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride2_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);

bool direct_avx2_stride1_int8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride1_int8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride1_int8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);

bool direct_avx2_stride2_int8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride2_int8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride2_int8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);

#if MEGDNN_X86_WITH_MKL_DNN
bool mkldnn_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_qint8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_qint8_usable_preferred(const ConvBiasImpl::NCBKernSizeParam&);

bool mkldnn_matmul_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_matmul_qint8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_matmul_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);
#endif

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
