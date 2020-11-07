/**
 * \file dnn/src/arm_common/convolution/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./int8x8x32/algos.h"
#include "./quint8/algos.h"

#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/common/opr_delegate.h"

using namespace megdnn;
using namespace arm_common;


/* ===================== ConvolutionBackwardData ===================== */
struct ConvolutionBackwardDataImpl::AlgoPack {
#if __ARM_FEATURE_DOTPROD
    AlgoSdot8DirectStride1 i8x8x32_direct_stride1_sdot;
    AlgoSdot8DirectStride2 i8x8x32_direct_stride2_sdot;
    AlgoUdot8DirectStride1 quint8_direct_stride1_udot;
    AlgoUdot8DirectStride2 quint8_direct_stride2_udot;
#endif
};
ConvolutionBackwardDataImpl::AlgoPack ConvolutionBackwardDataImpl::sm_algo_pack;

ConvolutionBackwardDataImpl::ncb_kern_t
ConvolutionBackwardDataImpl::ncb_1g_dispatch_kern(
        Algorithm* algo, const NCBKernSizeParam& param) {
    if (algo->handle_type() == Handle::HandleType::ARM_COMMON) {
        return static_cast<AlgoBase*>(algo)->dispatch_kern(this, param);
    }
    return fallback::ConvolutionBackwardDataImpl::ncb_1g_dispatch_kern(algo,
                                                                       param);
}

size_t ConvolutionBackwardDataImpl::ncb_1g_get_workspace(
        Algorithm* algo, const NCBKernSizeParam& param) {
    if (algo->handle_type() == Handle::HandleType::ARM_COMMON) {
        return static_cast<AlgoBase*>(algo)->get_workspace(this, param);
    }
    return fallback::ConvolutionBackwardDataImpl::ncb_1g_get_workspace(algo,
                                                                       param);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*>
ConvolutionBackwardDataImpl::ncb_1g_get_all_algorithms(
        const NCBKernSizeParam& param) {
    auto ret = fallback::ConvolutionBackwardDataImpl::ncb_1g_get_all_algorithms(
            param);

#if __ARM_FEATURE_DOTPROD
    if ((param.filter_type.enumv() == DTypeEnum::QuantizedS8 ||
         param.filter_type.enumv() == DTypeEnum::Int8) &&
        (param.grad_type.enumv() == DTypeEnum::QuantizedS32 ||
         param.grad_type.enumv() == DTypeEnum::Int32)) {
        if (sm_algo_pack.i8x8x32_direct_stride1_sdot.usable(this, param)) {
            ret.insert(ret.begin(), &sm_algo_pack.i8x8x32_direct_stride1_sdot);
        }
        if (sm_algo_pack.i8x8x32_direct_stride2_sdot.usable(this, param)) {
            ret.insert(ret.begin(), &sm_algo_pack.i8x8x32_direct_stride2_sdot);
        }
    } else if (param.filter_type.enumv() == DTypeEnum::Quantized8Asymm &&
               param.grad_type.enumv() == DTypeEnum::QuantizedS32) {
        if (sm_algo_pack.quint8_direct_stride1_udot.usable(this, param)) {
            ret.insert(ret.begin(), &sm_algo_pack.quint8_direct_stride1_udot);
        }
        if (sm_algo_pack.quint8_direct_stride2_udot.usable(this, param)) {
            ret.insert(ret.begin(), &sm_algo_pack.quint8_direct_stride2_udot);
        }
    }
#endif
    return ret;
}
const char* ConvolutionBackwardDataImpl::get_algorithm_set_name() const {
    // arm common version 0
    return "DeconvAC0";
}

// vim: syntax=cpp.doxygen
