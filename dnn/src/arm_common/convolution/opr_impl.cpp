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
class ConvolutionBackwardDataImpl::AlgoPack : NonCopyableObj {
#if __ARM_FEATURE_DOTPROD
    AlgoSdot8DirectStride1 i8x8x32_direct_stride1_sdot;
    AlgoSdot8DirectStride2 i8x8x32_direct_stride2_sdot;
    AlgoUdot8DirectStride1 quint8_direct_stride1_udot;
    AlgoUdot8DirectStride2 quint8_direct_stride2_udot;
#endif

    fallback::ConvolutionBackwardDataImpl::AlgoBase::Mapper m_all_algos_map;
    SmallVector<fallback::ConvolutionBackwardDataImpl::AlgoBase*>
            m_all_algos;

public:
    AlgoPack() {
#if __ARM_FEATURE_DOTPROD
        m_all_algos.emplace_back(&i8x8x32_direct_stride1_sdot);
        m_all_algos.emplace_back(&i8x8x32_direct_stride2_sdot);
        m_all_algos.emplace_back(&quint8_direct_stride1_udot);
        m_all_algos.emplace_back(&quint8_direct_stride2_udot);
#endif

        for (auto&& algo : m_all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<fallback::ConvolutionBackwardDataImpl::AlgoBase*>&
    all_algos() const {
        return m_all_algos;
    }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

const ConvolutionBackwardDataImpl::AlgoPack&
ConvolutionBackwardDataImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

MEGDNN_FB_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardDataImpl)

SmallVector<fallback::ConvolutionBackwardDataImpl::AlgoBase*>
ConvolutionBackwardDataImpl::get_all_packed_algo() {
    auto&& algos = fallback::ConvolutionBackwardDataImpl::get_all_packed_algo();
    algos.insert(algos.begin(), algo_pack().all_algos().begin(),
                 algo_pack().all_algos().end());
    return std::move(algos);
}

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

const char* ConvolutionBackwardDataImpl::get_algorithm_set_name() const {
    // arm common version 0
    return "DeconvAC0";
}

// vim: syntax=cpp.doxygen
