/**
 * \file dnn/src/armv7/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/conv_bias/opr_impl.h"
#include "src/armv7/conv_bias/int8/algos.h"
#include "src/armv7/conv_bias/quint8/algos.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/common/metahelper.h"

#include "src/fallback/convolution/opr_impl.h"

using namespace megdnn;
using namespace armv7;

class ConvBiasImpl::AlgoPack : NonCopyableObj {
    AlgoS8MatrixMul s8_matrix_mul;
    AlgoQU8MatrixMul qu8_matrix_mul;
    fallback::ConvBiasImpl::AlgoBase::Mapper m_all_algos_map;
    SmallVector<fallback::ConvBiasImpl::AlgoBase*> m_all_algos;
public:
    AlgoPack() {
        m_all_algos.emplace_back(&qu8_matrix_mul);
        m_all_algos.emplace_back(&s8_matrix_mul);

        for (auto&& algo : m_all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<fallback::ConvBiasImpl::AlgoBase*>& all_algos()
            const {
        return m_all_algos;
    }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

const ConvBiasImpl::AlgoPack& ConvBiasImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

MEGDNN_FB_DEF_GET_ALGO_FROM_DESC(ConvBiasImpl)

SmallVector<fallback::ConvBiasImpl::AlgoBase*>
ConvBiasImpl::get_all_packed_algo() {
    auto&& algos = arm_common::ConvBiasImpl::get_all_packed_algo();
    //! TODO fused matmul bias is slower than matmul + elemwise in armv7 now,
    //! and nearly equal in aarch64, because of the waste of register in
    //! postprocess
    algos.insert(algos.end(), algo_pack().all_algos().begin(),
                 algo_pack().all_algos().end());
    return std::move(algos);
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    return "ARMV7";
}

// vim: syntax=cpp.doxygen
