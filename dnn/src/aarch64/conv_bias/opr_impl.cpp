/**
 * \file dnn/src/aarch64/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/conv_bias/opr_impl.h"
#include "src/aarch64/conv_bias/int8/algos.h"
#include "src/aarch64/conv_bias/quint8/algos.h"

#include "src/naive/handle.h"
#include "src/common/utils.h"
#include "src/common/metahelper.h"

#include "src/fallback/convolution/opr_impl.h"
#include "src/aarch64/conv_bias/fp32/algos.h"
#include "src/aarch64/conv_bias/fp16/algos.h"

using namespace megdnn;
using namespace aarch64;

class ConvBiasImpl::AlgoPack : NonCopyableObj {
    AlgoF32DirectStride2 f32_direct_stride2;
    AlgoS8MatrixMul s8_matrix_mul;
    AlgoQU8MatrixMul qu8_matrix_mul;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16DirectStride2 f16_direct_stride2;
#endif

    fallback::ConvBiasImpl::AlgoBase::Mapper m_all_algos_map;
    SmallVector<fallback::ConvBiasImpl::AlgoBase*> m_direct_algos;
    SmallVector<fallback::ConvBiasImpl::AlgoBase*> m_matmul_algos;

public:
    AlgoPack() {
        m_matmul_algos.emplace_back(&qu8_matrix_mul);
        m_matmul_algos.emplace_back(&s8_matrix_mul);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        m_direct_algos.emplace_back(&f16_direct_stride2);
#endif
        m_direct_algos.emplace_back(&f32_direct_stride2);

        for (auto&& algo : m_direct_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
        for (auto&& algo : m_matmul_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<fallback::ConvBiasImpl::AlgoBase*>& direct_algos() const {
        return m_direct_algos;
    }
    const SmallVector<fallback::ConvBiasImpl::AlgoBase*>& matmul_algos()
            const {
        return m_matmul_algos;
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
    algos.insert(algos.begin(), algo_pack().direct_algos().begin(),
                 algo_pack().direct_algos().end());
    //! We put matmul algos at the begin. Because matmul will get privilege when
    //! prefer return true. See
    algos.insert(algos.begin(), algo_pack().matmul_algos().begin(),
                 algo_pack().matmul_algos().end());
    return std::move(algos);
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    return "AARCH64";
}

// vim: syntax=cpp.doxygen
