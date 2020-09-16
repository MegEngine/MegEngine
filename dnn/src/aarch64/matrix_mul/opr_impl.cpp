/**
 * \file dnn/src/aarch64/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/aarch64/matrix_mul/algos.h"
#include "src/aarch64/matrix_mul/opr_impl.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace aarch64;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoF32K8x12x1 f32K8x12x1;
    AlgoF32MK4_8x12x1 f32_mk4_8x12x1;
    AlgoF32K4x16x1 f32k4x16x1;
    AlgoF32MK4_4x16 f32mk4_4x16;
    AlgoF32Gemv f32_gemv;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16K8x24x1 f16_k8x24x1;
    AlgoF16MK8_8x8 f16_mk8_8x8;
#endif
#if __ARM_FEATURE_DOTPROD
    AlgoInt8x8x32K8x12x4DotProd int8x8x32_k8x12x4_dotprod;
    AlgoInt8x8x32MK4_8x12x4DotProd int8x8x32_mk4_8x12x4_dotprod;
#else
    AlgoInt8x8x32MK4_4x4x16 int8x8x32_mk4_4x4x16;
    AlgoInt8x8x32K4x4x16 int8x8x32_k4x4x16;
    AlgoInt8x8x32K8x8x8 int8x8x32_k8x8x8;
#endif
    AlgoInt8x8x16K8x8x8 int8x8x16_k8x8x8;
    AlgoInt8x8x16K4x4x16 int8x8x16_k4x4x16;
    AlgoInt8x8x16MK4_16x12x4 int8x8x16_mk4_16x12x4;
    AlgoInt8x8x16MK4_4x4x8 int8x8x16_mk4_4x4x8;
    AlgoInt8x8x16MK4_K8x8x8 int8x8x16_mk4_k8x8x8;

    AlgoInt16x16x32K12x8x1 int16x16x32_k12x8x1;
    AlgoInt16x16x32MK8_8x8 int16x16x32_mk8_8x8;

#if __ARM_FEATURE_DOTPROD
    AlgoQuint8K8x8x4DotProd quint8_k8x8x4_dotprod;
    AlgoQuint8GemvDotProd quint8_gemv_dotprod;
#else
    AlgoQuint8K8x8x8 quint8_k8x8x8;
#endif
    AlgoInt4x4x16K8x8x8 int4x4x16_k8x8x8;

    SmallVector<fallback::MatrixMulImpl::AlgoBase*> m_all_algos;
    fallback::MatrixMulImpl::AlgoBase::Mapper m_all_algos_map;
public:

    AlgoPack() {
        m_all_algos.emplace_back(&f32_gemv);
        m_all_algos.emplace_back(&f32K8x12x1);
        m_all_algos.emplace_back(&f32_mk4_8x12x1);
        m_all_algos.emplace_back(&f32k4x16x1);
        m_all_algos.emplace_back(&f32mk4_4x16);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        m_all_algos.emplace_back(&f16_k8x24x1);
        m_all_algos.emplace_back(&f16_mk8_8x8);
#endif
#if __ARM_FEATURE_DOTPROD
        m_all_algos.emplace_back(&int8x8x32_k8x12x4_dotprod);
        m_all_algos.emplace_back(&int8x8x32_mk4_8x12x4_dotprod);
#else
        m_all_algos.emplace_back(&int8x8x32_k4x4x16);
        m_all_algos.emplace_back(&int8x8x32_k8x8x8);
        m_all_algos.emplace_back(&int8x8x32_mk4_4x4x16);
#endif
        m_all_algos.emplace_back(&int8x8x16_k4x4x16);
        m_all_algos.emplace_back(&int8x8x16_k8x8x8);
        m_all_algos.emplace_back(&int8x8x16_mk4_k8x8x8);
        m_all_algos.emplace_back(&int8x8x16_mk4_4x4x8);
        m_all_algos.emplace_back(&int8x8x16_mk4_16x12x4);

        m_all_algos.emplace_back(&int16x16x32_k12x8x1);
        m_all_algos.emplace_back(&int16x16x32_mk8_8x8);
#if __ARM_FEATURE_DOTPROD
        m_all_algos.emplace_back(&quint8_gemv_dotprod);
        m_all_algos.emplace_back(&quint8_k8x8x4_dotprod);
#else
        m_all_algos.emplace_back(&quint8_k8x8x8);
#endif
        m_all_algos.emplace_back(&int4x4x16_k8x8x8);

        for (auto&& algo : m_all_algos) {
            m_all_algos_map.emplace(algo->info().desc, algo);
        }
    }

    const SmallVector<fallback::MatrixMulImpl::AlgoBase*>& all_algos() const {
        return m_all_algos;
    }
    const AlgoBase::Mapper& all_algos_map() const { return m_all_algos_map; }
};

const MatrixMulImpl::AlgoPack& MatrixMulImpl::algo_pack() {
    static AlgoPack algo_pack;
    return algo_pack;
}

MEGDNN_FB_DEF_GET_ALGO_FROM_DESC(MatrixMulImpl)

SmallVector<fallback::MatrixMulImpl::AlgoBase*>
MatrixMulImpl::get_all_packed_algo() {
    auto&& algos = arm_common::MatrixMulImpl::get_all_packed_algo();
    algos.insert(algos.begin(), algo_pack().all_algos().begin(),
                 algo_pack().all_algos().end());
    return std::move(algos);
}

// vim: syntax=cpp.doxygen
