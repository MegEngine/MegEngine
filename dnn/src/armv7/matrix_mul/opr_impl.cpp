/**
 * \file dnn/src/armv7/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/armv7/matrix_mul/algos.h"
#include "src/armv7/matrix_mul/opr_impl.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_impl.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace armv7;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoF32 f32;
    AlgoF32MK4Pack4x12 f32_mk4_pack_4x12;
    AlgoF32MK4_4x8 f32_mk4_4x8;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16K4x16x1 f16_k4x16x1;
    AlgoF16MK8_4x8 f16_mk8_4x8;
#endif
#if __ARM_FEATURE_DOTPROD
    AlgoInt8x8x32K6x8x4 int8_k6x8x4;
    AlgoQuint8DotK4x8x4 quint8_k4x8x4;
    AlgoInt8x8x32MK4_8x4x4DotProd int8x8x32_mk4_8x4x4_dotprod;
#endif
    AlgoF32Gemv f32_gemv;
    AlgoInt8x8x32MK4_4x2x16 int8x8x32_mk4_4x2x16;
    AlgoInt8x8x32K4x2x16 int8x8x32_k4x2x16;
    AlgoInt8x8x32K4x8x8 int8x8x32_k4x8x8;
    AlgoQuint8K4x8x8 quint8_k4x8x8;
    AlgoInt8x8x16K4x2x16 int8x8x16_k4x2x16;
    AlgoInt8x8x16K4x8x8 int8x8x16_k4x8x8;
    AlgoInt8x8x16MK4_8x8x4 int8x8x16_mk4_8x8x4;
    AlgoInt16x16x32K12x4x1 int16x16x32_k12x4x1;
    AlgoInt16x16x32MK8_4x8 int16x16x32_mk8_4x8;

    SmallVector<fallback::MatrixMulImpl::AlgoBase*> m_all_algos;
    fallback::MatrixMulImpl::AlgoBase::Mapper m_all_algos_map;

public:

    AlgoPack() {
        m_all_algos.emplace_back(&f32_gemv);
        m_all_algos.emplace_back(&f32);
        m_all_algos.emplace_back(&f32_mk4_pack_4x12);
        m_all_algos.emplace_back(&f32_mk4_4x8);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        m_all_algos.emplace_back(&f16_k4x16x1);
        m_all_algos.emplace_back(&f16_mk8_4x8);
#endif
#if __ARM_FEATURE_DOTPROD
        m_all_algos.emplace_back(&int8x8x32_mk4_8x4x4_dotprod);
        m_all_algos.emplace_back(&int8_k6x8x4);
        m_all_algos.emplace_back(&quint8_k4x8x4);
#endif
        m_all_algos.emplace_back(&int8x8x32_mk4_4x2x16);
        m_all_algos.emplace_back(&int8x8x32_k4x2x16);
        m_all_algos.emplace_back(&int8x8x32_k4x8x8);
        m_all_algos.emplace_back(&quint8_k4x8x8);
        m_all_algos.emplace_back(&int8x8x16_mk4_8x8x4);
        m_all_algos.emplace_back(&int8x8x16_k4x2x16);
        m_all_algos.emplace_back(&int8x8x16_k4x8x8);

        m_all_algos.emplace_back(&int16x16x32_k12x4x1);
        m_all_algos.emplace_back(&int16x16x32_mk8_4x8);

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

SmallVector<fallback::MatrixMulImpl::AlgoBase*>
MatrixMulImpl::get_all_packed_algo() {
    auto algos = arm_common::MatrixMulImpl::get_all_packed_algo();
    algos.insert(algos.begin(), algo_pack().all_algos().begin(),
                 algo_pack().all_algos().end());
    return algos;
}

MEGDNN_FB_DEF_GET_ALGO_FROM_DESC(MatrixMulImpl)

// vim: syntax=cpp.doxygen
