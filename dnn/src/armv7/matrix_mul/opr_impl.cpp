/**
 * \file dnn/src/armv7/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/armv7/matrix_mul/opr_impl.h"
#include "src/armv7/matrix_mul/algos.h"
#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_impl.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace armv7;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoF32 f32;
    AlgoF32MK4_4x8  f32_mk4_4x8;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16K4x16x1 f16_k4x16x1;
    AlgoF16MK8_4x8 f16_mk8_4x8;
#endif
#if __ARM_FEATURE_DOTPROD
    AlgoInt8x8x32K6x8x4 int8_k6x8x4;
    AlgoQuint8DotK4x8x4 quint8_k4x8x4;
#endif
    AlgoF32Gemv f32_gemv;
    AlgoInt8x8x32MK4_4x2x16 int8x8x32_mk4_4x2x16;
    AlgoInt8x8x32K4x2x16 int8x8x32_k4x2x16;
    AlgoInt8x8x32K4x8x8 int8x8x32_k4x8x8;
#if !__ARM_FEATURE_DOTPROD
    AlgoInt8x8x32Gemv int8x8x32_gemv;
#endif
    AlgoQuint8K4x8x8 quint8_k4x8x8;
    AlgoInt8x8x16K4x2x16 int8x8x16_k4x2x16;
    AlgoInt8x8x16K4x8x8 int8x8x16_k4x8x8;
    AlgoInt16x16x32K12x4x1 int16x16x32_k12x4x1;
    AlgoInt16x16x32MK8_4x8 int16x16x32_mk8_4x8;

public:
    SmallVector<MatrixMulImpl::AlgoBase*> all_algos;

    AlgoPack() {
        all_algos.emplace_back(&f32_gemv);
        all_algos.emplace_back(&f32);
        all_algos.emplace_back(&f32_mk4_4x8);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        all_algos.emplace_back(&f16_k4x16x1);
        all_algos.emplace_back(&f16_mk8_4x8);
#endif
#if __ARM_FEATURE_DOTPROD
        all_algos.emplace_back(&int8_k6x8x4);
        all_algos.emplace_back(&quint8_k4x8x4);
#endif
#if !__ARM_FEATURE_DOTPROD
        all_algos.emplace_back(&int8x8x32_gemv);
#endif
        all_algos.emplace_back(&int8x8x32_mk4_4x2x16);
        all_algos.emplace_back(&int8x8x32_k4x8x8);
        all_algos.emplace_back(&int8x8x32_k4x2x16);
        all_algos.emplace_back(&quint8_k4x8x8);
        all_algos.emplace_back(&int8x8x16_k4x8x8);
        all_algos.emplace_back(&int8x8x16_k4x2x16);
        all_algos.emplace_back(&int16x16x32_k12x4x1);
        all_algos.emplace_back(&int16x16x32_mk8_4x8);
    }
};

SmallVector<MatrixMulImpl::AlgoBase*> MatrixMulImpl::algo_pack() {
    static AlgoPack s_algo_pack;
    auto algos = arm_common::MatrixMulImpl::algo_pack();
    algos.insert(algos.begin(), s_algo_pack.all_algos.begin(),
                 s_algo_pack.all_algos.end());
    return algos;
}

// vim: syntax=cpp.doxygen
