/**
 * \file dnn/src/arm_common/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/matrix_mul/opr_impl.h"
#include "src/arm_common/matrix_mul/algos.h"
#include "src/common/metahelper.h"

using namespace megdnn;
using namespace arm_common;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoInt8x8x16 int8x8x16;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    AlgoF16Gemv f16gemv;
#endif
    AlgoInt8x8x32Gemv int8x8x32_gemv;
    AlgoInt8x8x32GemvMK4 int8x8x32_gemv_mk4;
#if __ARM_FEATURE_DOTPROD
    AlgoInt8x8x32GemvMK4Dot int8x8x32_gemv_mk4_dot;
#endif
    AlgoGevm gevm;
    AlgoF32GemvMK4 f32_gemv_mk4;

public:
    AlgoPack() {
        all_algos.emplace_back(&int8x8x16);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        all_algos.emplace_back(&f16gemv);
#endif
#if __ARM_FEATURE_DOTPROD
        all_algos.emplace_back(&int8x8x32_gemv_mk4_dot);
#endif
        all_algos.emplace_back(&int8x8x32_gemv);
        all_algos.emplace_back(&int8x8x32_gemv_mk4);
        all_algos.emplace_back(&f32_gemv_mk4);
        all_algos.emplace_back(&gevm);
    }
    SmallVector<fallback::MatrixMulImpl::AlgoBase*> all_algos;
};

SmallVector<fallback::MatrixMulImpl::AlgoBase*> MatrixMulImpl::algo_pack() {
    static AlgoPack s_algo_pack;
    auto&& algos = fallback::MatrixMulImpl::algo_pack();
    algos.insert(algos.begin(), s_algo_pack.all_algos.begin(),
                 s_algo_pack.all_algos.end());
    return std::move(algos);
}

// vim: syntax=cpp.doxygen
