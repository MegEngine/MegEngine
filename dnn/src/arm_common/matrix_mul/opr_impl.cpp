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

namespace {
uint8_t arm_common_algo_type_storage;
}  // anonymous namespace

void* const MatrixMulImpl::sm_arm_common_algo_type =
        &arm_common_algo_type_storage;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoInt8x8x16 int8x8x16;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
	AlgoF16Gemv f16gemv;
#endif
    AlgoInt8x8x32Gemv int8x8x32_gemv; 
public:
    AlgoPack() {
        all_algos.emplace_back(&int8x8x16);
#if  __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        all_algos.emplace_back(&f16gemv);
#endif
        all_algos.emplace_back(&int8x8x32_gemv);
 }
    SmallVector<AlgoBase*> all_algos;
};

SmallVector<MatrixMulImpl::AlgoBase*> MatrixMulImpl::algo_pack() {
    static AlgoPack s_algo_pack;
    auto&& algos = fallback::MatrixMulImpl::algo_pack();
    algos.insert(algos.begin(), s_algo_pack.all_algos.begin(),
                 s_algo_pack.all_algos.end());
    return std::move(algos);
}

// vim: syntax=cpp.doxygen
