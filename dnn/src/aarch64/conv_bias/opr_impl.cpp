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

public:
    AlgoPack() {
        matmul_algos.emplace_back(&qu8_matrix_mul);
        matmul_algos.emplace_back(&s8_matrix_mul);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        direct_algos.emplace_back(&f16_direct_stride2);
#endif
        direct_algos.emplace_back(&f32_direct_stride2);
    }
    SmallVector<AlgoBase*> direct_algos;
    SmallVector<AlgoBase*> matmul_algos;
};

SmallVector<ConvBiasImpl::AlgoBase*> ConvBiasImpl::algo_pack() {
    static AlgoPack sl_algo_pack;
    auto&& algos = arm_common::ConvBiasImpl::algo_pack();
    algos.insert(algos.begin(), sl_algo_pack.direct_algos.begin(),
                 sl_algo_pack.direct_algos.end());
    //! We put matmul algos at the begin. Because matmul will get privilege when
    //! prefer return true. See
    algos.insert(algos.begin(), sl_algo_pack.matmul_algos.begin(),
                 sl_algo_pack.matmul_algos.end());
    return std::move(algos);
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    return "AARCH64";
}

// vim: syntax=cpp.doxygen
