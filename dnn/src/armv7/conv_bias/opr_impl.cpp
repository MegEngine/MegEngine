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
public:
    AlgoPack() {
        all_algos.emplace_back(&qu8_matrix_mul);
        all_algos.emplace_back(&s8_matrix_mul);
    }
    SmallVector<AlgoBase*> all_algos;
};

SmallVector<ConvBiasImpl::AlgoBase*> ConvBiasImpl::algo_pack() {
    static AlgoPack sl_algo_pack;
    auto&& algos = arm_common::ConvBiasImpl::algo_pack();
    //! TODO fused matmul bias is slower than matmul + elemwise in armv7 now,
    //! and nearly equal in aarch64, because of the waste of register in
    //! postprocess
    algos.insert(algos.end(), sl_algo_pack.all_algos.begin(),
                 sl_algo_pack.all_algos.end());
    return std::move(algos);
}

const char* ConvBiasImpl::get_algorithm_set_name() const {
    return "ARMV7";
}

// vim: syntax=cpp.doxygen
