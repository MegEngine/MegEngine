/**
 * \file dnn/src/arm_common/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/elemwise/opr_impl.h"
#include "src/arm_common/elemwise/binary/algo.h"
#include "src/arm_common/elemwise/ternary/algo.h"
#include "src/arm_common/elemwise/unary/algo.h"
#include "src/arm_common/elemwise_helper/elemwise_op.h"
#include "src/common/metahelper.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace arm_common;

class ElemwiseImpl::AlgoPack {
    AlgoUnary algo_unary;
    AlgoBinaryVecVec algo_binary_vec_vec;
    AlgoBinaryVecScalar algo_binary_vec_sca;
    AlgoBinaryVecBcast101 algo_binary_vec_bcast101;
    AlgoBinaryVecBcastX0X algo_binary_vec_bcastX0X;
    AlgoBinaryVecBcast111C algo_binary_vec_bcast110;
    AlgoBinaryVecBcast101xX algo_binary_VEC_BCAST101xX;
    AlgoTernaryFma3VecVecVec algo_ternaryfma3_vec_vec_vec;
    AlgoTernaryFma3VecVecScalar algo_ternaryfma3_vec_vecsca;
    AlgoTernaryFma3Bcast101VecBcast101 algo_ternaryfma3_bcast101_vec_bcast101;
    AlgoTernaryFma3Bcast111CVecBcast111C algo_ternaryfma3_bcast110_vec_bcast110;
    AlgoTernaryFma3Bcast101xXVecBcast101xX algo_ternaryfma3_bcast101xX_vec_bcast101xX;
    AlgoTernaryFma3VecBcast101Vec algo_ternaryfma3_vec_bcast101_vec;
    AlgoTernaryFma3VecBcast111CVec algo_ternaryfma3_vec_bcast110_vec;
    AlgoTernaryFma3VecBcast101xXVec algo_ternaryfma3_vec_bcast101xX_vec;
    AlgoTernaryFma3VecScalarVec algo_ternaryfma3_vec_sca_vec;
    AlgoTernaryFma3VecScalarScalar algo_ternaryfma3_vec_sca_sca;

public:
    AlgoPack() {
        all_algos.emplace_back(&algo_unary);
        all_algos.emplace_back(&algo_binary_vec_vec);
        all_algos.emplace_back(&algo_binary_vec_sca);
        all_algos.emplace_back(&algo_binary_vec_bcast101);
        all_algos.emplace_back(&algo_binary_vec_bcastX0X);
        all_algos.emplace_back(&algo_binary_vec_bcast110);
        all_algos.emplace_back(&algo_binary_VEC_BCAST101xX);
        all_algos.emplace_back(&algo_ternaryfma3_vec_vec_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_vecsca);
        all_algos.emplace_back(&algo_ternaryfma3_bcast101_vec_bcast101);
        all_algos.emplace_back(&algo_ternaryfma3_bcast110_vec_bcast110);
        all_algos.emplace_back(&algo_ternaryfma3_bcast101xX_vec_bcast101xX);
        all_algos.emplace_back(&algo_ternaryfma3_vec_bcast101_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_bcast110_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_bcast101xX_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_sca_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_sca_sca);
    }
    SmallVector<AlgoBase*> all_algos;
};

void ElemwiseImpl::exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) {
    m_src = &srcs;
    m_dst = &dst;

    if (!dst.layout.is_contiguous()) {
        return fallback::ElemwiseImpl::exec(srcs, dst);
    }

    if (m_dst->layout.dtype == dtype::Float32() ||
        DNN_FLOAT16_SELECT(m_dst->layout.dtype == dtype::Float16(), false) ||
        m_dst->layout.dtype == dtype::Int32() ||
        m_dst->layout.dtype == dtype::Int16() || m_dst->layout.dtype == dtype::Int8()) {
        auto kern_param = make_kern_param(this);
        kern_param.m_dst = &dst;
        static AlgoPack m_algo_pack;
        for (auto& m_algo : m_algo_pack.all_algos) {
            if (m_algo->is_available(kern_param)) {
                m_algo->exec(kern_param);
                return;
            }
        }
    }
    fallback::ElemwiseImpl::exec(srcs, dst);
}

// vim: syntax=cpp.doxygen
