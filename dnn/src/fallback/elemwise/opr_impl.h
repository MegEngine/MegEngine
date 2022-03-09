/**
 * \file dnn/src/fallback/elemwise/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/elemwise_helper/elemwise_op.h"
#include "src/naive/elemwise/opr_impl.h"

namespace megdnn {
namespace fallback {

class ElemwiseImpl : public naive::ElemwiseForwardImpl {
    template <typename dtype, uint32_t mode>
    void unary_kern(const ElemwiseOpParamN<1>& param);

    template <uint32_t mode>
    void exec_UNARY_INT();

    template <uint32_t mode>
    void exec_UNARY_FLOAT();

    template <typename dtype, uint32_t mode>
    void binary_kern(const ElemwiseOpParamN<2>& param);

    template <uint32_t mode>
    void exec_BINARY_INT();

    template <uint32_t mode>
    void exec_BINARY_FLOAT();

    void exec_fallback(const TensorNDArray& srcs, _megdnn_tensor_out dst);
    bool exec_gi_intrinsic(const TensorNDArray& srcs, _megdnn_tensor_out dst);

private:
    class AlgoUnary;
    class AlgoBinaryVecVec;
    class AlgoBinaryVecScalar;
    class AlgoBinaryVecBcast101;
    class AlgoBinaryVecBcastX0X;
    class AlgoBinaryVecBcast111C;
    class AlgoBinaryVecBcast101xX;
    class AlgoTernaryFma3VecVecVec;
    class AlgoTernaryFma3VecVecScalar;
    class AlgoTernaryFma3Bcast101VecBcast101;
    class AlgoTernaryFma3Bcast111CVecBcast111C;
    class AlgoTernaryFma3Bcast101xXVecBcast101xX;
    class AlgoTernaryFma3VecBcast101Vec;
    class AlgoTernaryFma3VecBcast111CVec;
    class AlgoTernaryFma3VecBcast101xXVec;
    class AlgoTernaryFma3VecScalarVec;
    class AlgoTernaryFma3VecScalarScalar;
    class AlgoPack;

public:
    class AlgoBase;
    struct KernParam {
        elemwise::BcastType broad_cast_type;
        Mode mode;
        const TensorND* m_dst;
        Handle* handle;
        ElemwiseOpParamN<3> ternary_elparam;
        ElemwiseOpParamN<2> binary_elparam;
        ElemwiseOpParamN<1> unary_elparam;
    };
    KernParam make_kern_param(ElemwiseImpl* opr);
    using naive::ElemwiseForwardImpl::ElemwiseForwardImpl;
    void exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) override;

    const char* get_algorithm_set_name() const { return "FALLBACK ELEMWISE"; }
    bool is_thread_safe() const override { return true; }
};

/*!
 * \brief base class for Elemwise algo
 */
class ElemwiseImpl::AlgoBase : public detail::Algorithm {
public:
    virtual bool is_available(const KernParam&) const = 0;
    virtual void exec(const KernParam&) const = 0;
    virtual ~AlgoBase() = default;
    uint32_t type() const override { return INVALID_ALGO_TYPE; };
};

//! fallback only support float, int32, int8
#define DISPATCH_TYPE_FALLBACK(_case)                 \
    if (src0.layout.dtype == dtype::Float32{}) {      \
        DISPATCH_MODE_FLOAT(_case, float, 0);         \
    } else if (src0.layout.dtype == dtype::Int32{}) { \
        DISPATCH_MODE_INT(_case, int, 2);             \
    } else if (src0.layout.dtype == dtype::Int8{}) {  \
        DISPATCH_MODE_INT(_case, dt_int8, 4);         \
    }

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
