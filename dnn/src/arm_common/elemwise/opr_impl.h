/**
 * \file dnn/src/arm_common/elemwise/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/fallback/elemwise/opr_impl.h"

#include "src/arm_common/elemwise_op.h"

namespace megdnn {
namespace arm_common {
class ElemwiseImpl final : public fallback::ElemwiseImpl {
public:
    using fallback::ElemwiseImpl::ElemwiseImpl;
    void exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) override;
    const char* get_algorithm_set_name() const { return "ARM COMMON ELEMWISE"; }

private:
    struct KernParam {
        BcastType broad_cast_type;
        Mode mode;
        const TensorND* m_dst;
        Handle* handle;
        ElemwiseOpParamN<3> ternary_elparam;
        ElemwiseOpParamN<2> binary_elparam;
        ElemwiseOpParamN<1> unary_elparam;
    };
    KernParam make_kern_param(ElemwiseImpl* opr);
    class AlgoBase;
    class AlgoUnary;
    class AlgoBinaryVecVec;
    class AlgoBinaryVecScalar;
    class AlgoBinaryVecBcast101;
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
};

/*!
 *
 * \brief base class for Elemwise algo
 *
 */
class ElemwiseImpl::AlgoBase : public detail::Algorithm {
public:
    virtual bool is_available(const KernParam&) const = 0;
    virtual void exec(const KernParam&) const = 0;
    virtual ~AlgoBase() = default;
    uint32_t type() const override { return INVALID_ALGO_TYPE; };
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define DISPATCH_TYPE(_case)                                                       \
    if (src0.layout.dtype == dtype::Float32{}) {                                   \
        DISPATCH_MODE_FLOAT(_case, float, 0);                                      \
    } else if (DNN_FLOAT16_SELECT(src0.layout.dtype == dtype::Float16{}, false)) { \
        DISPATCH_MODE_FLOAT(_case, __fp16, 1);                                     \
    } else if (src0.layout.dtype == dtype::Int32{}) {                              \
        DISPATCH_MODE_INT(_case, int, 2);                                          \
    } else if (src0.layout.dtype == dtype::Int16{}) {                              \
        DISPATCH_MODE_INT(_case, dt_int16, 3);                                     \
    } else if (src0.layout.dtype == dtype::Int8{}) {                               \
        DISPATCH_MODE_INT(_case, dt_int8, 4);                                      \
    }
#else
#define DISPATCH_TYPE(_case)                          \
    if (src0.layout.dtype == dtype::Float32{}) {      \
        DISPATCH_MODE_FLOAT(_case, float, 0);         \
    } else if (src0.layout.dtype == dtype::Int32{}) { \
        DISPATCH_MODE_INT(_case, int, 2);             \
    } else if (src0.layout.dtype == dtype::Int16{}) { \
        DISPATCH_MODE_INT(_case, dt_int16, 3);        \
    } else if (src0.layout.dtype == dtype::Int8{}) {  \
        DISPATCH_MODE_INT(_case, dt_int8, 4);         \
    }
#endif

}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
