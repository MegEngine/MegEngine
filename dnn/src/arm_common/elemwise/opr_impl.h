#pragma once
#include "src/fallback/elemwise/opr_impl.h"

#include "src/arm_common/elemwise_helper/elemwise_op.h"

namespace megdnn {
namespace arm_common {
class ElemwiseImpl final : public fallback::ElemwiseImpl {
public:
    using fallback::ElemwiseImpl::AlgoBase;
    using fallback::ElemwiseImpl::ElemwiseImpl;
    void exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) override;
    const char* get_algorithm_set_name() const { return "ARM COMMON ELEMWISE"; }

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
