#include "./opr_impl.h"
#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/elemwise_multi_type/kern_defs.cuh"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_elemwise_multi_type)

using namespace megdnn;
using namespace naive;

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<1>& param, const TensorND& dst, Elemwise::Mode mode) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    megdnn_assert(param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

    switch (mode) {
#define DISPATCH(_mode)                                                        \
    case Elemwise::Mode::_mode: {                                              \
        typedef ElemwiseKern<                                                  \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode, float> \
                KernImpl;                                                      \
        MIDOUT_BEGIN(                                                          \
                megdnn_naive_elemwise_multi_type, midout_iv(1),                \
                param_enumv::Elemwise::Mode::_mode) {                          \
            dispatch_qint_op_dtype<KernImpl, ElemwiseOpParamN<1>>(param, dst); \
        }                                                                      \
        MIDOUT_END();                                                          \
        break;                                                                 \
    }

        DISPATCH(RELU);
        DISPATCH(ABS);
        DISPATCH(ACOS);
        DISPATCH(ASIN);
        DISPATCH(CEIL);
        DISPATCH(COS);
        DISPATCH(EXP);
        DISPATCH(EXPM1);
        DISPATCH(FLOOR);
        DISPATCH(LOG);
        DISPATCH(LOG1P);
        DISPATCH(NEGATE);
        DISPATCH(SIGMOID);
        DISPATCH(SIN);
        DISPATCH(TANH);
        DISPATCH(FAST_TANH);
        DISPATCH(ROUND);
        DISPATCH(ERF);
        DISPATCH(ERFINV);
        DISPATCH(ERFC);
        DISPATCH(ERFCINV);
        DISPATCH(H_SWISH);
#undef DISPATCH
        default:
            megdnn_assert_internal(0);
    }
#else
    __builtin_trap();
#endif
}

// vim: syntax=cpp.doxygen
