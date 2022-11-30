#include "./opr_impl.h"
#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/elemwise_multi_type/kern_defs.cuh"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_elemwise_multi_type)

using namespace megdnn;
using namespace naive;

void ElemwiseMultiTypeImpl::on_quantized_mode(
        const ElemwiseOpParamN<2>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.dtype.enumv() == param[1].layout.dtype.enumv() &&
            param[0].layout.dtype.category() == DTypeCategory::QUANTIZED);
    megdnn_assert(dst.layout.dtype.category() == DTypeCategory::QUANTIZED);

    switch (mode) {
#define DISPATCH(_mode)                                                        \
    case Elemwise::Mode::_mode: {                                              \
        typedef ElemwiseKern<                                                  \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode, float> \
                KernImpl;                                                      \
        MIDOUT_BEGIN(                                                          \
                megdnn_naive_elemwise_multi_type, midout_iv(2),                \
                param_enumv::Elemwise::Mode::_mode) {                          \
            dispatch_qint_op_dtype<KernImpl, ElemwiseOpParamN<2>>(param, dst); \
        }                                                                      \
        MIDOUT_END();                                                          \
        break;                                                                 \
    }

        DISPATCH(ABS_GRAD);
        DISPATCH(ADD);
        DISPATCH(FLOOR_DIV);
        DISPATCH(MAX);
        DISPATCH(MIN);
        DISPATCH(MOD);
        DISPATCH(MUL);
        DISPATCH(POW);
        DISPATCH(SIGMOID_GRAD);
        DISPATCH(SUB);
        DISPATCH(SWITCH_GT0);
        DISPATCH(TANH_GRAD);
        DISPATCH(TRUE_DIV);
        DISPATCH(LOG_SUM_EXP);

        DISPATCH(LT);
        DISPATCH(LEQ);
        DISPATCH(EQ);

        DISPATCH(FUSE_ADD_RELU);
        DISPATCH(FUSE_ADD_SIGMOID);
        DISPATCH(FUSE_ADD_TANH);
        DISPATCH(FAST_TANH_GRAD);
        DISPATCH(ATAN2);
        DISPATCH(H_SWISH_GRAD);
        DISPATCH(FUSE_ADD_H_SWISH);
#undef DISPATCH
        default:
            megdnn_assert_internal(0);
    }
}

void ElemwiseMultiTypeImpl::dest_type_bool_mode(
        const ElemwiseOpParamN<1>& param, const TensorND& dst, Elemwise::Mode mode) {
    switch (mode) {
        case Elemwise::Mode::ISINF: {
            switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(_dt, _mode)                                                    \
    case DTypeTrait<_dt>::enumv: {                                              \
        typedef ElemwiseBoolKern<                                               \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode,         \
                typename DTypeTrait<_dt>::ctype, dt_bool>                       \
                KernImpl##_mode;                                                \
        using _ctype = typename DTypeTrait<_dt>::ctype;                         \
        MIDOUT_BEGIN(                                                           \
                megdnn_naive_elemwise_multi_type, midout_iv(0), _ctype,         \
                param_enumv::Elemwise::Mode::_mode) {                           \
            dispatch_dst_bool_op<KernImpl##_mode, _ctype, dt_bool>(param, dst); \
        }                                                                       \
        MIDOUT_END();                                                           \
        break;                                                                  \
    }
#define DISPATCH_MODE(_mode)                                  \
    DISPATCH(megdnn::dtype::Float32, _mode);                  \
    DNN_INC_FLOAT16(DISPATCH(megdnn::dtype::Float16, _mode);) \
    DNN_INC_FLOAT16(DISPATCH(megdnn::dtype::BFloat16, _mode);)
                DISPATCH_MODE(ISINF);
                default:
                    megdnn_throw(ssprintf(
                            "Unsupported input dtype %s for ElemwiseMultiType",
                            param[0].layout.dtype.name()));
            };
            break;
        };
        case Elemwise::Mode::ISNAN: {
            switch (param[0].layout.dtype.enumv()) {
                DISPATCH_MODE(ISNAN);
                default:
                    megdnn_throw(ssprintf(
                            "Unsupported input dtype %s for ElemwiseMultiType",
                            param[0].layout.dtype.name()));
            };
            break;
        };
        default:
            megdnn_assert_internal(0);
    }
#undef DISPATCH_MODE
#undef DISPATCH
}

void ElemwiseMultiTypeImpl::dest_type_bool_mode(
        const ElemwiseOpParamN<2>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(param[0].layout.dtype.enumv() == param[1].layout.dtype.enumv());
    switch (mode) {
        case Elemwise::Mode::EQ: {
            switch (param[0].layout.dtype.enumv()) {
#define DISPATCH(_dt, _mode)                                                    \
    case DTypeTrait<_dt>::enumv: {                                              \
        typedef ElemwiseBoolKern<                                               \
                megcorePlatformCPU, param_enumv::Elemwise::Mode::_mode,         \
                typename DTypeTrait<_dt>::ctype, dt_bool>                       \
                KernImpl##_mode;                                                \
        using _ctype = typename DTypeTrait<_dt>::ctype;                         \
        MIDOUT_BEGIN(                                                           \
                megdnn_naive_elemwise_multi_type, midout_iv(1), _ctype,         \
                param_enumv::Elemwise::Mode::_mode) {                           \
            dispatch_dst_bool_op<KernImpl##_mode, _ctype, dt_bool>(param, dst); \
        }                                                                       \
        MIDOUT_END();                                                           \
        break;                                                                  \
    };
#define DISPATCH_MODE(_mode)                                   \
    DISPATCH(megdnn::dtype::Float32, _mode);                   \
    DNN_INC_FLOAT16(DISPATCH(megdnn::dtype::Float16, _mode);)  \
    DNN_INC_FLOAT16(DISPATCH(megdnn::dtype::BFloat16, _mode);) \
    DISPATCH(megdnn::dtype::Int32, _mode);                     \
    DISPATCH(megdnn::dtype::Int16, _mode);                     \
    DISPATCH(megdnn::dtype::Int8, _mode);                      \
    DISPATCH(megdnn::dtype::Uint8, _mode);                     \
    DISPATCH(megdnn::dtype::Bool, _mode);
                DISPATCH_MODE(EQ);
                break;
                default:
                    megdnn_throw(ssprintf(
                            "Unsupported input dtype %s for ElemwiseMultiType",
                            param[0].layout.dtype.name()));
            };
            break;
        };
        case Elemwise::Mode::NEQ: {
            switch (param[0].layout.dtype.enumv()) {
                DISPATCH_MODE(NEQ);
                default:
                    megdnn_throw(ssprintf(
                            "Unsupported input dtype %s for ElemwiseMultiType",
                            param[0].layout.dtype.name()));
            };
            break;
        };
        case Elemwise::Mode::LT: {
            switch (param[0].layout.dtype.enumv()) {
                DISPATCH_MODE(LT);
                default:
                    megdnn_throw(ssprintf(
                            "Unsupported input dtype %s for ElemwiseMultiType",
                            param[0].layout.dtype.name()));
            };
            break;
        };
        case Elemwise::Mode::LEQ: {
            switch (param[0].layout.dtype.enumv()) {
                DISPATCH_MODE(LEQ);
                default:
                    megdnn_throw(ssprintf(
                            "Unsupported input dtype %s for ElemwiseMultiType",
                            param[0].layout.dtype.name()));
            };
            break;
        };
        default:
            megdnn_assert_internal(0);
    }
#undef DISPATCH_MODE
#undef DISPATCH
}

// vim: syntax=cpp.doxygen
