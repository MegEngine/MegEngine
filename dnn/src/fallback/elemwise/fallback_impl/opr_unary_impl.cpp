#include "src/fallback/elemwise/opr_impl.h"

#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_elemwise_unary)

namespace megdnn {
namespace fallback {

template <typename dtype, uint32_t mode>
void ElemwiseImpl::unary_kern(const ElemwiseOpParamN<1>& param) {
    using ctype = typename DTypeTrait<dtype>::ctype;
    using Kern = ElemwiseKern<megcorePlatformCPU, mode, ctype>;
    MIDOUT_BEGIN(megdnn_fallback_elemwise_unary, ctype, midout_iv(mode)) {
        // only specialize for the most common 1-dim case
        auto tot = param.size;
        auto stride = param[0].layout.stride[0];
        auto src0 = param[0];
        auto dst_tensor = *m_dst;
        if (param.max_ndim == 1) {
            MIDOUT_BEGIN(
                    megdnn_fallback_elemwise_unary, ctype, midout_iv(mode),
                    midout_iv(1)) {
                MEGDNN_DISPATCH_CPU_KERN_OPR({
                    ctype* __restrict src = static_cast<ctype*>(src0.raw_ptr());
                    ctype* __restrict dst = static_cast<ctype*>(dst_tensor.raw_ptr());
                    for (size_t i = 0; i < tot; ++i) {
                        dst[i] = Kern::apply(src[i * stride]);
                    }
                });
                return;
            }
            MIDOUT_END();
        }
        naive::ElemwiseForwardImpl::exec(*m_src, *m_dst);
    }
    MIDOUT_END();
}

#define SWITCH_DTYPE(_cat, _cb)                            \
    switch (m_dst->layout.dtype.enumv()) {                 \
        MEGDNN_FOREACH_COMPUTING_DTYPE_##_cat(_cb) default \
                : megdnn_throw("bad dtype");               \
    }

template <uint32_t mode>
void ElemwiseImpl::exec_UNARY_INT() {
    auto param = make_elemwise_op_param<1>();
#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return unary_kern<_dt, mode>(param);

    switch (m_dst->layout.dtype.enumv()) {
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
        cb(::megdnn::dtype::Uint16) default : megdnn_throw("bad dtype");
    }

#undef cb
}

template <uint32_t mode>
void ElemwiseImpl::exec_UNARY_FLOAT() {
    auto param = make_elemwise_op_param<1>();
#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return unary_kern<_dt, mode>(param);

    SWITCH_DTYPE(FLOAT, cb)

#undef cb
}

#undef SWITCH_DTYPE
using Mode = param_enumv::Elemwise::Mode;
#define INST(mode) template void megdnn::fallback::ElemwiseImpl::exec_UNARY_INT<mode>();
INST(Mode::RELU);
INST(Mode::ABS);
INST(Mode::NEGATE);
INST(Mode::RELU6);
INST(Mode::SQUARE);
INST(Mode::SIGN);
#undef INST

#define INST(mode) \
    template void megdnn::fallback::ElemwiseImpl::exec_UNARY_FLOAT<mode>();
INST(Mode::RELU);
INST(Mode::ABS);
INST(Mode::ACOS);
INST(Mode::ASIN);
INST(Mode::CEIL);
INST(Mode::COS);
INST(Mode::EXP);
INST(Mode::EXPM1);
INST(Mode::FLOOR);
INST(Mode::LOG);
INST(Mode::LOG1P);
INST(Mode::NEGATE);
INST(Mode::SIGMOID);
INST(Mode::SIN);
INST(Mode::TANH);
INST(Mode::FAST_TANH);
INST(Mode::ROUND);
INST(Mode::ERF);
INST(Mode::ERFINV);
INST(Mode::ERFC);
INST(Mode::ERFCINV);
INST(Mode::H_SWISH);
INST(Mode::SILU);
INST(Mode::GELU);
INST(Mode::SINH);
INST(Mode::COSH);
INST(Mode::ASINH);
INST(Mode::ACOSH);
INST(Mode::ATANH);
INST(Mode::TAN);
INST(Mode::SOFTPLUS);
INST(Mode::RELU6);
INST(Mode::HSIGMOID);
INST(Mode::LOGSIGMOID);
INST(Mode::SQRT);
INST(Mode::SQUARE);
INST(Mode::SIGN);
#undef INST
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
