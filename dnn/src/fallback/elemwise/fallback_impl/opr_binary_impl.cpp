#include "src/fallback/elemwise/opr_impl.h"

#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_elemwise_binary)

namespace megdnn {
namespace fallback {

template <typename dtype, uint32_t mode>
void ElemwiseImpl::binary_kern(const ElemwiseOpParamN<2>& param) {
    using ctype = typename DTypeTrait<dtype>::ctype;
    using Kern = ElemwiseKern<megcorePlatformCPU, mode, ctype>;

    MIDOUT_BEGIN(megdnn_fallback_elemwise_binary, ctype, midout_iv(mode)) {
        if (param.max_ndim == 1) {
            MIDOUT_BEGIN(
                    megdnn_fallback_elemwise_binary, ctype, midout_iv(mode),
                    midout_iv(1)) {
                auto tot = param.size;
                auto as = param[0].layout.stride[0], bs = param[1].layout.stride[0];
                auto src0 = param[0];
                auto src1 = param[1];
                auto dst_tensor = *m_dst;

                MEGDNN_DISPATCH_CPU_KERN_OPR({
                    ctype* __restrict a = static_cast<ctype*>(src0.raw_ptr());
                    ctype* __restrict b = static_cast<ctype*>(src1.raw_ptr());
                    ctype* __restrict dst = static_cast<ctype*>(dst_tensor.raw_ptr());
                    for (size_t i = 0; i < tot; ++i) {
                        dst[i] = Kern::apply(a[i * as], b[i * bs]);
                    }
                });
                return;
            }
            MIDOUT_END();
        }

        if (std::min(param[0].layout.ndim, param[1].layout.ndim) > 1) {
            return naive::ElemwiseForwardImpl::exec(*m_src, *m_dst);
        }

        if (param.max_ndim == 2) {
            if (param[0].layout.ndim == 1) {
                MIDOUT_BEGIN(
                        megdnn_fallback_elemwise_binary, ctype, midout_iv(mode),
                        midout_iv(21)) {
                    auto as = param[0].layout.stride[0],
                         bs0 = param[1].layout.stride[0],
                         bs1 = param[1].layout.stride[1];
                    auto n0 = param[1].layout.shape[0], n1 = param[1].layout.shape[1];
                    auto src0 = param[0];
                    auto src1 = param[1];
                    auto dst_tensor = *m_dst;

                    MEGDNN_DISPATCH_CPU_KERN_OPR({
                        ctype* __restrict a = static_cast<ctype*>(src0.raw_ptr());
                        ctype* __restrict b = static_cast<ctype*>(src1.raw_ptr());
                        ctype* __restrict dst =
                                static_cast<ctype*>(dst_tensor.raw_ptr());
                        ptrdiff_t toff = 0;
                        for (size_t i = 0; i < n0; ++i) {
                            for (size_t j = 0; j < n1; ++j) {
                                dst[toff] =
                                        Kern::apply(a[as * toff], b[bs0 * i + bs1 * j]);
                                ++toff;
                            }
                        }
                    });
                    return;
                }
                MIDOUT_END();
            }

            MIDOUT_BEGIN(
                    megdnn_fallback_elemwise_binary, ctype, midout_iv(mode),
                    midout_iv(22)) {
                megdnn_assert(param[1].layout.ndim == 1);
                auto bs = param[1].layout.stride[0], as0 = param[0].layout.stride[0],
                     as1 = param[0].layout.stride[1];
                auto n0 = param[0].layout.shape[0], n1 = param[0].layout.shape[1];
                auto src0 = param[0];
                auto src1 = param[1];
                auto dst_tensor = *m_dst;

                MEGDNN_DISPATCH_CPU_KERN_OPR({
                    ctype* __restrict a = static_cast<ctype*>(src0.raw_ptr());
                    ctype* __restrict b = static_cast<ctype*>(src1.raw_ptr());
                    ctype* __restrict dst = static_cast<ctype*>(dst_tensor.raw_ptr());
                    ptrdiff_t toff = 0;
                    for (size_t i = 0; i < n0; ++i) {
                        for (size_t j = 0; j < n1; ++j) {
                            dst[toff] = Kern::apply(a[as0 * i + as1 * j], b[toff * bs]);
                            ++toff;
                        }
                    }
                });
                return;
            }
            MIDOUT_END();
        }

        if (param.max_ndim == 3) {
            auto brd_101 = [](const TensorND& t) {
                auto&& l = t.layout;
                return l.ndim == 3 && l.stride[0] == 0 && l.stride[2] == 0;
            };
            if (param[0].layout.ndim == 1 && brd_101(param[1])) {
                MIDOUT_BEGIN(
                        megdnn_fallback_elemwise_binary, ctype, midout_iv(mode),
                        midout_iv(31)) {
                    auto as = param[0].layout.stride[0], bs = param[1].layout.stride[1];
                    auto n0 = param[1].layout.shape[0], n1 = param[1].layout.shape[1],
                         n2 = param[1].layout.shape[2];
                    auto src0 = param[0];
                    auto src1 = param[1];
                    auto dst_tensor = *m_dst;

                    MEGDNN_DISPATCH_CPU_KERN_OPR({
                        ctype* __restrict a = static_cast<ctype*>(src0.raw_ptr());
                        ctype* __restrict b = static_cast<ctype*>(src1.raw_ptr());
                        ctype* __restrict dst =
                                static_cast<ctype*>(dst_tensor.raw_ptr());
                        size_t toff = 0;
                        for (size_t i = 0; i < n0; ++i) {
                            for (size_t j = 0; j < n1; ++j) {
                                for (size_t k = 0; k < n2; ++k) {
                                    dst[toff] = Kern::apply(a[as * toff], b[bs * j]);
                                    ++toff;
                                }
                            }
                        }
                    });
                    return;
                }
                MIDOUT_END();
            }
            if (param[1].layout.ndim == 1 && brd_101(param[0])) {
                MIDOUT_BEGIN(
                        megdnn_fallback_elemwise_binary, ctype, midout_iv(mode),
                        midout_iv(32)) {
                    auto as = param[0].layout.stride[1], bs = param[1].layout.stride[0];
                    auto n0 = param[0].layout.shape[0], n1 = param[0].layout.shape[1],
                         n2 = param[0].layout.shape[2];
                    auto src0 = param[0];
                    auto src1 = param[1];
                    auto dst_tensor = *m_dst;
                    MEGDNN_DISPATCH_CPU_KERN_OPR({
                        ctype* __restrict a = static_cast<ctype*>(src0.raw_ptr());
                        ctype* __restrict b = static_cast<ctype*>(src1.raw_ptr());
                        ctype* __restrict dst =
                                static_cast<ctype*>(dst_tensor.raw_ptr());
                        size_t toff = 0;
                        for (size_t i = 0; i < n0; ++i) {
                            for (size_t j = 0; j < n1; ++j) {
                                for (size_t k = 0; k < n2; ++k) {
                                    dst[toff] = Kern::apply(a[as * j], b[bs * toff]);
                                    ++toff;
                                }
                            }
                        }
                    });
                    return;
                }
                MIDOUT_END();
            }
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
void ElemwiseImpl::exec_BINARY_INT() {
    auto param = make_elemwise_op_param<2>();
#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return binary_kern<_dt, mode>(param);

    switch (m_dst->layout.dtype.enumv()) {
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
        MEGDNN_ELEMWISE_INC_UINT16(cb(::megdnn::dtype::Uint16))
        default:
            megdnn_throw("bad dtype");
    }

#undef cb
}

template <uint32_t mode>
void ElemwiseImpl::exec_BINARY_FLOAT() {
    auto param = make_elemwise_op_param<2>();
#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return binary_kern<_dt, mode>(param);

    SWITCH_DTYPE(FLOAT, cb)

#undef cb
}

#undef SWITCH_DTYPE

#undef SWITCH_DTYPE
using Mode = param_enumv::Elemwise::Mode;
#define INST(mode) template void megdnn::fallback::ElemwiseImpl::exec_BINARY_INT<mode>()
INST(Mode::ABS_GRAD);
INST(Mode::ADD);
INST(Mode::FLOOR_DIV);
INST(Mode::MAX);
INST(Mode::MIN);
INST(Mode::MOD);
INST(Mode::MUL);
INST(Mode::SIGMOID_GRAD);
INST(Mode::SUB);
INST(Mode::SWITCH_GT0);
INST(Mode::TANH_GRAD);
INST(Mode::LT);
INST(Mode::LEQ);
INST(Mode::EQ);
INST(Mode::SHL);
INST(Mode::SHR);
INST(Mode::FUSE_ADD_RELU);
INST(Mode::RMULH);
INST(Mode::PRELU);
#undef INST

#define INST(mode) \
    template void megdnn::fallback::ElemwiseImpl::exec_BINARY_FLOAT<mode>()
INST(Mode::ABS_GRAD);
INST(Mode::ADD);
INST(Mode::FLOOR_DIV);
INST(Mode::MAX);
INST(Mode::MIN);
INST(Mode::MOD);
INST(Mode::MUL);
INST(Mode::POW);
INST(Mode::SIGMOID_GRAD);
INST(Mode::SUB);
INST(Mode::SWITCH_GT0);
INST(Mode::TANH_GRAD);
INST(Mode::TRUE_DIV);
INST(Mode::LOG_SUM_EXP);
INST(Mode::LT);
INST(Mode::LEQ);
INST(Mode::EQ);
INST(Mode::FUSE_ADD_RELU);
INST(Mode::FUSE_ADD_SIGMOID);
INST(Mode::FUSE_ADD_TANH);
INST(Mode::FAST_TANH_GRAD);
INST(Mode::ATAN2);
INST(Mode::H_SWISH_GRAD);
INST(Mode::FUSE_ADD_H_SWISH);
INST(Mode::SILU_GRAD);
INST(Mode::GELU_GRAD);
INST(Mode::PRELU);
INST(Mode::ASINH_GRAD);
INST(Mode::ACOSH_GRAD);
INST(Mode::ATANH_GRAD);
INST(Mode::SOFTPLUS_GRAD);
INST(Mode::RELU6_GRAD);
INST(Mode::HSIGMOID_GRAD);
#undef INST
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
