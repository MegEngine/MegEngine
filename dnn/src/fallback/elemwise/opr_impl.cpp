/**
 * \file dnn/src/fallback/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./opr_impl.h"

#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_elemwise_unary)
MIDOUT_DECL(megdnn_fallback_elemwise_binary)
MIDOUT_DECL(megdnn_fallback_elemwise_exec_UNARY_INT)
MIDOUT_DECL(megdnn_fallback_elemwise_exec_UNARY_FLOAT)
MIDOUT_DECL(megdnn_fallback_elemwise_exec_BINARY_INT)
MIDOUT_DECL(megdnn_fallback_elemwise_exec_BINARY_FLOAT)

namespace megdnn {
namespace fallback {

template <typename dtype, uint32_t mode>
void ElemwiseImpl::unary_kern(const ElemwiseOpParamN<1>& param) {
    using ctype = typename DTypeTrait<dtype>::ctype;
    using Kern = ElemwiseKern<megcorePlatformCPU, mode, ctype>;
    MIDOUT_BEGIN(megdnn_fallback_elemwise_unary, ctype, midout_iv(mode)) {
        ctype* __restrict src = param[0].ptr<ctype>();
        ctype* __restrict dst = m_dst->ptr<ctype>();

        // only specialize for the most common 1-dim case
        if (param.max_ndim == 1) {
            MIDOUT_BEGIN(megdnn_fallback_elemwise_unary, ctype, midout_iv(mode),
                         midout_iv(1)) {
                auto tot = param.size;
                auto stride = param[0].layout.stride[0];
                MEGDNN_DISPATCH_CPU_KERN_OPR({
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

template <typename dtype, uint32_t mode>
void ElemwiseImpl::binary_kern(const ElemwiseOpParamN<2>& param) {
    using ctype = typename DTypeTrait<dtype>::ctype;
    using Kern = ElemwiseKern<megcorePlatformCPU, mode, ctype>;

    MIDOUT_BEGIN(megdnn_fallback_elemwise_binary, ctype, midout_iv(mode)) {
        ctype* __restrict a = param[0].ptr<ctype>();
        ctype* __restrict b = param[1].ptr<ctype>();
        ctype* __restrict dst = m_dst->ptr<ctype>();

        if (param.max_ndim == 1) {
            MIDOUT_BEGIN(megdnn_fallback_elemwise_binary, ctype,
                         midout_iv(mode), midout_iv(1)) {
                auto tot = param.size;
                auto as = param[0].layout.stride[0],
                     bs = param[1].layout.stride[0];
                MEGDNN_DISPATCH_CPU_KERN_OPR({
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
                MIDOUT_BEGIN(megdnn_fallback_elemwise_binary, ctype,
                             midout_iv(mode), midout_iv(21)) {
                    auto as = param[0].layout.stride[0],
                         bs0 = param[1].layout.stride[0],
                         bs1 = param[1].layout.stride[1];
                    auto n0 = param[1].layout.shape[0],
                         n1 = param[1].layout.shape[1];
                    MEGDNN_DISPATCH_CPU_KERN_OPR({
                        ptrdiff_t toff = 0;
                        for (size_t i = 0; i < n0; ++i) {
                            for (size_t j = 0; j < n1; ++j) {
                                dst[toff] = Kern::apply(a[as * toff],
                                                        b[bs0 * i + bs1 * j]);
                                ++toff;
                            }
                        }
                    });
                    return;
                }
                MIDOUT_END();
            }

            MIDOUT_BEGIN(megdnn_fallback_elemwise_binary, ctype,
                         midout_iv(mode), midout_iv(22)) {
                megdnn_assert(param[1].layout.ndim == 1);
                auto bs = param[1].layout.stride[0],
                     as0 = param[0].layout.stride[0],
                     as1 = param[0].layout.stride[1];
                auto n0 = param[0].layout.shape[0],
                     n1 = param[0].layout.shape[1];

                MEGDNN_DISPATCH_CPU_KERN_OPR({
                    ptrdiff_t toff = 0;
                    for (size_t i = 0; i < n0; ++i) {
                        for (size_t j = 0; j < n1; ++j) {
                            dst[toff] = Kern::apply(a[as0 * i + as1 * j],
                                                    b[toff * bs]);
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
                MIDOUT_BEGIN(megdnn_fallback_elemwise_binary, ctype,
                             midout_iv(mode), midout_iv(31)) {
                    auto as = param[0].layout.stride[0],
                         bs = param[1].layout.stride[1];
                    auto n0 = param[1].layout.shape[0],
                         n1 = param[1].layout.shape[1],
                         n2 = param[1].layout.shape[2];
                    MEGDNN_DISPATCH_CPU_KERN_OPR({
                        size_t toff = 0;
                        for (size_t i = 0; i < n0; ++i) {
                            for (size_t j = 0; j < n1; ++j) {
                                for (size_t k = 0; k < n2; ++k) {
                                    dst[toff] = Kern::apply(a[as * toff],
                                                            b[bs * j]);
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
                MIDOUT_BEGIN(megdnn_fallback_elemwise_binary, ctype,
                             midout_iv(mode), midout_iv(32)) {
                    auto as = param[0].layout.stride[1],
                         bs = param[1].layout.stride[0];
                    auto n0 = param[0].layout.shape[0],
                         n1 = param[0].layout.shape[1],
                         n2 = param[0].layout.shape[2];
                    MEGDNN_DISPATCH_CPU_KERN_OPR({
                        size_t toff = 0;
                        for (size_t i = 0; i < n0; ++i) {
                            for (size_t j = 0; j < n1; ++j) {
                                for (size_t k = 0; k < n2; ++k) {
                                    dst[toff] = Kern::apply(a[as * j],
                                                            b[bs * toff]);
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

void ElemwiseImpl::exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) {
    if (!dst.layout.is_contiguous())
        return naive::ElemwiseForwardImpl::exec(srcs, dst);

    m_src = &srcs;
    m_dst = &dst;

#define CONCAT2(a, b, c) a##_##b##_##c
#define CONCAT(a, b, c) CONCAT2(a, b, c)
#define SWITCH_MODE_CB(_mode)                                           \
    case Mode::_mode:                                                   \
        MIDOUT_BEGIN(CONCAT(megdnn_fallback_elemwise_exec, ARITY, CAT), \
                     midout_iv(Mode::_mode)) {                          \
            return CONCAT(exec, ARITY,                                  \
                          CAT)<param_enumv::Elemwise::Mode::_mode>();   \
        }                                                               \
        MIDOUT_END();
#define SWITCH_MODE                                          \
    switch (m_param.mode) {                                  \
        CONCAT(MEGDNN_FOREACH_ELEMWISE_MODE, ARITY, CAT)     \
        (SWITCH_MODE_CB) default : megdnn_throw("bad mode"); \
    }

    if (dst.layout.dtype.category() == DTypeCategory::INT) {
#define CAT INT
        if (srcs.size() == 1) {
#define ARITY UNARY
            SWITCH_MODE
#undef ARITY
        }

        if (srcs.size() == 2) {
#define ARITY BINARY
            SWITCH_MODE
#undef ARITY
        }
#undef CAT
    } else if (dst.layout.dtype.category() == DTypeCategory::FLOAT) {
#define CAT FLOAT
        if (srcs.size() == 1) {
#define ARITY UNARY
            SWITCH_MODE
#undef ARITY
        }

        if (srcs.size() == 2) {
#define ARITY BINARY
            SWITCH_MODE
#undef ARITY
        }
#undef CAT
    }

#undef cb
    naive::ElemwiseForwardImpl::exec(srcs, dst);
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

    SWITCH_DTYPE(INT, cb)

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

template <uint32_t mode>
void ElemwiseImpl::exec_BINARY_INT() {
    auto param = make_elemwise_op_param<2>();
#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return binary_kern<_dt, mode>(param);

    SWITCH_DTYPE(INT, cb)

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

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
