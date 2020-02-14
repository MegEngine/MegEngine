/**
 * \file dnn/src/naive/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/elemwise/opr_impl.h"
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise_helper.cuh"
#include "src/common/utils.h"
#include "src/naive/elemwise/kern_caller.h"
#include "src/naive/handle.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_elemwise)

namespace megdnn {
namespace naive {
namespace {

template <bool c_is_scalar, typename ctype>
void fuse_mul_add3(ctype* dest, const ElemwiseOpParamN<3>& param) {
    auto iter0 = tensor_iter_valonly<ctype>(param[0]).begin();
    auto iter1 = tensor_iter_valonly<ctype>(param[1]).begin();
    auto p2 = param[2].ptr<ctype>();

    for (size_t i = 0; i < param.size; ++i) {
        auto off0 = iter0.offset();
        dest[i] = (*iter0) * (*iter1) + p2[c_is_scalar ? 0 : off0];
        ++iter0;
        ++iter1;
    }
}

template <typename ctype>
void fuse_mul_add4(ctype* dest, const ElemwiseOpParamN<4>& param) {
    auto iter0 = tensor_iter_valonly<ctype>(param[0]).begin();
    auto iter1 = tensor_iter_valonly<ctype>(param[1]).begin();
    auto p2 = param[2].ptr<ctype>(), p3 = param[3].ptr<ctype>();

    for (size_t i = 0; i < param.size; ++i) {
        auto off0 = iter0.offset(), off1 = iter1.offset();
        dest[i] = (*iter0) * (*iter1) + p2[off0] * p3[off1];
        ++iter0;
        ++iter1;
    }
}

}  // anonymous namespace

#define on_arity_dispatched_cb_dtype(_dt)                              \
    if (m_dst->layout.dtype == _dt()) {                                \
        using dtrait = DTypeTrait<_dt>;                                \
        using ctype = dtrait::ctype;                                   \
        return ModeDispatcher<arity, dtrait::category, ctype>::run(    \
                static_cast<HandleImpl*>(handle()), src, m_param.mode, \
                m_dst->ptr<ctype>());                                  \
    }

#define _cb_dispatch_mode(_m)                                                  \
    case Mode::_m:                                                             \
        do {                                                                   \
            using KernImpl =                                                   \
                    ElemwiseKern<megcorePlatformCPU,                           \
                                 param_enumv::Elemwise::Mode::_m, ctype>;      \
            MIDOUT_BEGIN(megdnn_naive_elemwise,                                \
                         midout_iv(param_enumv::Elemwise::Mode::_m)) {         \
                MEGDNN_DISPATCH_CPU_KERN(                                      \
                        handle,                                                \
                        ElemArithKernCaller<arity MEGDNN_COMMA KernImpl>::run( \
                                dst, src));                                    \
                return;                                                        \
            }                                                                  \
            MIDOUT_END();                                                      \
        } while (0);

#define IMPL_MODE_DISPATCHER(_arity, _dtype_cat)                            \
    template <typename ctype>                                               \
    struct ElemwiseForwardImpl::ModeDispatcher<_arity, _dtype_cat, ctype> { \
        static constexpr int arity = _arity;                                \
        static void run(HandleImpl* handle,                                 \
                        const ElemwiseOpParamN<arity>& src, Mode mode,      \
                        ctype* dst) {                                       \
            switch (mode) {                                                 \
                FOREACH(_cb_dispatch_mode)                                  \
                default:                                                    \
                    megdnn_throw("bad mode");                               \
            }                                                               \
        }                                                                   \
    }

#include "src/common/elemwise/opr_impl_body.inl"

template <typename ctype, bool c_is_scalar>
void ElemwiseForwardImpl::impl_fuse_mul_add3(
        const ElemwiseOpParamN<3>& params) {
    auto dptr = m_dst->ptr<ctype>();
    MEGDNN_DISPATCH_CPU_KERN_OPR(fuse_mul_add3<c_is_scalar>(dptr, params));
}

template <typename ctype>
void ElemwiseForwardImpl::impl_fuse_mul_add4(
        const ElemwiseOpParamN<4>& params) {
    auto dptr = m_dst->ptr<ctype>();
    MEGDNN_DISPATCH_CPU_KERN_OPR(fuse_mul_add4(dptr, params));
}
}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
