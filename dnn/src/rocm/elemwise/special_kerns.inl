/**
 * \file dnn/src/rocm/elemwise/special_kerns.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "./special_kerns.h.hip"

namespace megdnn {
namespace rocm {
namespace elemwise_intl {

template <typename ctype, bool c_is_scalar>
struct FuseMulAdd3Op {
    typedef ctype* __restrict__ bufptr_t;
    bufptr_t m_dst, m_src2;

    __device__ __forceinline__ void operator()(uint32_t idx, int off0, int off1,
                                               bufptr_t src0, bufptr_t src1) {
        m_dst[idx] = src0[off0] * src1[off1] + m_src2[c_is_scalar ? 0 : off0];
    }
};

template <typename ctype>
struct FuseMulAdd4Op {
    typedef ctype* __restrict__ bufptr_t;
    bufptr_t m_dst, m_src2, m_src3;

    __device__ __forceinline__ void operator()(uint32_t idx, int off0, int off1,
                                               bufptr_t src0, bufptr_t src1) {
        m_dst[idx] = static_cast<ctype>(src0[off0]) *
                             static_cast<ctype>(src1[off1]) +
                     static_cast<ctype>(m_src2[off0]) *
                             static_cast<ctype>(m_src3[off1]);
    }
};

//! wrap an op so the special OpCaller can be selected by template matching
template <class Op>
class FuseOpWrapper {
    const Op& m_op;

public:
    FuseOpWrapper(const Op& op) : m_op(op) {}

    operator const Op&() const { return m_op; }
};

template <class Op, class PVis0, class PVis1>
struct OpCallerBinary<FuseOpWrapper<Op>, PVis0, PVis1> {
    Op op;
    PVis0 par0;
    PVis1 par1;

    __device__ __forceinline__ void thread_init(uint32_t idx) {
        par0.thread_init(idx);
        par1.thread_init(idx);
    }

    __device__ __forceinline__ void on(uint32_t idx) {
        op(idx, par0.offset(idx), par1.offset(idx), par0.ptr(), par1.ptr());
    }

    __device__ __forceinline__ void next() {
        par0.next();
        par1.next();
    }
};

template <class Op, class PVis>
struct OpCallerUniform<FuseOpWrapper<Op>, 2, PVis> {
    Op op;
    PVis par[2];

    __device__ __forceinline__ void thread_init(uint32_t idx) {
        par[0].thread_init(idx);
        par[1].thread_init(idx);
    }

    __device__ __forceinline__ void on(uint32_t idx) {
        op(idx, par[0].offset(idx), par[1].offset(idx), par[0].ptr(),
           par[1].ptr());
    }

    __device__ __forceinline__ void next() {
        par[0].next();
        par[1].next();
    }
};

}  // namespace elemwise_intl

namespace {
template <typename ctype, class Op, int arity>
void run_fuse_elemwise(Op& op, const ElemwiseOpParamN<arity>& param,
                       hipStream_t stream) {
    param.assert_initialized();
    ElemwiseOpParamN<2> p2 = *static_cast<const ElemwiseOpParamN<2>*>(
            static_cast<const void*>(&param));
    elemwise_intl::UserOpInvoker<elemwise_intl::FuseOpWrapper<Op>, ctype, 2>(
            p2, stream, op);
}
}  // anonymous namespace

template <bool c_is_scalar, typename ctype>
void kern_fuse_mul_add3(ctype* dest, const ElemwiseOpParamN<3>& param,
                        hipStream_t stream) {
    elemwise_intl::FuseMulAdd3Op<ctype, c_is_scalar> op;
    op.m_dst = dest;
    op.m_src2 = param[2].ptr<ctype>();
    run_fuse_elemwise<ctype>(op, param, stream);
}

template <typename ctype>
void kern_fuse_mul_add4(ctype* dest, const ElemwiseOpParamN<4>& param,
                        hipStream_t stream) {
    elemwise_intl::FuseMulAdd4Op<ctype> op;
    op.m_dst = dest;
    op.m_src2 = param[2].ptr<ctype>();
    op.m_src3 = param[3].ptr<ctype>();
    run_fuse_elemwise<ctype>(op, param, stream);
}

#define INST(_dt)                                                              \
    template void kern_fuse_mul_add3<true>(                                    \
            DTypeTrait<_dt>::ctype*, const ElemwiseOpParamN<3>&, hipStream_t); \
    template void kern_fuse_mul_add3<false>(                                   \
            DTypeTrait<_dt>::ctype*, const ElemwiseOpParamN<3>&, hipStream_t); \
    template void kern_fuse_mul_add4(DTypeTrait<_dt>::ctype*,                  \
                                     const ElemwiseOpParamN<4>&, hipStream_t);


// vim: ft=cpp syntax=cpp.doxygen

