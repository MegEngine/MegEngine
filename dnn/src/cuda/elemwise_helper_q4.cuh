/**
 * \file dnn/src/cuda/elemwise_helper_q4.cuh
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

#include "src/cuda/elemwise_helper.cuh"

/*
 * please note that all arithmetics on GPU are 32-bit for best performance; this
 * limits max possible size
 */

namespace megdnn {
namespace cuda {

template <typename ctype>
struct IsNotTypeQ4 {
    static constexpr bool value = !(std::is_same<ctype, dt_qint4>::value ||
                                    std::is_same<ctype, dt_quint4>::value);
};

template <typename ctype>
struct IsTypeQ4 {
    static constexpr bool value = (std::is_same<ctype, dt_qint4>::value ||
                                   std::is_same<ctype, dt_quint4>::value);
};

//! internals for element-wise
namespace elemwise_intl {
#define devfunc __device__ __forceinline__

#if MEGDNN_CC_CUDA
/*!
 * \brief call an operator whose each param are promted to the same ndim and
 *      brdcast_mask
 * \tparam PVis ParamElemVisitor class
 */
template <class Op, int arity, class PVisSrc, class PVisDst, bool BetweenQ4>
struct OpCallerToQ4;

//! specialization for arity == 1
template <class Op, class PVisSrc, class PVisDst>
struct OpCallerToQ4<Op, 1, PVisSrc, PVisDst, false> {
    Op op;
    PVisSrc par_src[1];
    PVisDst par_dst[1];
    using src_ctype = typename PVisSrc::CType;

    devfunc void on(uint32_t access_idx) {
        int32_t idx0 = par_dst[0].idx(access_idx * 2);
        int32_t idx1 = par_dst[0].idx(access_idx * 2 + 1);
        src_ctype src0 = (idx0 >= 0) ? par_src[0].at(idx0) : (src_ctype)0;
        src_ctype src1 = (idx1 >= 0) ? par_src[0].at(idx1) : (src_ctype)0;
        op(access_idx, src0, src1);
    }
};
//! specialization for arity == 2
template <class Op, class PVisSrc, class PVisDst>
struct OpCallerToQ4<Op, 2, PVisSrc, PVisDst, false> {
    Op op;
    PVisSrc par_src[2];
    PVisDst par_dst[1];
    using src_ctype = typename PVisSrc::CType;

    devfunc void on(uint32_t access_idx) {
        int32_t idx0 = par_dst[0].idx(access_idx * 2);
        int32_t idx1 = par_dst[0].idx(access_idx * 2 + 1);
        src_ctype src00 = (idx0 >= 0) ? par_src[0].at(idx0) : (src_ctype)0;
        src_ctype src10 = (idx0 >= 0) ? par_src[1].at(idx0) : (src_ctype)0;
        src_ctype src01 = (idx0 >= 0) ? par_src[0].at(idx1) : (src_ctype)0;
        src_ctype src11 = (idx0 >= 0) ? par_src[1].at(idx1) : (src_ctype)0;

        op(access_idx, src00, src10, src01, src11);
    }
};

template <class Op, class PVisSrc, class PVisDst>
struct OpCallerToQ4<Op, 3, PVisSrc, PVisDst, false> {
    Op op;
    PVisSrc par_src[3];
    PVisDst par_dst[1];
    using src_ctype = typename PVisSrc::CType;

    devfunc void on(uint32_t access_idx) {
        int32_t idx0 = par_dst[0].idx(access_idx * 2);
        int32_t idx1 = par_dst[0].idx(access_idx * 2 + 1);
        src_ctype src00 = (idx0 >= 0) ? par_src[0].at(idx0) : (src_ctype)0;
        src_ctype src10 = (idx0 >= 0) ? par_src[1].at(idx0) : (src_ctype)0;
        src_ctype src20 = (idx0 >= 0) ? par_src[2].at(idx0) : (src_ctype)0;
        src_ctype src01 = (idx0 >= 0) ? par_src[0].at(idx1) : (src_ctype)0;
        src_ctype src11 = (idx0 >= 0) ? par_src[1].at(idx1) : (src_ctype)0;
        src_ctype src21 = (idx0 >= 0) ? par_src[2].at(idx1) : (src_ctype)0;

        op(access_idx, src00, src10, src20, src01, src11, src21);
    }
};

//! specialization for arity == 1
template <class Op, class PVisSrc, class PVisDst>
struct OpCallerToQ4<Op, 1, PVisSrc, PVisDst, true> {
    Op op;
    PVisSrc par_src[1];
    PVisDst par_dst[1];

    devfunc void on(uint32_t access_idx) {
        op(access_idx, par_src[0].at(access_idx));
    }
};
//! specialization for arity == 2
template <class Op, class PVisSrc, class PVisDst>
struct OpCallerToQ4<Op, 2, PVisSrc, PVisDst, true> {
    Op op;
    PVisSrc par_src[2];
    PVisDst par_dst[1];

    devfunc void on(uint32_t access_idx) {
        op(access_idx, par_src[0].at(access_idx), par_src[1].at(access_idx));
    }
};

template <class Op, class PVisSrc, class PVisDst>
struct OpCallerToQ4<Op, 3, PVisSrc, PVisDst, true> {
    Op op;
    PVisSrc par_src[3];
    PVisDst par_dst[1];

    devfunc void on(uint32_t access_idx) {
        op(access_idx, par_src[0].at(access_idx), par_src[1].at(access_idx),
           par_src[2].at(access_idx));
    }
};

/* f}}} */

template <class OpCaller>
__global__ void cuda_kern_q4(OpCaller op_caller, uint32_t size) {
    uint32_t access_idx = blockIdx.x * blockDim.x + threadIdx.x,
             delta = blockDim.x * gridDim.x;
    if (access_idx < size) {
        op_caller.on(access_idx);
        access_idx += delta;
        if (access_idx < size) {
            op_caller.on(access_idx);
            access_idx += delta;
            if (access_idx < size) {
                op_caller.on(access_idx);
            }
        }
    }
}

/* f{{{ UserOpInvoker specializations */

//! run op by promoting all params to same ndim
template <class Op, typename src_ctype, typename dst_ctype, int arity,
          bool BetweenQ4>
class UserOpInvokerQ4 {
    const ElemwiseOpParamN<arity>& m_src_param;
    const ElemwiseOpParamN<1>& m_dst_param;
    cudaStream_t m_stream;
    const Op& m_op;

    void dispatch0() {
        switch (m_dst_param.max_ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1<ndim>();
            MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        }
        on_bad_ndim(m_dst_param.max_ndim);
    }

    template <int ndim>
    void dispatch1() {
        using PVisSrc = typename std::conditional<
                BetweenQ4, ParamVectVisitor<ndim, src_ctype, BCAST_OTHER>,
                ParamElemVisitor<ndim, src_ctype, BCAST_OTHER>>::type;

        typedef OpCallerToQ4<Op, arity, PVisSrc,
                             ParamVectVisitor<ndim, dst_ctype, BCAST_OTHER>,
                             BetweenQ4>
                Caller;

        size_t size = m_dst_param[0].layout.access_bytes();
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern_q4<Caller>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);

        Caller caller;
        caller.op = m_op;
        for (int i = 0; i < arity; ++i)
            caller.par_src[i].host_init(m_src_param[i], grid_size, block_size);
        caller.par_dst[0].host_init(m_dst_param[0], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);
        after_kernel_launch();
    }

public:
    UserOpInvokerQ4(const ElemwiseOpParamN<arity>& src_param,
                    const ElemwiseOpParamN<1>& dst_param, cudaStream_t stream,
                    const Op& op)
            : m_src_param(src_param),
              m_dst_param(dst_param),
              m_stream(stream),
              m_op(op) {
        dispatch0();
    }
};
#endif
/* f}}} */

#undef devfunc
}  // namespace elemwise_intl

template <class Op, typename src_ctype, typename dst_ctype, int arity>
void run_elemwise(const ElemwiseOpParamN<arity>& src_param,
                  const ElemwiseOpParamN<1>& dst_param, cudaStream_t stream,
                  const Op& op = Op());
#if MEGDNN_CC_CUDA

template <class Op, typename src_ctype, typename dst_ctype, int arity>
void run_elemwise(const ElemwiseOpParamN<arity>& src_param,
                  const ElemwiseOpParamN<1>& dst_param, cudaStream_t stream,
                  const Op& op) {
    src_param.assert_initialized();
    dst_param.assert_initialized();
    // TODO: Maybe 2bit?
    megdnn_assert(dst_param[0].layout.dtype.is_low_bit());
    megdnn_assert(dst_param[0].layout.is_contiguous());

    elemwise_intl::UserOpInvokerQ4<Op, src_ctype, dst_ctype, arity,
                                   IsTypeQ4<src_ctype>::value>(
            src_param, dst_param, stream, op);
}

#define INST_RUN_ELEMWISE_LOWBIT(Op, src_ctype, dst_ctype, arity)       \
    template void run_elemwise<Op, src_ctype, dst_ctype, arity>(        \
            const ElemwiseOpParamN<arity>&, const ElemwiseOpParamN<1>&, \
            cudaStream_t, const Op&)
#endif

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
