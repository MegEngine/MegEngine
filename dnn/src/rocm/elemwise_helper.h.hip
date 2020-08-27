/**
 * \file src/rocm/elemwise_helper.h.hip
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief helper utilities for implementing element-wise kernels
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#pragma once

#include "hip_header.h"
#include "src/rocm/utils.h.hip"
#include "src/common/elemwise_helper.cuh"
#include "src/rocm/int_fastdiv.h.hip"

/*
 * please note that all arithmetics on GPU are 32-bit for best performance; this
 * limits max possible size
 */

namespace megdnn {
namespace rocm {

//! internals for element-wise
namespace elemwise_intl {
#define devfunc __device__ __forceinline__

/*!
 * \brief get hip launch specs for element-wise kernel
 * \param kern kernel function address
 * \param size total size of elements
 */
void get_launch_spec(const void* kern, size_t size, int* grid_size,
                     int* block_size);

MEGDNN_NORETURN void on_bad_ndim(int ndim);

/*!
 * \brief broadcast type
 * BCAST_x[0]x[1]...: x[i] == !stride[i]
 */
enum BcastType { BCAST_OTHER, BCAST_101, BCAST_10, BCAST_01, BCAST_FULL };

/*!
 * \brief visitor to access an elemeent in a tensor at given logic index
 * \tparam ctype plain element ctype (i.e. ctype in DTypeTrait)
 * \tparam brdcast_mask bit mask for broadcast of params; (i.e. stride[i] is
 *      0 iff (brdcast_mask & (1<<(ndim-1-i))) is 1.
 *
 * host interface:
 *      void host_init(
 *              const TensorND &tensor, int grid_size, int block_size)
 *
 * device interface:
 *      void thread_init(uint32_t idx)
 *          called on thread entrance, with logical indexing; the index may
 *          go beyond buffer range
 *
 *      ctype* ptr()
 *          return buffer pointer; can be used by specialized OpCaller
 *
 *      void next()
 *          called before moving to next chunk on each thread
 *
 *      int offset(uint32_t idx)
 *          get physical offset from logical index
 *
 *      ctype& at(uint32_t idx)
 *          ptr()[offset(idx)]
 *
 */
template <int ndim, typename ctype, BcastType brd_type>
class ParamElemVisitor;

#define PARAM_ELEM_VISITOR_COMMON_DEV      \
    devfunc ctype* ptr() { return m_ptr; } \
    devfunc ctype& at(uint32_t idx) { return m_ptr[offset(idx)]; }

//! specialization for BCAST_OTHER
template <int ndim, typename ctype>
class ParamElemVisitor<ndim, ctype, BCAST_OTHER> {
    ctype* __restrict m_ptr;
    int m_stride[ndim];

    //! m_shape_highdim[i] = original_shape[i + 1]
#ifdef _MSC_VER
    Uint32Fastdiv m_shape_highdim[ndim > 1 ? ndim - 1 : 1];
#else
    Uint32Fastdiv m_shape_highdim[ndim - 1];
#endif

public:
    static const int NDIM = ndim;

    void host_init(const TensorND& rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t) {}

    devfunc void next() {}

    devfunc int offset(uint32_t idx) {
        int offset = 0;
#pragma unroll
        for (int i = ndim - 1; i >= 1; --i) {
            Uint32Fastdiv& shp = m_shape_highdim[i - 1];
            uint32_t idx_div = idx / shp;
            offset += (idx - idx_div * shp.divisor()) * m_stride[i];
            idx = idx_div;
        }
        offset += idx * m_stride[0];
        return offset;
    }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

/*!
 * \brief specialization for ndim == 3 and BCAST_101
 * (for dimshuffle 'x', 0, 'x')
 *
 * visit: idx / m_shape2 % m_shape1
 */
template <typename ctype>
class ParamElemVisitor<3, ctype, BCAST_101> {
    ctype* __restrict m_ptr;
    StridedDivSeq2 m_shape12;
    int m_stride1;

public:
    static const int NDIM = 3;

    void host_init(const TensorND& rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t idx) { m_shape12.device_init(idx); }

    devfunc void next() { m_shape12.next(); }

    devfunc int offset(uint32_t /* idx */) {
        return m_shape12.get() * m_stride1;
    }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

/*!
 * \brief specialization for ndim == 2 and BCAST_10
 *
 * visit: idx % m_shape1
 */
template <typename ctype>
class ParamElemVisitor<2, ctype, BCAST_10> {
    ctype* __restrict m_ptr;
    StridedDivSeq<false> m_shape1;
    int m_stride1;

public:
    static const int NDIM = 2;

    void host_init(const TensorND& rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t idx) { m_shape1.device_init(idx); }

    devfunc void next() { m_shape1.next(); }

    devfunc int offset(uint32_t /* idx */) { return m_shape1.r() * m_stride1; }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

/*!
 * \brief specialization for ndim == 2 and BCAST_01
 *
 * visit: idx / shape1
 */
template <typename ctype>
class ParamElemVisitor<2, ctype, BCAST_01> {
    ctype* __restrict m_ptr;
    StridedDivSeq<true> m_shape1;
    int m_stride0;

public:
    static const int NDIM = 2;

    void host_init(const TensorND& rv, int grid_size, int block_size);

    devfunc void thread_init(uint32_t idx) { m_shape1.device_init(idx); }

    devfunc void next() { m_shape1.next(); }

    devfunc int offset(uint32_t /* idx */) { return m_shape1.q() * m_stride0; }

    PARAM_ELEM_VISITOR_COMMON_DEV
};

//! specialization for ndim == 1 and BCAST_FULL
template <typename ctype>
class ParamElemVisitor<1, ctype, BCAST_FULL> {
    ctype* __restrict m_ptr;

public:
    static const int NDIM = 1;

    void host_init(const TensorND& rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t) {}

    devfunc void next() {}

    devfunc int offset(uint32_t idx) {
        MEGDNN_MARK_USED_VAR(idx);
        return 0;
    }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

#undef PARAM_ELEM_VISITOR_COMMON_DEV

#if MEGDNN_CC_CUDA
/*
 * OpCaller is used to invoke user operator with loaded element arguments.
 *
 * device interface:
 *      void thread_init(uint32_t idx);
 *
 *      void on(uint32_t idx);
 *
 *      void next();
 */

/*!
 * \brief call user op directly without visiting any params (i.e. arity ==
 *      0)
 */
template <class Op>
struct OpCallerNull {
    Op op;

    devfunc void thread_init(uint32_t) {}

    devfunc void on(uint32_t idx) { op(idx); }

    devfunc void next() {}
};

/*!
 * \brief call an operator whose each param are promted to the same ndim and
 *      brdcast_mask
 * \tparam PVis ParamElemVisitor class
 */
template <class Op, int arity, class PVis>
struct OpCallerUniform;

//! specialization for arity == 1
template <class Op, class PVis>
struct OpCallerUniform<Op, 1, PVis> {
    Op op;
    PVis par[1];

    devfunc void thread_init(uint32_t idx) { par[0].thread_init(idx); }

    devfunc void on(uint32_t idx) { op(idx, par[0].at(idx)); }

    devfunc void next() { par[0].next(); }
};
//! specialization for arity == 2
template <class Op, class PVis>
struct OpCallerUniform<Op, 2, PVis> {
    Op op;
    PVis par[2];

    devfunc void thread_init(uint32_t idx) {
        par[0].thread_init(idx);
        par[1].thread_init(idx);
    }

    devfunc void on(uint32_t idx) { op(idx, par[0].at(idx), par[1].at(idx)); }

    devfunc void next() {
        par[0].next();
        par[1].next();
    }
};
//! specialization for arity == 3
template <class Op, class PVis>
struct OpCallerUniform<Op, 3, PVis> {
    Op op;
    PVis par[3];

    devfunc void thread_init(uint32_t idx) {
        par[0].thread_init(idx);
        par[1].thread_init(idx);
        par[2].thread_init(idx);
    }

    devfunc void on(uint32_t idx) {
        op(idx, par[0].at(idx), par[1].at(idx), par[2].at(idx));
    }

    devfunc void next() {
        par[0].next();
        par[1].next();
        par[2].next();
    }
};

/*!
 * \brief call binary (i.e. arity == 2) operator with different param
 *      visitors
 */
template <class Op, class PVis0, class PVis1>
struct OpCallerBinary {
    Op op;
    PVis0 par0;
    PVis1 par1;

    devfunc void thread_init(uint32_t idx) {
        par0.thread_init(idx);
        par1.thread_init(idx);
    }

    devfunc void on(uint32_t idx) { op(idx, par0.at(idx), par1.at(idx)); }

    devfunc void next() {
        par0.next();
        par1.next();
    }
};

template <class OpCaller>
__global__ void cuda_kern(OpCaller op_caller, uint32_t size) {
    uint32_t idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
             delta = hipBlockDim_x * hipGridDim_x;
    // each thread works on at most 3 elements; see get_launch_spec
    op_caller.thread_init(idx);
    if (idx < size) {
        op_caller.on(idx);
        idx += delta;
        if (idx < size) {
            op_caller.next();
            op_caller.on(idx);
            idx += delta;
            if (idx < size) {
                op_caller.next();
                op_caller.on(idx);
            }
        }
    }
}

//! invoke a user Op passed to run_elemwise
template <class Op, typename ctype, int arity>
class UserOpInvoker;

//! run op by promoting all params to same ndim
template <class Op, typename ctype, int arity>
class UserOpInvokerToSameNdim {
    const ElemwiseOpParamN<arity>& m_param;
    hipStream_t m_stream;
    const Op& m_op;

    void dispatch0() {
        switch (m_param.max_ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1<ndim>();
            MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        }
        on_bad_ndim(m_param.max_ndim);
    }

    template <int ndim>
    void dispatch1() {
        typedef OpCallerUniform<Op, arity,
                                ParamElemVisitor<ndim, ctype, BCAST_OTHER>>
                Caller;
        size_t size = m_param.size;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);

        Caller caller;
        caller.op = m_op;
        for (int i = 0; i < arity; ++i)
            caller.par[i].host_init(m_param[i], grid_size, block_size);

        hipLaunchKernelGGL(fptr,
                           dim3(grid_size), dim3(block_size), 0, m_stream,
                           caller, size);
        after_kernel_launch();
    }

public:
    UserOpInvokerToSameNdim(const ElemwiseOpParamN<arity>& param,
                            hipStream_t stream, const Op& op)
            : m_param(param), m_stream(stream), m_op(op) {
        dispatch0();
    }
};

//! implement general case by UserOpInvokerToSameNdim
template <class Op, typename ctype, int arity>
class UserOpInvoker : public UserOpInvokerToSameNdim<Op, ctype, arity> {
public:
    UserOpInvoker(const ElemwiseOpParamN<arity>& param, hipStream_t stream,
                  const Op& op)
            : UserOpInvokerToSameNdim<Op, ctype, arity>(param, stream, op) {}
};

//! specialization for arity == 0
template <class Op, typename ctype>
class UserOpInvoker<Op, ctype, 0> {
public:
    UserOpInvoker(const ElemwiseOpParamN<0>& param, hipStream_t stream,
                  const Op& op) {
        size_t size = param.size;
        typedef OpCallerNull<Op> Caller;
        Caller caller;
        caller.op = op;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);
        hipLaunchKernelGGL(fptr,
                           dim3(grid_size), dim3(block_size), 0, stream, caller,
                           size);
        after_kernel_launch();
    }
};

#define DEFINE_BRDCAST_DISPATCH_RECEIVERS(_cb_header, _cb_dispatch, _stride) \
    _cb_header(1) {                                                          \
        const ptrdiff_t* stride = _stride;                                   \
        if (!stride[0]) {                                                    \
            return _cb_dispatch(1, BCAST_FULL);                              \
        }                                                                    \
        _cb_dispatch(1, BCAST_OTHER);                                        \
    }                                                                        \
    _cb_header(2) {                                                          \
        const ptrdiff_t* stride = _stride;                                   \
        if (!stride[0] && stride[1]) {                                       \
            return _cb_dispatch(2, BCAST_10);                                \
        }                                                                    \
        if (stride[0] && !stride[1]) {                                       \
            return _cb_dispatch(2, BCAST_01);                                \
        }                                                                    \
        _cb_dispatch(2, BCAST_OTHER);                                        \
    }                                                                        \
    _cb_header(3) {                                                          \
        const ptrdiff_t* stride = _stride;                                   \
        if (!stride[0] && stride[1] && !stride[2]) {                         \
            return _cb_dispatch(3, BCAST_101);                               \
        }                                                                    \
        _cb_dispatch(3, BCAST_OTHER);                                        \
    }

//! specialization for binary opr
template <class Op, typename ctype>
class UserOpInvoker<Op, ctype, 2> {
    bool m_invoked;
    const ElemwiseOpParamN<2>& m_param;
    hipStream_t m_stream;
    const Op& m_op;

    void fallback() {
        megdnn_assert(!m_invoked);
        UserOpInvokerToSameNdim<Op, ctype, 2>(m_param, m_stream, m_op);
        m_invoked = true;
    }

    void dispatch0() {
        switch (m_param[0].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1_##ndim();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
        }
        fallback();
    }

#define cb_header(ndim) void dispatch1_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    dispatch2<ParamElemVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                      m_param[0].layout.stride)
#undef cb_header
#undef cb_dispatch

    template <class PVis0>
    void dispatch2() {
        switch (m_param[1].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch3_##ndim<PVis0>();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
        }
        fallback();
    }

#define cb_header(ndim)    \
    template <class PVis0> \
    void dispatch3_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    do_run<PVis0, ParamElemVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                      m_param[1].layout.stride)
#undef cb_header
#undef cb_dispatch

    template <class PVis0, class PVis1>
    void do_run() {
        megdnn_assert(!m_invoked);
        m_invoked = true;
        typedef OpCallerBinary<Op, PVis0, PVis1> Caller;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        size_t size = m_param.size;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);
        Caller caller;
        caller.op = m_op;
        caller.par0.host_init(m_param[0], grid_size, block_size);
        caller.par1.host_init(m_param[1], grid_size, block_size);
        hipLaunchKernelGGL(fptr,
                           dim3(grid_size), dim3(block_size), 0, m_stream,
                           caller, size);
        after_kernel_launch();
    }

public:
    UserOpInvoker(const ElemwiseOpParamN<2>& param, hipStream_t stream,
                  const Op& op)
            : m_param(param), m_stream(stream), m_op(op) {
        m_invoked = false;
        dispatch0();
        megdnn_assert(m_invoked);
    }
};

#undef DEFINE_BRDCAST_DISPATCH_RECEIVERS

#endif  // MEGDNN_CC_CUDA

#undef devfunc
}  // namespace elemwise_intl

/*!
 * \brief general element-wise kernel launcher
 *
 * \tparam arity number of params for the operator
 * \param param param values for the operator; must have been initialized (i.e.
 *      by calling ElemwiseOpParamN::init_from_given_tensor). The params
 *      can have arbitrary layouts, as long as they share the same total number
 *      of elements.
 * \param op callable with a signature compatible with
 *      `void op(uint32_t idx, ctype& param0, ..., ctype& param[arity - 1])`
 *      if arity == 0, there is only an `idx` input
 */
template <class Op, typename ctype, int arity>
void run_elemwise(const ElemwiseOpParamN<arity>& param, hipStream_t stream,
                  const Op& op = Op());

#if MEGDNN_CC_CUDA
template <class Op, typename ctype, int arity>
void run_elemwise(const ElemwiseOpParamN<arity>& param, hipStream_t stream,
                  const Op& op) {
    param.assert_initialized();
    elemwise_intl::UserOpInvoker<Op, ctype, arity>(param, stream, op);
}

/*!
 * \brief explicit instantialization of run_elemwise for given template params;
 *      used in .cu files, so corresponding run_elemwise can be called from .cpp
 */
#define INST_RUN_ELEMWISE(Op, ctype, arity)       \
    template void run_elemwise<Op, ctype, arity>( \
            const ElemwiseOpParamN<arity>&, hipStream_t, const Op&)
#endif  // MEGDNN_CC_CUDA

}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

