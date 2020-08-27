/**
 * \file src/rocm/relayout/relayout_contiguous.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#pragma once

#include "hip_header.h"
#include "megdnn/basic_types.h"
#include "src/rocm/int_fastdiv.h.hip"
#include "src/rocm/elemwise_helper.h.hip"

#include <stdio.h>

namespace megdnn {
namespace rocm {

void copy_last_contiguous(const TensorND &dst, const TensorND &src,
                          size_t contiguous_size, hipStream_t stream);


void get_last_contiguous_launch_spec(const void *kern, size_t size,
                                     size_t contiguous_size, int *grid_size,
                                     int *block_size);

//! internals for contiguous
namespace contiguous_intl {

#define devfunc __device__ __forceinline__
    /*!
     * \brief contiguous type
     * If the layout is contiguous, then the type is CONTIG_FULL, CONTIG_OTHER
     * otherwise.
     */
    enum ContigType {
        CONTIG_OTHER,
        CONTIG_FULL
    };

    /*!
    * \brief visitor to access an element in a tensor at given logic index
    * \tparam ctype plain element ctype (i.e. ctype in DTypeTrait)
    * \tparam contig_mask bit mask for contig of params;
    *
    * host interface:
    *      void host_init(
    *              const TensorND &tensor, int grid_size, int block_size)
    *
    * device interface:
    *      void thread_init(uint32_t idx)
    *          called on thread entrance, with logical indexing; the index
    y
    *          go beyond buffer range
    *
    *      ctype* ptr()
    *          return buffer pointer; can be used by specialized OpCaller
    *
    *      int offset(uint32_t idx)
    *          get physical offset from logical index
    *
    *      ctype& at(uint32_t idx)
    *          ptr()[offset(idx)]
    *
    */
    template<int ndim, typename ctype, ContigType contig_type>
    class ParamElemVisitor;

    /* f{{{ ParamElemVisitor specialization */

#define PARAM_ELEM_VISITOR_COMMON_DEV \
            devfunc ctype* ptr() { \
                return m_ptr; \
            } \
            devfunc ctype& at(uint32_t idx) { \
                return m_ptr[offset(idx)]; \
            } \

    //! specialization for CONTIG_OTHER
    template<int ndim, typename ctype>
    class ParamElemVisitor<ndim, ctype, CONTIG_OTHER> {
        ctype * __restrict m_ptr;
        int m_stride[ndim];

        //! m_shape_highdim[i] = original_shape[i + 1]
#ifdef _MSC_VER
        Uint32Fastdiv m_shape_highdim[ndim > 1 ? ndim - 1 : 1];
#else
        Uint32Fastdiv m_shape_highdim[ndim - 1];
#endif

        public:
            static const int NDIM = ndim;

            void host_init(const TensorND &rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
            devfunc void thread_init(uint32_t) {
            }

            devfunc void next() {
            }

            devfunc int offset(uint32_t idx) {
                int offset = 0;
#pragma unroll
                for (int i = ndim - 1; i >= 1; -- i) {
                    Uint32Fastdiv &shp = m_shape_highdim[i - 1];
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

    //! specialization for CONTIG_FULL
    template<int ndim, typename ctype>
    class ParamElemVisitor<ndim, ctype, CONTIG_FULL> {
        ctype * __restrict m_ptr;

        public:
            static const int NDIM = ndim;

            void host_init(const TensorND &rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
            devfunc void thread_init(uint32_t) {
            }

            devfunc void next() {
            }

            devfunc int offset(uint32_t idx) {
                return idx;
            }

            PARAM_ELEM_VISITOR_COMMON_DEV
#endif

    };

#undef PARAM_ELEM_VISITOR_COMMON_DEV

    /* f}}} */ 

    template <class PVis0, class PVis1>
    struct OpCallerBinaryContiguous {
        PVis0 par0;
        PVis1 par1;
    };

    /* f{{{ hip  kern */

#if MEGDNN_CC_CUDA
    /*!
     * \brief hip kern for the last axis stride is contiguous and
     *     contiguous_size is small
     */
    template <typename OpCaller>
    __global__ void hip_last_contiguous_kern(OpCaller op_caller,
                                              uint32_t contiguous_size,
                                              uint32_t size) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= contiguous_size;
        if (idx >= size) return;

        size_t remain_size = size - idx;
        remain_size =
            remain_size > contiguous_size ? contiguous_size : remain_size;

        int offset0 = op_caller.par0.offset(idx);
        int offset1 = op_caller.par1.offset(idx);

    #pragma unroll
        for (size_t i = 0; i < remain_size; ++i) {
            op_caller.par0.ptr()[offset0++] = op_caller.par1.ptr()[offset1++];
        }
    }

    /*!
     * \brief hip kern for the last axis stride is contiguous and
     *     contiguous_size is large
     *
     * \param op_caller user op caller
     * \param contiguous_size the size of last contiguous elements
     * \param size total number of elements
     * \param contig_size_each_block the contiguous elements visited in each
     *     block
     * \param contig_block_size the block size used to visit the contiguous
     *     element
     */
    template <typename OpCaller>
    __global__ void hip_last_contiguous_large_kern(
            OpCaller op_caller, uint32_t contiguous_size, uint32_t size,
            uint32_t contig_size_each_block, uint32_t contig_block_size) {
        // Every block manipulate a sub contiguous elements
        //! The \p contig_idx contiguous elements
        uint32_t contig_idx = blockIdx.x / contig_block_size;
        //! The idx in the current contiguous elemwise
        uint32_t contig_block_idx = blockIdx.x - contig_idx * contig_block_size;
        uint32_t idx = contig_idx * contiguous_size + contig_block_idx *
            contig_size_each_block;
        if (idx >= size) return;

        uint32_t remain = contiguous_size - contig_block_idx *
            contig_size_each_block;
        if (remain > contig_size_each_block) remain = contig_size_each_block;

        uint32_t physical_idx0 = op_caller.par0.offset(idx);
        uint32_t physical_idx1 = op_caller.par1.offset(idx);

        int i = threadIdx.x;
        while (i < remain) {
            op_caller.par0.ptr()[physical_idx0 + i] =
                op_caller.par1.ptr()[physical_idx1 + i];
            i += blockDim.x;
        }
    }

    /* f}}} */


#define DEFINE_CONTIG_RECEIVER(_ndim, _cb_header, _cb_dispatch, _layout) \
    _cb_header(_ndim) { \
        if (_layout.is_contiguous()) { \
            return _cb_dispatch(_ndim, CONTIG_FULL); \
        } \
        return _cb_dispatch(_ndim, CONTIG_OTHER); \
    } \

    //! invoke a user Op passed to run_elemwise
    template<typename ctype, int arity>
    class UserOpInvoker;

    /* f{{{ UserOpInvoker specializations */

    //! specialization for binary opr
    template<typename ctype>
    class UserOpInvoker<ctype, 2> {
        bool m_invoked;
        const ElemwiseOpParamN<2> &m_param;
        hipStream_t m_stream;
        const size_t m_contiguous_size;

        void dispatch0() {
            switch(m_param[0].layout.ndim) {
#define cb(ndim) \
                case ndim: return dispatch1_##ndim();
                MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
            }
        }

#define cb_header(ndim) void dispatch1_##ndim()
#define cb_dispatch(ndim, contig_mask) \
        dispatch2<ParamElemVisitor<ndim, ctype, contig_mask> >()
DEFINE_CONTIG_RECEIVER(1, cb_header, cb_dispatch, m_param[0].layout)
DEFINE_CONTIG_RECEIVER(2, cb_header, cb_dispatch, m_param[0].layout)
DEFINE_CONTIG_RECEIVER(3, cb_header, cb_dispatch, m_param[0].layout)
DEFINE_CONTIG_RECEIVER(4, cb_header, cb_dispatch, m_param[0].layout)
DEFINE_CONTIG_RECEIVER(5, cb_header, cb_dispatch, m_param[0].layout)
DEFINE_CONTIG_RECEIVER(6, cb_header, cb_dispatch, m_param[0].layout)
DEFINE_CONTIG_RECEIVER(7, cb_header, cb_dispatch, m_param[0].layout)
#undef cb_header
#undef cb_dispatch


        template<class PVis0>
        void dispatch2() {
            switch(m_param[1].layout.ndim) {
#define cb(ndim) \
                case ndim: return dispatch3_##ndim<PVis0>();
                MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
            }
        }

#define cb_header(ndim) \
    template<class PVis0> \
    void dispatch3_##ndim()
#define cb_dispatch(ndim, contig_mask) \
        do_run<PVis0, ParamElemVisitor<ndim, ctype, contig_mask> >()
DEFINE_CONTIG_RECEIVER(1, cb_header, cb_dispatch, m_param[1].layout)
DEFINE_CONTIG_RECEIVER(2, cb_header, cb_dispatch, m_param[1].layout)
DEFINE_CONTIG_RECEIVER(3, cb_header, cb_dispatch, m_param[1].layout)
DEFINE_CONTIG_RECEIVER(4, cb_header, cb_dispatch, m_param[1].layout)
DEFINE_CONTIG_RECEIVER(5, cb_header, cb_dispatch, m_param[1].layout)
DEFINE_CONTIG_RECEIVER(6, cb_header, cb_dispatch, m_param[1].layout)
DEFINE_CONTIG_RECEIVER(7, cb_header, cb_dispatch, m_param[1].layout)
#undef cb_header
#undef cb_dispatch

        template<class PVis0, class PVis1>
        void do_run() {
            megdnn_assert(!m_invoked);
            m_invoked = true;
            typedef OpCallerBinaryContiguous<PVis0, PVis1> Caller;
            size_t size = m_param.size;
            int grid_size, block_size;
            if (m_contiguous_size > 32) {
                void (*fptr)(Caller, uint32_t, uint32_t, uint32_t, uint32_t);
                fptr = hip_last_contiguous_large_kern<Caller>;
                safe_size_in_kern(size);
                block_size = m_contiguous_size;
                if (block_size > 256) {
                    block_size = 256;
                } else if (block_size > 128) {
                    block_size = 128;
                } else if (block_size > 64) {
                    block_size = 64;
                } else {
                    block_size = 32;
                }

                const uint32_t MAX_CONTIG_SIZE_EACH_BLOCK = 1024;
                uint32_t contig_block_size = 1;
                uint32_t contig_size_each_block = m_contiguous_size;
                if (m_contiguous_size > MAX_CONTIG_SIZE_EACH_BLOCK) {
                    contig_size_each_block = MAX_CONTIG_SIZE_EACH_BLOCK;
                    contig_block_size =
                            (m_contiguous_size + contig_size_each_block - 1) /
                            contig_size_each_block;
                }
                grid_size = (size + m_contiguous_size - 1) / m_contiguous_size *
                            contig_block_size;
                Caller caller;
                caller.par0.host_init(m_param[0], grid_size, block_size);
                caller.par1.host_init(m_param[1], grid_size, block_size);
                hipLaunchKernelGGL(fptr, dim3(grid_size), dim3(block_size), 0,
                                   m_stream, caller, m_contiguous_size, size,
                                   contig_size_each_block, contig_block_size);
            } else {
                void (*fptr)(Caller, uint32_t, uint32_t);
                fptr = hip_last_contiguous_kern<Caller>;
                get_last_contiguous_launch_spec(
                        reinterpret_cast<const void *>(fptr),
                        size, m_contiguous_size, &grid_size, &block_size);
                Caller caller;
                caller.par0.host_init(m_param[0], grid_size, block_size);
                caller.par1.host_init(m_param[1], grid_size, block_size);
                hipLaunchKernelGGL(fptr, dim3(grid_size), dim3(block_size), 0,
                                   m_stream, caller, m_contiguous_size, size);
            }
            after_kernel_launch();
        }

        public:
            UserOpInvoker(const ElemwiseOpParamN<2> &param, hipStream_t stream,
                    const size_t contiguous_size):
                m_param(param), m_stream(stream),
                m_contiguous_size(contiguous_size)
            {
                m_invoked = false;
                dispatch0();
                megdnn_assert(m_invoked);
            }
    };

#undef DEFINE_CONTIG_RECEIVER

    /* f}}} */ 

#endif

#undef devfunc

} // namespace contiguous_intl

} // rocm
} // megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
