/**
 * \file dnn/src/cuda/relayout/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/int_fastdiv.cuh"
#include "src/cuda/relayout/param_visitor.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

void copy_noncontig_general(const TensorND &dst, const TensorND &src, cudaStream_t stream);
void get_launch_spec_unroll16(const void *kern, size_t size, int *grid_size, int *block_size);
void get_launch_spec_unroll4(const void *kern, size_t size, int *grid_size, int *block_size);

//! internals for general
namespace noncontig_general_intl {

#define devfunc __device__ __forceinline__
/*!
 * \brief contiguous type
 * If the layout is contiguous, then the type is CONTIG_FULL, CONTIG_OTHER
 * otherwise.
 */

template <class PVis0, class PVis1>
struct OpCallerBinaryNoContiguous {
    PVis0 par0;
    PVis1 par1;
};

/* f{{{ cuda kern */

#if MEGDNN_CC_CUDA
/*!
 * \brief cuda kern for general case replacing elemwise_helper
 */
template <typename OpCaller>
__global__ void cuda_kern_general(OpCaller op_caller, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, delta = blockDim.x * gridDim.x;
    if (idx < size) {
        int offset0 = op_caller.par0.offset(idx);
        int offset1 = op_caller.par1.offset(idx);
        op_caller.par0.ptr()[offset0] = op_caller.par1.ptr()[offset1];
        idx += delta;
        if (idx < size) {
            offset0 = op_caller.par0.offset(idx);
            offset1 = op_caller.par1.offset(idx);
            op_caller.par0.ptr()[offset0] = op_caller.par1.ptr()[offset1];
            idx += delta;
            if (idx < size) {
                offset0 = op_caller.par0.offset(idx);
                offset1 = op_caller.par1.offset(idx);
                op_caller.par0.ptr()[offset0] = op_caller.par1.ptr()[offset1];
            }
        }
    }
}

/*!
 * \brief cuda kern for last two shape transpose
 *        type is byte, dst is last contig and %4 = 0
 */
template <typename OpCaller>
__global__ void dst_pack_kern(OpCaller op_caller, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, delta = blockDim.x * gridDim.x;
    //! each thread fetch 4 elements
    uint32_t num = size / 4;
    if (idx < num) {
        uint32_t offset = idx * 4;
        uchar1 val1 = *reinterpret_cast<uchar1 *>(&op_caller.par1.at(offset));
        uchar1 val2 = *reinterpret_cast<uchar1 *>(&op_caller.par1.at(offset + 1));
        uchar1 val3 = *reinterpret_cast<uchar1 *>(&op_caller.par1.at(offset + 2));
        uchar1 val4 = *reinterpret_cast<uchar1 *>(&op_caller.par1.at(offset + 3));

        *reinterpret_cast<uchar4 *>(&op_caller.par0.at(offset)) = uchar4{val1.x, val2.x, val3.x, val4.x};

        idx += delta;
    }
}

/*!
 * \brief cuda kern for last two shape transpose
 *        type is byte, src is last contig and %4 = 0
 */
template <typename OpCaller>
__global__ void src_pack_kern(OpCaller op_caller, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, delta = blockDim.x * gridDim.x;
    //! each thread fetch 4 elements
    uint32_t num = size / 4;
    if (idx < num) {
        uint32_t offset = idx * 4;
        uchar4 val = *reinterpret_cast<uchar4 *>(&op_caller.par1.at(offset));
        *reinterpret_cast<uchar1 *>(&op_caller.par0.at(offset)) = uchar1{val.x};
        *reinterpret_cast<uchar1 *>(&op_caller.par0.at(offset + 1)) = uchar1{val.y};
        *reinterpret_cast<uchar1 *>(&op_caller.par0.at(offset + 2)) = uchar1{val.z};
        *reinterpret_cast<uchar1 *>(&op_caller.par0.at(offset + 3)) = uchar1{val.w};

        idx += delta;
    }
}
/* f}}} */

#define DEFINE_CONTIG_RECEIVER(_ndim, _cb_header, _cb_dispatch, _layout) \
    _cb_header(_ndim) {                                                  \
        if (_layout.is_contiguous()) {                                   \
            return _cb_dispatch(_ndim, CONTIG_FULL);                     \
        }                                                                \
        return _cb_dispatch(_ndim, CONTIG_OTHER);                        \
    }

//! invoke a user Op passed to run_elemwise
template <typename ctype, int arity>
class UserOpInvoker;

/* f{{{ UserOpInvoker specializations */

//! specialization for binary opr
template <typename ctype>
class UserOpInvoker<ctype, 2> {
    bool m_invoked;
    const ElemwiseOpParamN<2> &m_param;
    cudaStream_t m_stream;
    size_t m_rw_size;

    void dispatch0() {
        switch (m_param[0].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1_##ndim();
            MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        }
    }

#define cb_header(ndim) void dispatch1_##ndim()
#define cb_dispatch(ndim, contig_mask) dispatch2<ParamElemVisitor<ndim, ctype, contig_mask>>()
    DEFINE_CONTIG_RECEIVER(1, cb_header, cb_dispatch, m_param[0].layout)
    DEFINE_CONTIG_RECEIVER(2, cb_header, cb_dispatch, m_param[0].layout)
    DEFINE_CONTIG_RECEIVER(3, cb_header, cb_dispatch, m_param[0].layout)
    DEFINE_CONTIG_RECEIVER(4, cb_header, cb_dispatch, m_param[0].layout)
    DEFINE_CONTIG_RECEIVER(5, cb_header, cb_dispatch, m_param[0].layout)
    DEFINE_CONTIG_RECEIVER(6, cb_header, cb_dispatch, m_param[0].layout)
    DEFINE_CONTIG_RECEIVER(7, cb_header, cb_dispatch, m_param[0].layout)
#undef cb_header
#undef cb_dispatch

    template <class PVis0>
    void dispatch2() {
        switch (m_param[1].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch3_##ndim<PVis0>();
            MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        }
    }

#define cb_header(ndim)    \
    template <class PVis0> \
    void dispatch3_##ndim()
#define cb_dispatch(ndim, contig_mask) do_run<PVis0, ParamElemVisitor<ndim, ctype, contig_mask>>()
    DEFINE_CONTIG_RECEIVER(1, cb_header, cb_dispatch, m_param[1].layout)
    DEFINE_CONTIG_RECEIVER(2, cb_header, cb_dispatch, m_param[1].layout)
    DEFINE_CONTIG_RECEIVER(3, cb_header, cb_dispatch, m_param[1].layout)
    DEFINE_CONTIG_RECEIVER(4, cb_header, cb_dispatch, m_param[1].layout)
    DEFINE_CONTIG_RECEIVER(5, cb_header, cb_dispatch, m_param[1].layout)
    DEFINE_CONTIG_RECEIVER(6, cb_header, cb_dispatch, m_param[1].layout)
    DEFINE_CONTIG_RECEIVER(7, cb_header, cb_dispatch, m_param[1].layout)
#undef cb_header
#undef cb_dispatch

    int try_int8_pack() {
        //! return-type -1: general kernel
        //!              0: src pack int8
        //!              1: dst pack int8
        auto src = m_param[1].layout;
        auto dst = m_param[0].layout;
        bool dst_contig = dst.stride[dst.ndim - 1] == 1;
        bool src_contig = src.stride[src.ndim - 1] == 1;
        if (!src_contig && !dst_contig) {
            return -1;
        }
        if (std::is_same<ctype, dt_float16>::value)
            return -1;
        else if (std::is_same<ctype, dt_int32>::value)
            return -1;
        else if (std::is_same<ctype, dt_byte>::value) {
            //! check if src or dst is one dim and contig
            auto check_one_dim = [&]() {
                if (dst.ndim == 1 && dst_contig)
                    return true;
                else if (src.ndim == 1 && src_contig)
                    return true;
                else 
                    return false;
            };
            if (check_one_dim()) {
                bool src_pack = src.shape[src.ndim - 1] % 4 == 0;
                bool dst_pack = dst.shape[dst.ndim - 1] % 4 == 0;
                if (src_pack && src_contig)
                    return 1;
                else if (dst_pack && dst_contig)
                    return 0;
            }
        }
        return -1;
    }
    int count = 0;
    template <class PVis0, class PVis1>
    void do_run() {
        megdnn_assert(!m_invoked);
        m_invoked = true;
        typedef OpCallerBinaryNoContiguous<PVis0, PVis1> Caller;
        size_t size = m_param.size;
        int grid_size, block_size;

        Caller caller;
        auto param_host_init = [&]() {
            caller.par0.host_init(m_param[0], grid_size, block_size);
            caller.par1.host_init(m_param[1], grid_size, block_size);
        };
        int kernel_type = try_int8_pack();
        if (kernel_type == 1) {
            //! src pack: read 1 uchar4, write 4 uchar
            auto fptr = src_pack_kern<Caller>;
            get_launch_spec_unroll4(reinterpret_cast<const void *>(fptr), size, &grid_size, &block_size);
            param_host_init();
            (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);

        } else if (kernel_type == 0) {
            //! dst pack: read 4 uchar, write 1 uchar4
            auto fptr = dst_pack_kern<Caller>;
            get_launch_spec_unroll4(reinterpret_cast<const void *>(fptr), size, &grid_size, &block_size);
            param_host_init();
            (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);

        } else {
            //! general
            auto fptr = cuda_kern_general<Caller>;
            elemwise_intl::get_launch_spec(reinterpret_cast<const void *>(fptr), size, &grid_size, &block_size);
            param_host_init();
            (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);
        }
        after_kernel_launch();
    }

public:
    UserOpInvoker(const ElemwiseOpParamN<2> &param, cudaStream_t stream)
        : m_rw_size(param.size), m_param(param), m_stream(stream) {
        m_invoked = false;
        dispatch0();
        megdnn_assert(m_invoked);
    }
};

#undef DEFINE_CONTIG_RECEIVER

/* f}}} */

#endif

#undef devfunc

} // namespace noncontig_general_intl
} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
