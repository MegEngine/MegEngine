#pragma once

#include "src/rocm/elemwise_helper.h.hip"
#include "src/common/elemwise/kern_defs.cuh"

namespace megdnn {
namespace rocm {

template <int arity, class KernImpl>
struct ElemArithKernWrapper;

template <class KernImpl>
struct ElemArithKernWrapper<1, KernImpl> {
    typedef typename KernImpl::ctype ctype;
    ctype* dst;

#if MEGDNN_CC_CUDA
    __device__ void operator()(uint32_t idx, ctype x) {
        dst[idx] = KernImpl::apply(x);
    }
#endif
};
template <class KernImpl>
struct ElemArithKernWrapper<2, KernImpl> {
    typedef typename KernImpl::ctype ctype;
    ctype* dst;

#if MEGDNN_CC_CUDA
    __device__ void operator()(uint32_t idx, ctype x, ctype y) {
        dst[idx] = KernImpl::apply(x, y);
    }
#endif
};
template <class KernImpl>
struct ElemArithKernWrapper<3, KernImpl> {
    typedef typename KernImpl::ctype ctype;
    ctype* dst;

#if MEGDNN_CC_CUDA
    __device__ void operator()(uint32_t idx, ctype x, ctype y, ctype z) {
        dst[idx] = KernImpl::apply(x, y, z);
    }
#endif
};

}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen


