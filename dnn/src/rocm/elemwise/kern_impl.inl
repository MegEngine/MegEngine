#pragma once

#ifndef KERN_IMPL_MODE
#error "KERN_IMPL_MODE, KERN_IMPL_ARITY and KERN_IMPL_CTYPE must be defined"
#endif

#include "src/rocm/elemwise/kern_wrapper.h.hip"

namespace megdnn {
namespace rocm {

#define cb(_mode)                                                                     \
    typedef ElemwiseKern<                                                             \
            megcorePlatformROCM, param_enumv::Elemwise::Mode::_mode, KERN_IMPL_CTYPE> \
            KernImpl##_mode;                                                          \
    typedef ElemArithKernWrapper<KERN_IMPL_ARITY, KernImpl##_mode> Wrapper##_mode;    \
    INST_RUN_ELEMWISE(Wrapper##_mode, KERN_IMPL_CTYPE, KERN_IMPL_ARITY);

KERN_IMPL_MODE(cb)

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
