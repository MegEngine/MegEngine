#pragma once

#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/elemwise_helper.cuh"

namespace megdnn {
namespace naive {

template <int arity, class KernImpl>
struct ElemArithKernCaller {
    typedef typename KernImpl::ctype ctype;
    static void run(ctype* dest, const ElemwiseOpParamN<arity>& param);
};

}  // namespace naive
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
