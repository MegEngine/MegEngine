/**
 * \file dnn/src/naive/elemwise/kern_impl.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#ifndef KERN_IMPL_MODE
#error "KERN_IMPL_MODE, KERN_IMPL_ARITY and KERN_IMPL_CTYPE must be defined"
#endif

#include "megdnn/tensor_iter.h"
#include "./kern_caller.h"

namespace megdnn {
namespace naive {

    template<class KernImpl>
    struct ElemArithKernCaller<1, KernImpl> {
        typedef typename KernImpl::ctype ctype;
        static void run(ctype *dest, const ElemwiseOpParamN<1> &param) {
            auto iter0 = tensor_iter_valonly<ctype>(param[0]).begin();
            for (size_t i = 0; i < param.size; ++ i) {
                dest[i] = KernImpl::apply(*iter0);
                ++ iter0;
            }
        }
    };
    template<class KernImpl>
    struct ElemArithKernCaller<2, KernImpl> {
        typedef typename KernImpl::ctype ctype;
        static void run(ctype *dest, const ElemwiseOpParamN<2> &param) {
            auto iter0 = tensor_iter_valonly<ctype>(param[0]).begin();
            auto iter1 = tensor_iter_valonly<ctype>(param[1]).begin();
            for (size_t i = 0; i < param.size; ++ i) {
                dest[i] = KernImpl::apply(*iter0, *iter1);
                ++ iter0;
                ++ iter1;
            }
        }
    };
    template<class KernImpl>
    struct ElemArithKernCaller<3, KernImpl> {
        typedef typename KernImpl::ctype ctype;
        static void run(ctype *dest, const ElemwiseOpParamN<3> &param) {
            auto iter0 = tensor_iter_valonly<ctype>(param[0]).begin();
            auto iter1 = tensor_iter_valonly<ctype>(param[1]).begin();
            auto iter2 = tensor_iter_valonly<ctype>(param[2]).begin();
            for (size_t i = 0; i < param.size; ++ i) {
                dest[i] = KernImpl::apply(*iter0, *iter1, *iter2);
                ++ iter0;
                ++ iter1;
                ++ iter2;
            }
        }
    };

#define cb(_m) \
    template struct ElemArithKernCaller<KERN_IMPL_ARITY, \
    ElemwiseKern< \
        megcorePlatformCPU, param_enumv::Elemwise::Mode::_m, KERN_IMPL_CTYPE>>; \



KERN_IMPL_MODE(cb)

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
