/**
 * \file dnn/src/cuda/elemwise_multi_type/kern_impl.inl
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
#error "KERN_IMPL_MODE, KERN_IMPL_ARITY, KERN_IMPL_STYPE, KERN_IMPL_DTYPE must be defined"
#endif

#include "src/cuda/elemwise_multi_type/kern_ops.cuh"

namespace megdnn {
namespace cuda {

#define cb(_m)                                                                 \
    typedef ElemwiseKern<megcorePlatformCUDA, param_enumv::Elemwise::Mode::_m, \
                         float>                                                \
            KernImpl;                                                          \
    typedef kern_ops_quantized::QuantizedMultiTypeOp<                          \
            KERN_IMPL_ARITY, KERN_IMPL_STYPE, KERN_IMPL_DTYPE, KernImpl>       \
            Op;                                                                \
    INST_RUN_ELEMWISE(Op, KERN_IMPL_STYPE, KERN_IMPL_ARITY);

KERN_IMPL_MODE(cb)

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
