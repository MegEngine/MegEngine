/**
 * \file dnn/src/naive/elemwise/kern_caller.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/elemwise_helper.cuh"

namespace megdnn {
namespace naive {

    template<int arity, class KernImpl>
    struct ElemArithKernCaller {
        typedef typename KernImpl::ctype ctype;
        static void run(ctype *dest, const ElemwiseOpParamN<arity> &param);
    };

} // namespace naive
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen


