/**
 *
 * \file dnn/src/rocm/add_update/add_update.h.hip
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "hip_header.h"
#include "src/rocm/elemwise_helper.h.hip"

#if MEGDNN_CC_HOST
#include "megdnn/oprs.h"
#endif

namespace megdnn {
namespace rocm {

    template<typename ctype>
    struct AddUpdateKernOp {
        ctype *dst;
        ctype alpha, beta, bias;

        __device__ void operator() (uint32_t idx, ctype delta) {
            dst[idx] = dst[idx] * alpha + delta * beta + bias;
        }

#if MEGDNN_CC_HOST
        AddUpdateKernOp(const TensorND &dest, const AddUpdate::Param &param):
            dst{dest.ptr<ctype>()},
            alpha(param.alpha), beta(param.beta), bias(param.bias)
        {
        }
#endif
    };

    template<typename ctype>
    struct AddUpdateKernOpNonContig {
        ctype alpha, beta, bias;

        __device__ void operator() (uint32_t /*idx*/, ctype &dst, ctype delta) {
            dst = dst * alpha + delta * beta + bias;
        }

#if MEGDNN_CC_HOST
        AddUpdateKernOpNonContig(const AddUpdate::Param &param):
            alpha(param.alpha), beta(param.beta), bias(param.bias)
        {
        }
#endif
    };

}
}

// vim: ft=cpp syntax=cpp.doxygen

