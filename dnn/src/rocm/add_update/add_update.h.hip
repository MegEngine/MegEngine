/**
 *
 * \file src/rocm/add_update/add_update.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
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

