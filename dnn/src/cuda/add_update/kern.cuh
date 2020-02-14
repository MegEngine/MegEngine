/**
 * \file dnn/src/cuda/add_update/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/cuda/utils.cuh"
#include "src/cuda/elemwise_helper.cuh"

#if MEGDNN_CC_HOST
#include "megdnn/oprs.h"
#endif

namespace megdnn{
namespace cuda {

    template<typename ctype, typename enable = void>
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

    template <typename ctype>
    struct AddUpdateKernOp<
            ctype, typename std::enable_if<
                           std::is_same<ctype, dt_int8>::value ||
                           std::is_same<ctype, dt_uint8>::value>::type> {
        typedef typename elemwise_intl::VectTypeTrait<ctype>::vect_type
                vect_type;
        ctype* dst;
        ctype alpha, beta, bias;
        __device__ void operator()(uint32_t idx, ctype delta) {
            dst[idx] = dst[idx] * alpha + delta * beta + bias;
        }
        __device__ void operator()(uint32_t idx, vect_type delta) {
            vect_type& x = *(vect_type*)(&dst[idx]);
            x.x = x.x * alpha + delta.x * beta + bias;
            x.y = x.y * alpha + delta.y * beta + bias;
            x.z = x.z * alpha + delta.z * beta + bias;
            x.w = x.w * alpha + delta.w * beta + bias;
        }
#if MEGDNN_CC_HOST
        AddUpdateKernOp(const TensorND& dest, const AddUpdate::Param& param)
                : dst{dest.ptr<ctype>()},
                  alpha(param.alpha),
                  beta(param.beta),
                  bias(param.bias){};
#endif
    };

    template<typename ctype, typename enable = void>
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

    template <typename ctype>
    struct AddUpdateKernOpNonContig<
            ctype, typename std::enable_if<
                           std::is_same<ctype, dt_int8>::value ||
                           std::is_same<ctype, dt_uint8>::value>::type> {
        typedef typename elemwise_intl::VectTypeTrait<ctype>::vect_type
                vect_type;
        ctype alpha, beta, bias;
        __device__ void operator()(uint32_t, ctype& dst, ctype delta) {
            dst = dst * alpha + delta * beta + bias;
        }
        __device__ void operator()(uint32_t, vect_type& dst, vect_type delta) {
            dst.x = dst.x * alpha + delta.x * beta + bias;
            dst.y = dst.y * alpha + delta.y * beta + bias;
            dst.z = dst.z * alpha + delta.z * beta + bias;
            dst.w = dst.w * alpha + delta.w * beta + bias;
        }
#if MEGDNN_CC_HOST
        AddUpdateKernOpNonContig(const AddUpdate::Param& param)
                : alpha(param.alpha), beta(param.beta), bias(param.bias) {}
#endif
    };

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

