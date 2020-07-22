/**
 * \file dnn/src/cuda/elemwise/kern_wrapper.cuh
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
#include "src/cuda/elemwise_helper.cuh"

namespace megdnn {
namespace cuda {

    template<int arity, class KernImpl, typename enable = void>
    struct ElemArithKernWrapper;

    template <class KernImpl>
    struct ElemArithKernWrapper<
            1, KernImpl,
            typename std::enable_if<
                    !std::is_same<typename KernImpl::ctype, dt_int8>::value &&
                    !std::is_same<typename KernImpl::ctype, dt_uint8>::value &&
					!std::is_same<typename KernImpl::ctype, 
									dt_bool>::value>::type> {
        typedef typename KernImpl::ctype ctype;
        ctype* dst;

#if MEGDNN_CC_CUDA
        __device__ void operator()(uint32_t idx, ctype x) {
            dst[idx] = KernImpl::apply(x);
        }
#endif
    };
    template <class KernImpl>
    struct ElemArithKernWrapper<
            2, KernImpl,
            typename std::enable_if<
                    !std::is_same<typename KernImpl::ctype, dt_int8>::value &&
                    !std::is_same<typename KernImpl::ctype, dt_uint8>::value &&
					!std::is_same<typename KernImpl::ctype,
									dt_bool>::value>::type> {
        typedef typename KernImpl::ctype ctype;
        ctype* dst;

#if MEGDNN_CC_CUDA
        __device__ void operator()(uint32_t idx, ctype x, ctype y) {
            dst[idx] = KernImpl::apply(x, y);
        }
#endif
    };
    template <class KernImpl>
    struct ElemArithKernWrapper<
            3, KernImpl,
            typename std::enable_if<
                    !std::is_same<typename KernImpl::ctype, dt_int8>::value &&
                    !std::is_same<typename KernImpl::ctype, dt_uint8>::value &&
					!std::is_same<typename KernImpl::ctype,
									dt_bool>::value>::type> {
        typedef typename KernImpl::ctype ctype;
        ctype* dst;

#if MEGDNN_CC_CUDA
        __device__ void operator()(uint32_t idx, ctype x, ctype y, ctype z) {
            dst[idx] = KernImpl::apply(x, y, z);
        }
#endif
    };

    template <class KernImpl>
    struct ElemArithKernWrapper<
            1, KernImpl,
            typename std::enable_if<
                    std::is_same<typename KernImpl::ctype, dt_int8>::value ||
                    std::is_same<typename KernImpl::ctype, dt_uint8>::value ||
					std::is_same<typename KernImpl::ctype, 
									dt_bool>::value>::type> {
        typedef typename KernImpl::ctype ctype;
        using VectTypeTrait = elemwise_intl::VectTypeTrait<ctype>;
        typedef typename VectTypeTrait::vect_type vect_type;
        ctype* dst;
#if MEGDNN_CC_CUDA
        __device__ __forceinline__ void operator()(uint32_t idx, ctype x) {
            dst[idx] = KernImpl::apply(x);
        }
        __device__ __forceinline__ void operator()(uint32_t idx, vect_type x) {
            ctype a = KernImpl::apply(x.x);
            ctype b = KernImpl::apply(x.y);
            ctype g = KernImpl::apply(x.z);
            ctype r = KernImpl::apply(x.w);
            *(vect_type*)(&dst[idx]) = VectTypeTrait::make_vector(a, b, g, r);
        }
#endif
    };

    template <class KernImpl>
    struct ElemArithKernWrapper<
            2, KernImpl,
            typename std::enable_if<
                    std::is_same<typename KernImpl::ctype, dt_int8>::value ||
                    std::is_same<typename KernImpl::ctype, dt_uint8>::value ||
					std::is_same<typename KernImpl::ctype,
									dt_bool>::value>::type> {
        typedef typename KernImpl::ctype ctype;
        using VectTypeTrait = elemwise_intl::VectTypeTrait<ctype>;
        typedef typename VectTypeTrait::vect_type vect_type;
        ctype* dst;
#if MEGDNN_CC_CUDA
        __device__ __forceinline__ void operator()(uint32_t idx, ctype x,
                                                   ctype y) {
            dst[idx] = KernImpl::apply(x, y);
        }
        __device__ __forceinline__ void operator()(uint32_t idx, vect_type x,
                                                   vect_type y) {
            ctype a = KernImpl::apply(x.x, y.x);
            ctype b = KernImpl::apply(x.y, y.y);
            ctype g = KernImpl::apply(x.z, y.z);
            ctype r = KernImpl::apply(x.w, y.w);
            *(vect_type*)(&dst[idx]) = VectTypeTrait::make_vector(a, b, g, r);
        }
#endif
    };

    template <class KernImpl>
    struct ElemArithKernWrapper<
            3, KernImpl,
            typename std::enable_if<
                    std::is_same<typename KernImpl::ctype, dt_int8>::value ||
                    std::is_same<typename KernImpl::ctype, dt_uint8>::value ||
					std::is_same<typename KernImpl::ctype,
									dt_bool>::value>::type> {
        typedef typename KernImpl::ctype ctype;
        using VectTypeTrait = elemwise_intl::VectTypeTrait<ctype>;
        typedef typename VectTypeTrait::vect_type vect_type;
        ctype* dst;
#if MEGDNN_CC_CUDA
        __device__ __forceinline__ void operator()(uint32_t idx, ctype x,
                                                   ctype y, ctype z) {
            dst[idx] = KernImpl::apply(x, y, z);
        }
        __device__ __forceinline__ void operator()(uint32_t idx, vect_type x,
                                                   vect_type y, vect_type z) {
            ctype a = KernImpl::apply(x.x, y.x, z.x);
            ctype b = KernImpl::apply(x.y, y.y, z.y);
            ctype g = KernImpl::apply(x.z, y.z, z.z);
            ctype r = KernImpl::apply(x.w, y.w, z.w);
            *(vect_type*)(&dst[idx]) = VectTypeTrait::make_vector(a, b, g, r);
        }
#endif
    };

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

