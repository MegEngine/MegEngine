/**
 * \file dnn/src/common/elemwise/kern_defs.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/common/opr_param_defs_enumv.cuh"
#include "src/common/elemwise_helper.cuh"
#include "src/common/utils.cuh"
#include "src/common/elemwise/erfinv.h"

#include "megcore_cdefs.h"
#include "megdnn/dtype.h"

#include <cmath>
#include <cstdlib>

#if MEGDNN_CC_HOST
#include <algorithm>
using std::max;
using std::min;
#endif

#ifndef MEGDNN_ELEMWISE_MODE_ENABLE
#define MEGDNN_ELEMWISE_MODE_ENABLE(_mode, _cb) _cb(_mode)
#define MEGDNN_ELEMWISE_MODE_ENABLE_ALL 1
#endif

#if MEGDNN_CC_HOST && !defined(__host__)
#define MEGDNN_HOST_DEVICE_SELF_DEFINE
#define __host__
#define __device__
#endif

namespace megdnn {


    template<typename T>
    __device__ __host__ inline T log_sum_exp(T x, T y) {
        T a, b;
        a = x < y ? x : y;
        b = x < y ? y : x;
        return T(b + log1pf(exp(a - b)));
    }

    __device__ __host__ inline float fast_tanh(float x) {
        return x * (27.f + x * x) / (27.f + 9.f * x * x);
    }

    //! use multiplying (1.f / 6.f) to replace dividing 6.f, because we didn't
    //! pass
    //! --use_fast_math to nvcc to enable --prec_div optimization, which will
    //! cause performance drop on Turing architecture
    __device__ __host__ inline float fuse_add_hswish(float x, float y) {
        float z = x + y;
        return z * min(max(z + 3, 0.f), 6.f) * (1.f / 6.f);
    }

    __device__ __host__ inline float fast_tanh_grad(float x, float dx) {
        float x_pow2 = x * x;
        float deno = 3.f + x_pow2;
        return ((-48.f * x_pow2) / deno + 27.f + x_pow2) / (deno * 9.f) * dx;
    }

#include "src/common/elemwise/each_mode.inl"

    template<megcorePlatform_t plat, uint32_t mode, typename dtype>
    struct ElemwiseKern;

//! define kernel for a single ctype
#define DEF_KERN(_ctype, _mode, _imp) \
    template<megcorePlatform_t plat> \
    struct ElemwiseKern<plat, param_enumv::Elemwise::Mode::_mode, _ctype> { \
        typedef _ctype ctype; \
        static __host__ __device__ _ctype apply(KERN_SIG) { \
            return ctype(_imp); \
        } \
    }

//! define kernel for all float types
#define DEF_KERN_FLOAT(_mode, _imp) \
    DEF_KERN(dt_float32, _mode, _imp); \
    MEGDNN_INC_FLOAT16(DEF_KERN(dt_float16, _mode, _imp);) \
    MEGDNN_INC_FLOAT16(DEF_KERN(dt_bfloat16, _mode, _imp);)

//! define kernel for all int types
#define DEF_KERN_INT(_mode, _imp) \
    DEF_KERN(dt_int32, _mode, _imp); \
    DEF_KERN(dt_int16, _mode, _imp); \
    DEF_KERN(dt_int8, _mode, _imp); \
    DEF_KERN(dt_uint8, _mode, _imp); \

//! define kernel for all ctypes
#define DEF_KERN_ALL(_mode, _imp) \
    DEF_KERN_INT(_mode, _imp); \
    DEF_KERN_FLOAT(_mode, _imp); \

    /* ================== unary kernels ================== */
#define KERN_SIG ctype x

    // int and float
    DEF_KERN_ALL(NEGATE, -x);
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
    DEF_KERN_INT(RELU, x <= ctype(0) ? ctype(0) : x);
    DEF_KERN_FLOAT(RELU, x <= 0.f ? ctype(0) : x);
#else
    DEF_KERN_ALL(RELU, x <= ctype(0) ? ctype(0) : x);
#endif
    DEF_KERN_INT(ABS, abs(int(x)));
    // DEF_KERN_INT(ABS, x > ctype(0) ? x : -x);
    DEF_KERN_FLOAT(ABS, fabsf(x));

    // float only
    DEF_KERN_FLOAT(ACOS, acosf(x));
    DEF_KERN_FLOAT(ASIN, asinf(x));
    DEF_KERN_FLOAT(CEIL, ceilf(x));
    DEF_KERN_FLOAT(COS, cosf(x));
    DEF_KERN_FLOAT(EXP, expf(x));
    DEF_KERN_FLOAT(EXPM1, expm1f(x));
    DEF_KERN_FLOAT(FLOOR, floorf(x));
    DEF_KERN_FLOAT(LOG, logf(x));
    DEF_KERN_FLOAT(LOG1P, log1pf(x));
    DEF_KERN_FLOAT(SIGMOID, 1.f / (expf(-x) + 1.f));
    DEF_KERN_FLOAT(SIN, sinf(x));
    DEF_KERN_FLOAT(TANH, tanhf(x));
    DEF_KERN_FLOAT(FAST_TANH, fast_tanh(x));
    DEF_KERN_FLOAT(ROUND, roundf(x));
    DEF_KERN_FLOAT(ERF, erff(x));
    DEF_KERN_FLOAT(ERFINV, erfinvf(x));
    DEF_KERN_FLOAT(ERFC, erfcf(x));
    DEF_KERN_FLOAT(ERFCINV, erfcinvf(x));
    DEF_KERN_FLOAT(H_SWISH, x * min(max(x + 3, 0.f), 6.f) * (1.f / 6.f));

    // int only
    DEF_KERN(dt_bool, NOT, x ^ 1);

#undef KERN_SIG

    /* ================== binary kernels ================== */
#define KERN_SIG ctype x, ctype y

    // int and float
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
    DEF_KERN_INT(ABS_GRAD, x > ctype(0) ? y : -y);
    DEF_KERN_FLOAT(ABS_GRAD, x > 0.f ? y : -y);
#else
    DEF_KERN_ALL(ABS_GRAD, x > ctype(0) ? y : -y);
#endif
    DEF_KERN_ALL(ADD, x + y);
    DEF_KERN_ALL(MAX, x > y ? x : y);
    DEF_KERN_ALL(MIN, x < y ? x : y);
    DEF_KERN_ALL(MUL, x* y);
    DEF_KERN(dt_bool, AND, x && y);
    DEF_KERN(dt_bool, OR, x || y);
    DEF_KERN(dt_bool, XOR, x ^ y);
    DEF_KERN_INT(RMULH, round_mulh_saturate(x, y));
    DEF_KERN_ALL(SIGMOID_GRAD, x*(ctype(1) - x) * y);
    DEF_KERN_ALL(SUB, x - y);
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
    DEF_KERN_INT(SWITCH_GT0, x > ctype(0) ? y : ctype(0));
    DEF_KERN_FLOAT(SWITCH_GT0, x > 0.f ? y : ctype(0));
#else
    DEF_KERN_ALL(SWITCH_GT0, x > ctype(0) ? y : ctype(0));
#endif
    DEF_KERN_ALL(TANH_GRAD, (ctype(1) - x * x) * y);
    DEF_KERN_ALL(LT, x < y);
    DEF_KERN_ALL(LEQ, x <= y);
    DEF_KERN_ALL(EQ, x == y);
    DEF_KERN(dt_bool, LT, x < y);
    DEF_KERN(dt_bool, LEQ, x <= y);
    DEF_KERN(dt_bool, EQ, x == y);

    DEF_KERN_INT(FLOOR_DIV, x / y);
    DEF_KERN_FLOAT(FLOOR_DIV, floorf(x / y));

    DEF_KERN_INT(MOD, x % y);
    DEF_KERN_FLOAT(MOD, fmodf(x, y));

    DEF_KERN_INT(SHL, x << y);
    DEF_KERN_INT(SHR, x >> y);
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
    DEF_KERN_INT(FUSE_ADD_RELU, (x + y) <= ctype(0) ? ctype(0) : (x + y));
    DEF_KERN_FLOAT(FUSE_ADD_RELU, (x + y) <= 0.f ? ctype(0) : (x + y));
#else
    DEF_KERN_ALL(FUSE_ADD_RELU,
                 (x + y) <= ctype(0) ? ctype(0) : (x + y));
#endif

    // float only
    DEF_KERN_FLOAT(TRUE_DIV, x / y);
    DEF_KERN_FLOAT(POW, powf(x, y));
    DEF_KERN_FLOAT(LOG_SUM_EXP, log_sum_exp(x, y));
    DEF_KERN_FLOAT(FAST_TANH_GRAD, fast_tanh_grad(x, y));

    DEF_KERN_FLOAT(FUSE_ADD_TANH, tanhf(x+y));
    DEF_KERN_FLOAT(FUSE_ADD_SIGMOID, 1.f / (expf(-(x+y)) + 1.f));

    DEF_KERN_FLOAT(ATAN2, atan2f(x, y));
    DEF_KERN_FLOAT(H_SWISH_GRAD,
                   x < -3.f ? 0.f : (x > 3.f ? y : (2.f * x + 3.f) / 6.f * y));

    DEF_KERN_FLOAT(FUSE_ADD_H_SWISH, fuse_add_hswish(x, y));
#undef KERN_SIG

    /* ================== ternary kernels ================== */
#define KERN_SIG ctype x, ctype y, ctype z

    // int and float
    DEF_KERN_ALL(COND_LEQ_MOV, x <= y ? z : ctype(0));
    DEF_KERN_ALL(FUSE_MUL_ADD3, x * y + z);

#undef KERN_SIG


#undef DEF_KERN_AD
#undef DEF_KERN

} // namespace megdnn

#if MEGDNN_CC_HOST && defined(MEGDNN_HOST_DEVICE_SELF_DEFINE)
#undef MEGDNN_HOST_DEVICE_SELF_DEFINE
#undef __host__
#undef __device__
#endif

// vim: ft=cpp syntax=cpp.doxygen
