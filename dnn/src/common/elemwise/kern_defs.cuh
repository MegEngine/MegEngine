#pragma once

#include "src/common/elemwise/erfinv.h"
#include "src/common/elemwise_helper.cuh"
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/common/utils.cuh"

#include "megcore_cdefs.h"
#include "megdnn/dtype.h"

#include <cmath>
#include <cstdlib>
#include "math.h"

#if MEGDNN_CC_HOST
#include <algorithm>
using std::max;
using std::min;

#define rsqrtf(x) (1.f / sqrt(x))
#endif

#ifndef MEGDNN_ELEMWISE_MODE_ENABLE
#define MEGDNN_ELEMWISE_MODE_ENABLE(_mode, _cb) _cb(_mode)
#define MEGDNN_ELEMWISE_MODE_ENABLE_ALL         1
#endif

#if MEGDNN_CC_HOST && !defined(__host__)
#define MEGDNN_HOST_DEVICE_SELF_DEFINE
#define __host__
#define __device__
#endif

namespace megdnn {

template <typename T>
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

//! grad of silu
__device__ __host__ inline float silu_grad(float x, float dy) {
    const float one = 1.0;
    float sigmoid = one / (one + expf(-x));
    return dy * sigmoid * (one + x * (one - sigmoid));
}

__device__ __host__ inline float normcdf(float x) {
#if MEGDNN_CC_HOST
    return 0.5f * (1.f + erff(x / sqrtf(2.f)));
#else
    //! use cuda build-in math
    return ::normcdff(x);
#endif
}

//! grad of gelu
__device__ __host__ inline float gelu_grad(float x, float dy) {
    //! 1/ sqrt(2 * pi)
    const float coeff = 0.3989422804014327f;
    float phi = coeff * expf(-0.5f * x * x);
    float normcdf_v = normcdf(x);
    return dy * (normcdf_v + x * phi);
}

//! grad of softplus
__device__ __host__ inline float softplus_grad(float x, float dy) {
    float logg = -dy * expf(-fabs(x)) / (1.f + expf(-fabs(x)));
    float grad0 = x > 0.f ? logg : -logg;
    float relux = x < 0.f ? 0.f : x;
    float grad1 = relux > 0.f ? dy : 0.f;
    return grad0 + grad1;
}

__device__ __host__ inline bool feq(float a, float b) {
    return fabsf(a - b) < 1e-6;
}

__device__ __host__ inline float dispatch_powf(float x, float y) {
#define CALL_IF(_v, _stmt) \
    do {                   \
        if (feq(y, _v)) {  \
            return _stmt;  \
        }                  \
    } while (0)

    CALL_IF(2.f, x * x);
    CALL_IF(0.5f, sqrtf(x));
    CALL_IF(-0.5f, rsqrtf(x));
    CALL_IF(0.f, 1.f);
    CALL_IF(1.f, x);
    CALL_IF(3.f, x * x * x);
    CALL_IF(-1.f, 1.f / x);
    CALL_IF(-2.f, 1.f / (x * x));
#undef CALL_IF
    return powf(x, y);
}

__device__ __host__ inline int dispatch_floordiv_int(int x, int y) {
    if ((x ^ y) < 0) {
        const auto quot = x / y;
        const auto rem = x % y;
        return rem ? quot - 1 : quot;
    }
    return x / y;
}

#include "src/common/elemwise/each_mode.inl"

template <megcorePlatform_t plat, uint32_t mode, typename dtype>
struct ElemwiseKern;

//! define kernel for a single ctype
#define DEF_KERN(_ctype, _mode, _imp)                                             \
    template <megcorePlatform_t plat>                                             \
    struct ElemwiseKern<plat, param_enumv::Elemwise::Mode::_mode, _ctype> {       \
        typedef _ctype ctype;                                                     \
        static __host__ __device__ _ctype apply(KERN_SIG) { return ctype(_imp); } \
    }

//! define kernel for all float types
#define DEF_KERN_FLOAT(_mode, _imp)                     \
    DEF_KERN(dt_float32, _mode, _imp);                  \
    DNN_INC_FLOAT16(DEF_KERN(dt_float16, _mode, _imp);) \
    DNN_INC_FLOAT16(DEF_KERN(dt_bfloat16, _mode, _imp);)

//! define kernel for all int types
#define DEF_KERN_INT(_mode, _imp)    \
    DEF_KERN(dt_int32, _mode, _imp); \
    DEF_KERN(dt_int16, _mode, _imp); \
    DEF_KERN(dt_int8, _mode, _imp);  \
    DEF_KERN(dt_uint8, _mode, _imp); \
    DEF_KERN(dt_uint16, _mode, _imp);

//! define kernel for all ctypes
#define DEF_KERN_ALL(_mode, _imp) \
    DEF_KERN_INT(_mode, _imp);    \
    DEF_KERN_FLOAT(_mode, _imp);

/* ================== unary kernels ================== */
#define KERN_SIG ctype x

// int and float
DEF_KERN_ALL(NEGATE, -x);
DEF_KERN_ALL(SQUARE, x* x);
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
DEF_KERN_INT(RELU, x <= ctype(0) ? ctype(0) : x);
DEF_KERN_INT(RELU6, x <= ctype(0) ? ctype(0) : (x <= ctype(6) ? x : ctype(6)));
DEF_KERN_INT(SIGN, x < ctype(0) ? ctype(-1) : (x > ctype(0) ? ctype(1) : ctype(0)));
DEF_KERN_FLOAT(RELU, x <= 0.f ? ctype(0) : x);
DEF_KERN_FLOAT(RELU6, x <= 6.f ? ctype(0) : (x <= 6.f ? x : ctype(6)));
DEF_KERN_FLOAT(SIGN, x < 0.f ? -1.f : (x > 0.f ? 1.f : 0.f));
#else
DEF_KERN_ALL(RELU, x <= ctype(0) ? ctype(0) : x);
DEF_KERN_ALL(RELU6, x <= ctype(0) ? ctype(0) : (x <= ctype(6) ? x : ctype(6)));
DEF_KERN_ALL(SIGN, x < ctype(0) ? ctype(-1) : (x > ctype(0) ? ctype(1) : ctype(0)));
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
DEF_KERN_FLOAT(H_SWISH, x* min(max(x + 3, 0.f), 6.f) * (1.f / 6.f));
DEF_KERN_FLOAT(SILU, x / (expf(-x) + 1.f));
DEF_KERN_FLOAT(GELU, x* normcdf(x));
DEF_KERN_FLOAT(SINH, sinhf(x));
DEF_KERN_FLOAT(COSH, coshf(x));
DEF_KERN_FLOAT(ASINH, asinhf(x));
DEF_KERN_FLOAT(ACOSH, acoshf(x));
DEF_KERN_FLOAT(ATANH, atanhf(x));
DEF_KERN_FLOAT(TAN, tanf(x));
DEF_KERN_FLOAT(SOFTPLUS, log1pf(expf(-fabsf(x))) + (x <= ctype(0) ? ctype(0) : x));
DEF_KERN_FLOAT(
        HSIGMOID,
        x <= ctype(-3) ? ctype(0) : (x >= ctype(3) ? ctype(1) : ((x + 3.f) / 6.f)));
DEF_KERN_FLOAT(SQRT, sqrtf(x));
DEF_KERN_FLOAT(LOGSIGMOID, -log1pf(expf(-fabsf(x))) + (x >= ctype(0) ? ctype(0) : x));

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
DEF_KERN(dt_bool, AND, x&& y);
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

DEF_KERN_INT(FLOOR_DIV, dispatch_floordiv_int(x, y));
DEF_KERN_FLOAT(FLOOR_DIV, floorf(x / y));

DEF_KERN_INT(MOD, ((y + x % y) % y));  // consistent with python modulo
DEF_KERN_FLOAT(MOD, fmodf(x, y));

DEF_KERN_INT(SHL, x << y);
DEF_KERN_INT(SHR, x >> y);
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
DEF_KERN_INT(FUSE_ADD_RELU, (x + y) <= ctype(0) ? ctype(0) : (x + y));
DEF_KERN_FLOAT(FUSE_ADD_RELU, (x + y) <= 0.f ? ctype(0) : (x + y));
#else
DEF_KERN_ALL(FUSE_ADD_RELU, (x + y) <= ctype(0) ? ctype(0) : (x + y));
#endif
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
DEF_KERN_INT(PRELU, x > ctype(0) ? x : (x * y));
DEF_KERN_FLOAT(PRELU, x > 0.f ? x : (x * y));
#else
DEF_KERN_ALL(PRELU, x > ctype(0) ? x : (x * y));
#endif

// float only
DEF_KERN_FLOAT(TRUE_DIV, x / y);
DEF_KERN_FLOAT(POW, powf(x, y));
DEF_KERN_FLOAT(LOG_SUM_EXP, log_sum_exp(x, y));
DEF_KERN_FLOAT(FAST_TANH_GRAD, fast_tanh_grad(x, y));

DEF_KERN_FLOAT(FUSE_ADD_TANH, tanhf(x + y));
DEF_KERN_FLOAT(FUSE_ADD_SIGMOID, 1.f / (expf(-(x + y)) + 1.f));

DEF_KERN_FLOAT(ATAN2, atan2f(x, y));
DEF_KERN_FLOAT(
        H_SWISH_GRAD,
        x < -3.f ? (ctype)0.f
                 : (ctype)(x > 3.f ? (ctype)y : (ctype)((2.f * x + 3.f) / 6.f * y)));

DEF_KERN_FLOAT(FUSE_ADD_H_SWISH, fuse_add_hswish(x, y));
DEF_KERN_FLOAT(SILU_GRAD, silu_grad(x, y));
DEF_KERN_FLOAT(GELU_GRAD, gelu_grad(x, y));
DEF_KERN_FLOAT(ASINH_GRAD, y / sqrt(x * x + 1.f));
DEF_KERN_FLOAT(ACOSH_GRAD, y / sqrt(x * x - 1.f));
DEF_KERN_FLOAT(ATANH_GRAD, y / (1.f - x * x));
DEF_KERN_FLOAT(SOFTPLUS_GRAD, softplus_grad(x, y));
DEF_KERN_FLOAT(RELU6_GRAD, x <= ctype(0) ? ctype(0) : (x >= ctype(6) ? ctype(0) : y));
DEF_KERN_FLOAT(
        HSIGMOID_GRAD,
        x <= ctype(-3) ? ctype(0) : (x >= ctype(3) ? ctype(0) : (y / 6.f)));
#undef KERN_SIG

/* ================== ternary kernels ================== */
#define KERN_SIG ctype x, ctype y, ctype z

// int and float
DEF_KERN_ALL(COND_LEQ_MOV, x <= y ? z : ctype(0));
DEF_KERN_ALL(COND_LT_MOV, x < y ? z : ctype(0));
DEF_KERN_ALL(FUSE_MUL_ADD3, x* y + z);
DEF_KERN_ALL(CLIP, x <= y ? y : (x <= z ? x : z));
DEF_KERN_FLOAT(PRELU_GRAD, x >= 0.f ? y : (y * z));

#undef KERN_SIG

#undef DEF_KERN_AD
#undef DEF_KERN
#undef DEF_KERN_FLOAT
#undef DEF_KERN_INT
#undef DEF_KERN_ALL

/* ================== bool kernels ================== */
//! define kernel
template <megcorePlatform_t plat, uint32_t mode, typename stype, typename dtype>
struct ElemwiseBoolKern;

#define DEF_KERN(_ctype, _dtype, _mode, _imp)                                      \
    template <megcorePlatform_t plat>                                              \
    struct ElemwiseBoolKern<                                                       \
            plat, param_enumv::Elemwise::Mode::_mode, _ctype, _dtype> {            \
        typedef _ctype ctype;                                                      \
        static __host__ __device__ _dtype apply(KERN_SIG) { return _dtype(_imp); } \
    }

//! define kernel for all float types
#define DEF_KERN_FLOAT(_mode, _imp)                              \
    DEF_KERN(dt_float32, dt_bool, _mode, _imp);                  \
    DNN_INC_FLOAT16(DEF_KERN(dt_float16, dt_bool, _mode, _imp);) \
    DNN_INC_FLOAT16(DEF_KERN(dt_bfloat16, dt_bool, _mode, _imp);)

//! define kernel for all int types
#define DEF_KERN_INT(_mode, _imp)             \
    DEF_KERN(dt_int32, dt_bool, _mode, _imp); \
    DEF_KERN(dt_int16, dt_bool, _mode, _imp); \
    DEF_KERN(dt_int8, dt_bool, _mode, _imp);  \
    DEF_KERN(dt_uint8, dt_bool, _mode, _imp);

//! define kernel for all ctypes
#define DEF_KERN_ALL(_mode, _imp) \
    DEF_KERN_INT(_mode, _imp);    \
    DEF_KERN_FLOAT(_mode, _imp);  \
    DEF_KERN(dt_bool, dt_bool, _mode, _imp);
#define KERN_SIG ctype x
DEF_KERN_FLOAT(ISNAN, isnan(float(x)));
DEF_KERN_FLOAT(ISINF, isinf(float(x)));
#undef KERN_SIG
#define KERN_SIG ctype x, ctype y
DEF_KERN_ALL(LT, x < y);
DEF_KERN_ALL(LEQ, x <= y);
DEF_KERN_ALL(EQ, x == y);
DEF_KERN_ALL(NEQ, x != y);
#undef KERN_SIG

#undef DEF_KERN_AD
#undef DEF_KERN
#undef DEF_KERN_FLOAT
#undef DEF_KERN_INT
#undef DEF_KERN_ALL

}  // namespace megdnn

#if MEGDNN_CC_HOST && defined(MEGDNN_HOST_DEVICE_SELF_DEFINE)
#undef MEGDNN_HOST_DEVICE_SELF_DEFINE
#undef __host__
#undef __device__
#endif

// vim: ft=cpp syntax=cpp.doxygen
