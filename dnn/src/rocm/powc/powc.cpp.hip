/**
 * \file src/rocm/powc/powc.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/powc/powc.h.hip"
#include "megdnn/dtype.h"
#include "src/rocm/elemwise_helper.h.hip"

#include <cmath>
#include <limits>

namespace megdnn {
namespace rocm {
// use a namespace (but not anonymous namespace) to avoid name confliction while
// maintaining readability of cuda kernel names
namespace hip_kern {

template <int>
struct PowCIntSmall;

template <>
struct PowCIntSmall<0> {
    template <typename T>
    static __device__ __forceinline__ T apply(T) {
        return static_cast<T>(1);
    }
};
template <>
struct PowCIntSmall<1> {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return x;
    }
};
template <>
struct PowCIntSmall<2> {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return x * x;
    }
};
template <>
struct PowCIntSmall<3> {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return x * x * x;
    }
};
template <>
struct PowCIntSmall<4> {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        x = x * x;
        return x * x;
    }
};
template <int n>
struct PowCIntSmall {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return PowCIntSmall<-n>::apply(static_cast<T>(1) / x);
    }
};

template <typename T>
struct PowCIntOdd {
    T exp;

    __device__ __forceinline__ T apply(T x) {
        return static_cast<T>(copysignf(powf(fabsf(x), exp), x));
    }
};

template <typename T>
struct PowCIntEven {
    T exp;

    __device__ __forceinline__ T apply(T x) {
        return static_cast<T>(powf(fabsf(x), exp));
    }
};

struct PowCFloatSqrt {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return static_cast<T>(sqrtf(x));
    }
};

struct PowCFloatCbrt {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return static_cast<T>(cbrtf(x));
    }
};

struct PowCFloatRSqrt {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return static_cast<T>(rsqrtf(x));
    }
};

struct PowCFloatRCbrt {
    template <typename T>
    static __device__ __forceinline__ T apply(T x) {
        return static_cast<T>(rcbrtf(x));
    }
};

template <typename T>
struct PowCFloat {
    T exp;

    __device__ __forceinline__ T apply(T x) {
        return static_cast<T>(powf(x, exp));
    }
};

template <typename T, typename PowOp>
struct PowCOp {
    T* dest;
    PowOp pow_op;

    __device__ __forceinline__ void operator()(uint32_t idx, T src) {
        dest[idx] = pow_op.apply(src);
    }
};

}  // namespace hip_kern

namespace {

template <typename T, typename PowOp>
void invoke(const TensorND& dest, const TensorND& src, PowOp pow_op,
            hipStream_t stream) {
    ElemwiseOpParamN<1> param;
    param[0] = src;
    param.init_from_given_tensor();
    typedef hip_kern::PowCOp<T, PowOp> Op;
    Op op;
    op.dest = dest.ptr<T>();
    op.pow_op = pow_op;
    run_elemwise<Op, T, 1>(param, stream, op);
}

bool feq(float a, float b) {
    return std::abs(a - b) < std::numeric_limits<float>::epsilon();
}

template <typename T>
void dispatch_op(const TensorND& dest, const TensorND& src, const float* exp_f,
                 const int* exp_i, hipStream_t stream) {
#define CALL(_op) invoke<T>(dest, src, _op, stream)
    if (exp_f) {
        float exp = *exp_f;
#define CALL_IF(_v, _op)    \
    do {                    \
        if (feq(exp, _v)) { \
            CALL(_op);      \
            return;         \
        }                   \
    } while (0)
        CALL_IF(.5f, hip_kern::PowCFloatSqrt());
        CALL_IF(1.f / 3.f, hip_kern::PowCFloatCbrt());
        CALL_IF(-.5f, hip_kern::PowCFloatRSqrt());
        CALL_IF(-1.f / 3.f, hip_kern::PowCFloatRCbrt());

        hip_kern::PowCFloat<T> op;
        op.exp = exp;
        CALL(op);
        return;
#undef CALL_IF
    }

    int exp = *exp_i;
    switch (exp) {
#define CASE(v)                  \
    case v:                      \
        CALL(hip_kern::PowCIntSmall<v>()); \
        return
        CASE(0);
        CASE(1);
        CASE(2);
        CASE(3);
        CASE(4);
        CASE(-1);
        CASE(-2);
        CASE(-3);
        CASE(-4);
#undef CASE
    }
    if (exp & 1) {
        hip_kern::PowCIntOdd<T> op;
        op.exp = exp;
        CALL(op);
    } else {
        hip_kern::PowCIntEven<T> op;
        op.exp = exp;
        CALL(op);
    }
#undef CALL
}
}  // anonymous namespace

void powc_kern(const TensorND& dest, const TensorND& src,
                     const float* exp_f, const int* exp_i,
                     hipStream_t stream) {
    switch (src.layout.dtype.enumv().ev) {
#define cb(dt)                                                             \
    case DTypeTrait<dt>::enumv:                                            \
        return dispatch_op<DTypeTrait<dt>::ctype>(dest, src, exp_f, exp_i, \
                                                  stream);
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("unsupported dtype for PowC");
    }
}
} // namespace rocm
} // namespace megdnn


// vim: syntax=cpp.doxygen
