/**
 * \file dnn/src/fallback/powc/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/naive/handle.h"


#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "src/arm_common/simd_macro/marm_neon.h"
#endif

#include <limits>

using namespace megdnn;
using namespace fallback;

namespace {

template <int exp>
struct powci;

template <>
struct powci<0> {
    template <typename T>
    static T apply(T) {
        return static_cast<T>(1);
    }
};
template <>
struct powci<1> {
    template <typename T>
    static T apply(T x) {
        return x;
    }
};
template <>
struct powci<2> {
    template <typename T>
    static T apply(T x) {
        return x * x;
    }
};
template <>
struct powci<3> {
    template <typename T>
    static T apply(T x) {
        return x * x * x;
    }
};
template <>
struct powci<4> {
    template <typename T>
    static T apply(T x) {
        x = x * x;
        return x * x;
    }
};
template <int exp>
struct powci {
    static_assert(exp < 0, "bad arg");
    template <typename T>
    static T apply(T x) {
        return powci<-exp>::apply(static_cast<T>(1) / x);
    }
};

struct powci_general_even {
    int exp;
    powci_general_even(int e) : exp{e} {}

    template <typename T>
    T apply(T x) {
        return static_cast<T>(std::pow(std::abs(x), static_cast<T>(exp)));
    }
};

template <size_t size>
struct float_itype;

#ifndef MEGDNN_DISABLE_FLOAT16
template <>
struct float_itype<2> {
    using type = uint16_t;
    static constexpr uint16_t mask = 1u << 15;
};
#endif

template <>
struct float_itype<4> {
    using type = uint32_t;
    static constexpr uint32_t mask = 1u << 31;
};

struct powci_general_odd {
    template <typename T>
    union fiu {
        T f;
        typename float_itype<sizeof(T)>::type i;

        fiu() {}
    };

    int exp;
    powci_general_odd(int e) : exp{e} {}

    template <typename T>
    T apply(T x) {
        fiu<T> iret, ix;
        iret.f = std::pow(std::abs(x), static_cast<T>(exp));
        ix.f = x;
        iret.i |= ix.i & float_itype<sizeof(T)>::mask;
        return iret.f;
    }
};

struct powcf_sqrt {
    template <typename T>
    static T apply(T x) {
        return static_cast<T>(std::sqrt(x));
    }
};

struct powcf_cbrt {
    template <typename T>
    static T apply(T x) {
        return static_cast<T>(std::cbrt(x));
    }
};

struct powcf_rep_sqrt {
    template <typename T>
    static T apply(T x) {
        return static_cast<T>(std::sqrt(static_cast<T>(1) / x));
    }
};

struct powcf_rep_cbrt {
    template <typename T>
    static T apply(T x) {
        return static_cast<T>(std::cbrt(static_cast<T>(1) / x));
    }
};

template <typename T>
struct powcf_general {
    float exp;

    powcf_general(float e) : exp{e} {}

    T apply(T x) { return static_cast<T>(std::pow(std::abs(x), exp)); }
};

template <typename T, class ExpFunc>
void pow_invoke(const T* src, T* dst, size_t size, ExpFunc expfunc) {
    size_t i;
    for (i = 0; i + 4 <= size; i += 4) {
        T a0 = src[i], a1 = src[i + 1], a2 = src[i + 2], a3 = src[i + 3];
        T b0 = expfunc.apply(a0), b1 = expfunc.apply(a1),
          b2 = expfunc.apply(a2), b3 = expfunc.apply(a3);
        dst[i] = b0;
        dst[i + 1] = b1;
        dst[i + 2] = b2;
        dst[i + 3] = b3;
    }
#if MEGDNN_FIX_AARCH32_BUG
    // FIXME: as llvm may cause cannot select error if enable vectorize
    #pragma clang loop vectorize(disable)
#endif
    for (; i < size; ++i) {
        dst[i] = expfunc.apply(src[i]);
    }
}

bool float_eq(float x, float y) {
    return std::abs(x - y) < std::numeric_limits<float>::epsilon();
}

}  // anonymous namespace

template <typename T>
void PowCImpl::do_exec_ct(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                          const float* exp_f, const int* exp_i) {
    auto handle = static_cast<naive::HandleImpl*>(this->handle());
    auto sptr = reinterpret_cast<T*>(src.raw_ptr);
    auto dptr = reinterpret_cast<T*>(dst.raw_ptr);
    auto size = src.layout.total_nr_elems();

#define CALL(_expfunc)                                           \
    do {                                                         \
        auto kern = [ sptr, dptr, size, expfunc = _expfunc ]() { \
            pow_invoke(sptr, dptr, size, expfunc);               \
        };                                                       \
        handle->dispatch_kern(kern);                             \
        return;                                                  \
    } while (0)
    if (exp_f) {
        float fv = *exp_f;

#define CALL_IF(_v, _expfunc) \
    if (float_eq(fv, _v)) {   \
        CALL(_expfunc);       \
        return;               \
    }

        constexpr float croot = 1.f / 3.f;
        CALL_IF(.5f, powcf_sqrt{});
        CALL_IF(croot, powcf_cbrt{});
        CALL_IF(-.5f, powcf_rep_sqrt{});
        CALL_IF(-croot, powcf_rep_cbrt{});
        CALL(powcf_general<T>{fv});

#undef CALL_IF
    }

    int iv = *exp_i;
    switch (iv) {
#define CASE(n)           \
    case n:               \
        CALL(powci<n>{}); \
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
    if (iv & 1) {
        CALL(powci_general_odd{iv});
    } else {
        CALL(powci_general_even{iv});
    }
#undef CALL
}

void PowCImpl::do_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                       const float* exp_f, const int* exp_i) {
    if (!src.layout.is_contiguous()) {
        naive::PowCImpl::do_exec(src, dst, exp_f, exp_i);
        return;
    }
    switch (src.layout.dtype.enumv()) {
#define cb(dt)                  \
    case DTypeTrait<dt>::enumv: \
        return do_exec_ct<DTypeTrait<dt>::ctype>(src, dst, exp_f, exp_i);
        cb(dtype::Float32);
#undef cb

#if !MEGDNN_DISABLE_FLOAT16
        case DTypeTrait<dtype::Float16>::enumv:
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            return MEGDNN_INC_FLOAT16(
                    do_exec_ct<__fp16>(src, dst, exp_f, exp_i));
#else
            return MEGDNN_INC_FLOAT16(
                    do_exec_ct<dt_float16>(src, dst, exp_f, exp_i));
#endif
#endif
        default:
            megdnn_throw("unsupported dtype for PowC");
    }
}

// vim: syntax=cpp.doxygen
