/**
 * \file dnn/src/x86/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/x86/elemwise/opr_impl.h"
#include "src/x86/elemwise_op.h"
#include "src/x86/utils.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

#if MEGDNN_X86_WITH_MKL
#include <mkl_vml.h>
#endif

using namespace megdnn;
using namespace x86;

namespace {
#if MEGDNN_X86_WITH_MKL
void check_mkl_error(const char* func) {
    MEGDNN_MARK_USED_VAR(func);
    int err = vmlClearErrStatus();
    if (err != VML_STATUS_OK) {
#if MEGDNN_ENABLE_MANGLING
        megdnn_throw("mkl error");
#else
        const char* name;
        switch (err) {
#define ON(x)      \
    case x:        \
        name = #x; \
        break
            ON(VML_STATUS_BADSIZE);
            ON(VML_STATUS_BADMEM);
            ON(VML_STATUS_ERRDOM);
            ON(VML_STATUS_SING);
            ON(VML_STATUS_OVERFLOW);
            ON(VML_STATUS_UNDERFLOW);
            ON(VML_STATUS_ACCURACYWARNING);
#undef ON
            default:
                name = "UNKNOWN";
        }
        MEGDNN_MARK_USED_VAR(name);
        megdnn_throw(
                ssprintf("MKL func %s reported error: code=%d(%s); "
                         "possibly due to input data corruption.",
                         func, err, name));
#endif
    }
}
#endif
}  // namespace

#if MEGDNN_X86_WITH_MKL
#define DISPATCH_MKL(_mode, _func)                             \
    case Mode::_mode:                                          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(_func(n, sptr, dptr);     \
                                     check_mkl_error(#_func)); \
        return true
#endif

#define DISPATCH_TYPE(simd_type)                      \
    if (src0.layout.dtype == dtype::Float32{}) {      \
        DISPATCH_MODE_FLOAT(dt_float32, simd_type);   \
    } else if (src0.layout.dtype == dtype::Int32{}) { \
        DISPATCH_MODE_INT(dt_int32, simd_type);       \
    } else if (src0.layout.dtype == dtype::Int16{}) { \
        DISPATCH_MODE_INT(dt_int16, simd_type);       \
    } else if (src0.layout.dtype == dtype::Int8{}) {  \
        DISPATCH_MODE_INT(dt_int8, simd_type);        \
    }

#define DISPATCH_SIMD_TYPE                           \
    do {                                             \
        if (is_supported(SIMDType::AVX2)) {          \
            DISPATCH_TYPE(SIMDType::AVX2);           \
        } else if (is_supported(SIMDType::SSE4_2)) { \
            DISPATCH_TYPE(SIMDType::SSE4_2);         \
        }                                            \
    } while (0)

bool ElemwiseImpl::exec_unary() {
#define DISPATCH_UNARY(_mode, _type, _simd_type, _op)                          \
    case Mode::_mode: {                                                        \
        thin_function<void(const _type*, _type*, DType, DType, size_t)> run =  \
                OpCallerUnary<_op<_simd_type, _type, _type>, _simd_type>::run; \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                run(static_cast<const _type*>(src0.raw_ptr),                   \
                    static_cast<_type*>(dst_tensor.raw_ptr),                   \
                    src0.layout.dtype, dst_tensor.layout.dtype, nr_elems));    \
        return true;                                                           \
    }

    if (m_src->size() != 1)
        return false;

    // some optr only takes input data of float_32
    if (m_dst->layout.dtype != dtype::Float32() &&
        (param().mode == Mode::EXP || param().mode == Mode::SIGMOID ||
         param().mode == Mode::TANH || param().mode == Mode::FAST_TANH ||
         param().mode == Mode::SIN || param().mode == Mode::COS ||
         param().mode == Mode::LOG || param().mode == Mode::FLOOR ||
         param().mode == Mode::CEIL || param().mode == Mode::H_SWISH))
        return false;

    auto elparam = make_elemwise_op_param<1>();
    if (!elparam[0].layout.is_contiguous())
        return false;
    megdnn_assert(elparam[0].layout.ndim == 1);
    auto& src0 = elparam[0];
    auto& dst_tensor = *m_dst;
    size_t nr_elems = src0.layout.total_nr_elems();

#define DISPATCH_MODE_FLOAT(_type, _simd_type)                    \
    switch (param().mode) {                                       \
        DISPATCH_UNARY(RELU, _type, _simd_type, ReluOp);          \
        DISPATCH_UNARY(SIGMOID, _type, _simd_type, SigmoidOp);    \
        DISPATCH_UNARY(EXP, _type, _simd_type, ExpOp);            \
        DISPATCH_UNARY(FAST_TANH, _type, _simd_type, FastTanhOp); \
        DISPATCH_UNARY(H_SWISH, _type, _simd_type, HSwishOp);     \
        default:                                                  \
            break;                                                \
    }

#define DISPATCH_MODE_INT(_type, _simd_type)             \
    switch (param().mode) {                              \
        DISPATCH_UNARY(RELU, _type, _simd_type, ReluOp); \
        DISPATCH_UNARY(ABS, _type, _simd_type, AbsOp);   \
        default:                                         \
            break;                                       \
    }

    DISPATCH_SIMD_TYPE;
#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT
#undef DISPATCH_UNARY

#if MEGDNN_X86_WITH_MKL
    if (m_dst->layout.dtype == dtype::Float32()) {
        auto n = elparam[0].layout.shape[0];
        auto sptr = elparam[0].ptr<dt_float32>(),
             dptr = m_dst->ptr<dt_float32>();

        auto mkl_dispatch = [&]() {
            switch (param().mode) {
                DISPATCH_MKL(ABS, vsAbs);
                //! Delete the calculation support of MKL LOG because it will
                //! cause VML_STATUS_SING error, the reason is At least one of
                //! the input array values causes a divide-by-zero exception or
                //! produces an invalid (QNaN) result.
                // DISPATCH_MKL(LOG, vsLn);
                DISPATCH_MKL(COS, vsCos);
                DISPATCH_MKL(SIN, vsSin);
                DISPATCH_MKL(TANH, vsTanh);
                DISPATCH_MKL(FLOOR, vsFloor);
                DISPATCH_MKL(CEIL, vsCeil);
                default:
                    return false;
            }
        };
        return mkl_dispatch();
    }
#undef DISPATCH_MKL
#endif
    return false;
}

bool ElemwiseImpl::exec_binary() {
    if (m_src->size() != 2 ||
        m_src->front().layout.dtype != m_dst->layout.dtype ||
        m_src->back().layout.dtype != m_dst->layout.dtype) {
        return false;
    }

    // Optrs only for float32

    auto elparam = make_elemwise_op_param<2>();
    auto &src0 = elparam[0], &src1 = elparam[1];
    size_t n = src0.layout.total_nr_elems();

#define DISPATCH_MODE_FLOAT(_type, _simd_type)                                 \
    switch (param().mode) {                                                    \
        DISPATCH_BINARY(MIN, _type, _simd_type, MinOp);                        \
        DISPATCH_BINARY(MAX, _type, _simd_type, MaxOp);                        \
        DISPATCH_BINARY(ADD, _type, _simd_type, AddOp);                        \
        DISPATCH_BINARY(SUB, _type, _simd_type, SubOp);                        \
        DISPATCH_BINARY(MUL, _type, _simd_type, MulOp);                        \
        DISPATCH_BINARY(FUSE_ADD_RELU, _type, _simd_type, FuseAddReluOp);      \
        DISPATCH_BINARY(FUSE_ADD_H_SWISH, _type, _simd_type, FuseAddHSwishOp); \
        default:                                                               \
            break;                                                             \
    }

#define DISPATCH_MODE_INT(_type, _simd_type)                              \
    switch (param().mode) {                                               \
        DISPATCH_BINARY(MIN, _type, _simd_type, MinOp);                   \
        DISPATCH_BINARY(MAX, _type, _simd_type, MaxOp);                   \
        DISPATCH_BINARY(ADD, _type, _simd_type, AddOp);                   \
        DISPATCH_BINARY(SUB, _type, _simd_type, SubOp);                   \
        DISPATCH_BINARY(FUSE_ADD_RELU, _type, _simd_type, FuseAddReluOp); \
        default:                                                          \
            break;                                                        \
    }

    // Case 1: size of src0 and src1 are exactly match
    if (is_vector(src0.layout) && is_vector(src1.layout)) {
        megdnn_assert(n == m_dst->layout.total_nr_elems());
#define DISPATCH_BINARY(_mode, _type, _simd_type, _op)                       \
    case Mode::_mode: {                                                      \
        thin_function<void(const _type*, const _type*, _type*, DType, DType, \
                           DType, size_t)>                                   \
                run = OpCallerBinary<_op<_simd_type, _type, _type>,          \
                                     _simd_type, VEC_VEC>::run;              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                run(static_cast<const _type*>(src0.raw_ptr),                 \
                    static_cast<const _type*>(src1.raw_ptr),                 \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,     \
                    src1.layout.dtype, dst.layout.dtype,                     \
                    src0.layout.total_nr_elems()));                          \
        return true;                                                         \
    }
        auto&& dst = *m_dst;
        DISPATCH_SIMD_TYPE;
#undef DISPATCH_BINARY
    }

    // Case 2: vector + scalar
    {
#define DISPATCH_BINARY(_mode, _type, _simd_type, _op)                      \
    case Mode::_mode: {                                                     \
        thin_function<void(const _type*, const _type, _type*, DType, DType, \
                           DType, size_t)>                                  \
                run = OpCallerBinary<_op<_simd_type, _type, _type>,         \
                                     _simd_type, VEC_SCALAR>::run;          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                run(static_cast<const _type*>(src0.raw_ptr),                \
                    static_cast<const _type*>(src1.raw_ptr)[0],             \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,    \
                    src1.layout.dtype, dst.layout.dtype,                    \
                    src0.layout.total_nr_elems()));                         \
        return true;                                                        \
    }

        bool normal_case =
                is_vector(src0.layout) && is_broadcasted_scalar(src1.layout);
        bool swap_case = false;
        bool commutable = mode_trait().commutable;
        if (!normal_case && commutable) {
            swap_case = is_vector(src1.layout) &&
                        is_broadcasted_scalar(src0.layout);
        }
        if (normal_case || swap_case) {
            auto &lhs = src0, &rhs = src1;
            if (swap_case)
                std::swap(lhs, rhs);
            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
        }
#undef DISPATCH_BINARY

        // scalar + vector : only for nonswap op
#define DISPATCH_BINARY(_mode, _type, _simd_type, _op)                      \
    case Mode::_mode: {                                                     \
        thin_function<void(const _type, const _type*, _type*, DType, DType, \
                           DType, size_t)>                                  \
                run = OpCallerBinary<_op<_simd_type, _type, _type>,         \
                                     _simd_type, SCALAR_VEC>::run;          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                run(static_cast<const _type*>(src0.raw_ptr)[0],             \
                    static_cast<const _type*>(src1.raw_ptr),                \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,    \
                    src1.layout.dtype, dst.layout.dtype,                    \
                    src1.layout.total_nr_elems()));                         \
        return true;                                                        \
    }

        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_scalar(src0.layout)) {
            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
        }
#undef DISPATCH_BINARY
    }

    // Case 3: NCHW + 1C11
    {
#define DISPATCH_BINARY(_mode, _type, _simd_type, _op)                       \
    case Mode::_mode: {                                                      \
        thin_function<void(const _type*, const _type*, _type*, DType, DType, \
                           DType, size_t, size_t, size_t)>                   \
                run = OpCallerBinary<_op<_simd_type, _type, _type>,          \
                                     _simd_type, VEC_BCAST101>::run;         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                run(static_cast<const _type*>(src0.raw_ptr),                 \
                    static_cast<const _type*>(src1.raw_ptr),                 \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,     \
                    src1.layout.dtype, dst.layout.dtype, binfo.x, binfo.y,   \
                    binfo.z));                                               \
        return true;                                                         \
    }

        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src0.layout) &&
                           is_broadcasted_channel_like(src1.layout, binfo);
        bool swap_case = false;
        bool commutable = mode_trait().commutable;
        if (!normal_case && commutable) {
            swap_case = is_vector(src1.layout) &&
                        is_broadcasted_channel_like(src0.layout, binfo);
        }

        if (normal_case || swap_case) {
            auto &lhs = src0, &rhs = src1;
            if (swap_case)
                std::swap(lhs, rhs);
            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
        }
#undef DISPATCH_BINARY

#define DISPATCH_BINARY(_mode, _type, _simd_type, _op)                       \
    case Mode::_mode: {                                                      \
        thin_function<void(const _type*, const _type*, _type*, DType, DType, \
                           DType, size_t, size_t, size_t)>                   \
                run = OpCallerBinary<_op<_simd_type, _type, _type>,          \
                                     _simd_type, BCAST101_VEC>::run;         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                run(static_cast<const _type*>(src0.raw_ptr),                 \
                    static_cast<const _type*>(src1.raw_ptr),                 \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,     \
                    src1.layout.dtype, dst.layout.dtype, binfo.x, binfo.y,   \
                    binfo.z));                                               \
        return true;                                                         \
    }
        // BCAST_101 + VEC : only for nonswap op
        if (!commutable && is_vector(src1.layout) &&
            is_broadcasted_channel_like(src0.layout, binfo)) {
            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
        }

#undef DISPATCH_BINARY

#define DISPATCH_BINARY(_mode, _type, _simd_type, _op)                        \
    case Mode::_mode: {                                                       \
        thin_function<void(const _type*, const _type*, _type*, DType, DType,  \
                           DType, size_t, size_t, size_t, size_t)>            \
                run = OpCallerBinary<_op<_simd_type, _type, _type>,           \
                                     _simd_type, BCAST101x_VEC>::run;         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                         \
                run(static_cast<const _type*>(src0.raw_ptr),                  \
                    static_cast<const _type*>(src1.raw_ptr),                  \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,      \
                    src1.layout.dtype, dst.layout.dtype, batch_size, binfo.x, \
                    binfo.y, binfo.z));                                       \
        return true;                                                          \
    }
        {
            bool normal_case =
                    is_vector(src1.layout) &&
                    is_broadcastedx_channel_like<8>(src0.layout, binfo);
            bool swap_case = false;
            bool commutable = mode_trait().commutable;
            if (!normal_case && commutable) {
                swap_case = is_vector(src0.layout) &&
                            is_broadcastedx_channel_like<8>(src1.layout, binfo);
            }

            if ((swap_case || normal_case) &&
                src0.layout.dtype == dtype::Float32() && binfo.z == 8) {
                auto &lhs = src0, &rhs = src1;
                if (swap_case) {
                    std::swap(lhs, rhs);
                }

                size_t batch_size =
                        src1.layout.shape[0] / (binfo.x * binfo.y * binfo.z);
                auto&& dst = *m_dst;
                if (is_supported(SIMDType::AVX2)) {
                    DISPATCH_MODE_FLOAT(dt_float32, SIMDType::AVX2)
                } else {
                    switch (param().mode) {
                        DISPATCH_BINARY(ADD, dt_float32, SIMDType::NONE, AddOp);
                        DISPATCH_BINARY(FUSE_ADD_RELU, dt_float32,
                                        SIMDType::NONE, FuseAddReluOp);
                        default:
                            break;
                    }
                }
            }
        }
        return false;
#undef DISPATCH_BINARY

#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT
    }
}

//////////////////////////////////////////Ternary/////////////////////////
/*
 * src2 should be either a scalar or has the same shape with src0
 * src0 should has the same shape with src1 or {1, C, 1, 1}
 */
bool ElemwiseImpl::exec_ternary_fma3() {
#define DISPATCH_MODE_FLOAT(_type, _simd_type)                             \
    switch (param().mode) {                                                \
        DISPATCH_TERNARY(FUSE_MUL_ADD3, _type, _simd_type, FuseMulAdd3Op); \
        default:                                                           \
            return false;                                                  \
    }
#define DISPATCH_MODE_INT(_type, _simd_type) return false;

    if (param().mode != Mode::FUSE_MUL_ADD3) {
        return false;
    }
    ElemwiseOpParamN<3> elparam;
    bool c_is_scalar;
    prepare_fma3(elparam, c_is_scalar);
    auto &src0 = elparam[0], &src1 = elparam[1], &src2 = elparam[2];

    // Case 1: shape of (src0, src2) and src1 are exactly match
    if (is_vector(src0.layout) && is_vector(src1.layout) &&
        is_vector(src2.layout)) {
#define DISPATCH_TERNARY(_mode, _type, _simd_type, _op)                      \
    case Mode::_mode: {                                                      \
        thin_function<void(const _type*, const _type*, const _type*, _type*, \
                           DType, DType, DType, DType, size_t)>              \
                run = OpCallerTernary<_op<_simd_type, _type, _type>,         \
                                      _simd_type, VEC_VEC_VEC>::run;         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                run(static_cast<const _type*>(src0.raw_ptr),                 \
                    static_cast<const _type*>(src1.raw_ptr),                 \
                    static_cast<const _type*>(src2.raw_ptr),                 \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,     \
                    src1.layout.dtype, src2.layout.dtype, dst.layout.dtype,  \
                    src0.layout.total_nr_elems()));                          \
        return true;                                                         \
    }

        auto&& dst = *m_dst;
        DISPATCH_SIMD_TYPE;
#undef DISPATCH_TERNARY
    }

    // Case 2: (src2 is a scalar) &&
    //         (src0 and src1 has the same shape)
    {
        bool normal_case =
                is_vector(src0.layout) && is_vector(src1.layout) && c_is_scalar;
        if (normal_case) {
#define DISPATCH_TERNARY(_mode, _type, _simd_type, _op)                     \
    case Mode::_mode: {                                                     \
        thin_function<void(const _type*, const _type*, const _type, _type*, \
                           DType, DType, DType, DType, size_t)>             \
                run = OpCallerTernary<_op<_simd_type, _type, _type>,        \
                                      _simd_type, VEC_VEC_SCALAR>::run;     \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                run(static_cast<const _type*>(src0.raw_ptr),                \
                    static_cast<const _type*>(src1.raw_ptr),                \
                    static_cast<const _type*>(src2.raw_ptr)[0],             \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,    \
                    src1.layout.dtype, src2.layout.dtype, dst.layout.dtype, \
                    src0.layout.total_nr_elems()));                         \
        return true;                                                        \
    }

            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
#undef DISPATCH_TERNARY
        }
    }

    // Case 3: shape of src0 and src2 is {1, C, 1, 1}
    {
        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src1.layout) &&
                           is_broadcasted_channel_like(src0.layout, binfo) &&
                           src0.layout.eq_layout(src2.layout);
        if (normal_case) {
#define DISPATCH_TERNARY(_mode, _type, _simd_type, _op)                        \
    case Mode::_mode: {                                                        \
        thin_function<void(const _type*, const _type*, const _type*, _type*,   \
                           DType, DType, DType, DType, size_t, size_t,         \
                           size_t)>                                            \
                run = OpCallerTernary<_op<_simd_type, _type, _type>,           \
                                      _simd_type, BCAST101_VEC_BCAST101>::run; \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                run(static_cast<const _type*>(src0.raw_ptr),                   \
                    static_cast<const _type*>(src1.raw_ptr),                   \
                    static_cast<const _type*>(src2.raw_ptr),                   \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,       \
                    src1.layout.dtype, src2.layout.dtype, dst.layout.dtype,    \
                    binfo.x, binfo.y, binfo.z));                               \
        return true;                                                           \
    }

            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
#undef DISPATCH_TERNARY
        }
    }

    // Case 4: shape of src1 is {1, C, 1, 1}, and src0 and src2 are contig
    {
        BroadcastChannelInfo binfo;
        bool normal_case = is_vector(src0.layout) &&
                           src0.layout.eq_layout(src2.layout) &&
                           is_broadcasted_channel_like(src1.layout, binfo);
        if (normal_case) {
#define DISPATCH_TERNARY(_mode, _type, _simd_type, _op)                      \
    case Mode::_mode: {                                                      \
        thin_function<void(const _type*, const _type*, const _type*, _type*, \
                           DType, DType, DType, DType, size_t, size_t,       \
                           size_t)>                                          \
                run = OpCallerTernary<_op<_simd_type, _type, _type>,         \
                                      _simd_type, VEC_BCAST101_VEC>::run;    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                run(static_cast<const _type*>(src0.raw_ptr),                 \
                    static_cast<const _type*>(src1.raw_ptr),                 \
                    static_cast<const _type*>(src2.raw_ptr),                 \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,     \
                    src1.layout.dtype, src2.layout.dtype, dst.layout.dtype,  \
                    binfo.x, binfo.y, binfo.z));                             \
        return true;                                                         \
    }

            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
#undef DISPATCH_TERNARY
        }
    }

    // Case 5: (src1 is a scalar) && (src0 and src2 has the same shape)
    {
        bool normal_case = is_vector(src0.layout) && is_vector(src2.layout) &&
                           is_broadcasted_scalar(src1.layout);
        if (normal_case) {
#define DISPATCH_TERNARY(_mode, _type, _simd_type, _op)                     \
    case Mode::_mode: {                                                     \
        thin_function<void(const _type*, const _type, const _type*, _type*, \
                           DType, DType, DType, DType, size_t)>             \
                run = OpCallerTernary<_op<_simd_type, _type, _type>,        \
                                      _simd_type, VEC_SCALAR_VEC>::run;     \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                run(static_cast<const _type*>(src0.raw_ptr),                \
                    static_cast<const _type*>(src1.raw_ptr)[0],             \
                    static_cast<const _type*>(src2.raw_ptr),                \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,    \
                    src1.layout.dtype, src2.layout.dtype, dst.layout.dtype, \
                    src0.layout.total_nr_elems()));                         \
        return true;                                                        \
    }

            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
#undef DISPATCH_TERNARY
        }
    }
    // Case 6: (src1 and src2 is scalar) && (src0 is vector)
    {
        bool normal_case = is_vector(src0.layout) &&
                           is_broadcasted_scalar(src1.layout) &&
                           is_broadcasted_scalar(src2.layout);
        if (normal_case) {
#define DISPATCH_TERNARY(_mode, _type, _simd_type, _op)                     \
    case Mode::_mode: {                                                     \
        thin_function<void(const _type*, const _type, const _type, _type*,  \
                           DType, DType, DType, DType, size_t)>             \
                run = OpCallerTernary<_op<_simd_type, _type, _type>,        \
                                      _simd_type, VEC_SCALAR_SCALAR>::run;  \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                       \
                run(static_cast<const _type*>(src0.raw_ptr),                \
                    static_cast<const _type*>(src1.raw_ptr)[0],             \
                    static_cast<const _type*>(src2.raw_ptr)[0],             \
                    static_cast<_type*>(dst.raw_ptr), src0.layout.dtype,    \
                    src1.layout.dtype, src2.layout.dtype, dst.layout.dtype, \
                    src0.layout.total_nr_elems()));                         \
        return true;                                                        \
    }
            auto&& dst = *m_dst;
            DISPATCH_SIMD_TYPE;
#undef DISPATCH_TERNARY
        }
    }
    return false;
#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_MODE_INT
}

void ElemwiseImpl::exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) {
    if (!dst.layout.is_contiguous())
        return fallback::ElemwiseImpl::exec(srcs, dst);

    m_src = &srcs;
    m_dst = &dst;

    bool optimizing = false;
    optimizing |= m_dst->layout.dtype == dtype::Float32();
    optimizing |= m_dst->layout.dtype == dtype::Int32();
    optimizing |= m_dst->layout.dtype == dtype::Int16();
    optimizing |= m_dst->layout.dtype == dtype::Int8();
    if (optimizing) {
        if (exec_unary()) {
            return;
        }

        if (exec_binary()) {
            return;
        }
        if (exec_ternary_fma3()) {
            return;
        }
    }

    fallback::ElemwiseImpl::exec(srcs, dst);
}

// vim: syntax=cpp.doxygen
