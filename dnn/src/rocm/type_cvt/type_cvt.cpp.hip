/**
 * \file src/rocm/type_cvt/type_cvt.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "./type_cvt.h.hip"
#include "megdnn/dtype.h"
#include "src/rocm/elemwise_helper.h.hip"

namespace megdnn {
namespace rocm {
namespace {

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOp {
    ctype_dest* dest;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = static_cast<ctype_dest>(src);
    }
};

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOpToQuantized {
    ctype_dest* dest;
    DTypeParam<ctype_dest> param;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = param.quantize(src);
    }
};

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOpFromQuantized {
    ctype_dest* dest;
    DTypeParam<ctype_src> param;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = static_cast<ctype_dest>(param.dequantize(src));
    }
};

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOpBetweenQuantized {
    ctype_dest* dest;
    DTypeParam<ctype_src> src_param;
    DTypeParam<ctype_dest> dst_param;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = dst_param.quantize(src_param.dequantize(src));
    }
};

}  // anonymous namespace

#define main_func(OpType, body)                                        \
    {                                                                  \
        typedef typename DTypeTrait<dtype_src>::ctype ctype_src;       \
        typedef typename DTypeTrait<dtype_dest>::ctype ctype_dest;     \
        typedef OpType<ctype_dest, ctype_src> Op;                      \
        ElemwiseOpParamN<1> param;                                     \
        param[0] = src;                                                \
        param.init_from_given_tensor();                                \
        megdnn_assert(DTypeTrait<ctype_src>::enumv ==                  \
                      src.layout.dtype.enumv().ev);                    \
        megdnn_assert(DTypeTrait<ctype_dest>::enumv ==                 \
                      dest.layout.dtype.enumv().ev);                   \
        Op op;                                                         \
        op.dest = dest.ptr<ctype_dest>();                              \
        body; \
        return run_elemwise<Op, ctype_src, 1>(param, stream, op); \
    }

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2q(
        const TensorND& dest, const TensorND& src,
        const DTypeParam<dtype_src>& src_param,
        const DTypeParam<dtype_dest>& dst_param,
        hipStream_t stream) {
    main_func(TypeCvtOpBetweenQuantized, op.dst_param = dst_param;
              op.src_param = src_param;)
}

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2q(
        const TensorND& dest, const TensorND& src,
        const DTypeParam<dtype_dest>& dst_param,
        hipStream_t stream) {
    main_func(TypeCvtOpToQuantized, op.param = dst_param;);
}

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2n(
        const TensorND& dest, const TensorND& src,
        const DTypeParam<dtype_src>& src_param,
        hipStream_t stream) {
    main_func(TypeCvtOpFromQuantized, op.param = src_param;);
}

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2n(const TensorND& dest, const TensorND& src,
                      hipStream_t stream) {
    main_func(TypeCvtOp, );
}

#define INST_Q2Q(dtype_src, dtype_dest)                               \
    template void typecvt_kern_q2q<dtype_src, dtype_dest>(            \
            const TensorND& dest, const TensorND& src,                \
            const DTypeParam<dtype_src>&  \
                    src_param,                                        \
            const DTypeParam<dtype_dest>& \
                    dst_param,                                        \
            hipStream_t stream);

#define INST_Q2N(dtype_src, dtype_dest)                              \
    template void typecvt_kern_q2n<dtype_src, dtype_dest>(           \
            const TensorND& dest, const TensorND& src,               \
            const DTypeParam<dtype_src>& \
                    src_param,                                       \
            hipStream_t stream);

#define INST_N2Q(dtype_src, dtype_dest)                               \
    template void typecvt_kern_n2q<dtype_src, dtype_dest>(            \
            const TensorND& dest, const TensorND& src,                \
            const DTypeParam<dtype_dest>& \
                    dst_param,                                        \
            hipStream_t stream);

#define INST_N2N(dtype_src, dtype_dest)                    \
    template void typecvt_kern_n2n<dtype_src, dtype_dest>( \
            const TensorND& dest, const TensorND& src, hipStream_t stream);

#if !MEGDNN_DISABLE_FLOAT16
#define MEGDNN_FOREACH_COMPUTING_DTYPE_WITH_DTYPE_SRC(dtype_src, cb) \
    cb(dtype_src, dt_int8) \
    cb(dtype_src, dt_int32) \
    cb(dtype_src, dt_int16) \
    cb(dtype_src, dt_uint8) \
    cb(dtype_src, dt_float32) \
    cb(dtype_src, dt_float16) \
    cb(dtype_src, dt_bfloat16) \

#else

#define MEGDNN_FOREACH_COMPUTING_DTYPE_WITH_DTYPE_SRC(dtype_src, cb) \
    cb(dtype_src, dt_int8) \
    cb(dtype_src, dt_int32) \
    cb(dtype_src, dt_int16) \
    cb(dtype_src, dt_uint8) \
    cb(dtype_src, dt_float32) \

#endif


#define MEGDNN_FOREACH_QUANTIZED_DTYPE_WITH_DTYPE_SRC(dtype_src, cb) \
    cb(dtype_src, dt_quint8) \
    cb(dtype_src, dt_qint32) \
    cb(dtype_src, dt_qint8) \

#define INST_SRC_QUANTIZED(dtype_src) \
    MEGDNN_FOREACH_COMPUTING_DTYPE_WITH_DTYPE_SRC(dtype_src, INST_Q2N) \
    MEGDNN_FOREACH_QUANTIZED_DTYPE_WITH_DTYPE_SRC(dtype_src, INST_Q2Q) \

#define INST_SRC_NORMAL(dtype_src) \
    MEGDNN_FOREACH_COMPUTING_DTYPE_WITH_DTYPE_SRC(dtype_src, INST_N2N) \
    MEGDNN_FOREACH_QUANTIZED_DTYPE_WITH_DTYPE_SRC(dtype_src, INST_N2Q) \

#if !MEGDNN_DISABLE_FLOAT16
#define MEGDNN_FOREACH_COMPUTING_CTYPE(cb) \
    cb(dt_int8) \
    cb(dt_int32) \
    cb(dt_int16) \
    cb(dt_uint8) \
    cb(dt_float32) \
    cb(dt_float16) \
    cb(dt_bfloat16) \

#else
#define MEGDNN_FOREACH_COMPUTING_CTYPE(cb) \
    cb(dt_int8) \
    cb(dt_int32) \
    cb(dt_int16) \
    cb(dt_uint8) \
    cb(dt_float32) \

#endif

#define MEGDNN_FOREACH_QUANTIZED_CTYPE(cb) \
    cb(dt_quint8) \
    cb(dt_qint32) \
    cb(dt_qint8)

MEGDNN_FOREACH_QUANTIZED_CTYPE(INST_SRC_QUANTIZED)
MEGDNN_FOREACH_COMPUTING_CTYPE(INST_SRC_NORMAL)

template void typecvt_kern_n2q<dtype::Int8, dtype::QuantizedS8>(
        const TensorND& src, const TensorND& dst,
        const DTypeParam<dt_qint8>& param, hipStream_t stream);

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
