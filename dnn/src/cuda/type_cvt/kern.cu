/**
 * \file dnn/src/cuda/type_cvt/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "megdnn/dtype.h"
#include "src/cuda/elemwise_helper.cuh"

using namespace megdnn;
using namespace cuda;
using namespace elemwise_intl;

namespace {
template <typename ctype_dest, typename ctype_src, typename enable = void>
struct TypeCvtOp {
    ctype_dest* dest;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = static_cast<ctype_dest>(src);
    }
};

template <typename ctype_dest, typename ctype_src, typename enable = void>
struct TypeCvtOpToQuantized {
    ctype_dest* dest;
    CudaDTypeParam<ctype_dest> param;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = param.quantize(src);
    }
};

template <typename ctype_dest, typename ctype_src, typename enable = void>
struct TypeCvtOpFromQuantized {
    ctype_dest* dest;
    CudaDTypeParam<ctype_src> param;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = static_cast<ctype_dest>(param.dequantize(src));
    }
};

template <typename ctype_dest, typename ctype_src, typename enable = void>
struct TypeCvtOpBetweenQuantized {
    ctype_dest* dest;
    CudaDTypeParam<ctype_src> src_param;
    CudaDTypeParam<ctype_dest> dst_param;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = dst_param.quantize(src_param.dequantize(src));
    }
};

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOp<ctype_dest, ctype_src,
                 typename std::enable_if<
                         std::is_same<ctype_src, dt_int8>::value ||
                         std::is_same<ctype_src, dt_uint8>::value ||
						 std::is_same<ctype_src, dt_bool>::value>::type> {
    ctype_dest* dest;
    using src_vect_type = typename VectTypeTrait<ctype_src>::vect_type;
    using dst_vect_type = typename VectTypeTrait<ctype_dest>::vect_type;
    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = static_cast<ctype_dest>(src);
    }
    __device__ __forceinline__ void operator()(uint32_t idx,
                                               src_vect_type src) {
        ctype_dest x = static_cast<ctype_dest>(src.x);
        ctype_dest y = static_cast<ctype_dest>(src.y);
        ctype_dest z = static_cast<ctype_dest>(src.z);
        ctype_dest w = static_cast<ctype_dest>(src.w);
        *(dst_vect_type*)(&dest[idx]) =
                VectTypeTrait<ctype_dest>::make_vector(x, y, z, w);
    }
};

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOpToQuantized<
        ctype_dest, ctype_src,
        typename std::enable_if<
                std::is_same<ctype_src, dt_int8>::value ||
                std::is_same<ctype_src, dt_uint8>::value ||
				std::is_same<ctype_src, dt_bool>::value>::type> {
    ctype_dest* dest;
    CudaDTypeParam<ctype_dest> param;
    using src_vect_type = typename VectTypeTrait<ctype_src>::vect_type;
    using dst_vect_type = typename VectTypeTrait<ctype_dest>::vect_type;
    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = param.quantize(src);
    }
    __device__ __forceinline__ void operator()(uint32_t idx,
                                               src_vect_type src) {
        ctype_dest x = param.quantize(src.x);
        ctype_dest y = param.quantize(src.y);
        ctype_dest z = param.quantize(src.z);
        ctype_dest w = param.quantize(src.w);
        *(dst_vect_type*)(&dest[idx]) =
                VectTypeTrait<ctype_dest>::make_vector(x, y, z, w);
    }
};

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOpFromQuantized<
        ctype_dest, ctype_src,
        typename std::enable_if<
                std::is_same<ctype_src, dt_qint8>::value ||
                std::is_same<ctype_src, dt_quint8>::value>::type> {
    ctype_dest* dest;
    CudaDTypeParam<ctype_src> param;
    using src_vect_type = typename VectTypeTrait<ctype_src>::vect_type;
    using dst_vect_type = typename VectTypeTrait<ctype_dest>::vect_type;
    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = static_cast<ctype_dest>(param.dequantize(src));
    }
    __device__ __forceinline__ void operator()(uint32_t idx,
                                               src_vect_type src) {
        ctype_dest x =
                static_cast<ctype_dest>(param.dequantize(ctype_src(src.x)));
        ctype_dest y =
                static_cast<ctype_dest>(param.dequantize(ctype_src(src.y)));
        ctype_dest z =
                static_cast<ctype_dest>(param.dequantize(ctype_src(src.z)));
        ctype_dest w =
                static_cast<ctype_dest>(param.dequantize(ctype_src(src.w)));
        *(dst_vect_type*)(&dest[idx]) =
                VectTypeTrait<ctype_dest>::make_vector(x, y, z, w);
    }
};

template <typename ctype_dest, typename ctype_src>
struct TypeCvtOpBetweenQuantized<
        ctype_dest, ctype_src,
        typename std::enable_if<
                std::is_same<ctype_src, dt_qint8>::value ||
                std::is_same<ctype_src, dt_quint8>::value>::type> {
    ctype_dest* dest;
    CudaDTypeParam<ctype_src> src_param;
    CudaDTypeParam<ctype_dest> dst_param;
    using src_vect_type = typename VectTypeTrait<ctype_src>::vect_type;
    using dst_vect_type = typename VectTypeTrait<ctype_dest>::vect_type;
    __device__ __forceinline__ ctype_dest apply(ctype_src in) {
        float inter = src_param.dequantize(in);
        return dst_param.quantize(inter);
    }
    __device__ __forceinline__ void operator()(uint32_t idx, ctype_src src) {
        dest[idx] = dst_param.quantize(src_param.dequantize(src));
    }
    __device__ __forceinline__ void operator()(uint32_t idx,
                                               src_vect_type src) {
        ctype_dest x = apply(ctype_src(src.x));
        ctype_dest y = apply(ctype_src(src.y));
        ctype_dest z = apply(ctype_src(src.z));
        ctype_dest w = apply(ctype_src(src.w));
        *(dst_vect_type*)(&dest[idx]) =
                VectTypeTrait<ctype_dest>::make_vector(x, y, z, w);
    }
};
}  // anonymous namespace

#define main_func(OpType, body)                                    \
    {                                                              \
        typedef typename DTypeTrait<dtype_src>::ctype ctype_src;   \
        typedef typename DTypeTrait<dtype_dest>::ctype ctype_dest; \
        typedef OpType<ctype_dest, ctype_src> Op;                  \
        ElemwiseOpParamN<1> param;                                 \
        param[0] = src;                                            \
        param.init_from_given_tensor();                            \
        megdnn_assert(DTypeTrait<ctype_src>::enumv ==              \
                      src.layout.dtype.enumv().ev);                \
        megdnn_assert(DTypeTrait<ctype_dest>::enumv ==             \
                      dest.layout.dtype.enumv().ev);               \
        Op op;                                                     \
        op.dest = dest.ptr<ctype_dest>();                          \
        body;                                                      \
        return run_elemwise<Op, ctype_src, 1>(param, stream, op);  \
    }

namespace megdnn {
namespace cuda {

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2q(
        const TensorND& dest, const TensorND& src,
        const CudaDTypeParam<dtype_src>& src_param,
        const CudaDTypeParam<dtype_dest>& dst_param,
        cudaStream_t stream) {
    main_func(TypeCvtOpBetweenQuantized, op.dst_param = dst_param;
              op.src_param = src_param;)
}

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2q(
        const TensorND& dest, const TensorND& src,
        const CudaDTypeParam<dtype_dest>& dst_param,
        cudaStream_t stream) {
    main_func(TypeCvtOpToQuantized, op.param = dst_param;);
}

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_q2n(
        const TensorND& dest, const TensorND& src,
        const CudaDTypeParam<dtype_src>& src_param,
        cudaStream_t stream) {
    main_func(TypeCvtOpFromQuantized, op.param = src_param;);
}

template <typename dtype_src, typename dtype_dest>
void typecvt_kern_n2n(const TensorND& dest, const TensorND& src,
                      cudaStream_t stream) {
    main_func(TypeCvtOp, );
}

#define INST_Q2Q(dtype_src, dtype_dest)                    \
    template void typecvt_kern_q2q<dtype_src, dtype_dest>( \
            const TensorND& dest, const TensorND& src,     \
            const CudaDTypeParam<dtype_src>& src_param,    \
            const CudaDTypeParam<dtype_dest>& dst_param, cudaStream_t stream);

#define INST_Q2N(dtype_src, dtype_dest)                    \
    template void typecvt_kern_q2n<dtype_src, dtype_dest>( \
            const TensorND& dest, const TensorND& src,     \
            const CudaDTypeParam<dtype_src>& src_param, cudaStream_t stream);

#define INST_N2Q(dtype_src, dtype_dest)                    \
    template void typecvt_kern_n2q<dtype_src, dtype_dest>( \
            const TensorND& dest, const TensorND& src,     \
            const CudaDTypeParam<dtype_dest>& dst_param, cudaStream_t stream);

#define INST_N2N(dtype_src, dtype_dest)                    \
    template void typecvt_kern_n2n<dtype_src, dtype_dest>( \
            const TensorND& dest, const TensorND& src, cudaStream_t stream);

#define MEGDNN_FOREACH_COMPUTING_DTYPE_WITH_DTYPE_SRC(dtype_src, cb) \
    cb(dtype_src, dt_int8) \
    cb(dtype_src, dt_int32) \
    cb(dtype_src, dt_int16) \
    cb(dtype_src, dt_uint8) \
    cb(dtype_src, dt_float32) \
    cb(dtype_src, dt_float16) \
    cb(dtype_src, dt_bfloat16) \
    cb(dtype_src, dt_bool) \

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

#define MEGDNN_FOREACH_COMPUTING_CTYPE(cb) \
    cb(dt_int8) \
    cb(dt_int32) \
    cb(dt_int16) \
    cb(dt_uint8) \
    cb(dt_float32) \
    cb(dt_float16) \
    cb(dt_bfloat16) \
    cb(dt_bool) \

#define MEGDNN_FOREACH_QUANTIZED_CTYPE(cb) \
    cb(dt_quint8) \
    cb(dt_qint32) \
    cb(dt_qint8)

MEGDNN_FOREACH_QUANTIZED_CTYPE(INST_SRC_QUANTIZED)
MEGDNN_FOREACH_COMPUTING_CTYPE(INST_SRC_NORMAL)

template void typecvt_kern_n2q<dtype::Int8, dtype::QuantizedS8>(
        const TensorND& src, const TensorND& dst,
        const CudaDTypeParam<dt_qint8>& param, cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
