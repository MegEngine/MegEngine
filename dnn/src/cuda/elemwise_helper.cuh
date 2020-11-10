/**
 * \file dnn/src/cuda/elemwise_helper.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "src/common/elemwise_helper.cuh"
#include "src/cuda/int_fastdiv.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.cuh"

/*
 * please note that all arithmetics on GPU are 32-bit for best performance; this
 * limits max possible size
 */

namespace megdnn {
namespace cuda {

//! internals for element-wise
namespace elemwise_intl {
#define devfunc __device__ __forceinline__

/*!
 * \brief get cuda launch specs for element-wise kernel
 * \param kern kernel function address
 * \param size total size of elements
 */
void get_launch_spec(const void* kern, size_t size, int* grid_size,
                     int* block_size);

MEGDNN_NORETURN void on_bad_ndim(int ndim);

/*!
 * \brief broadcast type
 * BCAST_x[0]x[1]...: x[i] == !stride[i]
 */
enum BcastType {
    BCAST_OTHER,
    BCAST_1010,
    BCAST_101,
    BCAST_10,
    BCAST_01,
    BCAST_FULL
};

/*!
 * \brief read and write type trait for byte width integer type
 */
template <typename ctype>
class VectTypeTrait;

struct ATTR_ALIGNED(8) half4 {
    dt_float16 x, y, z, w;
};

__device__ __forceinline__ half4 make_half4(dt_float16 x, dt_float16 y,
                                            dt_float16 z, dt_float16 w) {
    half4 t;
    t.x = x, t.y = y, t.z = z, t.w = w;
    return t;
}

struct ATTR_ALIGNED(8) bhalf4 {
    dt_bfloat16 x, y, z, w;
};

__device__ __forceinline__ bhalf4 make_bhalf4(dt_bfloat16 x, dt_bfloat16 y,
                                              dt_bfloat16 z, dt_bfloat16 w) {
    bhalf4 t;
    t.x = x, t.y = y, t.z = z, t.w = w;
    return t;
}

#define INST(_ctype, _vect_type)                                               \
    template <>                                                                \
    class VectTypeTrait<_ctype> {                                              \
    public:                                                                    \
        using vect_type = _vect_type;                                          \
        static const size_t packed_size = sizeof(_vect_type) / sizeof(_ctype); \
        static __device__ __forceinline__ vect_type make_vector(_ctype x,      \
                                                                _ctype y,      \
                                                                _ctype z,      \
                                                                _ctype w) {    \
            return make_##_vect_type(as_raw(x), as_raw(y), as_raw(z),          \
                                     as_raw(w));                               \
        }                                                                      \
    }
#define as_raw(x) x
INST(dt_int8, char4);
INST(dt_uint8, uchar4);
INST(dt_float32, float4);
INST(dt_float16, half4);
INST(dt_bfloat16, bhalf4);
INST(dt_int32, int4);
INST(dt_int16, short4);
INST(dt_bool, uchar4);
#undef as_raw
#define as_raw(x) x.as_int8()
INST(dt_qint8, char4);
#undef as_raw
#define as_raw(x) x.as_uint8()
INST(dt_quint8, uchar4);
#undef as_raw
#define as_raw(x) x.as_int32()
INST(dt_qint32, int4);
#undef as_raw
#undef INST

/*!
 * \brief visitor to access an elemeent in a tensor at given logic index
 * \tparam ctype plain element ctype (i.e. ctype in DTypeTrait)
 * \tparam brdcast_mask bit mask for broadcast of params; (i.e. stride[i] is
 *      0 iff (brdcast_mask & (1<<(ndim-1-i))) is 1.
 *
 * host interface:
 *      void host_init(
 *              const TensorND &tensor, int grid_size, int block_size)
 *
 * device interface:
 *      void thread_init(uint32_t idx)
 *          called on thread entrance, with logical indexing; the index may
 *          go beyond buffer range
 *
 *      ctype* ptr()
 *          return buffer pointer; can be used by specialized OpCaller
 *
 *      void next()
 *          called before moving to next chunk on each thread
 *
 *      int offset(uint32_t idx)
 *          get physical offset from logical index
 *
 *      ctype& at(uint32_t idx)
 *          ptr()[offset(idx)]
 *
 */
template <int ndim, typename ctype, BcastType brd_type>
class ParamVisitorBase;

template <int ndim, typename ctype, BcastType brd_type>
class ParamElemVisitor;

/*!
 * \brief visitor to access vector element in a tensor at given logic index
 * \tparam ctype same as ParamElemVisitor, vect_type packed vector type of
 * element ctype (i.e. vect_type in VectTypeTrait) \tparam brdcast_mask same
 * as ParamElemVisitor
 *
 *
 * device interface:
 *      vect_type& at(uint32_t idx)
 *          ptr()[offset(idx)]
 *
 */

template <int ndim, typename ctype, BcastType brd_type>
class ParamVectVisitor;

/* f{{{ ParamElemVisitor specializations */

#define PARAM_ELEM_VISITOR_COMMON_DEV      \
    devfunc ctype* ptr() { return m_ptr; } \
    devfunc ctype& at(uint32_t idx) { return m_ptr[offset(idx)]; }
#define PARAM_ELEM_VISITOR_COMMON_HOST static const int packed_size = 1;

//! specialization for BCAST_OTHER
template <int ndim, typename ctype>
class ParamVisitorBase<ndim, ctype, BCAST_OTHER> {
protected:
    ctype* __restrict m_ptr;
    int m_stride[ndim];

    //! m_shape_highdim[i] = original_shape[i + 1]
#ifdef _MSC_VER
    Uint32Fastdiv m_shape_highdim[ndim > 1 ? ndim - 1 : 1];
#else
    Uint32Fastdiv m_shape_highdim[ndim];
#endif

public:
    static const int NDIM = ndim;

    void host_init(const TensorND& rv, int grid_size, int block_size,
                   int packed_size);
#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t) {}

    devfunc void next() {}

    devfunc int offset(uint32_t idx) {
        int offset = 0;
#pragma unroll
        for (int i = ndim - 1; i >= 1; --i) {
            Uint32Fastdiv& shp = m_shape_highdim[i - 1];
            uint32_t idx_div = idx / shp;
            offset += (idx - idx_div * shp.divisor()) * m_stride[i];
            idx = idx_div;
        }
        offset += idx * m_stride[0];
        return offset;
    }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

template <int ndim, typename ctype>
class ParamElemVisitor<ndim, ctype, BCAST_OTHER>
        : public ParamVisitorBase<ndim, ctype, BCAST_OTHER> {
public:
    PARAM_ELEM_VISITOR_COMMON_HOST

    void host_init(const TensorND& rv, int grid_size, int block_size) {
        ParamVisitorBase<ndim, ctype, BCAST_OTHER>::host_init(
                rv, grid_size, block_size, packed_size);
    }
};

/*!
 * \brief specialization for ndim == 3 and BCAST_101
 * (for dimshuffle 'x', 0, 'x')
 *
 * visit: idx / m_shape2 % m_shape1
 */
template <typename ctype>
class ParamVisitorBase<3, ctype, BCAST_101> {
    StridedDivSeq2 m_shape12;
    int m_stride1;

protected:
    ctype* __restrict m_ptr;

public:
    static const int NDIM = 3;

    void host_init(const TensorND& rv, int grid_size, int block_size,
                   int packed_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t idx) { m_shape12.device_init(idx); }

    devfunc void next() { m_shape12.next(); }

    devfunc int offset(uint32_t idx) { return m_shape12.get() * m_stride1; }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

template <typename ctype>
class ParamElemVisitor<3, ctype, BCAST_101>
        : public ParamVisitorBase<3, ctype, BCAST_101> {
public:
    PARAM_ELEM_VISITOR_COMMON_HOST

    void host_init(const TensorND& rv, int grid_size, int block_size) {
        ParamVisitorBase<3, ctype, BCAST_101>::host_init(
                rv, grid_size, block_size, packed_size);
    }
};

/*!
 * \brief specialization for ndim == 2 and BCAST_10
 *
 * visit: idx % m_shape1
 */
template <typename ctype>
class ParamVisitorBase<2, ctype, BCAST_10> {
    StridedDivSeq<false> m_shape1;
    int m_stride1;

protected:
    ctype* __restrict m_ptr;

public:
    static const int NDIM = 2;

    void host_init(const TensorND& rv, int grid_size, int block_size,
                   int packed_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t idx) { m_shape1.device_init(idx); }

    devfunc void next() { m_shape1.next(); }

    devfunc int offset(uint32_t idx) { return m_shape1.r() * m_stride1; }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

template <typename ctype>
class ParamElemVisitor<2, ctype, BCAST_10>
        : public ParamVisitorBase<2, ctype, BCAST_10> {
public:
    PARAM_ELEM_VISITOR_COMMON_HOST

    void host_init(const TensorND& rv, int grid_size, int block_size) {
        ParamVisitorBase<2, ctype, BCAST_10>::host_init(
                rv, grid_size, block_size, packed_size);
    }
};

/*!
 * \brief specialization for ndim == 2 and BCAST_01
 *
 * visit: idx / shape1
 */
template <typename ctype>
class ParamVisitorBase<2, ctype, BCAST_01> {
    StridedDivSeq<true> m_shape1;
    int m_stride0;

protected:
    ctype* __restrict m_ptr;

public:
    static const int NDIM = 2;

    void host_init(const TensorND& rv, int grid_size, int block_size,
                   int packed_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t idx) { m_shape1.device_init(idx); }

    devfunc void next() { m_shape1.next(); }

    devfunc int offset(uint32_t idx) { return m_shape1.q() * m_stride0; }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

template <typename ctype>
class ParamElemVisitor<2, ctype, BCAST_01>
        : public ParamVisitorBase<2, ctype, BCAST_01> {
public:
    PARAM_ELEM_VISITOR_COMMON_HOST

    void host_init(const TensorND& rv, int grid_size, int block_size) {
        ParamVisitorBase<2, ctype, BCAST_01>::host_init(
                rv, grid_size, block_size, packed_size);
    }
};

//! specialization for ndim == 1 and BCAST_FULL
template <typename ctype>
class ParamVisitorBase<1, ctype, BCAST_FULL> {
protected:
    ctype* __restrict m_ptr;

public:
    static const int NDIM = 1;
    PARAM_ELEM_VISITOR_COMMON_HOST

    void host_init(const TensorND& rv, int grid_size, int block_size,
                   int packed_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t) {}

    devfunc void next() {}

    devfunc int offset(uint32_t idx) {
        MEGDNN_MARK_USED_VAR(idx);
        return 0;
    }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

template <typename ctype>
class ParamElemVisitor<1, ctype, BCAST_FULL>
        : public ParamVisitorBase<1, ctype, BCAST_FULL> {
public:
    PARAM_ELEM_VISITOR_COMMON_HOST

    void host_init(const TensorND& rv, int grid_size, int block_size) {
        ParamVisitorBase<1, ctype, BCAST_FULL>::host_init(
                rv, grid_size, block_size, packed_size);
    }
};

#undef PARAM_ELEM_VISITOR_COMMON_DEV
#undef PARAM_ELEM_VISITOR_COMMON_HOST

/* f}}} */

/* f{{{ ParamVectVisitor specializations */

#if MEGDNN_CC_CUDA
#define DEVICE_WRAPPER(x) x
#else
#define DEVICE_WRAPPER(x)
#endif
#define INST_PARAM_VECT_VISITOR                                             \
    template <int ndim, typename ctype>                                     \
    class ParamVectVisitor<ndim, ctype, _brdcast_mask>                      \
            : public ParamVisitorBase<ndim, ctype, _brdcast_mask> {         \
    public:                                                                 \
        using Super = ParamVisitorBase<ndim, ctype, _brdcast_mask>;         \
        using rwtype = typename VectTypeTrait<ctype>::vect_type;            \
        static const int packed_size = sizeof(rwtype) / sizeof(ctype);      \
        void host_init(const TensorND& rv, int grid_size, int block_size) { \
            ParamVisitorBase<ndim, ctype, _brdcast_mask>::host_init(        \
                    rv, grid_size, block_size, packed_size);                \
        }                                                                   \
        DEVICE_WRAPPER(devfunc rwtype& at(uint32_t idx) {                   \
            return *(rwtype*)(&Super::m_ptr[Super::offset(idx)]);           \
        })                                                                  \
    };
#define _brdcast_mask BCAST_OTHER
INST_PARAM_VECT_VISITOR;
#undef _brdcast_mask
#define _brdcast_mask BCAST_01
INST_PARAM_VECT_VISITOR;
#undef _brdcast_mask
#define _brdcast_mask BCAST_10
INST_PARAM_VECT_VISITOR;
#undef _brdcast_mask
#define _brdcast_mask BCAST_101
INST_PARAM_VECT_VISITOR;
#undef _brdcast_mask
#define INST_DT_IBYTE(ctype)                                                \
    template <int ndim>                                                     \
    class ParamVectVisitor<ndim, ctype, BCAST_FULL>                         \
            : public ParamVisitorBase<ndim, ctype, BCAST_FULL> {            \
    public:                                                                 \
        using Super = ParamVisitorBase<ndim, ctype, BCAST_FULL>;            \
        using rwtype = typename VectTypeTrait<ctype>::vect_type;            \
        static const int packed_size = sizeof(rwtype) / sizeof(ctype);      \
        void host_init(const TensorND& rv, int grid_size, int block_size) { \
            ParamVisitorBase<ndim, ctype, BCAST_FULL>::host_init(           \
                    rv, grid_size, block_size, packed_size);                \
        }                                                                   \
        DEVICE_WRAPPER(rwtype vect_scalar;                                  \
                       devfunc rwtype & at(uint32_t /* idx */) {            \
                           ctype v = Super::m_ptr[0];                       \
                           vect_scalar = VectTypeTrait<ctype>::make_vector( \
                                   v, v, v, v);                             \
                           return vect_scalar;                              \
                       })                                                   \
    }
INST_DT_IBYTE(dt_int8);
INST_DT_IBYTE(dt_uint8);
INST_DT_IBYTE(dt_qint8);
INST_DT_IBYTE(dt_quint8);
INST_DT_IBYTE(dt_bool);
#undef INST_DT_IBYTE
#undef DEVICE_WRAPPER
#undef INST_PARAM_VECT_VISITOR

/*!
 * \brief specialization for ndim == 4 and BCAST_1010
 *
 * visit: (idx % m_shape3) * m_stride3 + (idx / m_shape23 % m_shape1) *
 * m_stride1
 */
template <typename ctype>
class ParamVectVisitor<4, ctype, BCAST_1010> {
    StridedDivSeq2 m_shape123;
    StridedDivSeq<false> m_shape3;
    int m_stride3, m_stride1;
    ctype* __restrict m_ptr;

public:
    static const int NDIM = 4;
    using rwtype = typename VectTypeTrait<ctype>::vect_type;
    static const int packed_size = sizeof(rwtype) / sizeof(ctype);

    void host_init(const TensorND& rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t idx) {
        m_shape123.device_init(idx);
        m_shape3.device_init(idx);
    }

    devfunc void next() {
        m_shape123.next();
        m_shape3.next();
    }

    devfunc int offset(uint32_t idx) {
        return m_shape3.r() * m_stride3 + m_shape123.get() * m_stride1;
    }

    devfunc ctype* ptr() { return m_ptr; }
    devfunc rwtype& at(uint32_t idx) { return *(rwtype*)(&m_ptr[offset(idx)]); }
#endif
};

/* f}}} */

#if MEGDNN_CC_CUDA

/* f{{{ user operator callers */

/*
 * OpCaller is used to invoke user operator with loaded element arguments.
 *
 * device interface:
 *      void thread_init(uint32_t idx);
 *
 *      void on(uint32_t idx);
 *
 *      void next();
 */

/*!
 * \brief call user op directly without visiting any params (i.e. arity ==
 *      0)
 */
template <class Op>
struct OpCallerNull {
    Op op;

    devfunc void thread_init(uint32_t) {}

    devfunc void on(uint32_t idx) { op(idx); }

    devfunc void next() {}
};

/*!
 * \brief call an operator whose each param are promted to the same ndim and
 *      brdcast_mask
 * \tparam PVis ParamElemVisitor class
 */
template <class Op, int arity, class PVis>
struct OpCallerUniform;

//! specialization for arity == 1
template <class Op, class PVis>
struct OpCallerUniform<Op, 1, PVis> {
    Op op;
    PVis par[1];
    static const uint32_t packed_size = PVis::packed_size;

    devfunc void thread_init(uint32_t idx) {
        idx = idx * packed_size;
        par[0].thread_init(idx);
    }

    devfunc void on(uint32_t idx) {
        idx = idx * packed_size;
        op(idx, par[0].at(idx));
    }

    devfunc void on(uint32_t idx, uint32_t remain) {
        idx = idx * packed_size;
        if (remain >= packed_size) {
            op(idx, par[0].at(idx));
        } else {
            auto ptr0 = par[0].ptr();
            for (int i = 0; i < remain; i++) {
                op(idx + i, ptr0[par[0].offset(idx + i)]);
            }
        }
    }

    devfunc void next() { par[0].next(); }
};
//! specialization for arity == 2
template <class Op, class PVis>
struct OpCallerUniform<Op, 2, PVis> {
    Op op;
    PVis par[2];
    static const uint32_t packed_size = PVis::packed_size;

    devfunc void thread_init(uint32_t idx) {
        idx = idx * packed_size;
        par[0].thread_init(idx);
        par[1].thread_init(idx);
    }

    devfunc void on(uint32_t idx) {
        idx = idx * packed_size;
        op(idx, par[0].at(idx), par[1].at(idx));
    }

    devfunc void on(uint32_t idx, uint32_t remain) {
        idx = idx * packed_size;
        if (remain >= packed_size) {
            op(idx, par[0].at(idx), par[1].at(idx));
        } else {
            auto ptr0 = par[0].ptr();
            auto ptr1 = par[1].ptr();
            for (int i = 0; i < remain; i++) {
                op(idx + i, ptr0[par[0].offset(idx + i)],
                   ptr1[par[1].offset(idx + i)]);
            }
        }
    }

    devfunc void next() {
        par[0].next();
        par[1].next();
    }
};
//! specialization for arity == 3
template <class Op, class PVis>
struct OpCallerUniform<Op, 3, PVis> {
    Op op;
    PVis par[3];
    static const uint32_t packed_size = PVis::packed_size;

    devfunc void thread_init(uint32_t idx) {
        idx = idx * packed_size;
        par[0].thread_init(idx);
        par[1].thread_init(idx);
        par[2].thread_init(idx);
    }

    devfunc void on(uint32_t idx) {
        idx = idx * packed_size;
        op(idx, par[0].at(idx), par[1].at(idx), par[2].at(idx));
    }

    devfunc void on(uint32_t idx, uint32_t remain) {
        idx = idx * packed_size;
        if (remain >= packed_size) {
            op(idx, par[0].at(idx), par[1].at(idx), par[2].at(idx));
        } else {
            auto ptr0 = par[0].ptr();
            auto ptr1 = par[1].ptr();
            auto ptr2 = par[2].ptr();
            for (int i = 0; i < remain; i++) {
                op(idx + i, ptr0[par[0].offset(idx + i)],
                   ptr1[par[1].offset(idx + i)], ptr2[par[2].offset(idx + i)]);
            }
        }
    }

    devfunc void next() {
        par[0].next();
        par[1].next();
        par[2].next();
    }
};

//! specialization for arity == 4
template <class Op, class PVis>
struct OpCallerUniform<Op, 4, PVis> {
    Op op;
    PVis par[4];
    static const uint32_t packed_size = PVis::packed_size;

    devfunc void thread_init(uint32_t idx) {
        idx = idx * packed_size;
        par[0].thread_init(idx);
        par[1].thread_init(idx);
        par[2].thread_init(idx);
        par[3].thread_init(idx);
    }

    devfunc void on(uint32_t idx) {
        idx = idx * packed_size;
        op(idx, par[0].at(idx), par[1].at(idx), par[2].at(idx), par[3].at(idx));
    }

    devfunc void on(uint32_t idx, uint32_t remain) {
        idx = idx * packed_size;
        if (remain >= packed_size) {
            op(idx, par[0].at(idx), par[1].at(idx), par[2].at(idx),
               par[3].at(idx));
        } else {
            auto ptr0 = par[0].ptr();
            auto ptr1 = par[1].ptr();
            auto ptr2 = par[2].ptr();
            auto ptr3 = par[3].ptr();
            for (int i = 0; i < remain; i++) {
                op(idx + i, ptr0[par[0].offset(idx + i)],
                   ptr1[par[1].offset(idx + i)], ptr2[par[2].offset(idx + i)],
                   ptr3[par[3].offset(idx + i)]);
            }
        }
    }

    devfunc void next() {
        par[0].next();
        par[1].next();
        par[2].next();
        par[3].next();
    }
};

//! specialization for arity == 5
template <class Op, class PVis>
struct OpCallerUniform<Op, 5, PVis> {
    Op op;
    PVis par[5];
    static const uint32_t packed_size = PVis::packed_size;

    devfunc void thread_init(uint32_t idx) {
        idx = idx * packed_size;
        par[0].thread_init(idx);
        par[1].thread_init(idx);
        par[2].thread_init(idx);
        par[3].thread_init(idx);
        par[4].thread_init(idx);
    }

    devfunc void on(uint32_t idx) {
        idx = idx * packed_size;
        op(idx, par[0].at(idx), par[1].at(idx), par[2].at(idx), par[3].at(idx),
           par[4].at(idx));
    }

    devfunc void on(uint32_t idx, uint32_t remain) {
        idx = idx * packed_size;
        if (remain >= packed_size) {
            op(idx, par[0].at(idx), par[1].at(idx), par[2].at(idx),
               par[3].at(idx), par[4].at(idx));
        } else {
            auto ptr0 = par[0].ptr();
            auto ptr1 = par[1].ptr();
            auto ptr2 = par[2].ptr();
            auto ptr3 = par[3].ptr();
            auto ptr4 = par[4].ptr();
            for (int i = 0; i < remain; i++) {
                op(idx + i, ptr0[par[0].offset(idx + i)],
                   ptr1[par[1].offset(idx + i)], ptr2[par[2].offset(idx + i)],
                   ptr3[par[3].offset(idx + i)], ptr4[par[4].offset(idx + i)]);
            }
        }
    }

    devfunc void next() {
        par[0].next();
        par[1].next();
        par[2].next();
        par[3].next();
        par[4].next();
    }
};

/*!
 * \brief call binary (i.e. arity == 2) operator with different param
 *      visitors
 */
template <class Op, class PVis0, class PVis1>
struct OpCallerBinary {
    Op op;
    PVis0 par0;
    PVis1 par1;
    MEGDNN_STATIC_ASSERT(PVis0::packed_size == PVis1::packed_size,
                         "vector size mismatch")

    static const uint32_t packed_size = PVis0::packed_size;

    devfunc void thread_init(uint32_t idx) {
        idx = idx * packed_size;
        par0.thread_init(idx);
        par1.thread_init(idx);
    }

    devfunc void on(uint32_t idx) {
        idx = idx * packed_size;
        op(idx, par0.at(idx), par1.at(idx));
    }

    devfunc void next() {
        par0.next();
        par1.next();
    }
};

/* f}}} */

template <class OpCaller>
__global__ void cuda_kern(OpCaller op_caller, uint32_t size) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x,
             delta = blockDim.x * gridDim.x;
    // each thread works on at most 3 elements; see get_launch_spec
    op_caller.thread_init(idx);
    if (idx < size) {
        op_caller.on(idx);
        idx += delta;
        if (idx < size) {
            op_caller.next();
            op_caller.on(idx);
            idx += delta;
            if (idx < size) {
                op_caller.next();
                op_caller.on(idx);
            }
        }
    }
}

template <class Op, int arity, class PVis>
__global__ void cuda_kern(OpCallerUniform<Op, arity, PVis> op_caller,
                          uint32_t size) {
    constexpr uint32_t packed_size = PVis::packed_size;
    const uint32_t size_packed = DIVUP(size, packed_size);
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x,
             delta = blockDim.x * gridDim.x;
    if (idx < size_packed) {
        op_caller.on(idx, size - packed_size * idx);
        idx += delta;
        if (idx < size_packed) {
            op_caller.on(idx, size - packed_size * idx);
            idx += delta;
            if (idx < size_packed) {
                op_caller.on(idx, size - packed_size * idx);
            }
        }
    }
}

//! invoke a user Op passed to run_elemwise
template <class Op, typename ctype, int arity>
class UserOpInvoker;

/* f{{{ UserOpInvoker specializations */

//! run op by promoting all params to same ndim
template <class Op, typename ctype, int arity>
class UserOpInvokerToSameNdim {
    const ElemwiseOpParamN<arity>& m_param;
    cudaStream_t m_stream;
    const Op& m_op;

    void dispatch0() {
        switch (m_param.max_ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1<ndim>();
            MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        }
        on_bad_ndim(m_param.max_ndim);
    }

    template <int ndim>
    void dispatch1() {
        typedef OpCallerUniform<Op, arity,
                                ParamElemVisitor<ndim, ctype, BCAST_OTHER>>
                Caller;
        size_t size = m_param.size;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);

        Caller caller;
        caller.op = m_op;
        for (int i = 0; i < arity; ++i)
            caller.par[i].host_init(m_param[i], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);
        after_kernel_launch();
    }

public:
    UserOpInvokerToSameNdim(const ElemwiseOpParamN<arity>& param,
                            cudaStream_t stream, const Op& op)
            : m_param(param), m_stream(stream), m_op(op) {
        dispatch0();
    }
};

template <class Op, typename ctype, int arity>
class UserOpInvokerToSameNdimIByteHelper {
public:
    UserOpInvokerToSameNdimIByteHelper(const ElemwiseOpParamN<arity>& param,
                                       cudaStream_t stream, const Op& op)
            : m_rw_size(param.size),
              m_param(param),
              m_stream(stream),
              m_op(op) {
        if (!try_vect_load_store_contiguous() && !try_vect_load_store()) {
            dispatch0();
        }
    }

private:
    const ElemwiseOpParamN<arity>& m_param;
    size_t m_rw_size;
    cudaStream_t m_stream;
    const Op& m_op;
    using vect_type = typename VectTypeTrait<ctype>::vect_type;
    static const size_t packed_size = VectTypeTrait<ctype>::packed_size;

    void dispatch0() {
        switch (m_param.max_ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1<ndim>();
            MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        }
        on_bad_ndim(m_param.max_ndim);
    }

    void dispatch0_vect() {
        switch (m_param.max_ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1_vect<ndim>();
            MEGDNN_FOREACH_TENSOR_NDIM(cb)
#undef cb
        }
        on_bad_ndim(m_param.max_ndim);
    }

    void dispatch_contiguous() {
        typedef ParamVectVisitor<1, ctype, BCAST_OTHER> PVis;
        typedef OpCallerUniform<Op, arity, PVis> Caller;
        size_t size = m_rw_size;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Op, arity, PVis>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);

        Caller caller;
        caller.op = m_op;
        for (int i = 0; i < arity; ++i)
            caller.par[i].host_init(m_param[i], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, m_param.size);
        after_kernel_launch();
    }

    template <int ndim>
    void dispatch1() {
        typedef ParamElemVisitor<ndim, ctype, BCAST_OTHER> PVis;
        typedef OpCallerUniform<Op, arity, PVis> Caller;
        size_t size = m_rw_size;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);
        Caller caller;
        caller.op = m_op;
        for (int i = 0; i < arity; ++i)
            caller.par[i].host_init(m_param[i], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);
        after_kernel_launch();
    }

    template <int ndim>
    void dispatch1_vect() {
        typedef ParamVectVisitor<ndim, ctype, BCAST_OTHER> PVis;
        typedef OpCallerUniform<Op, arity, PVis> Caller;
        size_t size = m_rw_size;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);
        Caller caller;
        caller.op = m_op;
        for (int i = 0; i < arity; ++i)
            caller.par[i].host_init(m_param[i], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);
        after_kernel_launch();
    }

    bool try_vect_load_store() {
        auto try_last_contig = [](const TensorLayout& layout) {
            return layout.stride[layout.ndim - 1] == 1 &&
                   layout[layout.ndim - 1] % packed_size == 0;
        };
        /*
         * \NOTE: remove try_scalar() to adapt multi-type tenary op
         */
        for (int i = 0; i < arity; ++i) {
            if (!try_last_contig(m_param[i].layout))
                return false;
        }
        m_rw_size /= packed_size;
        dispatch0_vect();
        return true;
    }

    bool try_vect_load_store_contiguous() {
        auto try_contig = [](const TensorLayout& layout) {
            return (layout.is_contiguous());
        };
        for (int i = 0; i < arity; ++i) {
            if (!try_contig(m_param[i].layout))
                return false;
        }
        m_rw_size = DIVUP(m_rw_size, packed_size);
        dispatch_contiguous();
        return true;
    }
};

#define INST_DT_IBYTE(ctype)                                                \
    template <class Op, int arity>                                          \
    class UserOpInvokerToSameNdim<Op, ctype, arity>                         \
            : public UserOpInvokerToSameNdimIByteHelper<Op, ctype, arity> { \
        using Super = UserOpInvokerToSameNdimIByteHelper<Op, ctype, arity>; \
                                                                            \
    public:                                                                 \
        UserOpInvokerToSameNdim(const ElemwiseOpParamN<arity>& param,       \
                                cudaStream_t stream, const Op& op)          \
                : Super{param, stream, op} {}                               \
    }
INST_DT_IBYTE(dt_int8);
INST_DT_IBYTE(dt_uint8);
INST_DT_IBYTE(dt_qint8);
INST_DT_IBYTE(dt_quint8);
INST_DT_IBYTE(dt_bool);
#undef INST_DT_IBYTE

//! implement general case by UserOpInvokerToSameNdim
template <class Op, typename ctype, int arity>
class UserOpInvoker : public UserOpInvokerToSameNdim<Op, ctype, arity> {
public:
    UserOpInvoker(const ElemwiseOpParamN<arity>& param, cudaStream_t stream,
                  const Op& op)
            : UserOpInvokerToSameNdim<Op, ctype, arity>(param, stream, op) {}
};

//! specialization for arity == 0
template <class Op, typename ctype>
class UserOpInvoker<Op, ctype, 0> {
public:
    UserOpInvoker(const ElemwiseOpParamN<0>& param, cudaStream_t stream,
                  const Op& op) {
        size_t size = param.size;
        typedef OpCallerNull<Op> Caller;
        Caller caller;
        caller.op = op;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);
        (*fptr)<<<grid_size, block_size, 0, stream>>>(caller, size);
        after_kernel_launch();
    }
};

#define DEFINE_BRDCAST_DISPATCH_RECEIVERS(_cb_header, _cb_dispatch, _stride) \
    _cb_header(1) {                                                          \
        const ptrdiff_t* stride = _stride;                                   \
        if (!stride[0]) {                                                    \
            return _cb_dispatch(1, BCAST_FULL);                              \
        }                                                                    \
        _cb_dispatch(1, BCAST_OTHER);                                        \
    }                                                                        \
    _cb_header(2) {                                                          \
        const ptrdiff_t* stride = _stride;                                   \
        if (!stride[0] && stride[1]) {                                       \
            return _cb_dispatch(2, BCAST_10);                                \
        }                                                                    \
        if (stride[0] && !stride[1]) {                                       \
            return _cb_dispatch(2, BCAST_01);                                \
        }                                                                    \
        _cb_dispatch(2, BCAST_OTHER);                                        \
    }                                                                        \
    _cb_header(3) {                                                          \
        const ptrdiff_t* stride = _stride;                                   \
        if (!stride[0] && stride[1] && !stride[2]) {                         \
            return _cb_dispatch(3, BCAST_101);                               \
        }                                                                    \
        _cb_dispatch(3, BCAST_OTHER);                                        \
    }

//! specialization for binary opr
template <class Op, typename ctype>
class UserOpInvoker<Op, ctype, 2> {
    bool m_invoked;
    const ElemwiseOpParamN<2>& m_param;
    cudaStream_t m_stream;
    const Op& m_op;

    void fallback() {
        megdnn_assert(!m_invoked);
        UserOpInvokerToSameNdim<Op, ctype, 2>(m_param, m_stream, m_op);
        m_invoked = true;
    }

    void dispatch0() {
        switch (m_param[0].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1_##ndim();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
        }
        fallback();
    }

#define cb_header(ndim) void dispatch1_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    dispatch2<ParamElemVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                      m_param[0].layout.stride)
#undef cb_header
#undef cb_dispatch

    template <class PVis0>
    void dispatch2() {
        switch (m_param[1].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch3_##ndim<PVis0>();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
        }
        fallback();
    }

#define cb_header(ndim)    \
    template <class PVis0> \
    void dispatch3_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    do_run<PVis0, ParamElemVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                      m_param[1].layout.stride)
#undef cb_header
#undef cb_dispatch

    template <class PVis0, class PVis1>
    void do_run() {
        megdnn_assert(!m_invoked);
        m_invoked = true;
        typedef OpCallerBinary<Op, PVis0, PVis1> Caller;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        size_t size = m_param.size;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);
        Caller caller;
        caller.op = m_op;
        caller.par0.host_init(m_param[0], grid_size, block_size);
        caller.par1.host_init(m_param[1], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);
        after_kernel_launch();
    }

public:
    UserOpInvoker(const ElemwiseOpParamN<2>& param, cudaStream_t stream,
                  const Op& op)
            : m_param(param), m_stream(stream), m_op(op) {
        m_invoked = false;
        dispatch0();
        megdnn_assert(m_invoked);
    }
};

#define DEFINE_VECT_BRDCAST_DISPATCH_RECEIVERS(_cb_header, _cb_dispatch, \
                                               _stride)                  \
    DEFINE_BRDCAST_DISPATCH_RECEIVERS(_cb_header, _cb_dispatch, _stride) \
    _cb_header(4) {                                                      \
        const ptrdiff_t* stride = _stride;                               \
        if (!stride[0] && stride[1] && !stride[2] && stride[3]) {        \
            return _cb_dispatch(4, BCAST_1010);                          \
        }                                                                \
        _cb_dispatch(4, BCAST_OTHER);                                    \
    }

template <class Op, typename ctype>
class UserOpInvokerBinaryIByteHelper {
private:
    bool m_invoked;
    size_t m_rw_size;
    const ElemwiseOpParamN<2>& m_param;
    cudaStream_t m_stream;
    const Op& m_op;
    using vect_type = typename VectTypeTrait<ctype>::vect_type;
    static const size_t packed_size = VectTypeTrait<ctype>::packed_size;
    bool try_vect_load_store() {
        auto try_last_contig_or_scalar = [](const TensorLayout& layout) {
            return (layout.stride[layout.ndim - 1] == 1 &&
                    layout[layout.ndim - 1] % packed_size == 0) ||
                   (layout.ndim == 1 && layout.stride[0] == 0);
        };
        for (int i = 0; i < 2; ++i) {
            if (!try_last_contig_or_scalar(m_param[i].layout))
                return false;
        }
        m_rw_size /= packed_size;
        dispatch0_vect();
        return true;
    }

    bool try_vect_load_store_contiguous() {
        auto try_contig = [](const TensorLayout& layout) {
            return (layout.is_contiguous());
        };
        for (int i = 0; i < 2; ++i) {
            if (!try_contig(m_param[i].layout))
                return false;
        }
        m_rw_size = DIVUP(m_rw_size, packed_size);
        dispatch_contiguous();
        return true;
    }

    void dispatch0() {
        switch (m_param[0].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1_##ndim();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
        }
        fallback();
    }

    void dispatch0_vect() {
        switch (m_param[0].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch1_vect_##ndim();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
            case 4:
                return dispatch1_vect_4();
        }
        fallback();
    }

    void dispatch_contiguous() {
        m_invoked = true;
        typedef ParamVectVisitor<1, ctype, BCAST_OTHER> PVis;
        typedef OpCallerUniform<Op, 2, PVis> Caller;
        size_t size = m_rw_size;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Op, 2, PVis>;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);

        Caller caller;
        caller.op = m_op;
        for (int i = 0; i < 2; ++i)
            caller.par[i].host_init(m_param[i], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, m_param.size);
        after_kernel_launch();
    }

    void fallback() {
        megdnn_assert(!m_invoked);
        UserOpInvokerToSameNdim<Op, ctype, 2>(m_param, m_stream, m_op);
        m_invoked = true;
    }

#define cb_header(ndim) void dispatch1_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    dispatch2<ParamElemVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                      m_param[0].layout.stride)
#undef cb_header
#undef cb_dispatch

#define cb_header(ndim) void dispatch1_vect_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    dispatch2_vect<ParamVectVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_VECT_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                           m_param[0].layout.stride)
#undef cb_header
#undef cb_dispatch

    template <class PVis0>
    void dispatch2() {
        switch (m_param[1].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch3_##ndim<PVis0>();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
        }
        fallback();
    }

    template <class PVis0>
    void dispatch2_vect() {
        switch (m_param[1].layout.ndim) {
#define cb(ndim) \
    case ndim:   \
        return dispatch3_vect_##ndim<PVis0>();
            MEGDNN_FOREACH_TENSOR_NDIM_SMALL(cb)
#undef cb
            case 4:
                return dispatch3_vect_4<PVis0>();
        }
        fallback();
    }

#define cb_header(ndim)    \
    template <class PVis0> \
    void dispatch3_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    do_run<PVis0, ParamElemVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                      m_param[1].layout.stride)
#undef cb_header
#undef cb_dispatch

#define cb_header(ndim)    \
    template <class PVis0> \
    void dispatch3_vect_##ndim()
#define cb_dispatch(ndim, brdcast_mask) \
    do_run<PVis0, ParamVectVisitor<ndim, ctype, brdcast_mask>>()
    DEFINE_VECT_BRDCAST_DISPATCH_RECEIVERS(cb_header, cb_dispatch,
                                           m_param[1].layout.stride)
#undef cb_header
#undef cb_dispatch

    template <class PVis0, class PVis1>
    void do_run() {
        megdnn_assert(!m_invoked);
        m_invoked = true;
        typedef OpCallerBinary<Op, PVis0, PVis1> Caller;
        int grid_size, block_size;
        void (*fptr)(Caller, uint32_t) = cuda_kern<Caller>;
        size_t size = m_rw_size;
        get_launch_spec(reinterpret_cast<const void*>(fptr), size, &grid_size,
                        &block_size);
        Caller caller;
        caller.op = m_op;
        caller.par0.host_init(m_param[0], grid_size, block_size);
        caller.par1.host_init(m_param[1], grid_size, block_size);
        (*fptr)<<<grid_size, block_size, 0, m_stream>>>(caller, size);
        after_kernel_launch();
    }

public:
    UserOpInvokerBinaryIByteHelper(const ElemwiseOpParamN<2>& param,
                                   cudaStream_t stream, const Op& op)
            : m_rw_size(param.size),
              m_param(param),
              m_stream(stream),
              m_op(op) {
        m_invoked = false;
        if (!try_vect_load_store_contiguous() && !try_vect_load_store()) {
            dispatch0();
        }
        megdnn_assert(m_invoked);
    }
};

#define INST_DT_IBYTE(ctype)                                                 \
    template <class Op>                                                      \
    class UserOpInvoker<Op, ctype, 2>                                        \
            : public UserOpInvokerBinaryIByteHelper<Op, ctype> {             \
        using Super = UserOpInvokerBinaryIByteHelper<Op, ctype>;             \
                                                                             \
    public:                                                                  \
        UserOpInvoker(const ElemwiseOpParamN<2>& param, cudaStream_t stream, \
                      const Op& op)                                          \
                : Super{param, stream, op} {}                                \
    }
INST_DT_IBYTE(dt_int8);
INST_DT_IBYTE(dt_uint8);
INST_DT_IBYTE(dt_qint8);
INST_DT_IBYTE(dt_quint8);
INST_DT_IBYTE(dt_bool);
#undef INST_DT_IBYTE
#endif

#undef DEFINE_BRDCAST_DISPATCH_RECEIVERS
#undef DEFINE_VECT_BRDCAST_DISPATCH_RECEIVERS

/* f}}} */

#undef devfunc
}  // namespace elemwise_intl

/*!
 * \brief general element-wise kernel launcher
 *
 * \tparam arity number of params for the operator
 * \param param param values for the operator; must have been initialized (i.e.
 *      by calling ElemwiseOpParamN::init_from_given_tensor). The params
 *      can have arbitrary layouts, as long as they share the same total number
 *      of elements.
 * \param op callable with a signature compatible with
 *      `void op(uint32_t idx, ctype& param0, ..., ctype& param[arity - 1])`
 *      if arity == 0, there is only an `idx` input
 *      if ctype=dt_int8, dt_uint8, dt_qint8, dt_quint8, a signature compatible
 * with `void op(uint32_t idx, vect_type& param0, ..., ctype& param[arity - 1])`
 * should be implemented
 */
template <class Op, typename ctype, int arity>
void run_elemwise(const ElemwiseOpParamN<arity>& param, cudaStream_t stream,
                  const Op& op = Op());

#if MEGDNN_CC_CUDA
template <class Op, typename ctype, int arity>
void run_elemwise(const ElemwiseOpParamN<arity>& param, cudaStream_t stream,
                  const Op& op) {
    param.assert_initialized();
    elemwise_intl::UserOpInvoker<Op, ctype, arity>(param, stream, op);
}

/*!
 * \brief explicit instantialization of run_elemwise for given template params;
 *      used in .cu files, so corresponding run_elemwise can be called from .cpp
 */
#define INST_RUN_ELEMWISE(Op, ctype, arity)       \
    template void run_elemwise<Op, ctype, arity>( \
            const ElemwiseOpParamN<arity>&, cudaStream_t, const Op&)

#endif

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
