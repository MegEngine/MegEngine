/**
 * \file dnn/src/cuda/elemwise/special_kerns.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./special_kerns.cuh"

namespace megdnn {
namespace cuda {
namespace elemwise_intl {

    template <typename ctype, bool c_is_scalar, typename enable = void>
    struct FuseMulAdd3Op {
        typedef ctype* __restrict bufptr_t;
        bufptr_t m_dst, m_src2;

        __device__ __forceinline__ void operator()(uint32_t idx, int off0,
                                                   int /* off1 */, ctype x,
                                                   ctype y) {
            m_dst[idx] = x * y + m_src2[c_is_scalar ? 0 : off0];
        }
    };

    template <typename ctype>
    struct FuseMulAdd3Op<ctype, true,
                         typename std::enable_if<
                                 std::is_same<ctype, dt_int8>::value ||
                                 std::is_same<ctype, dt_uint8>::value>::type> {
        typedef ctype* __restrict bufptr_t;
        typedef typename VectTypeTrait<ctype>::vect_type vect_type;
        bufptr_t m_dst, m_src2;
        __device__ __forceinline__ void operator()(uint32_t idx, int off0, int,
                                                   ctype x, ctype y) {
            m_dst[idx] = x * y + m_src2[0];
        }
        __device__ __forceinline__ void operator()(int32_t idx, int off0, int,
                                                   vect_type x, vect_type y) {
            ctype a = x.x * y.x + m_src2[0];
            ctype b = x.y * y.y + m_src2[0];
            ctype g = x.z * y.z + m_src2[0];
            ctype r = x.w * y.w + m_src2[0];
            *(vect_type*)(&m_dst[idx]) =
                    VectTypeTrait<ctype>::make_vector(a, b, g, r);
        }
    };

    template <typename ctype>
    struct FuseMulAdd3Op<ctype, false,
                         typename std::enable_if<
                                 std::is_same<ctype, dt_int8>::value ||
                                 std::is_same<ctype, dt_uint8>::value>::type> {
        typedef ctype* __restrict bufptr_t;
        typedef typename VectTypeTrait<ctype>::vect_type vect_type;
        bufptr_t m_dst, m_src2;
        __device__ __forceinline__ void operator()(uint32_t idx, int off0, int,
                                                   ctype x, ctype y) {
            m_dst[idx] = x * y + m_src2[off0];
        }
        __device__ __forceinline__ void operator()(int32_t idx, int off0, int,
                                                   vect_type x, vect_type y) {
            vect_type z = *(vect_type*)(&m_src2[off0]);
            ctype a = x.x * y.x + z.x;
            ctype b = x.y * y.y + z.y;
            ctype g = x.z * y.z + z.z;
            ctype r = x.w * y.w + z.w;
            *(vect_type*)(&m_dst[idx]) =
                    VectTypeTrait<ctype>::make_vector(a, b, g, r);
        }
    };

    template <typename ctype, typename enable = void>
    struct FuseMulAdd4Op {
        typedef ctype* __restrict bufptr_t;
        bufptr_t m_dst, m_src2, m_src3;

        __device__ __forceinline__ void operator()(uint32_t idx, int off0, int off1,
                                                   ctype src0, ctype src1) {
            m_dst[idx] = src0 * src1 + m_src2[off0] * m_src3[off1];
        }
    };

    template <typename ctype>
    struct FuseMulAdd4Op<ctype,
                         typename std::enable_if<
                                 std::is_same<ctype, dt_int8>::value ||
                                 std::is_same<ctype, dt_uint8>::value>::type> {
        typedef ctype* __restrict bufptr_t;
        typedef typename VectTypeTrait<ctype>::vect_type vect_type;
        bufptr_t m_dst, m_src2, m_src3;
        __device__ __forceinline__ void operator()(uint32_t idx, int off0,
                                                   int off1, ctype x, ctype y) {
            m_dst[idx] = x * y + m_src2[off0] * m_src3[off1];
        }
        __device__ __forceinline__ void operator()(uint32_t idx, int off0,
                                                   int off1, vect_type x,
                                                   vect_type y) {
            vect_type z = *(vect_type*)(&m_src2[off0]);
            vect_type w = *(vect_type*)(&m_src3[off1]);
            ctype a = x.x * y.x + z.x * w.x;
            ctype b = x.y * y.y + z.y * w.y;
            ctype g = x.z * y.z + z.z * w.z;
            ctype r = x.w * y.w + z.w * w.w;
            *(vect_type*)(&m_dst[idx]) =
                    VectTypeTrait<ctype>::make_vector(a, b, g, r);
        }
    };

    //! wrap an op so the special OpCaller can be selected by template matching
    template <class Op>
    class FuseOpWrapper {
        const Op& m_op;

    public:
        FuseOpWrapper(const Op& op) : m_op(op) {}

        operator const Op&() const { return m_op; }
    };

    template <class Op, class PVis0, class PVis1>
    struct OpCallerBinary<FuseOpWrapper<Op>, PVis0, PVis1> {
        Op op;
        PVis0 par0;
        PVis1 par1;
        MEGDNN_STATIC_ASSERT(PVis0::packed_size == PVis1::packed_size,
                             "vector size mismatch");
        static const uint32_t packed_size = PVis0::packed_size;

        __device__ __forceinline__ void thread_init(uint32_t idx) {
            idx = idx * packed_size;
            par0.thread_init(idx);
            par1.thread_init(idx);
        }

        __device__ __forceinline__ void on(uint32_t idx) {
            idx = idx * packed_size;
            op(idx, par0.offset(idx), par1.offset(idx), par0.at(idx),
               par1.at(idx));
        }

        __device__ __forceinline__ void on(uint32_t idx, uint32_t remain) {
            idx = idx * packed_size;
            if (remain >= packed_size) {
                op(idx, par0.offset(idx), par1.offset(idx), par0.at(idx),
                   par1.at(idx));
            } else {
                auto ptr0 = par0.ptr();
                auto ptr1 = par1.ptr();
                for (int i = 0; i < remain; i++) {
                    op(idx + i, par0.offset(idx + i), par1.offset(idx + i),
                       ptr0[par0.offset(idx + i)], ptr1[par1.offset(idx + i)]);
                }
            }
        }

        __device__ __forceinline__ void next() {
            par0.next();
            par1.next();
        }
    };

    template <class Op, class PVis>
    struct OpCallerUniform<FuseOpWrapper<Op>, 2, PVis> {
        Op op;
        PVis par[2];
        static const uint32_t packed_size = PVis::packed_size;

        __device__ __forceinline__ void thread_init(uint32_t idx) {
            idx = idx * packed_size;
            par[0].thread_init(idx);
            par[1].thread_init(idx);
        }

        __device__ __forceinline__ void on(uint32_t idx) {
            idx = idx * packed_size;
            op(idx, par[0].offset(idx), par[1].offset(idx), par[0].at(idx),
               par[1].at(idx));
        }

        __device__ __forceinline__ void on(uint32_t idx, uint32_t remain) {
            idx = idx * packed_size;
            if (remain >= packed_size) {
                op(idx, par[0].offset(idx), par[1].offset(idx), par[0].at(idx),
                   par[1].at(idx));
            } else {
                auto ptr0 = par[0].ptr();
                auto ptr1 = par[1].ptr();
                for (int i = 0; i < remain; i++) {
                    op(idx + i, par[0].offset(idx + i), par[1].offset(idx + i),
                       ptr0[par[0].offset(idx + i)],
                       ptr1[par[1].offset(idx + i)]);
                }
            }
        }

        __device__ __forceinline__ void next() {
            par[0].next();
            par[1].next();
        }
    };

}  // namespace elemwise_intl

namespace {
    template <typename ctype, class Op, int arity>
    void run_fuse_elemwise(Op& op, const ElemwiseOpParamN<arity>& param,
                           cudaStream_t stream) {
        param.assert_initialized();
        ElemwiseOpParamN<2> p2 = *static_cast<const ElemwiseOpParamN<2>*>(
                static_cast<const void*>(&param));
        elemwise_intl::UserOpInvoker<elemwise_intl::FuseOpWrapper<Op>, ctype, 2>(
                p2, stream, op);
    }
}  // anonymous namespace

    template <bool c_is_scalar, typename ctype>
    void kern_fuse_mul_add3(ctype* dest, const ElemwiseOpParamN<3>& param,
                            cudaStream_t stream) {
        elemwise_intl::FuseMulAdd3Op<ctype, c_is_scalar> op;
        op.m_dst = dest;
        op.m_src2 = param[2].ptr<ctype>();
        run_fuse_elemwise<ctype>(op, param, stream);
    }

    template <typename ctype>
    void kern_fuse_mul_add4(ctype* dest, const ElemwiseOpParamN<4>& param,
                            cudaStream_t stream) {
        elemwise_intl::FuseMulAdd4Op<ctype> op;
        op.m_dst = dest;
        op.m_src2 = param[2].ptr<ctype>();
        op.m_src3 = param[3].ptr<ctype>();
        run_fuse_elemwise<ctype>(op, param, stream);
    }

#define INST(_dt)                                                       \
    template void kern_fuse_mul_add3<true>(DTypeTrait<_dt>::ctype*,     \
                                           const ElemwiseOpParamN<3>&,  \
                                           cudaStream_t);               \
    template void kern_fuse_mul_add3<false>(DTypeTrait<_dt>::ctype*,    \
                                            const ElemwiseOpParamN<3>&, \
                                            cudaStream_t);              \
    template void kern_fuse_mul_add4(DTypeTrait<_dt>::ctype*,           \
                                     const ElemwiseOpParamN<4>&,        \
                                     cudaStream_t);

// vim: ft=cuda syntax=cpp.doxygen

