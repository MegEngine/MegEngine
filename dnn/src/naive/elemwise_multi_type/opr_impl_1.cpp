#include "./opr_impl.h"
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise/kern_defs.cuh"
#include "src/common/elemwise_multi_type/kern_defs.cuh"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_int16x32x32x32(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_int16>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_int32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_int32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_int32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1) + (*i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
#else
    __builtin_trap();
#endif
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_int16xf32xf32xf32(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_int16>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_float32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_float32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1) + (*i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
#else
    __builtin_trap();
#endif
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_uint8xf32xf32xf32(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_uint8>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_float32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_float32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1) + (*i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
#else
    MEGDNN_MARK_USED_VAR(dst);
    __builtin_trap();
#endif
}

void ElemwiseMultiTypeImpl::on_mul_int16xf32xf32(
        const ElemwiseOpParamN<2>& param, const TensorND& dst) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto work = [src0, src1, size, dst]() {
        auto i0 = tensor_iter_valonly<dt_int16>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto dst_ptr = dst.ptr<dt_float32>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = (*i0) * (*i1);
            ++i0;
            ++i1;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
#else
    __builtin_trap();
#endif
}

void ElemwiseMultiTypeImpl::on_fuse_mul_add3_iXxf32xf32xi8(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    switch (param[0].layout.dtype.enumv()) {
#define cb(t)                  \
    case DTypeTrait<t>::enumv: \
        return dispatch_fma3_iXxf32xf32xi8<DTypeTrait<t>::ctype>(param, dst);
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
        default:
            megdnn_throw("unsupported src dtype");
    }
#else
    __builtin_trap();
#endif
}

template <typename ctype>
void ElemwiseMultiTypeImpl::dispatch_fma3_iXxf32xf32xi8(
        const ElemwiseOpParamN<3>& param, const TensorND& dst) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    auto size = param.size;
    auto src0 = param[0];
    auto src1 = param[1];
    auto src2 = param[2];
    auto work = [src0, src1, src2, size, dst]() {
        elemwise_multi_type::Fma3iXxf32xf32xiYOp<ctype, dt_int8> op;
        auto i0 = tensor_iter_valonly<ctype>(src0).begin();
        auto i1 = tensor_iter_valonly<dt_float32>(src1).begin();
        auto i2 = tensor_iter_valonly<dt_float32>(src2).begin();
        auto dst_ptr = dst.ptr<dt_int8>();
        for (size_t i = 0; i < size; ++i) {
            dst_ptr[i] = op(*i0, *i1, *i2);
            ++i0;
            ++i1;
            ++i2;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(work());
#else
    __builtin_trap();
#endif
}

// vim: syntax=cpp.doxygen
