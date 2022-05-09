#pragma once

#include "megdnn/tensor_iter.h"
#include "src/common/elemwise_multi_type/opr_impl_helper.h"
#include "src/naive/handle.h"
namespace megdnn {
namespace naive {

class ElemwiseMultiTypeImpl : public ElemwiseMultiTypeImplHelper {
    template <typename KernImpl, typename src_ctype, typename ElemParam>
    void dispatch_add_qint_op_dst(const ElemParam& param, const TensorND& dst) {
        switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                     \
    case DTypeTrait<_dt>::enumv:                                                    \
        dispatch_add_qint_op<KernImpl, src_ctype, typename DTypeTrait<_dt>::ctype>( \
                param, dst);                                                        \
        break;
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
            MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
#undef cb

            default:
                megdnn_assert(
                        0, "not support %s %s\n", param[0].layout.dtype.name(),
                        dst.layout.dtype.name());
        }
    }

    template <typename KernImpl, typename ElemParam>
    void dispatch_qint_op_dtype(const ElemParam& param, const TensorND& dst) {
        switch (param[0].layout.dtype.enumv()) {
#define cb(_dt)                                                                    \
    case DTypeTrait<_dt>::enumv:                                                   \
        dispatch_add_qint_op_dst<                                                  \
                KernImpl, typename DTypeTrait<_dt>::ctype, ElemParam>(param, dst); \
        break;
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
            MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
#undef cb

            default:
                megdnn_assert_internal(0);
        }
    }

    template <typename KernImpl, typename src_ctype, typename dst_ctype>
    void dispatch_add_qint_op(
            const ElemwiseOpParamN<1>& param, const TensorND& dst_tensor) {
        auto src = param[0];
        auto size = param.size;
        auto work = [src, size, dst_tensor]() {
            auto iA = tensor_iter_valonly<src_ctype>(src).begin();
            auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();

            auto param0 =
                    src.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
            auto dst_param = dst_tensor.layout.dtype
                                     .param<typename DTypeTrait<dst_ctype>::dtype>();
            for (size_t i = 0; i < size; i++) {
                src_ctype a = *iA;
                *pD = dst_param.quantize(KernImpl::apply(param0.dequantize(a)));
                ++iA;
                ++pD;
            }
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(work());
    }

    template <typename KernImpl, typename src_ctype, typename dst_ctype>
    void dispatch_dst_bool_op(
            const ElemwiseOpParamN<1>& param, const TensorND& dst_tensor) {
        auto size = param.size;
        auto src0 = param[0];
        auto work = [src0, size, dst_tensor]() {
            // This is needed as these iterators are captured as const value.
            auto iA = tensor_iter_valonly<src_ctype>(src0).begin();
            auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();
            for (size_t i = 0; i < size; i++) {
                src_ctype a = *iA;
                *pD = KernImpl::apply(a);
                ++iA;
                ++pD;
            }
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(work());
    }

    template <typename KernImpl, typename src_ctype, typename dst_ctype>
    void dispatch_add_qint_op(
            const ElemwiseOpParamN<2>& param, const TensorND& dst_tensor) {
        auto size = param.size;
        auto src0 = param[0];
        auto src1 = param[1];
        auto work = [src0, src1, size, dst_tensor]() {
            // This is needed as these iterators are captured as const value.
            auto iA = tensor_iter_valonly<src_ctype>(src0).begin();
            auto iB = tensor_iter_valonly<src_ctype>(src1).begin();
            auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();
            auto param0 =
                    src0.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
            auto param1 =
                    src1.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
            auto dst_param = dst_tensor.layout.dtype
                                     .param<typename DTypeTrait<dst_ctype>::dtype>();
            for (size_t i = 0; i < size; i++) {
                src_ctype a = *iA;
                src_ctype b = *iB;
                *pD = dst_param.quantize(
                        KernImpl::apply(param0.dequantize(a), param1.dequantize(b)));
                ++iA;
                ++iB;
                ++pD;
            }
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(work());
    }

    template <typename KernImpl, typename src_ctype, typename dst_ctype>
    void dispatch_dst_bool_op(
            const ElemwiseOpParamN<2>& param, const TensorND& dst_tensor) {
        auto size = param.size;
        auto src0 = param[0];
        auto src1 = param[1];
        auto work = [src0, src1, size, dst_tensor]() {
            // This is needed as these iterators are captured as const value.
            auto iA = tensor_iter_valonly<src_ctype>(src0).begin();
            auto iB = tensor_iter_valonly<src_ctype>(src1).begin();
            auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();
            for (size_t i = 0; i < size; i++) {
                src_ctype a = *iA;
                src_ctype b = *iB;
                *pD = KernImpl::apply(a, b);
                ++iA;
                ++iB;
                ++pD;
            }
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(work());
    }

    template <typename KernImpl, typename src_ctype, typename dst_ctype>
    void dispatch_add_qint_op(
            const ElemwiseOpParamN<3>& param, const TensorND& dst_tensor) {
        auto size = param.size;
        auto src0 = param[0];
        auto src1 = param[1];
        auto src2 = param[2];
        auto work = [src0, src1, src2, size, dst_tensor]() {
            // This is needed as these iterators are captured as const value.
            auto iA = tensor_iter_valonly<src_ctype>(src0).begin();
            auto iB = tensor_iter_valonly<src_ctype>(src1).begin();
            auto iC = tensor_iter_valonly<src_ctype>(src2).begin();
            auto pD = tensor_iter_valonly<dst_ctype>(dst_tensor).begin();
            auto param0 =
                    src0.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
            auto param1 =
                    src1.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
            auto param2 =
                    src2.layout.dtype.param<typename DTypeTrait<src_ctype>::dtype>();
            auto dst_param = dst_tensor.layout.dtype
                                     .param<typename DTypeTrait<dst_ctype>::dtype>();
            for (size_t i = 0; i < size; i++) {
                src_ctype a = *iA;
                src_ctype b = *iB;
                src_ctype c = *iC;
                *pD = dst_param.quantize(KernImpl::apply(
                        param0.dequantize(a), param1.dequantize(b),
                        param2.dequantize(c)));
                ++iA;
                ++iB;
                ++iC;
                ++pD;
            }
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(work());
    }

protected:
    template <typename ctype>
    void dispatch_fma3_iXxf32xf32xi8(
            const ElemwiseOpParamN<3>& param, const TensorND& dst);

    template <typename ctype, typename dst_ctype>
    void dispatch_round_shr_saturate_iXxi8xiX(
            const ElemwiseOpParamN<2>& param, const TensorND& dst);

    template <typename ctype>
    void dispatch_fuse_add_rmulh_round_shr_saturate(
            const ElemwiseOpParamN<6>& param, const TensorND& dst);

    void on_fuse_mul_add3_int16x32x32x32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) override;
    void on_fuse_mul_add3_iXxf32xf32xi8(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) override;
    void on_round_shr_saturate_iXxi8xi8(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) override;
    void on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) override;
    void on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) override;
    void on_round_shr_saturate_iXxi8xi16(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) override;
    void on_fuse_mul_add3_int16xf32xf32xf32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) override;
    void on_mul_int16xf32xf32(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) override;
    void on_fuse_mul_add3_uint8xf32xf32xf32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<1>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<2>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<3>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void dest_type_bool_mode(
            const ElemwiseOpParamN<1>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void dest_type_bool_mode(
            const ElemwiseOpParamN<2>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

public:
    using ElemwiseMultiTypeImplHelper::ElemwiseMultiTypeImplHelper;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
