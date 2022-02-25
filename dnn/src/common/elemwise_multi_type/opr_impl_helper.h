#pragma once

#include "megdnn/oprs/nn_int.h"
#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {

class ElemwiseMultiTypeImplHelper : public ElemwiseMultiType,
                                    protected ElemwiseLayoutHelper {
    static void call_check_layout_and_broadcast(
            void* opr, const TensorLayoutPtrArray& src, const TensorLayout& dst) {
        return static_cast<ElemwiseMultiTypeImplHelper*>(opr)
                ->check_layout_and_broadcast(src, dst);
    }

    template <int arity>
    ElemwiseOpParamN<arity> make_elemwise_op_param(
            const TensorNDArray& src, const TensorND& dst) {
        return ElemwiseLayoutHelper::make_elemwise_op_param<arity>(
                this, call_check_layout_and_broadcast, src, dst);
    }

protected:
    virtual void on_fuse_mul_add3_int16x32x32x32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) = 0;

    virtual void on_fuse_mul_add3_iXxf32xf32xi8(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) = 0;

    virtual void on_round_shr_saturate_iXxi8xi8(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) = 0;

    virtual void on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) = 0;

    virtual void on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) = 0;

    virtual void on_round_shr_saturate_iXxi8xi16(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) = 0;

    virtual void on_fuse_mul_add3_int16xf32xf32xf32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 int16xf32xf32xf32.");
    }

    virtual void on_mul_int16xf32xf32(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 int16xf32xf32.");
    }

    virtual void on_fuse_mul_add3_uint8xf32xf32xf32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 uint8xf32xf32xf32.");
    }

    virtual void on_quantized_mode(
            const ElemwiseOpParamN<1>& param, const TensorND& dst,
            Elemwise::Mode mode) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        MEGDNN_MARK_USED_VAR(mode);
        megdnn_throw("Unrealized except arm_common");
    }

    virtual void on_quantized_mode(
            const ElemwiseOpParamN<2>& param, const TensorND& dst,
            Elemwise::Mode mode) = 0;

    virtual void on_quantized_mode(
            const ElemwiseOpParamN<3>& param, const TensorND& dst,
            Elemwise::Mode mode) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        MEGDNN_MARK_USED_VAR(mode);
        megdnn_throw("Unrealized except arm_common");
    }

public:
    using ElemwiseMultiType::ElemwiseMultiType;

    void exec(
            _megdnn_in const TensorNDArray& src, _megdnn_tensor_out dst) override final;
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
