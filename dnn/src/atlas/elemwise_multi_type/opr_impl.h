#pragma once

#include "megdnn/tensor_iter.h"
#include "src/atlas/handle.h"
#include "src/common/elemwise_multi_type/opr_impl_helper.h"

namespace megdnn {
namespace atlas {

class ElemwiseMultiTypeImpl final : public ElemwiseMultiTypeImplHelper {
    virtual void on_fuse_mul_add3_int16x32x32x32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 int16x32x32x32.");
    }

    virtual void on_fuse_mul_add3_iXxf32xf32xi8(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 iXxf32xf32xi8.");
    }

    virtual void on_round_shr_saturate_iXxi8xi8(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 iXxi8xi8.");
    }

    virtual void on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 int16x16x16x8.");
    }

    virtual void on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 int32x32x32x8.");
    }
    virtual void on_quantized_mode(
            const ElemwiseOpParamN<2>& param, const TensorND& dst,
            Elemwise::Mode mode) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        MEGDNN_MARK_USED_VAR(mode);
        megdnn_throw("unsupported quantized mode");
    }

    virtual void on_round_shr_saturate_iXxi8xi16(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        megdnn_throw("unsupported ElemwiseMultiType fma3 iXxi8xi16.");
    }
    void dest_type_bool_mode(
            const ElemwiseOpParamN<2>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

public:
    using ElemwiseMultiTypeImplHelper::ElemwiseMultiTypeImplHelper;
};
}  // namespace atlas
}  // namespace megdnn