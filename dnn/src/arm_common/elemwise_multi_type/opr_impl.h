#pragma once

#include "src/fallback/elemwise_multi_type/opr_impl.h"

namespace megdnn {
namespace arm_common {

class ElemwiseMultiTypeImpl : public fallback::ElemwiseMultiTypeImpl {
    template <typename stype>
    void neon_round_shr_saturate_bcast_scalar(
            const stype* a_ptr, int8_t k, size_t size, dt_int8* dst_ptr);

    template <typename ctype>
    void dispatch_round_shr_saturate_iXxi8xi8_bcast_scalar(
            const ElemwiseOpParamN<2>& param, const TensorND& dst);

    bool dispatch_fuse_add_rmulh_rshr(
            const ElemwiseOpParamN<6>& param, const TensorND& dst);

protected:
    void on_round_shr_saturate_iXxi8xi8(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) override;
    void on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) override;
    void on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
            const ElemwiseOpParamN<6>& param, const TensorND& dst) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<1>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<2>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void on_quantized_mode(
            const ElemwiseOpParamN<3>& param, const TensorND& dst,
            Elemwise::Mode mode) override;

    void on_fuse_mul_add3_int16xf32xf32xf32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) override;

    void on_mul_int16xf32xf32(
            const ElemwiseOpParamN<2>& param, const TensorND& dst) override;

    void on_fuse_mul_add3_uint8xf32xf32xf32(
            const ElemwiseOpParamN<3>& param, const TensorND& dst) override;

public:
    using fallback::ElemwiseMultiTypeImpl::ElemwiseMultiTypeImpl;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
