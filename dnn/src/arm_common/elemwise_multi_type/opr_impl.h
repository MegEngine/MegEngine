/**
 * \file dnn/src/arm_common/elemwise_multi_type/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/fallback/elemwise_multi_type/opr_impl.h"

namespace megdnn {
namespace arm_common {

class ElemwiseMultiTypeImpl : public fallback::ElemwiseMultiTypeImpl {
    template <typename stype>
    void neon_round_shr_saturate_bcast_scalar(const stype* a_ptr, int8_t k,
                                              size_t size, dt_int8* dst_ptr);

    template <typename ctype>
    void dispatch_round_shr_saturate_iXxi8xi8_bcast_scalar(
            const ElemwiseOpParamN<2>& param, megdnn::dt_int8* dst);

    bool dispatch_fuse_add_rmulh_rshr(const ElemwiseOpParamN<6>& param,
                                      megdnn::dt_int8* dst);

protected:
    void on_round_shr_saturate_iXxi8xi8(const ElemwiseOpParamN<2>& param,
                                        dt_int8* dst) override;
    void on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
            const ElemwiseOpParamN<6>& param, dt_int8* dst) override;
    void on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
            const ElemwiseOpParamN<6>& param, dt_int8* dst) override;

    void on_quantized_mode(const ElemwiseOpParamN<1>& param,
                           const TensorND& dst, Elemwise::Mode mode) override;

    void on_quantized_mode(const ElemwiseOpParamN<2>& param,
                           const TensorND& dst, Elemwise::Mode mode) override;

    void on_quantized_mode(const ElemwiseOpParamN<3>& param,
                           const TensorND& dst, Elemwise::Mode mode) override;

public:
    using fallback::ElemwiseMultiTypeImpl::ElemwiseMultiTypeImpl;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
