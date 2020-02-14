/**
 * \file dnn/src/cuda/elemwise_multi_type/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/common/elemwise_multi_type/opr_impl_helper.h"

namespace megdnn {
namespace cuda {

class ElemwiseMultiTypeImpl final : public ElemwiseMultiTypeImplHelper {
    void on_fuse_mul_add3_int16x32x32x32(const ElemwiseOpParamN<3>& param,
                                         dt_int32* dst) override;

    void on_fuse_mul_add3_iXxf32xf32xi8(const ElemwiseOpParamN<3>& param,
                                        dt_int8* dst) override;

    void on_round_shr_saturate_iXxi8xi8(const ElemwiseOpParamN<2>& param,
                                        dt_int8* dst) override;

    void on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
            const ElemwiseOpParamN<6>& param, dt_int8* dst) override;

    void on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
            const ElemwiseOpParamN<6>& param, dt_int8* dst) override;

    void on_round_shr_saturate_iXxi8xi16(const ElemwiseOpParamN<2>& param,
                                         dt_int16* dst) override;
    
    void on_quantized_mode(const ElemwiseOpParamN<1>& param,
            const TensorND& dst, Elemwise::Mode mode) override;

    void on_quantized_mode(const ElemwiseOpParamN<2>& param,
            const TensorND& dst, Elemwise::Mode mode) override;
    
    void on_quantized_mode(const ElemwiseOpParamN<3>& param,
            const TensorND& dst, Elemwise::Mode mode) override;

public:
    using ElemwiseMultiTypeImplHelper::ElemwiseMultiTypeImplHelper;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
