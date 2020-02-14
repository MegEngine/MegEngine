/**
 * \file dnn/src/common/elemwise_multi_type/opr_impl_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/oprs/nn_int.h"
#include "src/common/elemwise/opr_impl_helper.h"

namespace megdnn {

class ElemwiseMultiTypeImplHelper : public ElemwiseMultiType,
                                    protected ElemwiseLayoutHelper {
    static void call_check_layout_and_broadcast(void* opr,
                                                const TensorLayoutPtrArray& src,
                                                const TensorLayout& dst) {
        return static_cast<ElemwiseMultiTypeImplHelper*>(opr)
                ->check_layout_and_broadcast(src, dst);
    }

    template <int arity>
    ElemwiseOpParamN<arity> make_elemwise_op_param(const TensorNDArray& src,
                                                   const TensorND& dst) {
        return ElemwiseLayoutHelper::make_elemwise_op_param<arity>(
                this, call_check_layout_and_broadcast, src, dst);
    }

protected:
    virtual void on_fuse_mul_add3_int16x32x32x32(
            const ElemwiseOpParamN<3>& param, dt_int32* dst) = 0;

    virtual void on_fuse_mul_add3_iXxf32xf32xi8(
            const ElemwiseOpParamN<3>& param, dt_int8* dst) = 0;

    virtual void on_round_shr_saturate_iXxi8xi8(
            const ElemwiseOpParamN<2>& param, dt_int8* dst) = 0;

    virtual void on_fuse_add_rmulh_round_shr_saturate_int16x16x16x8(
            const ElemwiseOpParamN<6>& param, dt_int8* dst) = 0;

    virtual void on_fuse_add_rmulh_round_shr_saturate_int32x32x32x8(
            const ElemwiseOpParamN<6>& param, dt_int8* dst) = 0;

    virtual void on_round_shr_saturate_iXxi8xi16(
            const ElemwiseOpParamN<2>& param, dt_int16* dst) = 0;

    virtual void on_quantized_mode(const ElemwiseOpParamN<1>& param,
                                   const TensorND& dst,
                                   Elemwise::Mode mode) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        MEGDNN_MARK_USED_VAR(mode);
        megdnn_throw("Unrealized except arm_common");
    }

    virtual void on_quantized_mode(const ElemwiseOpParamN<2>& param,
                                   const TensorND& dst,
                                   Elemwise::Mode mode) = 0;

    virtual void on_quantized_mode(const ElemwiseOpParamN<3>& param,
                                   const TensorND& dst,
                                   Elemwise::Mode mode) {
        MEGDNN_MARK_USED_VAR(param);
        MEGDNN_MARK_USED_VAR(dst);
        MEGDNN_MARK_USED_VAR(mode);
        megdnn_throw("Unrealized except arm_common");
    }

public:
    using ElemwiseMultiType::ElemwiseMultiType;

    void exec(_megdnn_in const TensorNDArray& src,
              _megdnn_tensor_out dst) override final;
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
