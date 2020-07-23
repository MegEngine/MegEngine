/**
 * \file dnn/src/common/batch_normalization.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void BNForward::deduce_layout(const TensorLayout& src, TensorLayout&,
                              TensorLayout&, TensorLayout&, TensorLayout&,
                              TensorLayout&, TensorLayout&, TensorLayout& dst) {
    dst = src;
}

void BNForward::check_exec(const TensorLayout& src, const TensorLayout& bn_scale,
                           const TensorLayout& bn_bias, const TensorLayout& mean,
                           const TensorLayout& variance,
                           const TensorLayout& batch_mean,
                           const TensorLayout& batch_inv_variance,
                           const TensorLayout& dst, size_t workspace_in_bytes) {
    megdnn_assert_contiguous(src);
    megdnn_assert_eq_layout(src, dst);
    megdnn_assert_eq_layout(bn_scale, bn_bias);

    megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(bn_scale.dtype.category() == DTypeCategory::FLOAT);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, bn_scale, bn_bias, mean, variance,
                                   batch_mean, batch_inv_variance, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void BNBackward::check_exec(const TensorLayout& x, const TensorLayout& dy,
                            const TensorLayout& saved_batch_mean,
                            const TensorLayout& saved_batch_variance,
                            const TensorLayout& bn_scale,
                            const TensorLayout& d_bn_scale,
                            const TensorLayout& d_bn_bias,
                            const TensorLayout& dx, size_t workspace_in_bytes) {
    megdnn_assert_contiguous(x);
    megdnn_assert_eq_layout(x, dy);
    megdnn_assert_eq_layout(x, dx);
    megdnn_assert_eq_layout(saved_batch_mean, d_bn_bias);
    megdnn_assert_eq_layout(saved_batch_mean, d_bn_scale);
    megdnn_assert_eq_layout(saved_batch_mean, saved_batch_variance);
    megdnn_assert_eq_layout(saved_batch_mean, bn_scale);
    megdnn_assert(x.dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(bn_scale.dtype.category() == DTypeCategory::FLOAT);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(x, dy, saved_batch_mean, saved_batch_variance,
                                   bn_scale, d_bn_scale, d_bn_bias, dx);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    megdnn_assert(param().fwd_mode == Param::FwdMode::TRAINING, "BNBackward only support TRAINING mode");
}

}  // namespace megdnn
// vim: syntax=cpp.doxygen
