/**
 * \file dnn/src/rocm/adaptive_pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/rocm/adaptive_pooling/opr_impl.h"

namespace megdnn {
namespace rocm {

void AdaptivePoolingForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto opr = handle()->create_operator<PoolingForward>();
    opr->param() = deduce_pooling_param(src.layout, dst.layout);
    opr->exec(src, dst, workspace);
}

size_t AdaptivePoolingForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    auto opr = handle()->create_operator<PoolingForward>();
    opr->param() = deduce_pooling_param(src, dst);
    return opr->get_workspace_in_bytes(src, dst);
}

void AdaptivePoolingBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
        _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    auto opr = handle()->create_operator<PoolingBackward>();
    opr->param() = deduce_pooling_param(src.layout, dst.layout);
    opr->exec(src, dst, diff, grad, workspace);
}

size_t AdaptivePoolingBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad) {
    auto opr = handle()->create_operator<PoolingBackward>();
    opr->param() = deduce_pooling_param(src, dst);
    return opr->get_workspace_in_bytes(src, dst, diff, grad);
}
}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
