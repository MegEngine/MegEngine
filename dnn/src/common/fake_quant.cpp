/**
 * \file dnn/src/common/fakequant.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void FakeQuantBase::deduce_layout_fwd(const TensorLayout& input,
                                      TensorLayout& output) {
    output = TensorLayout(input, input.dtype);
}

void FakeQuantBase::check_layout_fwd(const TensorLayout& input,
                                     const TensorLayout& scale,
                                     const TensorLayout& zero_point,
                                     const TensorLayout& output) {
    megdnn_assert(input.dtype == dtype::Float32());
    megdnn_assert(scale.dtype == dtype::Float32());
    megdnn_assert(zero_point.dtype == dtype::Float32());
    TensorLayout expected;
    deduce_layout_fwd(input, expected);
    megdnn_assert_eq_layout(expected, output);
}

void FakeQuantForward::deduce_layout(const TensorLayout& input,
                                     const TensorLayout& /*scale*/,
                                     const TensorLayout& /*zero_point*/,
                                     TensorLayout& output) {
    deduce_layout_fwd(input, output);
}

void FakeQuantForward::check_exec(const TensorLayout& input,
                                  const TensorLayout& scale,
                                  const TensorLayout& zero_point,
                                  const TensorLayout& output,
                                  size_t workspace_in_bytes) {
    check_layout_fwd(input, scale, zero_point, output);
    auto required_workspace_space =
            get_workspace_in_bytes(input, scale, zero_point, output);
    megdnn_assert(workspace_in_bytes >= required_workspace_space);
}

void FakeQuantBackward::check_exec(const TensorLayout& diff,
                                   const TensorLayout& input,
                                   const TensorLayout& scale,
                                   const TensorLayout& zero_point,
                                   const TensorLayout& grad,
                                   size_t workspace_in_bytes) {
    megdnn_assert_eq_shape(input, diff);
    megdnn_assert_eq_shape(input, grad);
    auto required_worspace_space =
            get_workspace_in_bytes(diff, input, scale, zero_point, grad);
    megdnn_assert(workspace_in_bytes >= required_worspace_space);
}

}  // namespace megdnn