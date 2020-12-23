/**
 * \file dnn/src/common/tqt.cpp
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

void TQTBase::deduce_layout_fwd(const TensorLayout& input,
                                TensorLayout& output) {
    output = TensorLayout(input, input.dtype);
}

void TQTBase::check_layout_fwd(const TensorLayout& input,
                               const TensorLayout& scale,
                               const TensorLayout& output) {
    megdnn_assert(input.dtype == dtype::Float32());
    megdnn_assert(scale.dtype == dtype::Float32());
    TensorLayout expected;
    deduce_layout_fwd(input, expected);
    megdnn_assert_eq_layout(expected, output);
}

void TQTForward::deduce_layout(const TensorLayout& input,
                               const TensorLayout& /* scale */,
                               TensorLayout& output) {
    deduce_layout_fwd(input, output);
}

void TQTForward::check_exec(const TensorLayout& input,
                            const TensorLayout& scale,
                            const TensorLayout& output,
                            size_t workspace_in_bytes) {
    check_layout_fwd(input, scale, output);
    auto required_workspace_space =
            get_workspace_in_bytes(input, scale, output);
    megdnn_assert(workspace_in_bytes >= required_workspace_space);
}

void TQTBackward::check_exec(const TensorLayout& diff,
                             const TensorLayout& input,
                             const TensorLayout& scale,
                             const TensorLayout& grad_x,
                             const TensorLayout& grad_s,
                             size_t workspace_in_bytes) {
    megdnn_assert_eq_shape(diff, input);
    megdnn_assert_eq_shape(grad_x, input);
    auto required_worspace_space =
            get_workspace_in_bytes(diff, input, scale, grad_x, grad_s);
    megdnn_assert(workspace_in_bytes >= required_worspace_space);
}

}  // namespace megdnn
