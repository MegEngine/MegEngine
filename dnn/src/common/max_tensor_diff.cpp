/**
 * \file dnn/src/common/max_tensor_diff.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"
#include "megdnn/tensor_format.h"
#include "src/common/utils.h"

using namespace megdnn;

void megdnn::MaxTensorDiff::check_exec(const TensorLayout& layout1,
                                       const TensorLayout& layout2,
                                       size_t workspace_in_bytes) {
    megdnn_assert(layout1.eq_layout(layout2), "layout1: %s, layout2: %s",
                  layout1.to_string().c_str(), layout2.to_string().c_str());
    if (Image2DPack4TensorFormat::is_valid_image(layout1)) {
        megdnn_assert(layout1.is_contiguous() && layout1.ndim == 2 &&
                              layout1.shape[0] && layout1.eq_layout(layout2),
                      "layout1: %s, layout2: %s", layout1.to_string().c_str(),
                      layout2.to_string().c_str());
    } else {
        megdnn_assert(layout1.is_contiguous() &&
                              (layout1.ndim == 1 || layout1.ndim == 2) &&
                              layout1.shape[0] && layout1.eq_layout(layout2),
                      "layout1: %s, layout2: %s", layout1.to_string().c_str(),
                      layout2.to_string().c_str());
    }
    auto required_workspace_in_bytes = get_workspace_in_bytes(layout1, layout2);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

// vim: syntax=cpp.doxygen
