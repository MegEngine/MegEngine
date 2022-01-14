/**
 * \file dnn/src/common/dropout.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <time.h>
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void DropoutForward::deduce_layout(
        const TensorLayout& inp, TensorLayout& oup, TensorLayout& mask) {
    oup = inp;
    size_t mask_size = get_mask_size_in_bytes(inp);
    mask = TensorLayout(TensorShape({mask_size}), dtype::Byte());
}

void DropoutForward::check_exec(
        const TensorLayout& inp, const TensorLayout& oup, const TensorLayout& mask,
        size_t workspace_in_bytes) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(inp) + ", " + megdnn_layout_msg(oup) + ", " +
               megdnn_layout_msg(mask);
    };
    MEGDNN_MARK_USED_VAR(errmsg);

    megdnn_assert_contiguous(inp);
    megdnn_assert_contiguous(oup);
    megdnn_assert_contiguous(mask);
    megdnn_assert(inp.eq_layout(oup), "%s", errmsg().c_str());
    megdnn_assert(inp.dtype.category() == DTypeCategory::FLOAT);

    auto required_workspace_in_bytes = get_workspace_in_bytes(inp, oup, mask);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    auto required_mask_size_in_bytes = get_mask_size_in_bytes(inp);
    megdnn_assert(mask.total_nr_elems() >= required_mask_size_in_bytes);
    megdnn_assert(mask.dtype == dtype::Byte());
}

void DropoutBackward::deduce_layout(
        const TensorLayout& doup, const TensorLayout&, TensorLayout& dinp) {
    dinp = doup;
}

void DropoutBackward::check_exec(
        const TensorLayout& doup, const TensorLayout& mask, const TensorLayout& dinp,
        size_t workspace_in_bytes) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(doup) + ", " + megdnn_layout_msg(mask) + ", " +
               megdnn_layout_msg(dinp);
    };
    MEGDNN_MARK_USED_VAR(errmsg);

    megdnn_assert_contiguous(doup);
    megdnn_assert_contiguous(mask);
    megdnn_assert_contiguous(dinp);
    megdnn_assert(doup.eq_layout(dinp), "%s", errmsg().c_str());

    auto required_workspace_in_bytes = get_workspace_in_bytes(doup, mask, dinp);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    megdnn_assert(doup.dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(mask.dtype == dtype::Byte());
    megdnn_assert(mask.ndim == 1);
}

}  // namespace megdnn
