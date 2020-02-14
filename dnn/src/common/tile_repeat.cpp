/**
 * \file dnn/src/common/tile_repeat.cpp
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

#include <numeric>

namespace megdnn {

void TileRepeatBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &dst)
{
    auto errmsg = megdnn_layout_msg(src) + ", " + megdnn_layout_msg(dst)
        + ", " + "times=" + param().times.to_string();
    auto errmsg_c = errmsg.c_str();
    MEGDNN_MARK_USED_VAR(errmsg_c);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(dst);
    auto expected_ndim = param().times.ndim;
    megdnn_assert(expected_ndim == src.ndim, "%s", errmsg_c);
    megdnn_assert(expected_ndim == dst.ndim, "%s", errmsg_c);
    rep(i, expected_ndim) {
        megdnn_assert(dst.shape[i] == param().times[i] * src.shape[i],
                "%s", errmsg_c);
    }

    megdnn_assert(src.dtype == dst.dtype);
}

void TileRepeatBase::deduce_layout_fwd(const TensorLayout &src,
        TensorLayout &dst)
{
    dst.ndim = src.ndim;
    rep(i, src.ndim) {
        dst.shape[i] = src.shape[i] * param().times[i];
    }
    dst.dtype = src.dtype;
    dst.init_contiguous_stride();
    check_layout_fwd(src, dst);
}

size_t TileRepeatBase::get_workspace_in_bytes_fwd(const TensorShape & /* src */,
        const TensorShape &dst,
        const TensorShape &times,
        DType dtype)
{
    size_t nr_workspace = 0;
    auto nr_reduces = count_not_ones_in_shape(times);
    if (nr_reduces == 0) {
        // case 1: no tile/repeat is needed, let alone workspace.
        nr_workspace = 0;
    } else if (nr_reduces == 1) {
        // case 2: only one tile/repeat is needed, so we don't need workspace.
        nr_workspace = 0;
    } else if (nr_reduces == 2) {
        // case 3: two tile/repeats are needed, so we need a single workspace.
        nr_workspace = 1;
    } else {
        // case 4: multiple tile/repeats are needed, so we need two workspace in
        // an alternate fashion.
        nr_workspace = 2;
    }
    if (nr_workspace == 0) {
        return 0;
    } else {
        WorkspaceBundle workspaces{
                nullptr, {nr_workspace, dst.total_nr_elems() * dtype.size()}};
        return workspaces.total_size_in_bytes();
    }
}

void TileBase::simplify_shape(const TensorShape &src,
        const TensorShape &dst,
        const TensorShape &times,
        TensorShape &src2,
        TensorShape &dst2,
        TensorShape &times2)
{
    size_t n = 0;
    for (size_t i = 0; i < src.ndim; ++i) {
        if (times.shape[i] == 1 && n > 0) {
            src2.shape[n-1] *= src.shape[i];
            dst2.shape[n-1] *= dst.shape[i];
        } else {
            src2.shape[n] = src.shape[i];
            dst2.shape[n] = dst.shape[i];
            times2.shape[n] = times.shape[i];
            ++n;
        }
    }
    src2.ndim = dst2.ndim = times2.ndim = n;
}

size_t TileBase::get_workspace_in_bytes_fwd(const TensorLayout &src_,
        const TensorLayout &dst_)
{
    TensorShape src, dst, times;
    simplify_shape(src_, dst_, param().times, src, dst, times);
    return TileRepeatBase::get_workspace_in_bytes_fwd(src, dst, times,
            src_.dtype);
}

void TileForward::deduce_layout(const TensorLayout &src,
        TensorLayout &dst)
{
    deduce_layout_fwd(src, dst);
}

void TileForward::check_exec(const TensorLayout &src, const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void TileBackward::check_exec(const TensorLayout &diff, const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void RepeatBase::simplify_shape(const TensorShape &src,
        const TensorShape & /* dst */,
        const TensorShape &times,
        TensorShape &src2,
        TensorShape &dst2,
        TensorShape &times2)
{
    auto n = 0u;
    size_t i = 0;
    while (i < times.ndim) {
        size_t j = i;
        while (j < times.ndim && times.shape[j] == 1) ++j;
        // Here: j is times.ndim, or times.shape[j] != 1
        if (j < times.ndim) ++j;
        src2.shape[n] = std::accumulate(src.shape + i, src.shape + j,
                1_z, SafeMultiplies<size_t>());
        times2.shape[n] = times.shape[j-1];
        dst2.shape[n] = src2.shape[n] * times2.shape[n];
        ++n;
        i = j;
    }
    src2.ndim = dst2.ndim = times2.ndim = n;
}

size_t RepeatBase::get_workspace_in_bytes_fwd(const TensorLayout &src_,
        const TensorLayout &dst_)
{
    TensorShape src, dst, times;
    simplify_shape(src_, dst_, param().times, src, dst, times);
    return TileRepeatBase::get_workspace_in_bytes_fwd(src, dst, times,
            src_.dtype);
}

void RepeatForward::deduce_layout(const TensorLayout &src,
        TensorLayout &dst)
{
    deduce_layout_fwd(src, dst);
}

void RepeatForward::check_exec(const TensorLayout &src, const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void RepeatBackward::check_exec(const TensorLayout &diff,
        const TensorLayout &grad, size_t workspace_in_bytes)
{
    check_layout_fwd(grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
