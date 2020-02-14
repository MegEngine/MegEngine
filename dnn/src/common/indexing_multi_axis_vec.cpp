/**
 * \file dnn/src/common/indexing_multi_axis_vec.cpp
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

using namespace megdnn;

namespace {
    size_t get_index_size_for_workspace(
            const TensorShape &shp, const size_t *axes, size_t nr_axes) {
        size_t idx_axis = axes[0];
        megdnn_assert(shp.ndim && nr_axes);
        for (size_t i = 1; i < nr_axes; ++ i) {
            megdnn_assert(axes[i] > axes[i - 1]);
            if (axes[i] != axes[i - 1] + 1) {
                idx_axis = 0;
                break;
            }
        }
        megdnn_assert(shp.ndim > idx_axis,
                "index on the %zuth axis; but shape is %s",
                idx_axis, shp.to_string().c_str());
        return shp.shape[idx_axis];
    }
} // anonymous namespace

IndexingMultiAxisVecBase::IndexDescLayoutOnly
IndexingMultiAxisVecBase::extract_index_layout(const IndexDesc &index) {
    IndexDescLayoutOnly ret(index.size());
    for (size_t i = 0; i < index.size(); ++ i) {
        ret[i].layout = index[i].vec.layout;
        ret[i].axis = index[i].axis;
    }
    return ret;
}

size_t IndexingMultiAxisVecBase::deduce_layout_fwd(
        const TensorLayout &data,
        const IndexDescLayoutOnly &index,
        TensorLayout &dst) {
    megdnn_assert(!index.empty());
    megdnn_assert(data.ndim >= index.size());
    dst.ndim = data.ndim - index.size() + 1;
    dst.shape[0] = 1;
    dst.dtype = data.dtype;

    auto brdcast = [&](const TensorLayout &ly) {
        if (ly.ndim != 1)
            return false;
        if (dst.shape[0] == ly.shape[0])
            return true;
        if (dst.shape[0] == 1) {
            dst.shape[0] = ly.shape[0];
            return true;
        }
        return ly.shape[0] == 1;
    };

    size_t dst_axis = 1;
    ptrdiff_t prev_axis = -1;
    for (size_t axis = 0; axis < index.size(); ++ axis) {
        auto &&idx = index[axis];
        megdnn_assert(idx.layout.dtype == dtype::Int32(),
                "invalid index dtype: %s", idx.layout.dtype.name());
        megdnn_assert(idx.axis < data.ndim &&
                static_cast<ptrdiff_t>(idx.axis) > prev_axis,
                "index %zu requests invalid axis %zu", axis, idx.axis);
        auto brd_succ = brdcast(idx.layout);
        megdnn_assert(brd_succ, "invalid layout at index %zu: %s",
                axis, idx.layout.to_string().c_str());

        for (size_t i = prev_axis + 1; i < idx.axis; ++ i) {
            dst.shape[dst_axis ++] = data.shape[i];
        }
        prev_axis = idx.axis;
    }
    for (size_t i = prev_axis + 1; i < data.ndim; ++ i) {
        dst.shape[dst_axis ++] = data.shape[i];
    }
    megdnn_assert(dst_axis == dst.ndim);

    size_t idx_axis = 0;
    {
        // fix idx_axis if index contains consecutive axes
        bool contig_idx = true;
        for (size_t i = 1; i < index.size(); ++ i) {
            if (index[i].axis != index[i - 1].axis + 1) {
                contig_idx = false;
                break;
            }
        }
        if (contig_idx) {
            auto shp0 = dst.shape[0];
            idx_axis = index[0].axis;
            for (size_t i = 0; i < idx_axis; ++ i) {
                dst.shape[i] = dst.shape[i + 1];
            }
            dst.shape[idx_axis] = shp0;
        }
    }

    dst.init_contiguous_stride();
    return idx_axis;
}

size_t IndexingMultiAxisVecBase::get_nonindex_axes(
        size_t src_ndim, const IndexDesc &index, size_t *out) {
    auto iter = index.begin();
    size_t nr = 0;
    for (size_t i = 0; i < src_ndim; ++ i) {
        if (iter != index.end() && i == iter->axis) {
            ++ iter;
        } else {
            out[nr ++] = i;
        }
    }
    megdnn_assert(nr + index.size() == src_ndim && iter == index.end());
    return nr;
}

IndexingMultiAxisVecBase::ExecInfo
IndexingMultiAxisVecBase::check_exec_noworkspace(
        const TensorLayout &data, const TensorLayout &value,
        const IndexDesc &index, IndexDescLayoutOnly &index_layout) {

    ExecInfo ret;
    index_layout = extract_index_layout(index);
    TensorLayout value_expect;
    ret.idx_axis = deduce_layout_fwd(data, index_layout, value_expect);
    megdnn_assert_eq_shape(value_expect, value);

    auto value_contig = value.collapse_contiguous();
    megdnn_assert(value_contig.ndim == 1,
            "value layout must be 1-dim contiguous; got %s",
            value.to_string().c_str());

    ret.value_stride = value_contig.stride[0];
    return ret;
}

std::pair<TensorLayout, size_t>
IndexingMultiAxisVecBase::get_value_iter_optimized_layout(
        const TensorLayout &data, const TensorLayout &value,
        const IndexDesc &index, size_t idx_axis) {
    size_t data_axes[TensorLayout::MAX_NDIM],
           nr_axes = get_nonindex_axes(data.ndim, index, data_axes);

    megdnn_assert(nr_axes == value.ndim - 1 && idx_axis < value.ndim &&
            nr_axes + index.size() == data.ndim);

    TensorLayout ret;
    if (idx_axis) {
        ret.ndim = idx_axis;
        for (size_t i = 0; i < idx_axis; ++ i) {
            ret.shape[i] = data.shape[data_axes[i]];
            ret.stride[i] = data.stride[data_axes[i]];
        }
        ret = ret.collapse_contiguous();
    }
    ret.shape[ret.ndim] = value.shape[idx_axis];
    ret.stride[ret.ndim] = 0;
    size_t ret_idx_axis = ret.ndim;
    ++ ret.ndim;

    if (idx_axis < nr_axes) {
        TensorLayout tail;
        tail.ndim = nr_axes - idx_axis;
        for (size_t i = idx_axis; i < nr_axes; ++ i) {
            tail.shape[i - idx_axis] = data.shape[data_axes[i]];
            tail.stride[i - idx_axis] = data.stride[data_axes[i]];
        }
        tail = tail.collapse_contiguous();
        for (size_t i = 0; i < tail.ndim; ++ i) {
            ret.shape[ret.ndim] = tail.shape[i];
            ret.stride[ret.ndim] = tail.stride[i];
            ++ ret.ndim;
        }
    }

    return {ret, ret_idx_axis};
}

size_t IndexingMultiAxisVec::get_workspace_in_bytes(
        const TensorShape &dst, const size_t *axes, size_t nr_axes) {
    return get_workspace_in_bytes(
            get_index_size_for_workspace(dst, axes, nr_axes));
}

IndexingMultiAxisVec::ExecInfo IndexingMultiAxisVec::check_exec(
        const TensorLayout &src, const IndexDesc &index,
        const TensorLayout &dst, size_t workspace_in_bytes) {
    IndexDescLayoutOnly index_layout;
    auto ret = check_exec_noworkspace(src, dst, index, index_layout);
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(
                dst.shape[ret.idx_axis]));
    megdnn_assert(ret.value_stride, "dst must be non-overlapping");
    return ret;
}

size_t IndexingModifyMultiAxisVecBase::get_workspace_in_bytes(
        const TensorShape &value, const size_t *axes, size_t nr_axes) {
    return get_workspace_in_bytes(
            get_index_size_for_workspace(value, axes, nr_axes));
}

IndexingModifyMultiAxisVecBase::ExecInfo
IndexingModifyMultiAxisVecBase::check_exec(
        const TensorLayout &data, const TensorLayout &value,
        const IndexDesc &index, size_t workspace_in_bytes) {
    megdnn_assert(data.is_non_overlapping_strong(),
            "data layout should not overlap: %s", data.to_string().c_str());
    IndexDescLayoutOnly index_layout;
    auto ret = check_exec_noworkspace(data, value, index, index_layout);
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(
                value.shape[ret.idx_axis]));
    return ret;
}

// vim: syntax=cpp.doxygen
