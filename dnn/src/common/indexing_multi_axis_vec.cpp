/**
 * \file dnn/src/common/indexing_multi_axis_vec.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"
#include "src/common/utils.h"

using namespace megdnn;

namespace {

// we need a workspace to store offset base table, which has same size with index
size_t get_index_size_for_workspace(
        const TensorShape& shp, const size_t* axes, size_t nr_axes, size_t idx_ndim) {
    size_t idx_axis = axes[0];
    megdnn_assert(shp.ndim && nr_axes);
    for (size_t i = 1; i < nr_axes; ++i) {
        megdnn_assert(axes[i] > axes[i - 1]);
        if (axes[i] != axes[i - 1] + 1) {
            idx_axis = 0;
            break;
        }
    }
    megdnn_assert(
            shp.ndim > idx_axis, "index on the %zuth axis; but shape is %s", idx_axis,
            shp.to_string().c_str());
    size_t idx_size = 1;
    for (size_t i = 0; i < idx_ndim; ++i) {
        idx_size *= shp.shape[idx_axis + i];
    }
    return idx_size;
}
}  // anonymous namespace

IndexingMultiAxisVecBase::IndexDescLayoutOnly IndexingMultiAxisVecBase::
        extract_index_layout(const IndexDesc& index) {
    IndexDescLayoutOnly ret(index.size());
    for (size_t i = 0; i < index.size(); ++i) {
        ret[i].layout = index[i].vec.layout;
        ret[i].axis = index[i].axis;
    }
    return ret;
}

size_t IndexingMultiAxisVecBase::deduce_layout_fwd(
        const TensorLayout& data, const IndexDescLayoutOnly& index, TensorLayout& dst) {
    megdnn_assert(!index.empty());
    megdnn_assert(data.ndim >= index.size());
    dst.ndim = data.ndim - index.size();
    dst.dtype = data.dtype;

    TensorShapeArray index_shapes;

    auto brdcast = [&](const TensorLayout& ly) {
        megdnn_assert(ly.dtype == dtype::Int32{});
        index_shapes.push_back(ly);
    };

    size_t dst_axis = 0;
    ptrdiff_t prev_axis = -1;
    for (size_t axis = 0; axis < index.size(); ++axis) {
        auto&& idx = index[axis];
        megdnn_assert(
                idx.layout.dtype == dtype::Int32(), "invalid index dtype: %s",
                idx.layout.dtype.name());
        megdnn_assert(
                idx.axis<data.ndim&& static_cast<ptrdiff_t>(idx.axis)> prev_axis,
                "index %zu requests invalid axis %zu", axis, idx.axis);
        brdcast(idx.layout);

        for (size_t i = prev_axis + 1; i < idx.axis; ++i) {
            dst.shape[dst_axis++] = data.shape[i];
        }
        prev_axis = idx.axis;
    }
    for (size_t i = prev_axis + 1; i < data.ndim; ++i) {
        dst.shape[dst_axis++] = data.shape[i];
    }
    megdnn_assert(dst_axis == dst.ndim);

    size_t idx_axis = 0;
    {
        // fix idx_axis if index contains consecutive axes
        bool contig_idx = true;
        for (size_t i = 1; i < index.size(); ++i) {
            if (index[i].axis != index[i - 1].axis + 1) {
                contig_idx = false;
                break;
            }
        }
        if (contig_idx) {
            idx_axis = index[0].axis;
        }
    }

    TensorShape index_shape;
    Elemwise::deduce_shape(index_shapes, index_shape);

    for (size_t i = 0; i < index_shape.ndim; ++i) {
        dst.add_axis_inplace(idx_axis + i, 1, 0);
        dst.shape[idx_axis + i] = index_shape.shape[i];
    }

    dst.init_contiguous_stride();
    return idx_axis;
}

size_t IndexingMultiAxisVecBase::get_nonindex_axes(
        size_t src_ndim, const IndexDesc& index, size_t* out) {
    auto iter = index.begin();
    size_t nr = 0;
    for (size_t i = 0; i < src_ndim; ++i) {
        if (iter != index.end() && i == iter->axis) {
            ++iter;
        } else {
            out[nr++] = i;
        }
    }
    megdnn_assert(nr + index.size() == src_ndim && iter == index.end());
    return nr;
}

IndexingMultiAxisVecBase::ExecInfo IndexingMultiAxisVecBase::check_exec_noworkspace(
        const TensorLayout& data, const TensorLayout& value, const IndexDesc& index,
        IndexDescLayoutOnly& index_layout) {
    ExecInfo ret;
    index_layout = extract_index_layout(index);
    TensorLayout value_expect;
    ret.idx_axis = deduce_layout_fwd(data, index_layout, value_expect);
    megdnn_assert_eq_shape(value_expect, value);

    auto value_contig = value.collapse_contiguous();
    megdnn_assert(
            value_contig.ndim == 1, "value layout must be 1-dim contiguous; got %s",
            value.to_string().c_str());

    ret.value_stride = value_contig.stride[0];
    return ret;
}

std::tuple<TensorLayout, size_t, TensorShape> IndexingMultiAxisVecBase::
        get_value_iter_optimized_layout(
                const TensorLayout& data, const TensorLayout& value,
                const IndexDesc& index, size_t idx_axis) {
    size_t data_axes[TensorLayout::MAX_NDIM],
            nr_axes = get_nonindex_axes(data.ndim, index, data_axes);

    // broadcast index shapes
    TensorLayout index_shape;
    {
        TensorShapeArray index_shapes;
        for (auto& idx : index) {
            megdnn_assert(idx.vec.layout.dtype == dtype::Int32{});
            index_shapes.push_back(idx.vec.layout);
        }
        Elemwise::deduce_shape(index_shapes, index_shape);
    }

    megdnn_assert(
            nr_axes == value.ndim - index_shape.ndim && idx_axis < value.ndim &&
            nr_axes + index.size() == data.ndim);

    TensorLayout ret;
    if (idx_axis) {
        ret.ndim = idx_axis;
        for (size_t i = 0; i < idx_axis; ++i) {
            ret.shape[i] = data.shape[data_axes[i]];
            ret.stride[i] = data.stride[data_axes[i]];
        }
        ret = ret.collapse_contiguous();
    }

    size_t ret_idx_axis = ret.ndim;
    for (size_t i = 0; i < index_shape.ndim; ++i) {
        ret.shape[ret.ndim] = value.shape[idx_axis + i];
        ret.stride[ret.ndim] = 0;
        ++ret.ndim;
    }

    if (idx_axis < nr_axes) {
        TensorLayout tail;
        tail.ndim = nr_axes - idx_axis;
        for (size_t i = idx_axis; i < nr_axes; ++i) {
            tail.shape[i - idx_axis] = data.shape[data_axes[i]];
            tail.stride[i - idx_axis] = data.stride[data_axes[i]];
        }
        tail = tail.collapse_contiguous();
        for (size_t i = 0; i < tail.ndim; ++i) {
            ret.shape[ret.ndim] = tail.shape[i];
            ret.stride[ret.ndim] = tail.stride[i];
            ++ret.ndim;
        }
    }

    return std::make_tuple(ret, ret_idx_axis, index_shape);
}

size_t IndexingMultiAxisVec::get_workspace_in_bytes(
        const TensorShape& dst, const size_t* axes, size_t nr_axes, size_t idx_ndim) {
    return get_workspace_in_bytes(
            get_index_size_for_workspace(dst, axes, nr_axes, idx_ndim));
}

IndexingMultiAxisVec::ExecInfo IndexingMultiAxisVec::check_exec(
        const TensorLayout& src, const IndexDesc& index, const TensorLayout& dst,
        size_t workspace_in_bytes) {
    IndexDescLayoutOnly index_layout;
    auto ret = check_exec_noworkspace(src, dst, index, index_layout);
    megdnn_assert(
            workspace_in_bytes >= get_workspace_in_bytes(dst.shape[ret.idx_axis]));
    megdnn_assert(ret.value_stride, "dst must be non-overlapping");
    return ret;
}

size_t IndexingModifyMultiAxisVecBase::get_workspace_in_bytes(
        const TensorShape& value, const size_t* axes, size_t nr_axes, size_t idx_ndim) {
    return get_workspace_in_bytes(
            get_index_size_for_workspace(value, axes, nr_axes, idx_ndim));
}

IndexingModifyMultiAxisVecBase::ExecInfo IndexingModifyMultiAxisVecBase::check_exec(
        const TensorLayout& data, const TensorLayout& value, const IndexDesc& index,
        size_t workspace_in_bytes) {
    megdnn_assert(
            data.is_non_overlapping_strong(), "data layout should not overlap: %s",
            data.to_string().c_str());
    IndexDescLayoutOnly index_layout;
    auto ret = check_exec_noworkspace(data, value, index, index_layout);
    megdnn_assert(
            workspace_in_bytes >= get_workspace_in_bytes(value.shape[ret.idx_axis]));
    return ret;
}

// vim: syntax=cpp.doxygen
