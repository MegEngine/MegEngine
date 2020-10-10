/**
 * \file dnn/src/naive/indexing_multi_axis_vec/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"

#include "megdnn/tensor_iter.h"
#include "src/naive/handle.h"

#include "src/common/utils.h"
#include "src/common/indexing_multi_axis_vec_kdef.h"

using namespace megdnn;
using namespace naive;

namespace {

template<typename data_type, class Opr, typename idx_type = dt_int32>
void do_exec(const TensorND &data, const TensorND &value,
        const IndexingMultiAxisVec::IndexDesc &index,
        const IndexingMultiAxisVec::ExecInfo &exec_info) {

    size_t nonidx_axes[TensorLayout::MAX_NDIM],
           nr_nonidx_axes = IndexingMultiAxisVec::get_nonindex_axes(
                   data.layout.ndim, index, nonidx_axes);

    auto data_layout = data.layout;
    auto data_ptr = data.ptr<data_type>();
    std::tuple<size_t, const idx_type*, ptrdiff_t>
        index_raw[TensorLayout::MAX_NDIM];
    size_t nr_index = index.size();
    for (size_t i = 0; i < nr_index; ++ i) {
        auto &&s = index[i];
        index_raw[i] = std::make_tuple(s.axis,
            s.vec.ptr<idx_type>(), s.vec.layout.stride[0]);

        if (s.vec.layout.shape[0] == 1)
            std::get<2>(index_raw[i]) = 0;
    }

    auto value_iter = tensor_iter<data_type>(value).begin();
    for (size_t _ = 0, _t = value.layout.total_nr_elems(); _ < _t; ++ _) {
        ptrdiff_t offset = 0;
        auto index_idx = value_iter.idx()[exec_info.idx_axis];
        for (size_t i = 0; i < nr_index; ++ i) {
            size_t axis = std::get<0>(index_raw[i]),
                   data_shape = data_layout.shape[axis];
            ptrdiff_t data_stride = data_layout.stride[axis];
            idx_type data_idx = std::get<1>(index_raw[i])[
                std::get<2>(index_raw[i]) * index_idx];
            if (data_idx < 0)
                data_idx += data_shape;
            megdnn_assert(data_idx >= 0 &&
                    static_cast<size_t>(data_idx) < data_shape,
                    "bad index value for index %zu at output %zu",
                    i, index_idx);
            offset += data_stride * data_idx;
        }
        for (size_t i = 0; i < nr_nonidx_axes; ++ i) {
            auto stride = data_layout.stride[nonidx_axes[i]];
            auto idx = value_iter.idx()[i + (i >= exec_info.idx_axis)];
            offset += stride * idx;
        }
        Opr::apply(data_ptr[offset], *value_iter);
        ++ value_iter;
    }
}

template<class Opr>
void dispatch_exec(HandleImpl *handle,
        const TensorND &data, const TensorND &value,
        const IndexingMultiAxisVec::IndexDesc &index,
        const IndexingMultiAxisVec::ExecInfo &exec_info) {
#define cb(_dt) \
    case DTypeTrait<_dt>::enumv: \
    { \
        MEGDNN_DISPATCH_CPU_KERN(handle, \
                do_exec<DTypeTrait<_dt>::ctype MEGDNN_COMMA Opr>( \
                    data, value, index, exec_info)); \
        return; \
    }
    switch (data.layout.dtype.enumv()) {
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
        default:
            megdnn_throw(megdnn_mangle("bad dtype"));
    }
#undef cb
}

} // anonymous namespace

void IndexingMultiAxisVecImpl::exec(
        _megdnn_tensor_in src, const IndexDesc &index,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {

    auto info = check_exec(src.layout, index, dst.layout, workspace.size);
    dispatch_exec<indexing_multi_axis_vec_kdef::OprFwd>(
            static_cast<HandleImpl*>(handle()), src, dst, index, info);
}

void IndexingSetMultiAxisVecImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_out value,
        const IndexDesc &index,
        _megdnn_workspace workspace) {

    auto info = check_exec(data.layout, value.layout, index, workspace.size);
    dispatch_exec<indexing_multi_axis_vec_kdef::OprSet>(
            static_cast<HandleImpl*>(handle()),
            data, value, index, info);
}

void IndexingIncrMultiAxisVecImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_out value,
        const IndexDesc &index,
        _megdnn_workspace workspace) {

    auto info = check_exec(data.layout, value.layout, index, workspace.size);
    dispatch_exec<indexing_multi_axis_vec_kdef::OprIncr>(
            static_cast<HandleImpl*>(handle()),
            data, value, index, info);
}

// vim: syntax=cpp.doxygen


