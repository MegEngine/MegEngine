/**
 * \file dnn/src/naive/indexing_one_hot/opr_impl.cpp
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

using namespace megdnn;
using namespace naive;

namespace {

    template<typename data_type, typename idx_type = dt_int32>
    void exec_get(const TensorND &src, const TensorND &index,
            const TensorND &dst, uint32_t axis) {

        TensorND src_nomid = src;
        src_nomid.layout.remove_axis_inplace(axis);
        auto src_mid_stride = src.layout.stride[axis];
        int src_mid_shape = src.layout.shape[axis];

        size_t nr_elems = src_nomid.layout.total_nr_elems();
        megdnn_assert(nr_elems == index.layout.total_nr_elems() &&
                nr_elems == dst.layout.total_nr_elems());
        auto src_iter = tensor_iter_valonly<data_type>(src_nomid).begin();
        auto idx_iter = tensor_iter_valonly<idx_type>(index).begin();
        auto dst_iter = tensor_iter_valonly<data_type>(dst).begin();

        data_type* sptr = src.ptr<data_type>();

        for (size_t i = 0; i < nr_elems; ++ i) {
            auto idx = *idx_iter;
            megdnn_assert(idx >= 0 && idx < src_mid_shape,
                    "bad value in IndexingOneHot index: input shape is %d, "
                    "index value is %d", src_mid_shape, idx);
            *dst_iter = sptr[src_iter.offset() + *idx_iter * src_mid_stride];
            ++ src_iter;
            ++ dst_iter;
            ++ idx_iter;
        }
    }

    template<typename data_type, typename idx_type = dt_int32>
    void exec_set(const TensorND &data, const TensorND &index,
            const TensorND &sub, uint32_t axis) {

        TensorND data_nomid = data;
        data_nomid.layout.remove_axis_inplace(axis);
        auto data_mid_stride = data.layout.stride[axis];
        int data_mid_shape = data.layout.shape[axis];

        size_t nr_elems = data_nomid.layout.total_nr_elems();
        megdnn_assert(nr_elems == index.layout.total_nr_elems() &&
                nr_elems == sub.layout.total_nr_elems());
        auto data_iter = tensor_iter_valonly<data_type>(data_nomid).begin();
        auto idx_iter = tensor_iter_valonly<idx_type>(index).begin();
        auto sub_iter = tensor_iter_valonly<data_type>(sub).begin();

        data_type* dptr = data.ptr<data_type>();

        for (size_t i = 0; i < nr_elems; ++ i) {
            auto idx = *idx_iter;
            megdnn_assert(idx >= 0 && idx < data_mid_shape);
            dptr[data_iter.offset() + *idx_iter * data_mid_stride] = *sub_iter;
            ++ data_iter;
            ++ sub_iter;
            ++ idx_iter;
        }
    }

} // anonymous namespace


void IndexingOneHotForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in index,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, index.layout, dst.layout, workspace.size);

#define cb(_dt) \
    case DTypeTrait<_dt>::enumv: \
    { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_get<DTypeTrait<_dt>::ctype>( \
                    src, index, dst, param().axis)); \
        return; \
    }
    switch (src.layout.dtype.enumv()) {
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(megdnn::dtype::Quantized8Asymm)
        default:
            megdnn_throw(megdnn_mangle("bad dtype"));
    }
#undef cb
}

void IndexingSetOneHotForwardImpl::exec(
        _megdnn_tensor_inout data, _megdnn_tensor_in index,
        _megdnn_tensor_in sub,
        _megdnn_workspace workspace) {
    check_exec(data.layout, index.layout, sub.layout, workspace.size);

#define cb(_dt) \
    case DTypeTrait<_dt>::enumv: \
    { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_set<DTypeTrait<_dt>::ctype>( \
                    data, index, sub, param().axis)); \
        return; \
    }
    switch (data.layout.dtype.enumv()) {
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(megdnn::dtype::Quantized8Asymm)
        default:
            megdnn_throw(megdnn_mangle("bad dtype"));
    }
#undef cb
}

// vim: syntax=cpp.doxygen


