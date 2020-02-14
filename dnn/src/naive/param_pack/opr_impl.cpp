/**
 * \file dnn/src/naive/param_pack/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/param_pack/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

template <typename T>
void ParamPackSplitImpl::exec_internal(_megdnn_tensor_in src, int32_t* table,
                                       _megdnn_tensor_out dsts,
                                       _megdnn_workspace) {
    auto dsts_ptr = static_cast<T**>(dsts.raw_ptr);
    auto src_ptr = src.ptr<T>();

    auto inp_size = src.layout.total_nr_elems();
    auto table_outer = table, table_inner = table_outer + inp_size;

    for (size_t j = 0; j < inp_size; j++) {
        int32_t i = table_outer[j];
        int32_t idx = table_inner[j];
        if (idx != -1) {
            dsts_ptr[i][idx] = src_ptr[j];
        }
    }
}

void ParamPackSplitImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in table,
                              _megdnn_tensor_out dsts,
                              _megdnn_workspace workspace) {
    check_exec(src.layout, table.layout, dsts.layout);
    auto table_ptr = table.ptr<int32_t>();

#define cb(DType)                                                       \
    if (src.layout.dtype == DType()) {                                  \
        using ctype = typename DTypeTrait<DType>::ctype;                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                   \
                exec_internal<ctype>(src, table_ptr, dsts, workspace)); \
        return;                                                         \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_throw("bad type");
#undef cb
}

template <typename T>
void ParamPackConcatImpl::exec_internal(_megdnn_tensor_in srcs, int32_t* table,
                                        _megdnn_tensor_out dst,
                                        _megdnn_workspace) {
    size_t out_size = dst.layout.total_nr_elems();

    auto srcs_ptr = static_cast<const T**>(srcs.raw_ptr);
    auto dst_ptr = dst.ptr<T>();

    auto table_outer = table, table_inner = table_outer + out_size;

    for (size_t j = 0; j < out_size; j++) {
        int32_t i = table_outer[j];
        int32_t idx = table_inner[j];
        if (idx != -1)
            dst_ptr[j] = srcs_ptr[i][idx];
        else
            dst_ptr[j] = 0;
    }
}

void ParamPackConcatImpl::exec(_megdnn_tensor_in srcs, _megdnn_tensor_in table,
                               _megdnn_tensor_out dst,
                               _megdnn_workspace workspace) {
    check_exec(dst.layout, table.layout, srcs.layout);
    auto table_ptr = table.ptr<int32_t>();

#define cb(DType)                                                       \
    if (dst.layout.dtype == DType()) {                                  \
        using ctype = typename DTypeTrait<DType>::ctype;                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                   \
                exec_internal<ctype>(srcs, table_ptr, dst, workspace)); \
        return;                                                         \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_throw("bad type");
#undef cb
}
