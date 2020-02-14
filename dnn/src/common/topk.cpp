/**
 * \file dnn/src/common/topk.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"

#include "src/common/utils.h"

#include <cmath>

using namespace megdnn;

void TopK::deduce_layout(int k, const TensorLayout& data, TensorLayout& values,
                         TensorLayout& indices) {
    megdnn_assert(k && data.ndim == 2 && data.stride[1] == 1,
                  "invalid k=%d data=%s", k, data.to_string().c_str());
    values.dtype = data.dtype;
    indices.dtype = dtype::Int32{};
    switch (param().mode) {
        case Param::Mode::KTH_ONLY:
            values.init_contiguous_stride({data[0]});
            indices.ndim = 0;
            break;
        case Param::Mode::VALUE_IDX_NOSORT:
        case Param::Mode::VALUE_IDX_SORTED:
            values.init_contiguous_stride(
                    {data[0], std::min<size_t>(std::abs(k), data.shape[1])});
            indices.init_contiguous_stride(values);
            break;
        default:
            megdnn_throw("invalid TopK mode");
    }
}

void TopK::exec(int k, _megdnn_tensor_in data, _megdnn_tensor_out values,
                _megdnn_tensor_out indices, _megdnn_workspace workspace) {
    TensorLayout oval, oidx;
    deduce_layout(k, data.layout, oval, oidx);
    megdnn_assert_eq_layout(oval, values.layout);
    int32_t* iptr = nullptr;
    if (param().mode == Param::Mode::KTH_ONLY) {
        megdnn_assert_eq_shape(indices.layout, TensorShape{});
    } else {
        iptr = indices.ptr<int32_t>();
        megdnn_assert_eq_layout(oidx, indices.layout);
    }
    megdnn_assert(workspace.size >= get_workspace_in_bytes(k, data.layout,
                                                           values.layout,
                                                           indices.layout));
    if (static_cast<size_t>(std::abs(k)) > data.layout[1]) {
        if (k > 0) {
            k = data.layout[1];
        } else {
            k = -static_cast<int>(data.layout[1]);
        }
    }
    do_exec(k, data, values, iptr, workspace);
}

// vim: syntax=cpp.doxygen

