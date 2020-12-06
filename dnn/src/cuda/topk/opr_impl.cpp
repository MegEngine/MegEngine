/**
 * \file dnn/src/cuda/topk/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./topk_radix.cuh"
#include "src/common/utils.h"
#include "src/cuda/argsort/argsort.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

template <typename ctype>
void TopKImpl::dispatch_with_ctype(int k, size_t m, size_t n, ptrdiff_t lda,
                                   const ctype* data, ctype* values,
                                   int* indices, void* workspace) {
    auto _handle = concrete_handle(handle());
    auto stream = _handle->stream();
    size_t grid_dim_y_limit = _handle->device_prop().maxGridSize[1];
    switch (param().mode) {
        case Param::Mode::KTH_ONLY:
            cuda_check(topk::find_kth_radix<ctype>(data, values, workspace, m,
                                                   n, lda, k, grid_dim_y_limit,
                                                   stream));
            return;
        case Param::Mode::VALUE_IDX_NOSORT: {
            WorkspaceBundle wk_bundle{workspace, {m * sizeof(ctype), 1}};
            auto thresh = static_cast<ctype*>(wk_bundle.get(0));
            auto real_wk = wk_bundle.get(1);
            cuda_check(topk::find_kth_radix<ctype>(data, thresh, real_wk, m, n,
                                                   lda, k, grid_dim_y_limit,
                                                   stream));
            cuda_check(topk::topk_select<ctype>(data, thresh, values, indices,
                                                real_wk, m, n, lda, k,
                                                grid_dim_y_limit, stream));
            return;
        }
        case Param::Mode::VALUE_IDX_SORTED: {
            WorkspaceBundle wk_bundle{
                    workspace,
                    {m * sizeof(ctype), m * std::abs(k) * sizeof(ctype),
                     m * std::abs(k) * sizeof(int32_t), 1}};
            auto thresh = static_cast<ctype*>(wk_bundle.get(0)),
                 nosort_values = static_cast<ctype*>(wk_bundle.get(1));
            auto nosort_idx = static_cast<int32_t*>(wk_bundle.get(2));
            auto real_wk = wk_bundle.get(3);
            cuda_check(topk::find_kth_radix<ctype>(data, thresh, real_wk, m, n,
                                                   lda, k, grid_dim_y_limit,
                                                   stream));
            cuda_check(topk::topk_select<ctype>(data, thresh, nosort_values,
                                                nosort_idx, real_wk, m, n, lda,
                                                k, grid_dim_y_limit, stream));
            argsort::forward(nosort_values, values, indices, real_wk, m,
                             std::abs(k), k > 0, stream, nosort_idx);
            return;
        }
    }
    megdnn_throw("bad topk mode");
}

void TopKImpl::do_exec(int k, _megdnn_tensor_in data, _megdnn_tensor_out values,
                       int32_t* indices, _megdnn_workspace workspace) {
    switch (data.layout.dtype.enumv()) {
        case DTypeEnum::Float32:
            dispatch_with_ctype<float>(k, data.layout[0], data.layout[1],
                                       data.layout.stride[0], data.ptr<float>(),
                                       values.ptr<float>(), indices,
                                       workspace.raw_ptr);
            return;
        case DTypeEnum::Int32:
            dispatch_with_ctype<int32_t>(k, data.layout[0], data.layout[1],
                                       data.layout.stride[0], data.ptr<int32_t>(),
                                       values.ptr<int32_t>(), indices,
                                       workspace.raw_ptr);
            return;
        default:
            megdnn_throw(
                    ssprintf("only float32 and int32 supported for cuda topk, got: %s",
                             data.layout.dtype.name()));
    }
}

size_t TopKImpl::get_workspace_in_bytes(int k, const TensorLayout& data,
                                        const TensorLayout& values,
                                        const TensorLayout& indices) {
    MEGDNN_MARK_USED_VAR(values);
    MEGDNN_MARK_USED_VAR(indices);
    size_t m = data[0], n = data[1];
    size_t kabs = std::abs(k);
    size_t grid_dim_y_limit =
            concrete_handle(handle())->device_prop().maxGridSize[1];
    megdnn_assert(std::max(m, n) <=
                  static_cast<size_t>(std::numeric_limits<int>::max()));
    size_t kth = topk::find_kth_radix_workspace(m, n, grid_dim_y_limit),
           sel = topk::topk_select_workspace(m, n);
    auto ctsize = data.dtype.size();
    switch (param().mode) {
        case Param::Mode::KTH_ONLY:
            return kth;
        case Param::Mode::VALUE_IDX_NOSORT:
            return WorkspaceBundle{nullptr, {m * ctsize, std::max(kth, sel)}}
                    .total_size_in_bytes();
        case Param::Mode::VALUE_IDX_SORTED:
            return WorkspaceBundle{
                    nullptr,
                    {m * ctsize, m * kabs * ctsize, m * kabs * sizeof(int32_t),
                     std::max(std::max(kth, sel),
                              argsort::get_fwd_workspace_in_bytes(
                                      m, kabs, data.dtype, k > 0, true))}}
                    .total_size_in_bytes();
    }
    megdnn_throw("bad topk mode");
}

// vim: syntax=cpp.doxygen

