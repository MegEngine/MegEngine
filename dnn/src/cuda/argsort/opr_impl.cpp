/**
 * \file dnn/src/cuda/argsort/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./argsort.cuh"
#include "./backward.cuh"

#include "src/common/utils.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

void ArgsortForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                              _megdnn_tensor_out indices,
                              _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, indices.layout, workspace.size);
    auto M = src.layout.shape[0], N = src.layout.shape[1];
    auto iptr = indices.ptr<dt_int32>();
    auto wptr = static_cast<void*>(workspace.raw_ptr);
    bool is_ascending = (param().order == Order::ASCENDING);
    auto stream = cuda_stream(this->handle());
    switch (src.layout.dtype.enumv()) {
#define cb(t)                                                          \
    case DTypeTrait<t>::enumv:                                         \
        argsort::forward(src.ptr<t>(), dst.ptr<t>(), iptr, wptr, M, N, \
                         is_ascending, stream);                        \
        break;
        ARGSORT_FOREACH_CTYPE(cb);
#undef cb
        default:
            megdnn_throw(ssprintf("unsupported argsort dtype on cuda: %s",
                                  src.layout.dtype.name()));
    }
}

size_t ArgsortForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                  const TensorLayout&,
                                                  const TensorLayout&) {
    megdnn_assert(src.ndim == 2, "invalid src layout: %s",
                  src.to_string().c_str());
    auto M = src.shape[0], N = src.shape[1];
    auto&& dtype = src.dtype;
    megdnn_assert(std::max(M, N) <=
                  static_cast<size_t>(std::numeric_limits<int>::max()));
    return argsort::get_fwd_workspace_in_bytes(
            M, N, dtype, param().order == Param::Order::ASCENDING);
}

void ArgsortBackwardImpl::exec(_megdnn_tensor_in diff,
                               _megdnn_tensor_in indices,
                               _megdnn_tensor_out grad,
                               _megdnn_workspace workspace) {
    check_exec(diff.layout, indices.layout, grad.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    switch (diff.layout.dtype.enumv()) {
#define cb(t)                                                                 \
    case DTypeTrait<t>::enumv:                                                \
        argsort::backward_proxy(grad.layout[0], grad.layout[1],               \
                                diff.layout[1], grad.ptr<t>(), diff.ptr<t>(), \
                                indices.ptr<int>(), stream);                  \
        break;
        ARGSORT_FOREACH_CTYPE(cb);
#undef cb
        default:
            megdnn_throw(ssprintf("unsupported argsort dtype on cuda: %s",
                                  diff.layout.dtype.name()));
    }
}

// vim: syntax=cpp.doxygen
