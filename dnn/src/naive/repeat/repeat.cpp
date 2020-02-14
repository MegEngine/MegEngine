/**
 * \file dnn/src/naive/repeat/repeat.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/repeat/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>

namespace megdnn {
namespace naive {

template <typename T>
void RepeatForwardImpl::exec_internal(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace /* workspace */)
{
    auto ndim = src.layout.ndim;
    auto sptr = src.ptr<T>(), dptr = dst.ptr<T>();
    auto sshape = src.layout.shape, dshape = dst.layout.shape;
    auto tshape = param().times.shape;
    size_t didx[TensorShape::MAX_NDIM];
    std::memset(didx, 0, sizeof(didx));
    do {
        size_t sidx[TensorShape::MAX_NDIM];
        rep(i, ndim) sidx[i] = didx[i] / tshape[i];
        auto si = get_linear_addr(sidx, sshape, ndim);
        auto di = get_linear_addr(didx, dshape, ndim);
        dptr[di] = sptr[si];
    } while (get_next_addr(didx, dshape, ndim));
}

void RepeatForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_internal<ctype>(src, dst, workspace)); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

template <typename T>
void RepeatBackwardImpl::exec_internal(_megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace /* workspace */)
{
    auto ndim = diff.layout.ndim;
    auto hptr = diff.ptr<T>(), gptr = grad.ptr<T>();
    auto dshape = diff.layout.shape, sshape = grad.layout.shape;
    auto tshape = param().times.shape;
    size_t didx[TensorShape::MAX_NDIM], sidx[TensorShape::MAX_NDIM];
    std::memset(didx, 0, sizeof(didx));
    std::memset(sidx, 0, sizeof(sidx));
    std::memset(gptr, 0, sizeof(T) * grad.layout.total_nr_elems());
    do {
        size_t sidx[TensorShape::MAX_NDIM];
        rep(i, ndim) sidx[i] = didx[i] / tshape[i];
        auto si = get_linear_addr(sidx, sshape, ndim);
        auto di = get_linear_addr(didx, dshape, ndim);
        gptr[si] += hptr[di];
    } while (get_next_addr(didx, dshape, ndim));
}

void RepeatBackwardImpl::exec(_megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(diff.layout, grad.layout, workspace.size);
#define cb(DType) \
    if (diff.layout.dtype == DType()) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                exec_internal<ctype>(diff, grad, workspace)); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
