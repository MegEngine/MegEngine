/**
 * \file dnn/src/cuda/cumsum/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./kern.cuh"

#include "src/common/reduce_helper.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace cumsum;

namespace {

/*!
 * \brief compute cumsum reduction on (A, B, C) tensor to (A, 1, C)
 */
template <typename T, class Op>
void dispatch(T* dst, T* workspace, size_t workspace_size, size_t A, size_t B,
              size_t C, bool exclusive, bool reverse, const Op& op,
              cudaStream_t stream) {
#define IF(exclusive_v, reverse_v)                                    \
    if (exclusive == exclusive_v && reverse == reverse_v) {           \
        run_kern<T, Op, exclusive_v, reverse_v>(                      \
                dst, workspace, workspace_size, A, B, C, op, stream); \
        return;                                                       \
    }
    IF(true, true)
    IF(true, false)
    IF(false, true)
    IF(false, false)
    megdnn_assert_internal(false);
#undef IF
}

}  // anonymous namespace

void CumsumForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                             _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param().axis);
    auto stream = cuda_stream(handle());
#define cb(DType)                                                            \
    if (src.layout.dtype == DType()) {                                       \
        using ctype = DTypeTrait<DType>::ctype;                              \
        dispatch<ctype, SumOp<ctype>>(                                       \
                dst.ptr<ctype>(), workspace.ptr<ctype>(), workspace.size, A, \
                B, C, param().exclusive, param().reverse, src.ptr<ctype>(),  \
                stream);                                                     \
        return;                                                              \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(false);
}

size_t CumsumForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                 const TensorLayout&) {
    size_t A, B, C;
    reduce::get_ABC(src, A, B, C, param().axis);
    cuda_check(cudaSetDevice(concrete_handle(handle())->device_id()));
    return cumsum::get_workspace_in_bytes(A, B, C, src.dtype.size());
}

// vim: syntax=cpp.doxygen
