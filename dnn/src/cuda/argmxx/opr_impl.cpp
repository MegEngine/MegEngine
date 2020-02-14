/**
 * \file dnn/src/cuda/argmxx/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/argmxx/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/common/reduce_helper.h"
#include "src/common/argmxx_helper.h"
#include "src/cuda/reduce_helper.cuh"

namespace {

using namespace megdnn;
using namespace cuda;
using namespace argmxx;

template <typename T, bool is_max>
size_t get_workspace_in_bytes_impl(const TensorLayout &src,
        const TensorLayout & /* dst */,
        size_t axis)
{
    size_t A, B, C;
    reduce::get_ABC(src, A, B, C, axis);
    return get_reduce_workspace_in_bytes<argmxx::ArgmxxOp<T, is_max>>(
            A, B, C);
}

template <typename T, bool is_max>
void exec_impl(const T *src, int *dst, void *workspace,
        size_t A, size_t B, size_t C,
        cudaStream_t stream)
{
    argmxx::ArgmxxOp<T, is_max> opr(const_cast<T *>(src), dst, A, B, C);
    run_reduce<argmxx::ArgmxxOp<T, is_max>, false>(
            (typename argmxx::ArgmxxOp<T, is_max>::wtype *)workspace,
            A, B, C,
            stream, opr);
    after_kernel_launch();
}

} // anonymous namespace

namespace megdnn {
namespace cuda {

size_t ArgmaxForwardImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayout &dst)
{
#define cb(DType) \
    if (src.dtype == DType()) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        return get_workspace_in_bytes_impl<ctype, true>(src, dst, param().axis); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(false);
}

void ArgmaxForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param().axis);
    auto stream = cuda_stream(handle());
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        exec_impl<ctype, true>(src.ptr<ctype>(), \
                dst.ptr<dt_int32>(), \
                workspace.raw_ptr, \
                A, B, C, stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

#undef cb
}

size_t ArgminForwardImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayout &dst)
{
#define cb(DType) \
    if (src.dtype == DType()) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        return get_workspace_in_bytes_impl<ctype, false>(src, dst, param().axis); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(false);
}

void ArgminForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    size_t A, B, C;
    reduce::get_ABC(src.layout, A, B, C, param().axis);
    auto stream = cuda_stream(handle());
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        exec_impl<ctype, false>(src.ptr<ctype>(), \
                dst.ptr<dt_int32>(), \
                workspace.raw_ptr, \
                A, B, C, stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    
#undef cb
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
