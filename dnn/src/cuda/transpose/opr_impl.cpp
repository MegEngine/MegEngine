/**
 * \file dnn/src/cuda/transpose/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/transpose/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/cuda/transpose/transpose.cuh"

namespace megdnn {
namespace cuda {

void TransposeForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    if (src.layout.dtype == dtype::Float32()) {
        auto handle = concrete_handle(this->handle());
        cublas_check(cublasSgeam(handle->cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_T,
                    dst.layout.shape[1], dst.layout.shape[0],
                    handle->one_device(),
                    src.ptr<dt_float32>(), src.layout.stride[0],
                    handle->zero_device(),
                    src.ptr<dt_float32>(), src.layout.stride[0],
                    dst.ptr<dt_float32>(), dst.layout.stride[0]));
    } else {
        auto stream = cuda_stream(handle());
#define cb(DType) \
        if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
            using T = typename DTypeTrait<DType>::ctype; \
            transpose<T>(src.ptr<T>(), \
                    dst.ptr<T>(), src.layout.shape[0], src.layout.shape[1], \
                    src.layout.stride[0], dst.layout.stride[0], stream); \
        }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    }
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
