/**
 * \file dnn/src/cuda/matrix_inverse/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./helper.cuh"
#include "./opr_impl.h"
#include "src/cuda/batched_matrix_mul/helper.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

size_t MatrixInverseImpl::get_workspace_in_bytes(size_t batch, size_t, size_t) {
    return batch * (sizeof(int) + sizeof(void*) + sizeof(void*));
}

void MatrixInverseImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                             _megdnn_workspace workspace) {
    megdnn_assert(src.layout.dtype == dtype::Float32(),
                  "Matrix Inverse only support Float32 dtype, got: %s",
                  src.layout.dtype.name());
    size_t batch, n;
    check_exec(src.layout, dst.layout, workspace, &batch, &n);
    auto handle = concrete_handle(this->handle());
    megdnn_assert(n < 32, "currently only n < 32 supported on cuda");
    const float** psrc_batch = workspace.ptr<const float*>();
    float** pdst_batch = const_cast<float**>(psrc_batch + batch);
    int* info = reinterpret_cast<int*>(pdst_batch + batch);
    auto stream = handle->stream();
    batched_matrix_mul::arange<uintptr_t>(
            reinterpret_cast<uintptr_t*>(psrc_batch),
            reinterpret_cast<uintptr_t>(src.raw_ptr), n * n * sizeof(float),
            batch, stream);
    batched_matrix_mul::arange<uintptr_t>(
            reinterpret_cast<uintptr_t*>(pdst_batch),
            reinterpret_cast<uintptr_t>(dst.raw_ptr), n * n * sizeof(float),
            batch, stream);
    cublas_check(cublasSmatinvBatched(handle->cublas_handle(), n, psrc_batch, n,
                                      pdst_batch, n, info, batch));
    matrix_inverse::check_error(info, batch,
                                handle->megcore_context().error_info,
                                m_error_tracker, stream);
}

// vim: syntax=cpp.doxygen
