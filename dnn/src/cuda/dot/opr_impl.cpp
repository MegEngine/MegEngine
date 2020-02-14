/**
 * \file dnn/src/cuda/dot/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/dot/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/cuda/dot/dot.cuh"

namespace megdnn {
namespace cuda {

void DotForwardImpl::exec(_megdnn_tensor_in A,
        _megdnn_tensor_in B,
        _megdnn_tensor_out C,
        _megdnn_workspace workspace)
{
    check_exec(A.layout, B.layout, C.layout, workspace.size);
    megdnn_assert(A.layout.dtype.category() == DTypeCategory::FLOAT);
    auto handle = cublas_handle(this->handle());
    if (A.layout.dtype == dtype::Float32()) {
        cublas_check(cublasSdot(handle, A.layout.total_nr_elems(),
                    A.ptr<dt_float32>(), A.layout.stride[0],
                    B.ptr<dt_float32>(), B.layout.stride[0],
                    C.ptr<dt_float32>()));
    } else {
        megdnn_assert_internal(A.layout.dtype == dtype::Float16());
        dot::run<dt_float16>(A.ptr<dt_float16>(),
                B.ptr<dt_float16>(),
                C.ptr<dt_float16>(),
                workspace.ptr<dt_float32>(),
                A.layout.total_nr_elems(),
                A.layout.stride[0], B.layout.stride[0],
                cuda_stream(this->handle()));
    }
}

} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen
