/**
 * \file dnn/src/cuda/check_non_finite/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/check_non_finite/opr_impl.h"
#include "src/cuda/reduce_helper.cuh"

#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

#include "src/common/reduce_helper.h"

namespace megdnn {
namespace cuda {

using reduce::CheckNonFiniteOp;

size_t CheckNonFiniteImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    typedef CheckNonFiniteOp<dt_float32, dt_int32, dt_int32> Op;
    return get_reduce_workspace_in_bytes<Op>(1, src.total_nr_elems(), 1);
}

void CheckNonFiniteImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    typedef CheckNonFiniteOp<dt_float32, dt_int32, dt_int32> Op;
    auto stream = cuda_stream(this->handle());
    auto B = src.layout.total_nr_elems();
    return run_reduce<Op, false>(
            workspace.ptr<dt_int32>(), 1, B, 1, stream,
            Op(src.ptr<dt_float32>(), dst.ptr<dt_int32>(), B));
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
