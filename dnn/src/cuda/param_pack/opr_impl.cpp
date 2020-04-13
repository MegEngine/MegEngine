/**
 * \file dnn/src/cuda/param_pack/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/param_pack/opr_impl.h"
#include "src/cuda/param_pack/param_pack.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

size_t ParamPackConcatImpl::get_workspace_in_bytes(const TensorShapeArray& srcs,
                                                   const TensorShape&,
                                                   const TensorShape&) {
    return sizeof(size_t) * srcs.size();
}

template <typename T>
void ParamPackConcatImpl::exec_internal(_megdnn_tensor_in srcs,
                                        _megdnn_tensor_in offsets,
                                        _megdnn_tensor_out dst,
                                        _megdnn_workspace workspace) {
    size_t inp_size = srcs.layout.shape[0],
           out_size = dst.layout.total_nr_elems();
    auto stream = cuda_stream(this->handle());

    auto src_cpu = static_cast<const T**>(srcs.raw_ptr);
    megdnn_assert_internal(src_cpu);
    auto src_gpu = reinterpret_cast<const T**>(workspace.raw_ptr);

    auto offsets_gpu = offsets.ptr<int32_t>();

    cuda_check(cudaMemcpyAsync(src_gpu, src_cpu, sizeof(const T*) * inp_size,
                               cudaMemcpyHostToDevice, stream));

    param_pack::concat_proxy<T>(src_gpu, dst.ptr<T>(), inp_size, out_size,
                                offsets_gpu, stream);
}

void ParamPackConcatImpl::exec(_megdnn_tensor_in srcs,
                               _megdnn_tensor_in offsets,
                               _megdnn_tensor_out dst,
                               _megdnn_workspace workspace) {
    check_exec(dst.layout, offsets.layout, srcs.layout);
#define cb(DType)                                            \
    if (dst.layout.dtype == DType()) {                       \
        using ctype = typename DTypeTrait<DType>::ctype;     \
        exec_internal<ctype>(srcs, offsets, dst, workspace); \
        return;                                              \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_throw("bad type");
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
