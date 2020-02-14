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
                                        _megdnn_tensor_in table,
                                        _megdnn_tensor_out dst,
                                        _megdnn_workspace workspace) {
    size_t inp_size = srcs.layout.shape[0],
           out_size = dst.layout.total_nr_elems();
    auto stream = cuda_stream(this->handle());

    auto src_cpu = static_cast<const T**>(srcs.raw_ptr);
    megdnn_assert_internal(src_cpu);
    auto src_gpu = reinterpret_cast<const T**>(workspace.raw_ptr);

    auto table_outer_gpu = table.ptr<int32_t>(),
         table_inner_gpu = table_outer_gpu + out_size;

    cuda_check(cudaMemcpyAsync(src_gpu, src_cpu, sizeof(const T*) * inp_size,
                               cudaMemcpyHostToDevice, stream));

    param_pack::concat_proxy<T>(src_gpu, dst.ptr<T>(), out_size,
                                table_outer_gpu, table_inner_gpu, stream);
}

void ParamPackConcatImpl::exec(_megdnn_tensor_in srcs, _megdnn_tensor_in table,
                               _megdnn_tensor_out dst,
                               _megdnn_workspace workspace) {
    check_exec(dst.layout, table.layout, srcs.layout);
#define cb(DType)                                          \
    if (dst.layout.dtype == DType()) {                     \
        using ctype = typename DTypeTrait<DType>::ctype;   \
        exec_internal<ctype>(srcs, table, dst, workspace); \
        return;                                            \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_throw("bad type");
#undef cb
}

size_t ParamPackSplitImpl::get_workspace_in_bytes(
        const TensorShape&, const TensorShape&, const TensorShapeArray& dsts) {
    return sizeof(size_t) * dsts.size();
}

template <typename T>
void ParamPackSplitImpl::exec_internal(_megdnn_tensor_in src,
                                       _megdnn_tensor_in table,
                                       _megdnn_tensor_out dsts,
                                       _megdnn_workspace workspace) {
    // inner and outer table must be  int32
    megdnn_assert(table.layout.dtype == dtype::Int32());
    // dsts is src pointer, ndim must be 1
    megdnn_assert(dsts.layout.ndim == 1);

    auto out_size = dsts.layout.shape[0],
         inp_size = src.layout.total_nr_elems();

    auto stream = cuda_stream(this->handle());

    auto total_workspace_size = sizeof(T*) * out_size;
    auto dsts_cpu = static_cast<T**>(dsts.raw_ptr);
    megdnn_assert_internal(dsts_cpu);
    auto dsts_gpu = reinterpret_cast<T**>(workspace.raw_ptr);

    auto table_outer_gpu = table.ptr<int32_t>();
    auto table_inner_gpu = table_outer_gpu + inp_size;

    cuda_check(cudaMemcpyAsync(dsts_gpu, dsts_cpu, total_workspace_size,
                               cudaMemcpyHostToDevice, stream));

    // param_pack_split_proxy()
    param_pack::split_proxy<T>(src.ptr<T>(), dsts_gpu, inp_size,
                               table_outer_gpu, table_inner_gpu, stream);
}

void ParamPackSplitImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in table,
                              _megdnn_tensor_out dsts,
                              _megdnn_workspace workspace) {
    check_exec(src.layout, table.layout, dsts.layout);
#define cb(DType)                                          \
    if (src.layout.dtype == DType()) {                     \
        using ctype = typename DTypeTrait<DType>::ctype;   \
        exec_internal<ctype>(src, table, dsts, workspace); \
        return;                                            \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_throw("bad type");
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
