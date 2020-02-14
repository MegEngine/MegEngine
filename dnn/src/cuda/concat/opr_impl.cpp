/**
 * \file dnn/src/cuda/concat/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/concat/opr_impl.h"
#include "src/cuda/utils.h"
#include "src/cuda/concat/concat.cuh"

namespace megdnn {
namespace cuda {

size_t ConcatForwardImpl::get_workspace_in_bytes(
        const TensorLayoutArray &srcs,
        const TensorLayout &dst)
{
    auto B = dst.shape[param().axis];
    // Please refer to the comment in ConcatForwardImpl::exec for detail.
    WorkspaceBundle bundle(nullptr, {
            sizeof(uintptr_t) * srcs.size(),
            sizeof(size_t) * srcs.size(),
            sizeof(size_t) * B,
            sizeof(size_t) * B});
    return bundle.total_size_in_bytes();
}

template <typename T>
void ConcatForwardImpl::exec_internal(
        _megdnn_in const TensorNDArray &srcs,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    auto srcs_layout = apply_vector<TensorLayout>(m_get_layout, srcs);
    auto srcs_shape = apply_vector<TensorShape>(m_get_shape, srcs_layout);
    check_exec(srcs_layout, dst.layout, workspace.size);
    size_t A, B, C;
    auto stream = cuda_stream(this->handle());

    // Pre-calculate B to determine cpu-side workspace size.
    B = dst.layout.shape[param().axis];

    // workspace_cpu will be freed by cuda callback.
    SmallVector<size_t> workspace_sizes{
        sizeof(const T *) * srcs.size(),
        sizeof(size_t) * srcs.size(),
        sizeof(size_t) * B,
        sizeof(size_t) * B,
    };

    // What do we need:
    //  1. An const T * array of length src.size(), the i-th element of
    //     which stores the address of the i-th srcs.
    //  2. A size_t array of length srcs.size(), the i-th element of which
    //     stores the shape of the param().axis-th axis of the i-th src.
    //  3. A size_t array of length B, the i-th element of which stores the
    //     index of the src tensor that the i-th element along the
    //     param().axis-th axis of dst belongs to.
    //  4. A size_t array of length B, the i-th element of which stores the
    //     intra-offset inside the corresponding src tensor of the i-th element
    //     along the param().axis-th axis of dst.
    //
    // These temporary spaces reside in the device side.
    // The actually work is delegated to concat::forward_proxy.
    WorkspaceBundle workspace_cpu(nullptr, workspace_sizes),
                    workspace_gpu(nullptr, workspace_sizes);
    auto total_workspace_size = workspace_cpu.total_size_in_bytes();
    void *workspace_cpu_raw = malloc(total_workspace_size);
    megdnn_assert_internal(workspace_cpu_raw);
    void *workspace_gpu_raw = workspace.raw_ptr;
    workspace_cpu = WorkspaceBundle(workspace_cpu_raw, workspace_sizes);
    workspace_gpu = WorkspaceBundle(workspace_gpu_raw, workspace_sizes);
    // srcs
   	auto srcs_cpu = static_cast<const T **>(workspace_cpu.get(0));
    auto srcs_gpu = static_cast<const T **>(workspace_gpu.get(0));
    for (size_t i = 0; i < srcs.size(); ++i) {
        srcs_cpu[i] = srcs[i].ptr<T>();
    }

    // Bv
    auto Bv_cpu = static_cast<size_t *>(workspace_cpu.get(1));
    auto Bv_gpu = static_cast<size_t *>(workspace_gpu.get(1));
    get_ABC(srcs_shape, A, Bv_cpu, C);

    // table_outer
    auto table_outer_cpu = static_cast<size_t *>(workspace_cpu.get(2));
    auto table_outer_gpu = static_cast<size_t *>(workspace_gpu.get(2));
    auto table_inner_cpu = static_cast<size_t *>(workspace_cpu.get(3));
    auto table_inner_gpu = static_cast<size_t *>(workspace_gpu.get(3));
    {
        size_t outer_idx = 0, inner_idx = 0;

        for (size_t i = 0; i < B; ++i) {
            table_outer_cpu[i] = outer_idx;
            table_inner_cpu[i] = inner_idx;
            ++inner_idx;
            if (inner_idx == Bv_cpu[outer_idx]) {
                ++outer_idx;
                inner_idx = 0;
            }
        }
    }
    for (size_t i = 0; i < workspace_cpu.nr_workspace(); ++i) {
        cuda_check(cudaMemcpyAsync(workspace_gpu.get(i),
                    workspace_cpu.get(i),
                    workspace_cpu.get_size(i),
                    cudaMemcpyHostToDevice,
                    stream));
    }
    /*
    CUDA_CK(cudaMemcpyAsync(workspace_gpu_raw, workspace_cpu_raw,
                workspace_cpu.total_size_in_bytes(),
                cudaMemcpyHostToDevice,
                stream));
    */
    cuda_check(cudaStreamAddCallback(stream, callback_free,
                static_cast<void *>(workspace_cpu_raw), 0));
    concat::forward_proxy<T>(srcs_gpu, dst.ptr<T>(), srcs.size(),
            A, B, C,
            Bv_gpu,
            table_outer_gpu,
            table_inner_gpu,
            stream);
}

void ConcatForwardImpl::exec(_megdnn_in const TensorNDArray &srcs,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
#define cb(DType) \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        exec_internal<ctype>(srcs, dst, workspace); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
