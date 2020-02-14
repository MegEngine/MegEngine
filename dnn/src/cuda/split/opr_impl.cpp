/**
 * \file dnn/src/cuda/split/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/split/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/cuda/split/split.cuh"
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {

size_t SplitForwardImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayoutArray &dsts)
{
    check_layout_common(dsts, src);
    auto B = src.shape[param().axis];
    // Please refer to ConcatForwardImpl. Implementations are similar.
    WorkspaceBundle bundle(nullptr, {
        sizeof(uintptr_t) * dsts.size(),
        sizeof(size_t) * dsts.size(),
        sizeof(size_t) * B,
        sizeof(size_t) * B,
    });
    return bundle.total_size_in_bytes();
}

template <typename T>
void SplitForwardImpl::exec_internal(_megdnn_tensor_in src,
        const TensorNDArray &dsts,
        _megdnn_workspace workspace)
{
    // Please refer to ConcatForwardImpl. Implementations are similar.
    auto dsts_layout = apply_vector<TensorLayout>(m_get_layout, dsts);
    auto dsts_shape = apply_vector<TensorShape>(m_get_shape, dsts_layout);
    check_exec(src.layout, dsts_layout, workspace.size);
    size_t A, B, C;
    auto stream = cuda_stream(this->handle());

    // Pre-calculate B to determine cpu-side workspace size.
    B = src.layout.shape[param().axis];

	// workspace_cpu will be freed by cuda callback.
    SmallVector<size_t> workspace_sizes {
        sizeof(const T *) * dsts.size(),
        sizeof(size_t) * dsts.size(),
        sizeof(size_t) * B,
        sizeof(size_t) * B,
    };

    WorkspaceBundle workspace_cpu(nullptr, workspace_sizes),
                    workspace_gpu(nullptr, workspace_sizes);

    auto total_workspace_size = workspace_cpu.total_size_in_bytes();
    void *workspace_cpu_raw = malloc(total_workspace_size);
    megdnn_assert_internal(workspace_cpu_raw);
    void *workspace_gpu_raw = static_cast<void *>(workspace.raw_ptr);
    workspace_cpu = WorkspaceBundle(workspace_cpu_raw, workspace_sizes);
    workspace_gpu = WorkspaceBundle(workspace_gpu_raw, workspace_sizes);

    auto dsts_cpu = static_cast<T **>(workspace_cpu.get(0));
    auto dsts_gpu = static_cast<T **>(workspace_gpu.get(0));
    for (size_t i = 0; i < dsts.size(); ++i) {
        dsts_cpu[i] = dsts[i].ptr<T>();
    }

    auto Bv_cpu = static_cast<size_t *>(workspace_cpu.get(1));
    auto Bv_gpu = static_cast<size_t *>(workspace_gpu.get(1));
    get_ABC(dsts_shape, A, Bv_cpu, C);

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
    split::forward_proxy<T>(src.ptr<T>(), dsts_gpu, dsts.size(),
            A, B, C,
            Bv_gpu,
            table_outer_gpu,
            table_inner_gpu,
            stream);
}

void SplitForwardImpl::exec(_megdnn_tensor_in src,
        const TensorNDArray &dsts,
        _megdnn_workspace workspace)
{
#define cb(DType) \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) { \
        using ctype = typename DTypeTrait<DType>::ctype; \
        exec_internal<ctype>(src, dsts, workspace); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

