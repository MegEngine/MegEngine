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

#include "src/common/reduce_helper_device.h"

namespace megdnn {
namespace cuda {

using device_reduce::CheckNonFiniteOp;
#define total_nr_elems_max 2048
size_t CheckNonFiniteImpl::_get_workspace_in_bytes() {
    // Call the _get_workspace_in_bytes to reduce the loop fetch workspace bytes
    typedef CheckNonFiniteOp<dt_float32, size_t, dt_int32, dt_int32> Op;
    megdnn_assert(m_size > 0);
    WorkspaceBundle bundle(
            nullptr, {
                             sizeof(dt_float32*) * m_size,
                             sizeof(size_t) * m_size,
                     });
    return get_reduce_workspace_in_bytes<Op>(1, m_size * total_nr_elems_max, 1) +
           bundle.total_size_in_bytes();
}

size_t CheckNonFiniteImpl::get_workspace_in_bytes(
        const TensorNDArray& srcs, const TensorLayout&) {
    m_size = 0;
    for (const auto& src : srcs) {
        m_size += DIVUP(src.layout.total_nr_elems(), total_nr_elems_max);
    }
    return _get_workspace_in_bytes();
}

void CheckNonFiniteImpl::exec(
        _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(srcs, dst, workspace.size);
    typedef CheckNonFiniteOp<dt_float32, size_t, dt_int32, dt_int32> Op;
    auto stream = cuda_stream(this->handle());
    SmallVector<size_t> workspace_sizes{
            sizeof(dt_float32*) * m_size,
            sizeof(size_t) * m_size,
    };
    WorkspaceBundle workspace_cpu(nullptr, workspace_sizes),
            workspace_gpu(nullptr, workspace_sizes);
    auto total_workspace_size = workspace_cpu.total_size_in_bytes();
    void* workspace_cpu_raw = malloc(total_workspace_size);
    megdnn_assert_internal(workspace_cpu_raw);
    void* workspace_gpu_raw = workspace.raw_ptr;
    workspace_cpu = WorkspaceBundle(workspace_cpu_raw, workspace_sizes);
    workspace_gpu = WorkspaceBundle(workspace_gpu_raw, workspace_sizes);

    auto srcs_cpu = static_cast<dt_float32**>(workspace_cpu.get(0));
    auto srcs_gpu = static_cast<dt_float32**>(workspace_gpu.get(0));
    auto srcs_total_nr_elems_cpu = static_cast<size_t*>(workspace_cpu.get(1));
    auto srcs_total_nr_elems_gpu = static_cast<size_t*>(workspace_gpu.get(1));

    // srcs
    // cut the tensor to a fixed length of total_nr_elems_max
    size_t i = 0;
    for (const auto& src : srcs) {
        size_t src_nr_elems = src.layout.total_nr_elems();
        size_t nr_elems = DIVUP(src_nr_elems, total_nr_elems_max);
        for (size_t j = 0; j < nr_elems; ++j, ++i) {
            srcs_cpu[i] = src.ptr<dt_float32>() + j * total_nr_elems_max;
            if (j + 1 == nr_elems && src_nr_elems % total_nr_elems_max) {
                srcs_total_nr_elems_cpu[i] = src_nr_elems % total_nr_elems_max;
            } else {
                srcs_total_nr_elems_cpu[i] = total_nr_elems_max;
            }
        }
    }
    for (size_t i = 0; i < workspace_cpu.nr_workspace(); ++i) {
        cuda_check(cudaMemcpyAsync(
                workspace_gpu.get(i), workspace_cpu.get(i), workspace_cpu.get_size(i),
                cudaMemcpyHostToDevice, stream));
    }
    cuda_check(cudaStreamAddCallback(
            stream, callback_free, static_cast<void*>(workspace_cpu_raw), 0));

    return run_reduce<Op, false>(
            static_cast<dt_int32*>(
                    (void*)((char*)workspace_gpu_raw +
                            workspace_gpu.total_size_in_bytes())),
            1, m_size * total_nr_elems_max, 1, stream,
            Op(srcs_gpu, srcs_total_nr_elems_gpu, dst.ptr<dt_int32>(),
               total_nr_elems_max, param().scale));
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
