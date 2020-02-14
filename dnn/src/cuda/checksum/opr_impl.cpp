/**
 * \file dnn/src/cuda/checksum/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "./opr_impl.h"

#include "src/cuda/reduce_helper.cuh"
#include "src/common/utils.h"

#include <algorithm>

using namespace megdnn;
using namespace cuda;

namespace {

WorkspaceBundle get_wbundle(const TensorLayout &data)
{
    size_t size_all = data.shape[0],
           size_ints = size_all / sizeof(uint32_t);
    size_t part1 = checksum::get_workspace_in_bytes(size_ints);
    size_t part2 = sizeof(ChecksumForward::Result::checksum);
    return {nullptr, {part1, part2}};
}

} // anonymous namespace

size_t ChecksumForwardImpl::get_workspace_in_bytes(const TensorLayout &data) {
    auto wbundle = get_wbundle(data);
    return wbundle.total_size_in_bytes();
}


ChecksumForward::Result ChecksumForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_workspace workspace) {
    auto wbundle = get_wbundle(data.layout);
    wbundle.set(workspace.raw_ptr);
    Result result;
    memset(&result, 0, sizeof(result));
    check_exec(data.layout, workspace.size);
    auto stream = cuda_stream(handle());

    auto ptr = static_cast<uint8_t*>(data.raw_ptr);
    size_t size_all = data.layout.shape[0],
           size_ints = size_all / sizeof(uint32_t);
    auto last_val_size = std::min<size_t>(size_all, 4);
    cuda_check(cudaMemcpyAsync(
                &result.last_val, ptr + size_all - last_val_size, last_val_size,
                cudaMemcpyDeviceToHost, stream));
    if (size_ints) {
        checksum::calc(static_cast<uint32_t *>(wbundle.get(1)),
                static_cast<uint32_t *>(data.raw_ptr),
                static_cast<uint32_t *>(wbundle.get(0)),
                size_ints, stream);
        cuda_check(cudaMemcpyAsync(&result.checksum, wbundle.get(1),
                    sizeof(result.checksum), cudaMemcpyDeviceToHost, stream));
    }
    cuda_check(cudaStreamSynchronize(stream));
    return result;
}

// vim: syntax=cpp.doxygen
