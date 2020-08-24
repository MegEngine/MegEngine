/**
 * \file dnn/src/cambricon/checksum/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cambricon/checksum/checksum.mlu.h"
#include "src/cambricon/checksum/opr_impl.h"

#include "src/cambricon/utils.h"

#include <algorithm>

using namespace megdnn;
using namespace cambricon;

namespace {
void bang_c_wrapper(uint32_t* dst, const uint32_t* src, int nr_elems,
                    cnrtQueue_t queue, cnrtCoreVersion_t core_version) {
    cnrtKernelParamsBuffer_t params;
    cnrt_check(cnrtGetKernelParamsBuffer(&params));
    cnrt_check(cnrtKernelParamsBufferAddParam(params, &dst, sizeof(uint32_t*)));
    cnrt_check(cnrtKernelParamsBufferAddParam(params, &src, sizeof(uint32_t*)));
    cnrt_check(cnrtKernelParamsBufferAddParam(params, &nr_elems, sizeof(int)));
    if (core_version == CNRT_MLU270) {
        cnrtDim3_t dim;
        dim.x = 16;
        dim.y = 1;
        dim.z = 1;
        cnrtFunctionType_t c = CNRT_FUNC_TYPE_UNION4;
        cnrt_check(cnrtInvokeKernel_V2((void*)&checksum_kernel_union4, dim,
                                       params, c, queue));
    } else if (core_version == CNRT_MLU220) {
        cnrtDim3_t dim;
        dim.x = 4;
        dim.y = 1;
        dim.z = 1;
        cnrtFunctionType_t c = CNRT_FUNC_TYPE_UNION1;
        cnrt_check(cnrtInvokeKernel_V2((void*)&checksum_kernel_union1, dim,
                                       params, c, queue));
    }
    after_kernel_launch();
    cnrt_check(cnrtDestroyKernelParamsBuffer(params));
}
}  // namespace

size_t ChecksumForwardImpl::get_workspace_in_bytes(const TensorLayout& /* data */) {
    size_t ws_size = sizeof(ChecksumForward::Result::checksum);
    return ws_size;
}

ChecksumForward::Result ChecksumForwardImpl::exec(_megdnn_tensor_in data,
                                                  _megdnn_workspace workspace) {
    Result result;
    memset(&result, 0, sizeof(result));
    check_exec(data.layout, workspace.size);
    auto queue = cnrt_queue(handle());

    auto ptr = static_cast<uint8_t*>(data.raw_ptr);
    size_t size_all = data.layout.shape[0],
           size_ints = size_all / sizeof(uint32_t);
    auto last_val_size = std::min<size_t>(size_all, 4);
    cnrt_check(cnrtMemcpyAsync(&result.last_val, ptr + size_all - last_val_size,
                               last_val_size, queue,
                               CNRT_MEM_TRANS_DIR_DEV2HOST));
    if (size_ints) {
        auto&& device_info = current_device_info();
        bang_c_wrapper(reinterpret_cast<uint32_t*>(workspace.raw_ptr),
                       static_cast<uint32_t*>(data.raw_ptr), size_ints, queue,
                       device_info.core_version);
        cnrt_check(cnrtMemcpyAsync(&result.checksum, workspace.raw_ptr,
                                   sizeof(result.checksum), queue,
                                   CNRT_MEM_TRANS_DIR_DEV2HOST));
    }
    cnrt_check(cnrtSyncQueue(queue));
    return result;
}

// vim: syntax=cpp.doxygen

