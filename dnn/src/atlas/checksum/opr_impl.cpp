/**
 * \file dnn/src/atlas/checksum/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/atlas/checksum/opr_impl.h"
#include "src/atlas/utils.h"
#include "src/naive/handle.h"

#include "src/common/utils.h"
#include "src/common/opr_delegate.h"

#include <cstring>

using namespace megdnn;
using namespace atlas;

size_t ChecksumForwardImpl::get_workspace_in_bytes(const TensorLayout&) {
    return 0;
}

ChecksumForward::Result ChecksumForwardImpl::exec(_megdnn_tensor_in data,
                                                  _megdnn_workspace workspace) {
    check_exec(data.layout, workspace.size);
    //! FIXME currently the cce programming interface is not so stable, here i
    //! just allocate some memory of cpu here and compute the result in cpu
    std::vector<uint8_t> cpu_data(data.layout.span().dist_byte(), 0);

    megcoreDeviceHandle_t dev_handle;
    megcoreComputingHandle_t comp_handle = handle()->megcore_computing_handle();
    megcoreGetDeviceHandle(comp_handle, &dev_handle);
    megcoreMemcpy(comp_handle, cpu_data.data(), data.raw_ptr, cpu_data.size(),
                  megcoreMemcpyDeviceToHost);
    megcoreSynchronize(comp_handle);

    auto opr = inplace_cpu_handle()->create_operator<ChecksumForward>();
    size_t workspace_size = opr->get_workspace_in_bytes(data.layout);
    std::vector<uint8_t> cpu_workspace_data(workspace_size, 0);

    Workspace cpu_workspace(
            reinterpret_cast<dt_byte*>(cpu_workspace_data.data()),
            cpu_workspace_data.size());

    return opr->exec(TensorND{cpu_data.data(), data.layout}, cpu_workspace);
}

// vim: syntax=cpp.doxygen
