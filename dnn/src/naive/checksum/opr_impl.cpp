/**
 * \file dnn/src/naive/checksum/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/naive/handle.h"

#include "src/common/utils.h"

#include <cstring>

using namespace megdnn;
using namespace naive;

size_t ChecksumForwardImpl::get_workspace_in_bytes(const TensorLayout &) {
    return 0;
}

ChecksumForward::Result ChecksumForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_workspace workspace) {

    check_exec(data.layout, workspace.size);

    Result result;
    bool finished = false;
    auto run = [&]() {
        auto ptr = static_cast<uint8_t*>(data.raw_ptr);
        size_t size_all = data.layout.shape[0],
        size_ints = size_all / sizeof(uint32_t);
        result.last_val.iv = 0;
        auto last_val_size = std::min<size_t>(size_all, 4);
        memcpy(&result.last_val, ptr + size_all - last_val_size, last_val_size);
        result.checksum = 0;
        auto iptr = static_cast<uint32_t*>(data.raw_ptr);
        for (size_t i = 0; i < size_ints; ++ i)
            result.checksum += iptr[i] * (i + 1);

        finished = true;
    };
    auto handle = static_cast<HandleImpl*>(this->handle());
    handle->dispatch_kern(run);
    handle->megcore_dispatcher()->sync();
    megdnn_assert(finished);
    return result;
}

// vim: syntax=cpp.doxygen
