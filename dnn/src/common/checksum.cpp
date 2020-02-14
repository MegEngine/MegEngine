/**
 * \file dnn/src/common/checksum.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"
#include "src/common/utils.h"

using namespace megdnn;

void megdnn::ChecksumForward::check_exec(const TensorLayout &layout,
        size_t workspace_in_bytes) {
    megdnn_assert(layout.is_contiguous() &&
            layout.ndim == 1 &&
            layout.dtype == dtype::Byte() &&
            layout.shape[0], "%s", layout.to_string().c_str());
    auto required_workspace_in_bytes = get_workspace_in_bytes(layout);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

// vim: syntax=cpp.doxygen

