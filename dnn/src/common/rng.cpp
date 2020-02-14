/**
 * \file dnn/src/common/rng.cpp
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

namespace megdnn {

void RNGBase::check_exec(
        const TensorLayout &dst, size_t workspace_in_bytes) {
    megdnn_assert(dst.dtype.category() == DTypeCategory::FLOAT &&
            dst.is_contiguous());
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(dst));
}

} // namespace megdnn

// vim: syntax=cpp.doxygen

