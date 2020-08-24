/**
 * \file dnn/src/atlas/checksum/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class ChecksumForwardImpl final : public ChecksumForward {
public:
    using ChecksumForward::ChecksumForward;

    bool is_thread_safe() const override { return true; }

    size_t get_workspace_in_bytes(const TensorLayout& data) override;

    Result exec(_megdnn_tensor_in data, _megdnn_workspace workspace) override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
