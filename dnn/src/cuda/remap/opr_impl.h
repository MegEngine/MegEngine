/**
 * \file dnn/src/cuda/remap/opr_impl.h
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
namespace cuda {
class RemapImpl final : public Remap {
public:
    using Remap::Remap;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_in map_xy,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& map_xy,
                                  const TensorLayout& dst) override {
        return 0;
    }
};

class RemapBackwardDataImpl final : public RemapBackwardData {
public:
    using RemapBackwardData::RemapBackwardData;
    void exec(_megdnn_tensor_in map_xy, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& map_xy,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override {
        return 0;
    }
};

class RemapBackwardMatImpl final : public RemapBackwardMat {
public:
    using RemapBackwardMat::RemapBackwardMat;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in map_xy,
              _megdnn_tensor_in diff, _megdnn_tensor_out grad,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& map_xy,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
