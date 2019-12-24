/**
 * \file dnn/src/cuda/pooling/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class PoolingForwardImpl final : public PoolingForward {
public:
    using PoolingForward::PoolingForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& dst) override {
        return get_workspace_bundle(nullptr, src, dst).total_size_in_bytes();
    }

private:
    TensorDesc src_desc, dst_desc;
    PoolingDesc pooling_desc;
    void setup_descs(const TensorLayout& src, const TensorLayout& dst);
    WorkspaceBundle get_workspace_bundle(void* ptr, const TensorLayout& src,
                                         const TensorLayout& dst) const;
};

class PoolingBackwardImpl final : public PoolingBackward {
public:
    using PoolingBackward::PoolingBackward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
              _megdnn_tensor_in diff, _megdnn_tensor_out grad,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& dst,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override {
        return get_workspace_bundle(nullptr, src, dst, diff, grad)
                .total_size_in_bytes();
    }

private:
    TensorDesc src_desc, dst_desc, diff_desc, grad_desc;
    PoolingDesc pooling_desc;
    void setup_descs(const TensorLayout& src, const TensorLayout& dst,
                     const TensorLayout& diff, const TensorLayout& grad);
    WorkspaceBundle get_workspace_bundle(void* ptr, const TensorLayout& src,
                                         const TensorLayout& dst,
                                         const TensorLayout& diff,
                                         const TensorLayout& grad) const;
};

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
