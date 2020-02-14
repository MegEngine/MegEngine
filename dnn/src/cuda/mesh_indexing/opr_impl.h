/**
 * \file dnn/src/cuda/mesh_indexing/opr_impl.h
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
#include "src/common/utils.h"

namespace megdnn {
namespace cuda {

class MeshIndexingImpl : public MeshIndexing {
    void* m_error_tracker = nullptr;

public:
    using MeshIndexing::MeshIndexing;

    void exec(_megdnn_tensor_in src, const IndexDesc& desc,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override {
        m_error_tracker = tracker;
    }
};

class IncrMeshIndexingImpl : public IncrMeshIndexing {
    void* m_error_tracker = nullptr;

public:
    using IncrMeshIndexing::IncrMeshIndexing;

    void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
              const IndexDesc& desc, _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override {
        m_error_tracker = tracker;
    }
};

class SetMeshIndexingImpl : public SetMeshIndexing {
    void* m_error_tracker = nullptr;

public:
    using SetMeshIndexing::SetMeshIndexing;

    void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
              const IndexDesc& desc, _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override {
        m_error_tracker = tracker;
    }
};

class BatchedMeshIndexingImpl : public BatchedMeshIndexing {
    void* m_error_tracker = nullptr;

public:
    using BatchedMeshIndexing::BatchedMeshIndexing;

    void exec(_megdnn_tensor_in src, const IndexDesc& desc,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override {
        m_error_tracker = tracker;
    }
};

class BatchedIncrMeshIndexingImpl : public BatchedIncrMeshIndexing {
    void* m_error_tracker = nullptr;

public:
    using BatchedIncrMeshIndexing::BatchedIncrMeshIndexing;

    void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
              const IndexDesc& desc, _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override {
        m_error_tracker = tracker;
    }
};

class BatchedSetMeshIndexingImpl : public BatchedSetMeshIndexing {
    void* m_error_tracker = nullptr;

public:
    using BatchedSetMeshIndexing::BatchedSetMeshIndexing;

    void exec(_megdnn_tensor_inout data, _megdnn_tensor_in value,
              const IndexDesc& desc, _megdnn_workspace workspace) override;

    void set_error_tracker(void* tracker) override {
        m_error_tracker = tracker;
    }
};

}  // namespace cuda
}  // namespace megdnn
