/**
 * \file dnn/src/cuda/dropout/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class DropoutDesc {
public:
    DropoutDesc() { cudnn_check(cudnnCreateDropoutDescriptor(&desc)); }
    ~DropoutDesc() { cudnn_check(cudnnDestroyDropoutDescriptor(desc)); }
    void set(
            cudnnHandle_t handle, void* status, size_t states_size_in_bytes,
            uint64_t seed, float drop_prob) {
        cudnn_check(cudnnSetDropoutDescriptor(
                desc, handle, drop_prob, status, states_size_in_bytes, seed));
    }
    void restore(
            cudnnHandle_t handle, float drop_prob, void* status,
            size_t states_size_in_bytes, uint64_t seed) {
#if CUDNN_VERSION >= 7000
        cudnn_check(cudnnRestoreDropoutDescriptor(
                desc, handle, drop_prob, status, states_size_in_bytes, 0));
#else
        // cudnnDropoutRestore is not support when cudnn version < 7000
        // so we set the dropoutDesc rather than restore
        cudnn_check(cudnnSetDropoutDescriptor(
                desc, handle, drop_prob, status, states_size_in_bytes, seed));
#endif
    }
    cudnnDropoutDescriptor_t desc;
};

class DropoutStatus {
    void* status;
    uint64_t status_size;
    uint64_t seed;
    float drop_prob;
    DropoutDesc desc;

public:
    DropoutStatus() {
        status = nullptr;
        status_size = 0;
    }
    ~DropoutStatus() {
        if (status != nullptr)
            cuda_check(cudaFree(status));
    }
    void set(cudnnHandle_t handle, uint64_t seed, float drop_prob) {
        this->seed = seed;
        this->drop_prob = drop_prob;
        cudnn_check(cudnnDropoutGetStatesSize(handle, &status_size));
        cuda_check(cudaMalloc(&status, status_size));
        desc.set(handle, status, status_size, seed, drop_prob);
    }
    void restore_desc(cudnnHandle_t handle) {
        desc.restore(handle, drop_prob, status, status_size, seed);
    }
    bool initialized() { return status != nullptr; }
    friend class DropoutForwardImpl;
    friend class DropoutBackwardImpl;
};

// similar to RNG operator, dropout operator also have status
class DropoutForwardImpl final : public DropoutForward {
    DropoutStatus dropout_status;

public:
    using DropoutForward::DropoutForward;
    void exec(
            _megdnn_tensor_in inp, _megdnn_tensor_out oup, _megdnn_tensor_out mask,
            _megdnn_workspace workspace) override;
    size_t get_mask_size_in_bytes(const TensorLayout& inp) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

class DropoutBackwardImpl final : public DropoutBackward {
#if CUDNN_VERSION >= 7000
    DropoutDesc op_desc;
#else
    // cudnnDropoutRestore is not support when cudnn version < 7000
    // so we need save the dropout status and set the dropoutDesc
    // rather than restore
    DropoutStatus dropout_status;
#endif

public:
    using DropoutBackward::DropoutBackward;
    void exec(
            _megdnn_tensor_in doup, _megdnn_tensor_in mask, _megdnn_tensor_out dinp,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
