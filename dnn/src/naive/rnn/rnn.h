/**
 * \file dnn/src/naive/rnn/rnn.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "megdnn/oprs/general.h"

namespace megdnn {
namespace naive {
namespace rnn {

class CellWeightsWrapperBase {
private:
    size_t _weight_size, _workspace_size;

public:
    TensorND weight_ih, weight_hh, bias_ih, bias_hh;
    // if no bias, will create dummy bias tensor from workspace
    CellWeightsWrapperBase(
            void* weight_ptr, size_t hidden_size, size_t input_size, size_t num_chunks,
            bool has_bias, DType dtype, _megdnn_workspace workspace);
    size_t weight_size_in_bytes() const;
    size_t workspace_size_in_bytes() const;
    static size_t backward_workspace_size_in_bytes(
            Handle* handle, size_t batch_size, size_t hidden_size, size_t input_size,
            size_t num_chunks, DType dtype);
    virtual void backward(
            Handle* handle, param::RNNCell::NonlineMode nonlineMode,
            _megdnn_tensor_in x, const TensorNDArray& states, _megdnn_tensor_in y,
            const TensorNDArray& douts, _megdnn_tensor_out dx, TensorNDArray& dstates,
            _megdnn_tensor_out dwi, _megdnn_tensor_out dwh, _megdnn_tensor_out dbias,
            _megdnn_workspace workspace) const;
    virtual size_t num_states() const;
    virtual ~CellWeightsWrapperBase() {}
};

class RNNCellWeightWrapper : public CellWeightsWrapperBase {
public:
    RNNCellWeightWrapper(
            void* weight_ptr, size_t hidden_size, size_t input_size, bool has_bias,
            DType dtype, _megdnn_workspace workspace);

    static size_t backward_workspace_size_in_bytes(
            Handle* handle, size_t batch_size, size_t hidden_size, size_t input_size,
            DType dtype);
};

class LSTMCellWeightWrapper : public CellWeightsWrapperBase {
public:
    LSTMCellWeightWrapper(
            void* weight_ptr, size_t hidden_size, size_t input_size, bool has_bias,
            DType dtype, _megdnn_workspace workspace);
    static size_t backward_workspace_size_in_bytes(
            Handle* handle, size_t batch_size, size_t hidden_size, size_t input_size,
            DType dtype);
    size_t num_states() const override;
    void backward(
            Handle* handle, param::RNNCell::NonlineMode nonlineMode,
            _megdnn_tensor_in x, const TensorNDArray& states, _megdnn_tensor_in y,
            const TensorNDArray& douts, _megdnn_tensor_out dx, TensorNDArray& dstates,
            _megdnn_tensor_out dwi, _megdnn_tensor_out dwh, _megdnn_tensor_out dbias,
            _megdnn_workspace workspace) const override;
};

}  // namespace rnn
}  // namespace naive
}  // namespace megdnn