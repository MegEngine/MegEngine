/**
 * \file dnn/src/cuda/convpooling/opr_impl.h
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

namespace megdnn {
namespace cuda {

// This method is out-of-date.
// Use shared memory to store (a part of) the input data.
/*
void start_gpu_xcorr_pool_with_shared_mem(
        cudaStream_t stream,
        float *input,
        const float *kernel,
        float *output, 
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        size_t pool_shape, 
        PoolModeCu poolMode = AVERAGE,
        bool relu = true,
        const float *bias = NULL);
*/

class ConvPoolingForwardImpl final: public ConvPoolingForward {
    public:
        ConvPoolingForwardImpl(Handle *handle);
        void exec( const _megdnn_in TensorND src,
                   const _megdnn_in TensorND filter, 
                   const _megdnn_in TensorND bias,
                  _megdnn_out TensorND dst,
                  _megdnn_out Workspace workspace) override;
        void deduce_layout(
                const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                TensorLayout & dst) override;
        void check_layout(
                const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                TensorLayout & dst,
                size_t workspace_limit_in_bytes) override;
        size_t get_workspace_in_bytes(const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                const TensorLayout & dst) override;
};

} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen