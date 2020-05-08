/**
 * \file dnn/src/cuda/local/local.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace megdnn {
namespace cuda {
namespace local {

size_t forward_proxy_default_share_mem_in_bytes(size_t IH, size_t IW);

void forward_proxy_default(const float *src, const float *filter, float *dst,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        bool is_xcorr,
        cudaStream_t stream);

/// forward

bool can_forward_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW);

void forward_proxy_convnet(const float *src, const float *filter, float *dst,
        float *workspace,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs, // IN stride and ON stride
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        float *one, float *zero);

size_t get_workspace_in_floats_forward_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW);

/// bwd data

bool can_backward_data_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW);

void backward_data_proxy_convnet(const float *filter,
        const float *diff,
        float *grad,
        float *workspace,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs, // IN stride and ON stride
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        float *one, float *zero);

size_t get_workspace_in_floats_backward_data_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW);

/// bwd filter

bool can_backward_filter_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW);

void backward_filter_proxy_convnet(const float *src,
        const float *diff,
        float *grad,
        float *workspace,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs, // IN stride and ON stride
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        float *one, float *zero);

size_t get_workspace_in_floats_backward_filter_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW);

} // namespace local
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
