/**
 * \file dnn/src/cuda/utils.h
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

#include "megcore_cdefs.h"
#include "src/common/utils.h"
#include "megdnn/handle.h"

#include "src/cuda/handle.h"
#include "src/cuda/utils.cuh"

#include "src/cuda/cudnn_with_check.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace megdnn {
namespace cuda {

static inline HandleImpl* concrete_handle(Handle* handle) {
    return static_cast<cuda::HandleImpl*>(handle);
}

static inline cudnnHandle_t cudnn_handle(Handle* handle) {
    return concrete_handle(handle)->cudnn_handle();
}

static inline cublasHandle_t cublas_handle(Handle* handle) {
    return concrete_handle(handle)->cublas_handle();
}

static inline cudaStream_t cuda_stream(Handle* handle) {
    return concrete_handle(handle)->stream();
}

static inline megcore::AsyncErrorInfo* async_error_info(Handle* handle) {
    return concrete_handle(handle)->megcore_context().error_info;
}

static inline void CUDART_CB callback_free(cudaStream_t /* stream */,
                                           cudaError_t status, void* userData) {
    cuda_check(status);
    free(userData);
}

//! get property of currently active device
const cudaDeviceProp& current_device_prop();

//! get property of device specified by device
const cudaDeviceProp* get_device_prop(int device);

//! check compute capability satisfied with given sm version
bool is_compute_capability_required(int major, int minor);

//! check compute capability equal to the given sm version
bool is_compute_capability_equalto(int major, int minor);

//! get the CUDNN_MAX_BATCH_X_CHANNEL_SIZE, it's just return the max size of the
//! third demension
size_t max_batch_x_channel_size();

//! get param buffer start address at cmem[0]
uint32_t param_buffer_start_address();

const char* current_device_arch_name();

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
